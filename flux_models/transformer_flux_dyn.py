# Copyright 2024 Black Forest Labs, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from math import e
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
# from diffusers.models.attention import FeedForward
# from diffusers.models.attention_processor import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from diffusers.models.modeling_outputs import Transformer2DModelOutput


from .attention_processor_dyn import Attention, FluxAttnProcessor2_0, FluxSingleAttnProcessor2_0

from .dy_utils import TokenSelect_baesd_on_x_t_text, DynFeedForward, DynaLinear, DynaLinear_FluxSingleAttnOut

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# YiYi to-do: refactor rope related functions/classes
def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


# YiYi to-do: refactor rope related functions/classes
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0,
                do_token_select=True):
        super().__init__()
        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.token_selection = TokenSelect_baesd_on_x_t_text(dim_in=dim, conditioning_dim=dim) if do_token_select else None

        # self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.proj_mlp = DynaLinear(dim, self.mlp_hidden_dim,
                                   num_heads=int(num_attention_heads * mlp_ratio), head_dim=attention_head_dim, 
                                   dyna_dim=[False, True])
        self.act_mlp = nn.GELU(approximate="tanh")
        # self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
        self.proj_out = DynaLinear_FluxSingleAttnOut(dim + self.mlp_hidden_dim, dim,
                                   num_heads=int(num_attention_heads * mlp_ratio), head_dim=attention_head_dim, 
                                   dyna_dim=[True, False])

        self.do_token_select = do_token_select

        processor = FluxSingleAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,

        tau=5.0,
        text_len=256,
        img_len=1024
    ):
        residual = hidden_states

        ff_output_img = torch.zeros((hidden_states.shape[0] * img_len, hidden_states.shape[-1]), device=hidden_states.device, dtype=hidden_states.dtype)


        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        attn_output=self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,

                channel_mask_attn=None
            )


        if self.do_token_select:
            token_mask=self.token_selection(hidden_states[:, text_len:, :], temb, hidden_states[:, :text_len, :], tau=tau) 
            token_act_rate = token_mask.mean()
        else:
            token_mask = None
            token_act_rate = 1.0

        text, img = norm_hidden_states[:, :text_len, :], norm_hidden_states[:, text_len:, :]
        
        text=self.proj_mlp(text, channel_mask=None)
        text = self.act_mlp(text)
        

        text = torch.cat([attn_output[:, :text_len, :], text], dim=2)
        text = self.proj_out(text, text_len)

        b_, n_, c_ = img.shape

        token_mask = token_mask.reshape(-1, *token_mask.shape[2:])
        token_idx = token_mask.nonzero()[:, 0]

        if token_idx.numel() > 0:
            img = img.reshape(-1, *img.shape[2:])
            img=self.proj_mlp.forward_inference(img[token_idx, :], channel_mask=None)
            img = self.act_mlp(img)

            img_attn_out = attn_output[:, text_len:, :]
            img_attn_out = img_attn_out.reshape(-1, *img_attn_out.shape[2:])
            img_attn_out = img_attn_out[token_idx, :]

            img = torch.cat([img_attn_out, img], dim=1)

            img_token_len = int(token_act_rate * n_)  # Yizeng: n_是静态img_len
            img = self.proj_out(img, img_token_len)
        
            ff_output_img[token_idx, :] = img
        
        
        ff_output_img = ff_output_img.reshape(hidden_states.shape[0], img_len, hidden_states.shape[-1])
        hidden_states = torch.cat([text, ff_output_img], dim=1)
        
        gate = gate.unsqueeze(1)
        hidden_states = gate * hidden_states
        hidden_states = residual + hidden_states    # Yizeng: 是否fuse scatter-add有待ablation
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6,
                 do_token_select=True):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        
        self.token_selection = TokenSelect_baesd_on_x_t_text(dim_in=dim, conditioning_dim=dim) if do_token_select else None
        self.do_token_select = do_token_select

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )
        
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = DynFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", 
                                 num_heads=num_attention_heads, 
                                 head_dim=attention_head_dim)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = DynFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate",
                                         num_heads=num_attention_heads, 
                                         head_dim=attention_head_dim)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        tau=5.0
    ):

        # Adaptive layer norm
        batch_size, seq_len_img, hidden_dim = hidden_states.shape
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        seq_len_text = encoder_hidden_states.shape[1]
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,

            channel_mask_attn=None
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        # Process attention outputs for the `encoder_hidden_states`.
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        # Till now, attn layer output is ready


        # Yizeng: Token selection before norm2, using separate adaln layers
        if self.do_token_select:
            token_mask=self.token_selection(hidden_states, temb, encoder_hidden_states, tau=tau)
        else:
            token_mask = None
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        # Feed-forward.
        # img

        # Yizeng: FFN，c -> 4c -> c
        b_, n_, c_ = norm_hidden_states.shape
        ff_output = torch.zeros_like(norm_hidden_states, device=norm_hidden_states.device, dtype=norm_hidden_states.dtype)
        token_mask = token_mask.reshape(-1, *token_mask.shape[2:])
        token_idx = token_mask.nonzero()[:, 0]
        # print(f"In a DoubleBlock, token_idx.numel() = {token_idx.numel()}, act_rate = {token_act_rate}")
        if token_idx.numel() > 0:
            ff_output = ff_output.reshape(-1, *ff_output.shape[2:])                            # 合并前2维
            norm_hidden_states = norm_hidden_states.reshape(-1, *norm_hidden_states.shape[2:])
            
            norm_hidden_states = self.ff(norm_hidden_states[token_idx, :], channel_mask=None)
            
            ff_output[token_idx, :] = norm_hidden_states
            
            ff_output = ff_output.reshape(b_, n_, c_)
        
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        
        
        # text
        # norm2
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        # ffn
        context_ff_output = self.ff_context(norm_encoder_hidden_states, channel_mask=None)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class DynFluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],

        do_token_select: bool = True
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = EmbedND(dim=self.inner_dim, theta=10000, axes_dim=axes_dims_rope)
        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,

                    do_token_select=do_token_select
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,

                    do_token_select=do_token_select
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.init_routers()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


    def init_routers(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and "router" in name:
                # print(f"init router: {name}")
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 5.0)
        # print("init routers done!!!")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,

        tau=5.0,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,

                tau=tau,
            )


        text_len = encoder_hidden_states.shape[1]
        img_len = hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,

                tau=tau,
                text_len=text_len,
                img_len=img_len,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return output

        return Transformer2DModelOutput(sample=output)
