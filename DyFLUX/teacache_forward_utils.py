import torch
# from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from typing import Any, Dict, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, scale_lora_layers, unscale_lora_layers

def teacache_forward(
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
    assert self.enable_teacache == True

    inp = hidden_states.clone()
    temb_ = temb.clone()
    modulated_inp, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.transformer_blocks[0].norm1(inp, emb=temb_)
    if self.cnt == 0 or self.cnt == self.num_steps - 1:
        should_calc = True
        self.accumulated_rel_l1_distance = 0
    else:
        # coefficients = [4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01]
        # rescale_func = np.poly1d(coefficients)
        # self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())

        self.accumulated_rel_l1_distance += ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
        if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
            should_calc = False
        else:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
    self.previous_modulated_input = modulated_inp  
    self.cnt += 1
    if self.cnt == self.num_steps:
        self.cnt = 0


    # TODO: calculate head flops
    flops_dyn, flops_stat, flops_dyn_attn, flops_stat_attn, flops_dyn_mlp, flops_stat_mlp = 0, 1e-10, 0, 1e-10, 0, 1e-10

    if not should_calc:
        hidden_states += self.previous_residual
    else:
        ori_hidden_states = hidden_states.clone()
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
        self.previous_residual = hidden_states - ori_hidden_states

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)


    if not return_dict:
        return output

    return Transformer2DModelOutput(sample=output)
