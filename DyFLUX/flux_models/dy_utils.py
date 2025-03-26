import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from torch.nn import functional as F

from einops import rearrange, reduce, repeat
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU

def _gumbel_sigmoid(logits, tau=1, hard=False, eps=1e-10, training=True, threshold=0.5):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        
        # print(f'tau = {tau}')
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret



class TokenSelect_baesd_on_x_t_text(nn.Module):
    def __init__(self, dim_in, conditioning_dim, is_hard=True, threshold=0.5,
                 elementwise_affine: bool = True,
                 eps: float = 1e-5,):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(conditioning_dim, dim_in * 4)
        self.norm = nn.LayerNorm(dim_in, eps=eps, elementwise_affine=elementwise_affine)
        self.full_dim_in = dim_in
        self.router = nn.Linear(dim_in, 1, bias=True)

        self.is_hard = is_hard
        self.threshold = threshold
        
        

    def forward(self, x, temb, text, tau=5.0):
        # if self.training or not is_single_block:
        b, l, c = x.shape
        l_text = text.shape[1]
                
        shift, scale, shift_text, scale_text = self.linear_t(self.silu(temb)).chunk(4, dim=1)     
        
        # if self.training or not is_single_block:
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        text = self.norm(text) * (1 + scale_text)[:, None, :] + shift_text[:, None, :]
            
        text = text.mean(dim=1).unsqueeze(1)
        
        logits = self.router(x + text)
        
        token_select_mask = _gumbel_sigmoid(logits, tau, self.is_hard, threshold=self.threshold, training=self.training)
        
        return token_select_mask


class DynaLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, head_dim, bias=True, dyna_dim=[False, False]):
        super(DynaLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.dyna_dim = dyna_dim
        self.static_flops_per_token = self.in_features_max * self.out_features_max
    
    def forward(self, input_x, channel_mask=None):
        # if self.training or channel_mask is None:
        # width_mult = channel_mask.mean() if channel_mask is not None else 1.0        
        # n_tokens = input_x.shape[1]
        # flops_static = n_tokens * self.in_features_max * self.out_features_max
        # flops = flops_static #* width_mult**sum(self.dyna_dim)
        
        out = super().forward(input_x)

        return out
            
    def forward_inference(self, input_x, channel_mask=None):
        if channel_mask is None:
            return self.forward(input_x, channel_mask=None)
        else:
            print("For now, we only release Dyn-FLUX with token-dynamic mode.")
            raise NotImplementedError

class DynaLinear_FluxSingleAttnOut(nn.Linear):
    def __init__(self, in_features, out_features, num_heads, head_dim, bias=True, dyna_dim=[True, False]):
        super(DynaLinear_FluxSingleAttnOut, self).__init__(
            in_features, out_features, bias=bias)
        self.in_features_max = in_features
        self.out_features_max = out_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        self.dyna_dim = dyna_dim
        self.static_flops_per_token = self.in_features_max * self.out_features_max
    
    
    def forward(self, tokens, token_len=1024, channel_mask_attn=None, channel_mask_ffn=None):        
        # For now, we release Dy-FLUX with token-dynamic mode.
        c = self.out_features_max
        
        weight = self.weight
        bias = self.bias
        
        tokens = F.linear(tokens, weight, bias)

        # flops_stat = token_len * c**2 * 5
        # flops_dyn = flops_stat
        return tokens
        

class DynGELU(GELU):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dyna_dim=[False, False]):
        super().__init__(dim_in=dim_in, dim_out=dim_out, approximate=approximate, bias=bias)
        
        self.dim_in_max = dim_in
        self.dim_out_max = dim_out
        self.dyna_dim = dyna_dim
        self.static_flops_per_token = dim_in * dim_out
    
    def forward(self, hidden_states, channel_mask=None):
        # width_mult = channel_mask.mean() if channel_mask is not None else 1.0
        # in_features, out_features = self.dim_in_max, self.dim_out_max    
        # n_tokens = hidden_states.shape[1]
        # flops_static = n_tokens * out_features * in_features
        # flops = flops_static #* width_mult**sum(self.dyna_dim)
        
        
        hidden_states = super().forward(hidden_states)
            
        return hidden_states
            
    def forward_inference(self, hidden_states, channel_mask=None):
        if channel_mask is None:
            return self.forward(hidden_states, channel_mask=None)
        else:
            print("For now, we only release Dyn-FLUX with token-dynamic mode.")
            raise NotImplementedError
         

class DynFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        
        num_heads: int = 1,
        head_dim: int = 64
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = DynGELU(dim, inner_dim, bias=bias, dyna_dim=[False, True])
        if activation_fn == "gelu-approximate":
            act_fn = DynGELU(dim, inner_dim, approximate="tanh", bias=bias, dyna_dim=[False, True])
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        # self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        self.net.append(DynaLinear(in_features=inner_dim, out_features=dim_out, num_heads=num_heads, head_dim=head_dim, bias=bias, dyna_dim=[True, False]))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, 
                      channel_mask: torch.Tensor, 
                      *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        
        # 0: DynGELU, 1: Dropout, 2: DynaLinear
        for i, module in enumerate(self.net):
            if i == 0 or i == 2:
                hidden_states = module(hidden_states, channel_mask)
            else:
                hidden_states = module(hidden_states)
                
        return hidden_states