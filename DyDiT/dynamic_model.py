import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from easydict import EasyDict

from torch.nn import functional as F


def convert_list_to_tensor(list_convert):
    if len(list_convert):
        result = torch.stack(list_convert, dim=1)
    else :
        result = None
    return result 

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
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
        # Difference of two` gumbels because we apply a sigmoid
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


def gumbel_softmax(logits, tau=5.0, dim = -1):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    
    return y_soft


class TokenSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.mlp_head = nn.Sequential(nn.Linear(dim_in, dim_in // 16, bias=bias),
                                      nn.ReLU(),
                                      nn.Linear(dim_in // 16, 1, bias=bias),
                                      )

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, x):
        b, l = x.shape[:2]
        # x = x.mean(dim=1)
        logits = self.mlp_head(x)
        
        token_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        # token_select = token_select.unsqueeze(dim=1)

        
        return token_select, logits
    


class STE_Min(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in1, x_in2, x_in3=math.inf):
        x = min(x_in1, x_in2, x_in3)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return None, g, g
    
class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None
    
    
ste_ceil = STE_Ceil.apply
ste_min = STE_Min.apply

class Router(nn.Module):
    def __init__(self, dim=768, channel_number=196, tau=5, is_hard=True, threshold=0.5) -> None:

        super().__init__()
        # self.channel_number = channel_number
        self.dim = dim
        self.channel_number = channel_number
        self.router = nn.Linear(dim, channel_number)
        
        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold
        
    def forward(self, x):
        # b, l = x.shape[:2]
        # x = x.mean(dim=1)
        
        logits = self.router(x)
        channel_select = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)

        
        return channel_select
        
