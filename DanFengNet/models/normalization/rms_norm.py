#!/usr/bin/env python
# -*- coding:utf-8 -*-
from torch import nn
import torch


class RMSNorm(nn.Module):
    # root-mean-square layer normalization
    # RMSNorm = x * weight / (sqrt(mean(x^2) + eps))
    def __init__(self, hidden_dim, eps=1e-6, bias=False):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(dtype=torch.float32, device=x.device)
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps)
        return self.weight * output.to(input_dtype)
