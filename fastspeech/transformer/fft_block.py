import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_1d import Conv1DNet
from .multi_head_attention import MultiHeadAttention

class FFTBlock(nn.Module):
    def __init__(self, emb_dim: int, num_head: int, h_dim: int, inner_dim: int, dropout: float = 0.1):
        super(FFTBlock, self).__init__()
        self.attn_out = MultiHeadAttention(num_head, emb_dim, h_dim, dropout=dropout)
        self.conv_out = Conv1DNet(emb_dim, inner_dim, dropout=dropout)
        
    def forward(self, input: torch.Tensor, non_pad_mask: torch.Tensor, attn_mask: torch.Tensor):
        non_pad_mask = non_pad_mask.float()
        
        output = self.attn_out(input, input, input, attn_mask) * non_pad_mask

        output = self.conv_out(output) * non_pad_mask

        return output

