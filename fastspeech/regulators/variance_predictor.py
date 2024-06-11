import numpy as np
import torch
import torch.nn as nn


# class VariancePredictor(nn.Module):
#     def __init__(self, inp_dim, inner_dim, kernel_size, padding_size, dropout=0.1):
#         super(VariancePredictor, self).__init__()
        
#         self.conv1 = nn.Conv1d(inp_dim, inner_dim, kernel_size, padding=padding_size)
#         self.layer_norm1 = nn.LayerNorm(inner_dim)
        
#         self.conv2 = nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding_size)
#         self.layer_norm2 = nn.LayerNorm(inner_dim)

#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
        
#         self.linear = nn.Linear(inner_dim, 1)


#     def forward(self, x: torch.Tensor):   
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.layer_norm1(x)
#         x = self.dropout(x)

#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.layer_norm2(x)
#         x = self.dropout(x)
        
#         x = self.linear(x)        

#         return x.squeeze(-1)
    
class VariancePredictor(nn.Module):
    def __init__(self, inp_dim: int, inner_dim: int, kernel_size: int, padding_size: int, dropout: float = 0.1):
        super(VariancePredictor, self).__init__()

        self.conv1d_1 = nn.Conv1d(
            inp_dim, inner_dim, kernel_size, padding=padding_size)
        self.layer_norm_1 = nn.LayerNorm(inner_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.conv1d_2 = nn.Conv1d(
            inner_dim, inner_dim, kernel_size, padding=padding_size)
        self.layer_norm_2 = nn.LayerNorm(inner_dim)
        self.linear_layer = nn.Linear(inner_dim, 1)

    def forward(self, encoder_output: torch.Tensor):

        x = encoder_output.contiguous().transpose(1, 2)
        x = self.conv1d_1(x)
        x = x.contiguous().transpose(1, 2)
        x = self.relu(self.layer_norm_1(x))
        x = self.dropout(x)
        
        x = x.contiguous().transpose(1, 2)
        x = self.conv1d_2(x)
        x = x.contiguous().transpose(1, 2)
        x = self.relu(self.layer_norm_2(x))
        x = self.dropout(x)
        
        out = self.relu(self.linear_layer(x))

        out = out.squeeze(-1)

        return out
