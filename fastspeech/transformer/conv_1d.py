import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1DNet(nn.Module):
    '''
    1D convolutional network with residual connection and layer normalization. Used in the FFT block of the encoder and decoder.

    :param inp_dim: Input dimension.
    :type inp_dim: int
    :param inner_dim: Inner dimension of the convolutional layers (output dimension of the first convolutional layer).
    :type inner_dim: int
    :param dropout: Dropout probability. Default is 0.1.
    :type dropout: float, optional
    '''

    def __init__(self, inp_dim: int, inner_dim: int, dropout: float = 0.1):
        super(Conv1DNet, self).__init__()
        self.conv1 = nn.Conv1d(inp_dim, inner_dim, kernel_size=9, padding=4)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(inner_dim, inp_dim, kernel_size=1, padding=0)
        self.layer_norm = nn.LayerNorm(inp_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
       Pass given input through the 1D convolutional network. The input comes from the Multi-Head Attention module.

       :param x: Input tensor of shape [batch_size, seq_len, d_in].
       :type x: torch.Tensor
       :returns: Output tensor of shape [batch_size, seq_len, d_in].
       :rtype: torch.Tensor
       '''
        residual = x
        
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        
        x = self.conv2(x)
        x = x.transpose(1, 2)
        
        output = self.dropout(x)
        output = self.layer_norm(output + residual)

        return output
