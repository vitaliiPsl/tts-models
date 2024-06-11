import numpy as np
import torch
import torch.nn as nn

from .sinusoidal_encoder import SinusoidEncodingTable
from .fft_block import FFTBlock

class Decoder(nn.Module):
    def __init__(self, max_seq_len: int, emb_dim: int, num_layer: int, num_head: int, h_dim: int, d_inner: int, mel_num: int, dropout: float):
        super(Decoder, self).__init__()

        table = SinusoidEncodingTable(max_seq_len, emb_dim, padding_idx=0).sinusoid_table
        self.position_encoding = nn.Embedding.from_pretrained(table, freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            emb_dim, num_head, h_dim, d_inner, dropout=dropout) for _ in range(num_layer)])

        self.linear = nn.Linear(emb_dim, mel_num)

    def forward(self, inp_seq: torch.Tensor):
        '''
        :param inp_seq: input tensor of shape [batch_size, seq_length, emb_dim].
        :type input: torch.Tensor
        '''
        batch_size, seq_len, emb_dim = inp_seq.size()

        # [batch_size, seq_len, 1]
        non_pad_mask = (inp_seq.sum(dim=-1, keepdim=True) != 0)

        # [batch_size, seq_len, seq_len]
        attn_mask = (inp_seq.sum(dim=-1).unsqueeze(1) == 0).repeat(1, seq_len, 1)

        # [batch_size, seq_len]
        pos_indices = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, -1).to(inp_seq.device)

        output = inp_seq + self.position_encoding(pos_indices)
                
        for layer in self.layer_stack:
            output = layer(output, non_pad_mask=non_pad_mask,
                           attn_mask=attn_mask)

        output = self.linear(output)
    
        return output.transpose(1, 2)
