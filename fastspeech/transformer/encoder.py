import numpy as np
import torch
import torch.nn as nn

from .sinusoidal_encoder import SinusoidEncodingTable
from .fft_block import FFTBlock

class Encoder(nn.Module):
    def __init__(self, vocab_dim: int, max_seq_len: int, emb_dim: int, num_layer: int, num_head: int, h_dim: int, d_inner: int, dropout: float):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_dim, emb_dim, padding_idx=0)

        table = SinusoidEncodingTable(max_seq_len, emb_dim, padding_idx=0).sinusoid_table
        self.position_encoding = nn.Embedding.from_pretrained(table, freeze=True)

        self.layer_stack = nn.ModuleList(
            [ FFTBlock(emb_dim, num_head, h_dim, d_inner, dropout=dropout) for _ in range(num_layer) ]
        )

    def forward(self, inp_seq: torch.Tensor):
        '''
        :param inp_seq: input tensor of shape [batch_size, seq_length].
        :type input: torch.Tensor
        '''
        batch_size, inp_seq_length = inp_seq.size()
        
        # [batch_size, seq_length, 1]
        non_pad_mask = (inp_seq.unsqueeze(2) != 0)
        
        # [batch_size, seq_length, seq_length]
        attn_mask = (inp_seq.unsqueeze(1) == 0).repeat(1, inp_seq.size(1), 1)
        
        # [batch_size, seq_length]
        inp_seq_inds = torch.arange(0, inp_seq_length).unsqueeze(0).expand(batch_size, -1)
        
        # [batch_size, seq_length, emb_dim(256)]
        output = self.embedding(inp_seq) + self.position_encoding(inp_seq_inds)
        
        # [batch_size, seq_length, emb_dim(256)]
        for layer in self.layer_stack:
            output = layer(output, non_pad_mask=non_pad_mask, attn_mask=attn_mask)

        return output
