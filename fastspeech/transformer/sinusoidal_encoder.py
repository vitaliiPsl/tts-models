import numpy as np
import torch

class SinusoidEncodingTable:
    def __init__(self, max_seq_len, inp_dim, padding_idx=None):
        self.max_seq_len = max_seq_len + 1
        self.inp_dim = inp_dim
        self.padding_idx = padding_idx
        
        self.sinusoid_table = self.build_table()

    def build_table(self):
        inds = np.arange(self.inp_dim) // 2
        div_term = np.power(10000, 2 * inds / self.inp_dim)

        positions = np.arange(self.max_seq_len)

        sinusoid_table = np.outer(positions, 1/div_term)

        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  

        if self.padding_idx is not None:
            sinusoid_table[self.padding_idx] = 0.

        return torch.Tensor(sinusoid_table)