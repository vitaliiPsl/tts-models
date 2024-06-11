import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, emb_dim, h_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_head = num_head
        self.h_dim = h_dim
        self.head_dim = h_dim // num_head

        self.Wq = nn.Linear(emb_dim, h_dim)
        self.Wk = nn.Linear(emb_dim, h_dim)
        self.Wv = nn.Linear(emb_dim, h_dim)

        self.softmax = nn.Softmax(dim=2)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(h_dim, emb_dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        residual = q
        batch_size, num_head, seq_len, head_dim = q.size(0), self.num_head, q.size(1), self.head_dim
        Q, K, V = self.Wq(q), self.Wk(k), self.Wv(v)
        
        review = lambda X: X.view(batch_size, seq_len, num_head, head_dim).transpose(1, 2).contiguous()
        Q, K, V = review(Q), review(K), review(V)
        
        reshape = lambda X: X.view(batch_size * num_head, seq_len, head_dim)
        Q, K, V = reshape(Q), reshape(K), reshape(V)
        
        mask = mask.repeat(num_head, 1, 1)
        
        a = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        
        a = a.masked_fill(mask, -np.inf)
        
        a = self.softmax(a)
        
        a = self.dropout(a)
        
        output = torch.bmm(a, V)
        
        output = output.view(batch_size, num_head, seq_len, head_dim).transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, num_head * head_dim)
        
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        return output
        
        
