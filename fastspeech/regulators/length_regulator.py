import numpy as np
import torch
import torch.nn as nn

class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.slowdown = 0.5

    def forward(self, sequences: torch.Tensor, durations: torch.Tensor, max_mel_length: int = None):
        durations = (durations + self.slowdown).clamp(min=1).int()

        expanded_sequences = self.expand_sequences(sequences, durations)
        
        if max_mel_length is None:
            max_mel_length = max([sequence.size(0)for sequence in expanded_sequences])
        padded_seq = self.pad_sequences(expanded_sequences, max_mel_length)

        return padded_seq, durations

    def expand_sequences(self, sequences: torch.Tensor, durations: torch.Tensor):
        batch_size, seq_len, enc_dim = sequences.size()

        max_duration = durations.sum(dim=1).max().item()
        expanded_sequences = torch.zeros(batch_size, max_duration, enc_dim, device=sequences.device)

        for i in range(batch_size):
            pos = 0
            for j in range(seq_len):
                token = sequences[i, j]
                duration = durations[i, j].item()
                if duration > 0:
                    expanded_sequences[i, pos:pos+duration] = token
                    pos += duration

        return expanded_sequences

    def pad_sequences(self, sequences, max_mel_length):
        padded_sequences = []

        for seq in sequences:
            seq_len, enc_dim = seq.size()
            if seq_len < max_mel_length:
                padding = torch.zeros(
                    (max_mel_length - seq_len, enc_dim), device=seq.device)
                padded_seq = torch.cat([seq, padding], dim=0)
            else:
                padded_seq = seq[:max_mel_length]
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)
