import torch

def collate_fn(batch):
    def pad_tensor(tensor, length, dim=0):
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat([tensor, torch.zeros(*pad_size, dtype=tensor.dtype)], dim=dim)

    max_seq_len = max([len(item['sequence']) for item in batch])
    max_mel_len = max([item['mel_spectrogram'].size(1) for item in batch])

    for item in batch:
        item['sequence'] = pad_tensor(item['sequence'], max_seq_len, dim=0)
        item['durations'] = pad_tensor(item['durations'], max_seq_len, dim=0)
        item['mel_spectrogram'] = pad_tensor(item['mel_spectrogram'], max_mel_len, dim=1)

    sequences = torch.stack([item['sequence'] for item in batch])
    durations = torch.stack([item['durations'] for item in batch])
    mel_spectrograms = torch.stack([item['mel_spectrogram'] for item in batch])

    return {
        'sequence': sequences,
        'durations': durations,
        'mel_spectrogram': mel_spectrograms
    }
