import torch
import torch.nn as nn


class PostNet(nn.Module):
    def __init__(self, num_mel_bins, postnet_embedding_dim=512, postnet_kernel_size=5, postnet_num_layers=5):
        super(PostNet, self).__init__()

        layers = []
        for i in range(postnet_num_layers):
            in_channels = num_mel_bins if i == 0 else postnet_embedding_dim
            out_channels = num_mel_bins if i == postnet_num_layers - 1 else postnet_embedding_dim
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, postnet_kernel_size, padding=(
                        postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(out_channels),
                    nn.Tanh() if i < postnet_num_layers - 1 else nn.Identity()
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
