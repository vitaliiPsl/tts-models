import torch
import torch.nn as nn

from .length_regulator import LengthRegulator
from .variance_predictor import VariancePredictor


class VarianceAdaptor(nn.Module):
    def __init__(self, config):
        super(VarianceAdaptor, self).__init__()

        self.duration_predictor = VariancePredictor(
            inp_dim=config["enc_emb_dim"],
            inner_dim=config["inner_dim"],
            kernel_size=config["dp_kernel_size"],
            padding_size=config["dp_padding"],
            dropout=config["dropout_prob"]
        )

        self.length_regulator = LengthRegulator()

    def forward(self, x: torch.Tensor, max_mel_length: int = None):
        duration_output = self.duration_predictor(x)

        lr_output, durations = self.length_regulator(x, duration_output, max_mel_length)

        return lr_output, durations
