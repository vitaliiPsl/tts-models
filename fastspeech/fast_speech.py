import numpy as np
import torch
import torch.nn as nn

from .transformer.encoder import Encoder
from .transformer.decoder import Decoder
from .regulators.variance_adaptor import VarianceAdaptor
from .postnet.postnet import PostNet

class FastSpeech(nn.Module):
    def __init__(self, config):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(
            vocab_dim=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            emb_dim=config["enc_emb_dim"],
            num_layer=config["enc_num_layer"],
            num_head=config["enc_num_head"],
            h_dim=config["enc_emb_dim"],
            d_inner=config["enc_1d_filter_size"],
            dropout=config["dropout_prob"]
        )

        self.variance_adaptor = VarianceAdaptor(config)

        self.decoder = Decoder(
            max_seq_len=config["max_seq_len"],
            emb_dim=config["dec_emb_dim"],
            num_layer=config["dec_num_layer"],
            num_head=config["dec_num_head"],
            h_dim=config["dec_emb_dim"],
            d_inner=config["dec_1d_filter_size"],
            mel_num=config["mel_num"],
            dropout=config["dropout_prob"]
        )

        self.postnet = PostNet(
            num_mel_bins=config["mel_num"],
            postnet_embedding_dim=config.get("postnet_embedding_dim", 512),
            postnet_kernel_size=config.get("postnet_kernel_size", 5),
            postnet_num_layers=config.get("postnet_num_layers", 5)
        )

    def forward(self, text_seq: torch.Tensor, max_mel_length: int = None):
        # [batch_size, seq_length, emb_dim]
        encoder_output = self.encoder(text_seq)   
        
        # [batch_size, expanded_seq_length, emb_dim], [batch_size, seq_length]
        lr_output, durations_output = self.variance_adaptor(encoder_output, max_mel_length)
        
        # [batch_size, predicted_length, mel_num]
        mel_spectrogram = self.decoder(lr_output)
        postnet_output = self.postnet(mel_spectrogram)
        mel_spectrogram_post = mel_spectrogram + postnet_output

        return mel_spectrogram_post, durations_output
