import os
import json
import librosa

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dictionary.dictionary_uk import PhonemeDictionaryUk
from preprocessor.preprocessor_uk import UkrainianProcessor

from dataloader.dataset import TextMelDataset
from dataloader.dataloader import collate_fn

from fastspeech.fast_speech import FastSpeech
from loss.fast_speech_loss import FastSpeechLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TTS():
    def __init__(self, config):
        self.dictionary = PhonemeDictionaryUk()
        self.preprocessor = UkrainianProcessor(self.dictionary)
        self.config = config

    def infer(self, text):
        sequence = self.preprocessor.preprocess(text)
        sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sequence = sequence.to(device)

        model = FastSpeech(self.config)
        model.to(device)

        with torch.no_grad():
            mel_output, _ = model(sequence)

        return mel_output

    def train(self):
        with open('./config/fast_speech.json', 'r') as f:
            config = json.load(f)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FastSpeech(config['model']).to(device)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss = FastSpeechLoss().to(device)

        with open(config["dataset"]["metadata_path"], 'r') as f:
            metadata = json.load(f)

        dictionary = PhonemeDictionaryUk()
        preprocessor = UkrainianProcessor(dictionary)

        dataset = TextMelDataset(preprocessor, config, metadata)
        dataloader = DataLoader(
            dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)

        num_steps = 70000
        global_step = 0
        log_interval = 1000
        save_interval = 10000

        model.train()
        while global_step < num_steps:
            total_loss = 0

            for batch in dataloader:
                if global_step >= num_steps:
                    break

                optimizer.zero_grad()

                text_seq = batch['sequence'].to(device)
                mel_target = batch['mel_spectrogram'].to(device)
                duration_target = batch['durations'].to(device)

                mel_pred, duration_pred = model(text_seq, mel_target.size(2))

                loss = loss(mel_pred, mel_target,
                            duration_pred, duration_target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                global_step += 1

                if global_step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    print(
                        f'Step [{global_step}/{num_steps}], Loss: {avg_loss:.4f}')
                    total_loss = 0

                if global_step % save_interval == 0:
                    torch.save(model.module.state_dict() if torch.cuda.device_count(
                    ) > 1 else model.state_dict(), f'fastspeech_step_{global_step}.pth')

        torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), 'fastspeech_final.pth')
