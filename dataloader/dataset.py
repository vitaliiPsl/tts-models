import os
import json
import librosa

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessor.preprocessor_uk import UkrainianProcessor

class TextMelDataset(Dataset):
    def __init__(self, preprocessor, config, metadata):
        self.preprocessor = preprocessor

        self.metadata = metadata
        self.audio_dir = config["dataset"]["audio_path"]
        self.durations_dir = config["dataset"]["duration_path"]
        
        self.sr = config["training"]["sr"]
        self.n_mels = config["training"]["n_mels"]
        self.n_fft = config["training"]["n_fft"]
        self.hop_length = config["training"]["hop_length"]
        self.win_length = config["training"]["win_length"]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        entry = self.metadata[idx]
        filename = entry['file']

        audio_path = os.path.join(self.audio_dir, filename)
        durations_path = os.path.join(self.durations_dir, os.path.splitext(filename)[0] + '.json')

        audio = self._load_audio(audio_path, sr=self.sr)
        mel_spectrogram = self._compute_mel_spectrogram(
            audio, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

        sequence, durations = self._load_phonemes(durations_path)

        return {
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'durations': torch.tensor(durations, dtype=torch.float),
            'mel_spectrogram': torch.tensor(mel_spectrogram, dtype=torch.float),
        }

    def _load_audio(self, audio_path, sr=22050):
        audio, _ = librosa.load(audio_path, sr=sr)
        return audio
    
    def _load_phonemes(self, durations_path):
        with open(durations_path, 'r', encoding='utf-8') as f:
            duration_data = json.load(f)
        
        phonemes = [d['phoneme'] if d['phoneme'] else 'sp' for d in duration_data]
        durations = np.array([d['duration'] for d in duration_data])

        sequence = self.preprocessor.dictionary.tokens_to_sequences(phonemes)

        return sequence, durations

    def _compute_mel_spectrogram(self, audio, sr=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return mel_spectrogram
