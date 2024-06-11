import re
import numpy as np

from .uk_ipa import ipa
from .preprocessor import Preprocessor

class UkrainianProcessor(Preprocessor):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def preprocess(self, text):
        normalized_text = self.normalize_text(text)

        tokens = self.tokenize(normalized_text)

        phonemes = self.dictionary.tokens_to_phonemes(tokens)

        sequence = self.dictionary.tokens_to_sequences(phonemes)
        
        return np.array(sequence)

    def normalize_text(self, text: str):
        text = text.lower()
        text = text.rstrip(r"""!"#$%&()*+,-./:;<=>?@[\]^_{|}~""")
        text = re.sub(r"[^а-щьюяґєіїa-zA-Z\s.,!?́]", "", text)
        return text

    def tokenize(self, text: str):
        tokens = re.findall(r'\w+|[.,!?]', text)
        return tokens
