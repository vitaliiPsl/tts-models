import json
from .uk_ipa import ipa

_phonemes = ['b', 'bʲ', 'bʲː', 'bː', 'c', 'cː', 'd', 'dzʲ', 'dʒ', 'dʲ', 'dʲː', 'd̪', 'd̪z̪', 'd̪z̪ː', 'd̪ː', 'd͡z', 'd͡zʲ', 'd͡ʒ', 'e', 'f', 'fʲ', 'fː', 'i', 'iˈ', 'iː', 'i̯', 'i̯ˈ', 'j', 'k', 'kʲ', 'kː', 'l', 'lʲ', 'lʲˈ', 'lˈ', 'lː', 'm', 'mʲ', 'mʲː', 'mˈ', 'n', 'nʲ', 'nʲˈ', 'nˈ', 'n̪', 'n̪ː', 'o', 'oˈ', 'p', 'pʲ', 'pʲː', 'r', 'rʲ', 'rˈ', 's', 'spn', 'sʲ', 'sʲt͡sʲ', 'sʲː', 's̪', 's̪ː', 't', 'tsʲ', 'tsʲː', 'tʃ',
             'tʃʲ', 'tʃʲː', 'tʃː', 'tʲ', 'tʲː', 't̪', 't̪s̪', 't̪s̪ː', 't̪ː', 't͡s', 't͡sʲː', 't͡ʃʃ', 't͡ʃː', 'u', 'uˈ', 'u̯', 'u̯ʲ', 'u̯ˈ', 'x', 'xʲ', 'z', 'zʲ', 'zʲt͡sʲ', 'zʲː', 'z̪', 'z̪ː', 'ç', 'ɐ', 'ɐˈ', 'ɑ', 'ɑˈ', 'ɔ', 'ɔʲ', 'ɔˈ', 'ɛ', 'ɛˈ', 'ɟ', 'ɡ', 'ɡʲ', 'ɦ', 'ɦʲ', 'ɪ', 'ɪˈ', 'ɲ', 'ɲː', 'ɾ', 'ɾʲ', 'ɾʲː', 'ɾː', 'ʃ', 'ʃt͡ʃ', 'ʃʲ', 'ʊ', 'ʊˈ', 'ʋ', 'ʋʲ', 'ʋʲː', 'ʋː', 'ʎ', 'ʎː', 'ʒ', 'ʒd͡ʒ', 'ʒt͡s', 'ʒʲ', 'ʒʲː', 'ʝ', 'ʲ']
_silences = ['sp', 'eps', 'unk']


class PhonemeDictionaryUk:
    def __init__(self):
        with open('dictionary/ukrainian_dictionary.json', 'r', encoding='utf-8') as file:
            self.phoneme_dict = json.load(file)

        self.dictionary = _phonemes + _silences
        self.phoneme_to_id = {s: i+1 for i, s in enumerate(self.dictionary)}

    def tokens_to_phonemes(self, tokens):
        phonemes = []
        for token in tokens:
            if token in ['.', ',', '!', '?']:
                phonemes += ['sp']
            elif token in self.phoneme_dict:
                phonemes.extend(self.phoneme_dict[token])
            else:
                phonemes.extend(ipa(token))

        return phonemes

    def tokens_to_sequences(self, phonemes):
        sequence = []
        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                sequence += [self.phoneme_to_id[phoneme]]
            else:
                raise Exception(f'Phoneme "{phoneme}" not found')

        return sequence
