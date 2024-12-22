from torch import Tensor
import torch


class CustomTokenizer():
    def __init__(self, vocab=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'x', ',', '<pad>', '<unk>']):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char,
                            idx in self.char_to_idx.items()}

    def encode(self, input_sequence):
        split_on_padding = input_sequence.split("<pad>")
        tokens = []
        attention_mask = []
        for unpadded_seq in split_on_padding:
            if not unpadded_seq:
                tokens.append(self.char_to_idx["<pad>"])
                attention_mask.append(0)
            else:
                if tokens:
                    tokens.append(self.char_to_idx["<pad>"])
                    attention_mask.append(0)
                for char in unpadded_seq:
                    tokens.append(self.char_to_idx.get(
                        char, self.char_to_idx['<unk>']))
                    attention_mask.append(1)
        return torch.IntTensor(tokens), torch.IntTensor(attention_mask)

    def decode(self, tokens):
        input_sequence = ""
        for tok in tokens:
            input_sequence += self.idx_to_char[tok.item()]
        return input_sequence


class CustomTokenizerAlt():
    def __init__(self, vocab=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'x', ',', '<pad>', '<unk>']):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char,
                            idx in self.char_to_idx.items()}

    def encode(self, input_sequence):
        split_on_padding = input_sequence.split("<pad>")
        tokens = []
        for unpadded_seq in split_on_padding:
            if not unpadded_seq:
                tokens.append(self.char_to_idx["<pad>"])
            else:
                if tokens:
                    tokens.append(self.char_to_idx["<pad>"])
                for char in unpadded_seq:
                    tokens.append(self.char_to_idx.get(
                        char, self.char_to_idx['<unk>']))
        return torch.IntTensor(tokens)

    def decode(self, tokens):
        input_sequence = ""
        for tok in tokens:
            input_sequence += self.idx_to_char[tok.item()]
        return input_sequence


if __name__ == "__main__":
    tknzr = CustomTokenizer()
    text = "1-10,<pad>4-12,10x12<pad><pad>"
    print(tknzr.encode(text))
    print(tknzr.decode(tknzr.encode(text)[0]))
