import torch
import re


class CustomTokenizer():
    def __init__(self, vocab=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'x', ',', '<eos>', '<pad>', '<unk>']):
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char,
                            idx in self.char_to_idx.items()}

    def encode_single(self, input_sequence):  # legacy, used only for batch_size = 1
        split_on_padding_and_eos = re.split(r"(<[^>]+>)", input_sequence)
        tokens = []
        for seq in split_on_padding_and_eos:
            if seq:  # Ignore empty strings
                if seq[0] == "<":  # Special char such as <eos> or <pad>
                    tokens.append(self.char_to_idx.get(
                        seq, self.char_to_idx['<unk>']))
                else:
                    for char in seq:
                        tokens.append(self.char_to_idx.get(
                            char, self.char_to_idx['<unk>']))
        return torch.IntTensor(tokens).reshape((1, len(tokens)))

    def encode(self, input_sequences):
        output_matrix = []
        for input_sequence in input_sequences:
            split_on_padding_and_eos = re.split(r"(<[^>]+>)", input_sequence)
            tokens = []
            for seq in split_on_padding_and_eos:
                if seq:  # Ignore empty strings
                    if seq[0] == "<":  # Special char such as <eos> or <pad>
                        tokens.append(self.char_to_idx.get(
                            seq, self.char_to_idx['<unk>']))
                    else:
                        for char in seq:
                            tokens.append(self.char_to_idx.get(
                                char, self.char_to_idx['<unk>']))
            output_matrix.append(tokens)
        return torch.IntTensor(output_matrix)

    def decode_single(self, tokens):  # legacy, used only for batch_size = 1
        input_sequence = ""
        for tok in tokens[0]:
            input_sequence += self.idx_to_char[tok.item()]
        return input_sequence

    def decode(self, tokens):
        output_sequences = []
        for sequence_index in range(tokens.shape[0]):
            input_sequence = ""
            for tok in tokens[sequence_index]:
                input_sequence += self.idx_to_char[tok.item()]
            output_sequences.append(input_sequence)
        return output_sequences


if __name__ == "__main__":
    tknzr = CustomTokenizer()
    text = "1-10,<pad>4<7>-12,10x12<eos><pad>"
    print(text)
    tokenized_text = tknzr.encode_single(text)
    print(tokenized_text.shape)
    print(tknzr.decode_single(tknzr.encode_single(text)))

    texts = ["1-10,<pad>4-12,10x12<eos><pad>",
             "8x12,14-6,11x12<eos><pad><pad>"]
    print(texts)
    tokenized_texts = tknzr.encode(texts)
    print(tokenized_texts)
    print(tknzr.decode(tokenized_texts))
