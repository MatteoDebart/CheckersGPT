import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import CustomTokenizer


class CheckersDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # TODO add padding to get fixed size
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)
        return encoding


texts = ["1-10,4-12,10x12", "1-14, jgf"]
tokenizer = CustomTokenizer()
dataset = CheckersDataset(texts, tokenizer, max_length=100)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
