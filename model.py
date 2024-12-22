import torch.nn as nn
import torch
from tokenizer import CustomTokenizer


class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_count):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size  # Size of word embeddings
        self.head_count = head_count  # Number of attention heads

        # Create linear layers for query, key and value projections for each head
        self.query_layers = nn.ModuleList(
            [nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.key_layers = nn.ModuleList(
            [nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])
        self.value_layers = nn.ModuleList(
            [nn.Linear(embed_size, embed_size, bias=False) for _ in range(head_count)])

        self.fc_out = nn.Linear(head_count * embed_size, embed_size)

    def forward(self, embeddings):
        batch_size, token_count = embeddings.shape[:2]
        qkvs = torch.zeros(self.head_count, 3, batch_size,
                           token_count, self.embed_size).to(embeddings.device)

        # Loop over heads and compute query, key and value projections
        for i in range(self.head_count):
            qkvs[i, 0] = self.query_layers[i](embeddings)
            qkvs[i, 1] = self.key_layers[i](embeddings)
            qkvs[i, 2] = self.value_layers[i](embeddings)

        # Compute energy terms for each head, batch, and pair of tokens
        energy = torch.zeros(self.head_count, batch_size,
                             token_count, token_count).to(embeddings.device)
        # Create a mask with false on and below the diagonal, and true above the diagonal
        mask = torch.triu(torch.ones(
            (token_count, token_count)), diagonal=1).bool()

        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        energy[h, b, i, j] = torch.dot(
                            qkvs[h, 0, b, i], qkvs[h, 1, b, j])
                energy[h, b] = energy[h, b].masked_fill(
                    mask, float('-inf'))  # Apply mask

        # Compute attention scores
        attention = torch.nn.functional.softmax(energy, dim=3)

        # Compute weighted sum of values for each head and token
        out = torch.zeros(batch_size, token_count, self.head_count,
                          self.embed_size).to(embeddings.device)
        for h in range(self.head_count):
            for b in range(batch_size):
                for i in range(token_count):
                    for j in range(token_count):
                        out[b, i, h] += (attention[h, b, i, j]
                                         * qkvs[h, 2, b, j])

        # Reshape and pass through final linear layer
        out = out.reshape(batch_size, token_count,
                          self.head_count * self.embed_size)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_count):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(
            embed_size, head_count)  # Self-attention layer
        self.norm1 = nn.LayerNorm(embed_size)  # Layer normalization
        self.norm2 = nn.LayerNorm(embed_size)  # Layer normalization

        # Feed-forward neural network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size)
        )

    def forward(self, embeddings):
        attention = self.attention(embeddings)

        # Apply residual connections and layer normalization
        out = self.norm1(attention + embeddings)
        out = attention + self.feed_forward(out)
        out = self.norm2(out)
        return out


class CheckersGPT(nn.Module):
    def __init__(self, vocab_size, num_positions, embed_size, num_layers, head_count):
        super(CheckersGPT, self).__init__()
        self.embed_size = embed_size  # Size of word embeddings
        self.vocab_size = vocab_size  # Size of vocabulary
        self.num_positions = num_positions
        self.word_embedding = nn.Embedding(
            vocab_size, embed_size)  # Embedding layer
        '''self.position_encoding = nn.Embedding(
            self.num_positions, embed_size)'''

        # List of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, head_count)
             for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self.tokenizer = CustomTokenizer()

    def forward(self, input, mask=None):

        input_tokens, attention_mask = self.tokenizer.encode(input)
        batch_size = 1  # TODO define with tokenizer

        token_count = len(input_tokens)  # TODO Assert ça a pas cassé
        print(token_count)

        out = self.word_embedding(input_tokens)
        out = out.expand(
            batch_size, *out.shape[-2:])  # Obtain word embeddings

        # print(out)
        print("out", out.shape)

        # Compute position encodings and add to word embeddings
        positions = torch.arange(0, token_count).expand(
            batch_size, token_count).to(input_tokens.device)

        # print(positions)
        print("pos ", positions.shape)
        position_encoding = self.position_encoding(positions)

        print(position_encoding.shape)
        print("pos enc", position_encoding.shape)
        # print("pos enc reshaped ",  position_encoding.reshape(out.shape))
        out += position_encoding[:out.shape[0], :, :]

        # Pass through each transformer block
        for layer in self.layers:
            out = layer(out)
        print(out.shape)
        # Produce logits for the final token in each sequence
        out = self.fc_out(out[:, -1, :].reshape(batch_size,
                          self.embed_size)).reshape(batch_size, self.vocab_size)
        return torch.nn.functional.softmax(out, dim=1)


if __name__ == "__main__":
    model = CheckersGPT(vocab_size=15, num_positions=400, embed_size=128,
                        num_layers=8, head_count=4)
    test_input = "1-10,<pad>4-12,10x12<pad>"
    print(model.forward(test_input))
