import torch
import torch.nn as nn
import math
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model)


    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):

    # seq_len : maximum len of the sentence
    # dropout : to make the model less overfit
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)