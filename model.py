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

        # create a matrix of dimension : (seq_len, d_model)
        self.pe = torch.zeros(seq_len, d_model)
        # create a vector of shape(seq_len , 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(-10000.0, )/ d_model))
        # apply sin to even position
        self.pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd position
        self.pe[:, 1::2] = torch.sin(position * div_term)

        self.pe = self.pe.unsqueeze(0) # (1, seq_lem , d_model)
        self.register_buffer(self.pe, 'pe')


    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        """
        In the LayerNormalization the mean and the variance of each batch is calculated independently of the other batches.
        Then the values in the batch are updated with new value computed with the mean, variance, aplha and bias parameters.
        """
        super().__init__()
        self.eps = eps # eps will provide a numerical stability and to avoid division by zero
        self.aplha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.ones(1)) # Additive


    def forward(self, x):
        mean = x.mean(dim =1, keepdim =True)
        std = x.std(dim =1, keepdim =True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, dff: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff) # w1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model) # w2 and B2


    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, seq_len, dff) --> (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.dk = d_model // h
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        dk = query.shape[-1]
        



    def forward(self, q, k, v, mask):
        # Mask is used when we dont want certain words to interact with some words
        query = self.wq(q) # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        key = self.wk(k) # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)
        value = self.wv(v) # (Batch, Seq_len, d_model) -> (Batch, Seq_len, d_model)


        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, dk) -> (Batch, h, Seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1,2)

