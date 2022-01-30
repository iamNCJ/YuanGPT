"""
Derived from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""

import math

import torch
from torch import nn
from torchtyping import TensorType


class SinCosEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: TensorType["batch_size", "seq_length", "hidden_size"]) \
            -> TensorType["batch_size", "seq_length", "hidden_size"]:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Return:
            H, H = E + P
        """
        x = x + self.pe[0, :x.size(0)]
        return self.dropout(x)
