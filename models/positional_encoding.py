import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding

    Attention Is All You Need
    Section 3.5
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, seq_len, d_model)

        Returns
        -------
        Tensor
            (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
    