import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer

    Attention Is All You Need
    Figure 1
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        #Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(
            d_model = d_model,
            num_heads = num_heads,
            dropout = dropout
        )

        #Position-wise Feed Forward
        self.feed_forward = PositionwiseFeedForward(
            d_model = d_model,
            d_ff = d_ff,
            dropout = dropout
        )

        #LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        #Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, seq_len, d_model)

        mask : Tensor, optional

        Returns
        -------
        output : Tensor
            (batch_size, seq_len, d_model)

        attention : Tensor
            (batch_size, num_heads, seq_len, seq_len)
        """
        # Multi-Head Self-Attention
        attention_output, attention = self.self_attention(
            query = x,
            key = x,
            value = x,
            mask = mask
        )

        # Residual + LayerNorm (Post-Norm)
        x = self.norm1(
            x + self.dropout1(attention_output)
        )

        # Position-wise Feed Forward
        feed_forward_output = self.feed_forward(x)

        # Residual + LayerNorm (Post-Norm)
        x = self.norm2(
            x + self.dropout2(feed_forward_output)
        )

        return x, attention