import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class EncoderLayerPreNorm(nn.Module):
    """
    Pre-Norm Transformer Encoder Layer

    x -> LayerNorm -> Self-Attention -> Residual
      -> LayerNorm -> Feed Forward -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        # Pre-Norm 的两个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        x:
            (batch_size, seq_len, d_model)

        mask:
            (batch_size, 1, 1, seq_len)

        Returns
        -------
        output:
            (batch_size, seq_len, d_model)

        attention:
            (batch_size, num_heads, seq_len, seq_len)
        """

        # ==================================================
        # Pre-Norm Self-Attention
        # ==================================================

        normalized_x = self.norm1(x)

        attention_output, attention = self.self_attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            mask=mask,
        )

        # Residual
        x = x + self.dropout1(attention_output)

        # ==================================================
        # Pre-Norm Feed Forward
        # ==================================================

        normalized_x = self.norm2(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual
        x = x + self.dropout2(feed_forward_output)

        return x, attention