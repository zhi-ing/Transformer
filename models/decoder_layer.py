import torch
import torch.nn as nn

from .feed_forward import PositionwiseFeedForward
from .multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    Attention Is All You Need
    Figure 1
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

        self.cross_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, tgt_len, d_model)

        encoder_output : Tensor
            (batch_size, src_len, d_model)

        src_mask : Tensor
            (batch_size, 1, 1, src_len)

        tgt_mask : Tensor
            (batch_size, 1, tgt_len, tgt_len)

        Returns
        -------
        output : Tensor
            (batch_size, tgt_len, d_model)

        self_attention : Tensor
            (batch_size, num_heads, tgt_len, tgt_len)

        cross_attention : Tensor
            (batch_size, num_heads, tgt_len, src_len)
        """
        self_attention_output, self_attention = self.self_attention(
            query=x,
            key=x,
            value=x,
            mask=tgt_mask,
        )

        x = self.norm1(x + self.dropout1(self_attention_output))

        cross_attention_output, cross_attention = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=src_mask,
        )

        x = self.norm2(x + self.dropout2(cross_attention_output))

        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(feed_forward_output))

        return x, self_attention, cross_attention
        