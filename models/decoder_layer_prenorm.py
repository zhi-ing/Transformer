import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class DecoderLayerPreNorm(nn.Module):
    """
    Pre-Norm Transformer Decoder Layer
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

        # ==================================================
        # 1. Masked Self-Attention
        # ==================================================

        normalized_x = self.norm1(x)

        self_attention_output, self_attention = (
            self.self_attention(
                query=normalized_x,
                key=normalized_x,
                value=normalized_x,
                mask=tgt_mask,
            )
        )

        # Residual
        x = x + self.dropout1(
            self_attention_output
        )

        # ==================================================
        # 2. Cross Attention
        # ==================================================

        normalized_x = self.norm2(x)

        cross_attention_output, cross_attention = (
            self.cross_attention(
                query=normalized_x,
                key=encoder_output,
                value=encoder_output,
                mask=src_mask,
            )
        )

        # Residual
        x = x + self.dropout2(
            cross_attention_output
        )

        # ==================================================
        # 3. Feed Forward
        # ==================================================

        normalized_x = self.norm3(x)

        feed_forward_output = self.feed_forward(
            normalized_x
        )

        # Residual
        x = x + self.dropout3(
            feed_forward_output
        )

        return (
            x,
            self_attention,
            cross_attention,
        )