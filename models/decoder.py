import torch
import torch.nn as nn

from .decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Transformer Decoder

    Attention Is All You Need
    Figure 1
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

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

        self_attentions : list
            每层 Decoder Self-Attention 权重

        cross_attentions : list
            每层 Cross-Attention 权重
        """
        self_attentions = []
        cross_attentions = []

        for layer in self.layers:
            x, self_attention, cross_attention = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

            self_attentions.append(self_attention)
            cross_attentions.append(cross_attention)

        return x, self_attentions, cross_attentions

