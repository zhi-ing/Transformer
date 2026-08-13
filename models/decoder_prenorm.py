import torch
import torch.nn as nn

from .decoder_layer_prenorm import DecoderLayerPreNorm


class DecoderPreNorm(nn.Module):
    """
    Pre-Norm Transformer Decoder
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
                DecoderLayerPreNorm(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Decoder 最后 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ):

        self_attentions = []
        cross_attentions = []

        for layer in self.layers:

            (
                x,
                self_attention,
                cross_attention,
            ) = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

            self_attentions.append(
                self_attention
            )

            cross_attentions.append(
                cross_attention
            )

        x = self.norm(x)

        return (
            x,
            self_attentions,
            cross_attentions,
        )