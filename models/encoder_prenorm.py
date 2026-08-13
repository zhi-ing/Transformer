import torch
import torch.nn as nn

from .encoder_layer_prenorm import EncoderLayerPreNorm


class EncoderPreNorm(nn.Module):
    """
    Pre-Norm Transformer Encoder
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
                EncoderLayerPreNorm(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # 原始 Transformer 最后一层输出需要 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        attentions = []

        for layer in self.layers:

            x, attention = layer(
                x,
                mask,
            )

            attentions.append(attention)

        # Pre-Norm Transformer 最后再做一次 LayerNorm
        x = self.norm(x)

        return x, attentions