import torch
import torch.nn as nn

from .encoder_layer import EncoderLayer

class Encoder(nn.Module):
    """
    Transformer Encoder

    Attention Is All You Need
    Figure 1
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        #N个Encoder Layer
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model = d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    dropout = dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, src_len, d_model)

        mask : Tensor
            (batch_size, 1, 1, src_len)

        Returns
        -------
        output : Tensor
            (batch_size, src_len, d_model)

        attentions : list
            每层Attention权重
        """
        attentions = []

        for layer in self.layers:
            x, attention = layer(
                x,
                mask
            )

            attentions.append(attention)

        return x, attentions