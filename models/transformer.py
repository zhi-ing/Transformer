import math
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer

    Attention Is All You Need
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        #Embedding
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        #Position Encoding
        self.position = PositionalEncoding(
            d_model = d_model,
            max_len = max_len,
            dropout = dropout
        )

        #Encoder
        self.encoder = Encoder(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff,
            dropout = dropout
        )

        #Decoder
        self.decoder = Decoder(
            num_layers = num_layers,
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff,
            dropout = dropout
        )

        #Output Projection
        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        #Source Padding Mask
    def make_src_mask(
            self,
            src: torch.Tensor
    ):
            """
        src:
            (batch_size, src_len)

        return:
            (batch_size,1,1,src_len)
        """
            src_mask = (
                src != self.src_pad_idx
            ).unsqueeze(1).unsqueeze(2)

            return src_mask

        #Target Mask
    def make_tgt_mask(
            self,
            tgt: torch.Tensor
    ):
            """
        tgt:
            (batch_size,tgt_len)

        return:
            (batch_size,1,tgt_len,tgt_len)
        """
            batch_size = tgt.size(0)
            tgt_len = tgt.size(1)

            #Padding Mask
            padding_mask = (
                tgt != self.tgt_pad_idx
            ).unsqueeze(1).unsqueeze(2)

            #Look Ahead Mask
            look_ahead_mask = torch.tril(
                torch.ones(
                    (tgt_len, tgt_len),
                    device = tgt.device,
                    dtype = torch.bool
                )
            )

            look_ahead_mask = (
                look_ahead_mask
                .unsqueeze(0)
                .unsqueeze(1)
            )

            tgt_mask = padding_mask & look_ahead_mask

            return tgt_mask

    def forward(
            self,
            src: torch.Tensor,
            tgt: torch.Tensor
    ):
             #Masks
             src_mask = self.make_src_mask(src)
             tgt_mask = self.make_tgt_mask(tgt)

             #Source Embedding
             src = self.src_embedding(src)
             src = src * math.sqrt(self.d_model)
             src = self.position(src)

             #Encoder
             encoder_output, encoder_attention = self.encoder(
                 src,
                 src_mask
             )

             #Target Embedding
             tgt = self.tgt_embedding(tgt)
             tgt = tgt * math.sqrt(self.d_model)
             tgt = self.position(tgt)

             #Decoder
             decoder_output, self_attentions, cross_attentions = self.decoder(
                 x = tgt,
                 encoder_output = encoder_output,
                 src_mask = src_mask,
                 tgt_mask = tgt_mask
             )

             #Linear
             logits = self.fc_out(
                 decoder_output
             )

             return (
                 logits,
                 encoder_attention,
                 self_attentions,
                 cross_attentions
             )


