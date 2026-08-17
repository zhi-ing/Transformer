import math
import torch
import torch.nn as nn

from .encoder_prenorm import EncoderPreNorm
from .decoder_prenorm import DecoderPreNorm
from .positional_encoding import PositionalEncoding


class TransformerPreNorm(nn.Module):
    """
    Pre-Norm Encoder-Decoder Transformer

    Ablation experiments:
        baseline
        no_pe
        one_head
        three_layers
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
        max_len: int = 5000,
        ablation: str = "baseline",
    ):
        super().__init__()

        # ==================================================
        # 基本信息
        # ==================================================

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = d_model

        self.ablation = ablation

        # 检查实验名称
        valid_ablations = {
            "baseline",
            "no_pe",
            "one_head",
            "three_layers",
        }

        if ablation not in valid_ablations:
            raise ValueError(
                f"Unknown ablation: {ablation}\n"
                f"Available options: {valid_ablations}"
            )

        # ==================================================
        # Ablation: one_head
        #
        # 只改变 Attention Head 数量
        # d_model 保持 512 不变
        # ==================================================

        if ablation == "one_head":
            num_heads = 1

        # ==================================================
        # Ablation: three_layers
        #
        # Encoder / Decoder:
        # 6 -> 3
        # ==================================================

        if ablation == "three_layers":
            num_layers = 3

        # 保存最终实际配置
        self.num_layers = num_layers
        self.num_heads = num_heads

        # ==================================================
        # Embedding
        # ==================================================

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
        )

        # ==================================================
        # Positional Encoding
        #
        # baseline:
        #     正常使用
        #
        # no_pe:
        #     Identity
        # ==================================================

        if ablation == "no_pe":

            self.position = nn.Identity()

        else:

            self.position = PositionalEncoding(
                d_model=d_model,
                max_len=max_len,
                dropout=dropout,
            )

        # ==================================================
        # Pre-Norm Encoder
        # ==================================================

        self.encoder = EncoderPreNorm(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ==================================================
        # Pre-Norm Decoder
        # ==================================================

        self.decoder = DecoderPreNorm(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ==================================================
        # Output Projection
        # ==================================================

        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size,
        )

    # ======================================================
    # Source Padding Mask
    # ======================================================

    def make_src_mask(
        self,
        src: torch.Tensor,
    ):
        """
        src:
            (batch_size, src_len)

        return:
            (batch_size, 1, 1, src_len)
        """

        return (
            src != self.src_pad_idx
        ).unsqueeze(1).unsqueeze(2)

    # ======================================================
    # Target Mask
    # ======================================================

    def make_tgt_mask(
        self,
        tgt: torch.Tensor,
    ):
        """
        tgt:
            (batch_size, tgt_len)

        return:
            (batch_size, 1, tgt_len, tgt_len)
        """

        tgt_len = tgt.size(1)

        # Padding mask

        padding_mask = (
            tgt != self.tgt_pad_idx
        ).unsqueeze(1).unsqueeze(2)

        # Causal mask

        look_ahead_mask = torch.tril(
            torch.ones(
                tgt_len,
                tgt_len,
                device=tgt.device,
                dtype=torch.bool,
            )
        )

        look_ahead_mask = (
            look_ahead_mask
            .unsqueeze(0)
            .unsqueeze(1)
        )

        return (
            padding_mask
            & look_ahead_mask
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ):

        # ==================================================
        # Masks
        # ==================================================

        src_mask = self.make_src_mask(src)

        tgt_mask = self.make_tgt_mask(tgt)

        # ==================================================
        # Encoder
        # ==================================================

        src = self.src_embedding(src)

        src = src * math.sqrt(
            self.d_model
        )

        # baseline:
        #     Embedding -> Positional Encoding
        #
        # no_pe:
        #     Embedding -> Identity
        #
        src = self.position(src)

        encoder_output, encoder_attention = (
            self.encoder(
                src,
                src_mask,
            )
        )

        # ==================================================
        # Decoder
        # ==================================================

        tgt = self.tgt_embedding(tgt)

        tgt = tgt * math.sqrt(
            self.d_model
        )

        # 与 Encoder 使用相同的实验设置
        tgt = self.position(tgt)

        (
            decoder_output,
            self_attentions,
            cross_attentions,
        ) = self.decoder(
            x=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

        # ==================================================
        # Output
        # ==================================================

        logits = self.fc_out(
            decoder_output
        )

        return (
            logits,
            encoder_attention,
            self_attentions,
            cross_attentions,
        )