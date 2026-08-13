import torch
import torch.nn as nn

from .attention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Attention Is All You Need
    Section 3.2.2
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = None
    ):
            """
        Parameters
        ----------
        query : (batch_size, query_len, d_model)

        key : (batch_size, key_len, d_model)

        value : (batch_size, value_len, d_model)

        mask : optional

        Returns
        -------
        output :
            (batch_size, query_len, d_model)

        attention :
            (batch_size, num_heads, query_len, key_len)
            """
            batch_size = query.size(0)

            # Linear Projection
            Q = self.W_q(query)
            K = self.W_k(key)
            V = self.W_v(value)

            # Split Heads
            Q = Q.view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k
            ).transpose(1,2)

            K = K.view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k
            ).transpose(1, 2)

            V = V.view(
                batch_size,
                -1,
                self.num_heads,
                self.d_k
            ).transpose(1, 2)

            # Scaled Dot-Product Attention
            output, attention = self.attention(
                Q,
                K,
                V,
                mask
            )

            # Concat Heads
            output = output.transpose(1,2).contiguous()

            output = output.view(
                batch_size,
                -1,
                self.d_model
            )
            # Final Linear
            output = self.W_o(output)
            output = self.dropout(output)

            return output, attention