#计算 Query 和 Key 的相似度（注意力分数），经过 Softmax 归一化后，
# 根据权重对 Value 进行加权求和，输出注意力结果和注意力权重。
import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention

    Attention(Q, K, V) =
        Softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(
        self,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Parameters
        ----------
        q : Tensor
            (batch_size, num_heads, seq_len_q, d_k)

        k : Tensor
            (batch_size, num_heads, seq_len_k, d_k)

        v : Tensor
            (batch_size, num_heads, seq_len_k, d_k)

        mask : Tensor, optional
            (batch_size, 1, seq_len_q, seq_len_k)
            或
            (batch_size, num_heads, seq_len_q, seq_len_k)

        Returns
        -------
        output : Tensor
            (batch_size, num_heads, seq_len_q, d_k)

        attention : Tensor
            (batch_size, num_heads, seq_len_q, seq_len_k)
        """

        #Key的维度 d_k
        d_k = q.size(-1)


        # Step1: 计算 Attention Score
        # score = QK^T / sqrt(d_k)
        scores = torch.matmul(
            q,
            k.transpose(-2,-1)
        ) / math.sqrt(d_k)

        # Step2: Mask(Decoder使用)
        if mask is not None:
           scores = scores.masked_fill(
               mask == 0,
               float("-inf")
           )

        # Step3: Softmax
        attention = torch.softmax(
            scores,
            dim = -1
        )

        # Step4: Dropout
        attention = self.dropout(attention)

        # Step5: Attention x V
        output = torch.matmul(
            attention,
            v
        )

        return output, attention
