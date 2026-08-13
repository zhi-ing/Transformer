import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network

    Attention Is All You Need
    Section 3.3
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        """
        Parameters
        ----------
        d_model : int
            Transformer隐藏层维度

        d_ff : int
            前馈网络隐藏层维度
            原论文默认：
            d_model = 512
            d_ff = 2048

        dropout : float
            Dropout概率
        """
        super().__init__()

        #第一层线性变换
        self.linear1 = nn.Linear(
            d_model,
            d_ff
        )

        #ReLu激活函数
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        #第二层线性变换
        self.linear2 = nn.Linear(
            d_ff,
            d_model
        )

    def forward(
        self,
        x: torch.Tensor
    ):
        """
        Parameters
        ----------
        x : Tensor
            (batch_size, seq_len, d_model)

        Returns
        -------
        Tensor
            (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x
    