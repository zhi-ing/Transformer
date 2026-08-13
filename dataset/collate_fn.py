#将一个 batch 中的多个样本（源序列、解码器输入序列、解码器目标序列）进行 padding 对齐，
# 并转换为 PyTorch Tensor，返回给模型训练使用
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch, src_pad_id, tgt_pad_id=None):
    if tgt_pad_id is None:
        tgt_pad_id = src_pad_id

    # -------------------------
    # 分别取出三个序列
    # -------------------------
    src = [
        torch.tensor(sample["src"], dtype=torch.long)
        for sample in batch
    ]

    tgt_input = [
        torch.tensor(sample["tgt_input"], dtype=torch.long)
        for sample in batch
    ]

    tgt_output = [
        torch.tensor(sample["tgt_output"], dtype=torch.long)
        for sample in batch
    ]

    # -------------------------
    # Padding
    # batch_first=True
    # 输出形状：(batch_size, seq_len)
    # -------------------------
    src = pad_sequence(
        src,
        batch_first=True,
        padding_value=src_pad_id,
    )

    tgt_input = pad_sequence(
        tgt_input,
        batch_first=True,
        padding_value=tgt_pad_id,
    )

    tgt_output = pad_sequence(
        tgt_output,
        batch_first=True,
        padding_value=tgt_pad_id,
    )

    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
    }