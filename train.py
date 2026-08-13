#加载数据 → 构建数据集和 DataLoader → 初始化 Transformer 模型 → 
# 使用 Noam 调度器的 Adam 优化器 → 训练 20 个 epoch → 
# 每个 epoch 训练并验证 → 保存最佳和最后一个模型。
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.collate_fn import collate_fn as collate_batch
from dataset.tokenizer import Tokenizer
from dataset.translation_dataset import TranslationDataset
from models.transformer import Transformer
from utils import config
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.scheduler import NoamScheduler
from utils.seed import set_seed

#解析数据文件路径
def resolve_data_path(path):
    if os.path.exists(path):
        return path

    base_name = os.path.basename(path)
    fallback = os.path.join("data", base_name)
    if os.path.exists(fallback):
        return fallback

    return path

#创建中文和英文两个 SentencePiece tokenizer，并返回给后续的数据集处理使用
def create_tokenizer():
    src_tokenizer = Tokenizer(config.SRC_SP_MODEL)
    tgt_tokenizer = Tokenizer(config.TGT_SP_MODEL)
    return src_tokenizer, tgt_tokenizer

#根据已经创建好的中文和英文 tokenizer，
# 创建训练集 Dataset 和验证集 Dataset，供 Transformer 的 DataLoader 加载。
def create_dataset(src_tokenizer, tgt_tokenizer):
    train_dataset = TranslationDataset(
        resolve_data_path(config.TRAIN_SRC),
        resolve_data_path(config.TRAIN_TGT),
        src_tokenizer,
        tgt_tokenizer,
    )
    valid_dataset = TranslationDataset(
        resolve_data_path(config.VALID_SRC),
        resolve_data_path(config.VALID_TGT),
        src_tokenizer,
        tgt_tokenizer,
    )
    return train_dataset, valid_dataset

#把 TranslationDataset 生成的单个样本，组合成 batch，并通过 padding 补齐长度，
# 最终创建 PyTorch 的 DataLoader，供 Transformer 训练使用。
def create_dataloader(train_dataset, valid_dataset, src_pad_id, tgt_pad_id):
    def collate(batch):
        return collate_batch(batch, src_pad_id, tgt_pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, valid_loader

#根据配置文件中的 Transformer 超参数，创建一个 Encoder-Decoder Transformer 模型，
# 并将模型移动到指定设备（CPU/GPU）。
def create_model(src_vocab_size, tgt_vocab_size, src_pad_id, tgt_pad_id):
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=src_pad_id,
        tgt_pad_idx=tgt_pad_id,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_len=config.MAX_LEN,
    )
    return model.to(config.DEVICE)

#创建 Transformer 损失函数（Loss Function），
# 用于比较模型预测结果和真实英文目标序列之间的差异。
def create_loss(tgt_pad_id):
    return nn.CrossEntropyLoss(
        ignore_index=tgt_pad_id,
        label_smoothing=getattr(config, "LABEL_SMOOTHING", 0.0),
    )

#创建 Adam 优化器，用于根据反向传播计算出的梯度更新模型参数。
def create_optimizer(model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.BETAS,
        eps=config.EPS,
    )

#创建 Noam 学习率调度器（Learning Rate Scheduler）
def create_scheduler(optimizer):
    return NoamScheduler(
        optimizer,
        d_model=config.D_MODEL,
        warmup_steps=config.WARMUP_STEPS,
    )

#完成 Transformer 模型一个 epoch 的训练过程：遍历训练数据 → 前向传播 → 
# 计算 Loss → 反向传播 → 更新参数 → 更新学习率 → 返回平均训练损失
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    #将模型切换到训练模式。
    model.train()
    #用于累计整个epoch所有batch的loss。
    total_loss = 0.0

    #创建进度条
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        src = batch["src"].to(device)
        tgt_input = batch["tgt_input"].to(device)
        tgt_output = batch["tgt_output"].to(device)

        optimizer.zero_grad()

        logits, _, _, _ = model(src, tgt_input)
        logits = logits.contiguous().view(-1, logits.size(-1))
        targets = tgt_output.contiguous().view(-1)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_lr():.6f}")

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    #切换模型到评估模式
    model.eval()
    #初始化loss累计变量
    total_loss = 0.0

    #禁止梯度计算
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt_input = batch["tgt_input"].to(device)
            tgt_output = batch["tgt_output"].to(device)

            logits, _, _, _ = model(src, tgt_input)
            logits = logits.contiguous().view(-1, logits.size(-1))
            targets = tgt_output.contiguous().view(-1)
            total_loss += criterion(logits, targets).item()

    return total_loss / len(dataloader)


def main():
    set_seed(config.SEED)

    src_tokenizer, tgt_tokenizer = create_tokenizer()
    train_dataset, valid_dataset = create_dataset(src_tokenizer, tgt_tokenizer)

    train_loader, valid_loader = create_dataloader(
        train_dataset,
        valid_dataset,
        src_tokenizer.pad_id,
        tgt_tokenizer.pad_id,
    )

    src_vocab_size = src_tokenizer.vocab_size
    tgt_vocab_size = tgt_tokenizer.vocab_size

    model = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_id=src_tokenizer.pad_id,
        tgt_pad_id=tgt_tokenizer.pad_id,
    )
    criterion = create_loss(tgt_tokenizer.pad_id)
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    best_val_loss = float("inf")
    start_epoch = 1

    if os.path.exists(config.LAST_MODEL_PATH):
        try:
            loaded_epoch = load_checkpoint(
                config.LAST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                config.DEVICE,
            )
            start_epoch = loaded_epoch + 1
        except Exception as exc:
            print(f"Failed to load checkpoint: {exc}")

    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            config.DEVICE,
        )
        val_loss = evaluate(model, valid_loader, criterion, config.DEVICE)

        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            config.LAST_MODEL_PATH,
            best_val_loss=best_val_loss,
        )

        if is_best:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                config.BEST_MODEL_PATH,
                best_val_loss=best_val_loss,
            )


if __name__ == "__main__":
    main()

