# ============================================================
# Pre-Norm Transformer Training
#
# 与原来的 train.py 保持相同训练流程，
# 唯一核心区别：
#
# 原模型：
#     models.transformer.Transformer
#     -> Post-Norm
#
# 新模型：
#     models.transformer_prenorm.TransformerPreNorm
#     -> Pre-Norm
#
# 新模型的 checkpoint 不覆盖原模型：
#
#     checkpoints/best_model_prenorm.pth
#     checkpoints/last_model_prenorm.pth
# ============================================================

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.collate_fn import collate_fn as collate_batch
from dataset.tokenizer import Tokenizer
from dataset.translation_dataset import TranslationDataset

# ============================================================
# 注意：
# 这里使用新的 Pre-Norm Transformer
# ============================================================

from models.transformer_prenorm import TransformerPreNorm

from utils import config
from utils.scheduler import NoamScheduler
from utils.seed import set_seed


# ============================================================
# 1. 数据路径解析
# ============================================================

def resolve_data_path(path):

    if os.path.exists(path):
        return path

    base_name = os.path.basename(path)

    fallback = os.path.join(
        "data",
        base_name
    )

    if os.path.exists(fallback):
        return fallback

    return path


# ============================================================
# 2. 创建 Tokenizer
# ============================================================

def create_tokenizer():

    src_tokenizer = Tokenizer(
        config.SRC_SP_MODEL
    )

    tgt_tokenizer = Tokenizer(
        config.TGT_SP_MODEL
    )

    return src_tokenizer, tgt_tokenizer


# ============================================================
# 3. 创建 Dataset
# ============================================================

def create_dataset(
    src_tokenizer,
    tgt_tokenizer
):

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


# ============================================================
# 4. 创建 DataLoader
# ============================================================

def create_dataloader(
    train_dataset,
    valid_dataset,
    src_pad_id,
    tgt_pad_id
):

    def collate(batch):

        return collate_batch(
            batch,
            src_pad_id,
            tgt_pad_id,
        )

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


# ============================================================
# 5. 创建 Pre-Norm Transformer
# ============================================================

def create_model(
    src_vocab_size,
    tgt_vocab_size,
    src_pad_id,
    tgt_pad_id
):

    model = TransformerPreNorm(

        src_vocab_size=src_vocab_size,

        tgt_vocab_size=tgt_vocab_size,

        src_pad_idx=src_pad_id,

        tgt_pad_idx=tgt_pad_id,

        d_model=config.D_MODEL,

        num_layers=config.NUM_LAYERS,

        num_heads=config.NUM_HEADS,

        d_ff=config.D_FF,

        dropout=config.DROPOUT,

        max_len=config.MAX_LEN,
    )

    return model.to(config.DEVICE)


# ============================================================
# 6. Loss
# ============================================================

def create_loss(tgt_pad_id):

    return nn.CrossEntropyLoss(

        ignore_index=tgt_pad_id,

        label_smoothing=getattr(
            config,
            "LABEL_SMOOTHING",
            0.0
        ),
    )


# ============================================================
# 7. Adam Optimizer
# ============================================================

def create_optimizer(model):

    return torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE,

        betas=config.BETAS,

        eps=config.EPS,
    )


# ============================================================
# 8. Noam Scheduler
# ============================================================

def create_scheduler(optimizer):

    return NoamScheduler(

        optimizer,

        d_model=config.D_MODEL,

        warmup_steps=config.WARMUP_STEPS,
    )


# ============================================================
# 9. 单个 Epoch 训练
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch in progress_bar:

        # ----------------------------------------------------
        # 数据
        # ----------------------------------------------------

        src = batch["src"].to(device)

        tgt_input = batch["tgt_input"].to(device)

        tgt_output = batch["tgt_output"].to(device)

        # ----------------------------------------------------
        # 清空梯度
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, _, _, _ = model(
            src,
            tgt_input
        )

        # ----------------------------------------------------
        # reshape
        #
        # logits:
        # (batch, tgt_len, vocab_size)
        #
        # ->
        # (batch * tgt_len, vocab_size)
        # ----------------------------------------------------

        logits = logits.contiguous().view(
            -1,
            logits.size(-1)
        )

        targets = tgt_output.contiguous().view(
            -1
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = criterion(
            logits,
            targets
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Noam Scheduler
        # ----------------------------------------------------

        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{scheduler.get_lr():.6f}"
        )

    return total_loss / len(dataloader)


# ============================================================
# 10. Validation
# ============================================================

def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in dataloader:

            src = batch["src"].to(device)

            tgt_input = batch["tgt_input"].to(device)

            tgt_output = batch["tgt_output"].to(device)

            # Forward
            logits, _, _, _ = model(
                src,
                tgt_input
            )

            logits = logits.contiguous().view(
                -1,
                logits.size(-1)
            )

            targets = tgt_output.contiguous().view(
                -1
            )

            loss = criterion(
                logits,
                targets
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)


# ============================================================
# 11. 保存 Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path,
    best_val_loss
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "best_val_loss":
            best_val_loss,
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved to: {path}"
    )


# ============================================================
# 12. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------

    set_seed(config.SEED)

    print(
        "============================================================"
    )

    print(
        "Pre-Norm Transformer Training"
    )

    print(
        "============================================================"
    )

    print(
        f"Device: {config.DEVICE}"
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    src_tokenizer, tgt_tokenizer = create_tokenizer()

    print(
        "Tokenizer loaded."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset, valid_dataset = create_dataset(
        src_tokenizer,
        tgt_tokenizer
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(valid_dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader, valid_loader = create_dataloader(

        train_dataset,

        valid_dataset,

        src_tokenizer.pad_id,

        tgt_tokenizer.pad_id,
    )

    print(
        f"Training batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(valid_loader)}"
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    src_vocab_size = (
        src_tokenizer.vocab_size
    )

    tgt_vocab_size = (
        tgt_tokenizer.vocab_size
    )

    print(
        f"Source vocabulary: {src_vocab_size}"
    )

    print(
        f"Target vocabulary: {tgt_vocab_size}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(

        src_vocab_size,

        tgt_vocab_size,

        src_tokenizer.pad_id,

        tgt_tokenizer.pad_id,
    )

    print(
        "Pre-Norm Transformer created."
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = create_loss(
        tgt_tokenizer.pad_id
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = create_scheduler(
        optimizer
    )

    # --------------------------------------------------------
    # 新的 checkpoint 路径
    #
    # 注意：
    # 不使用原来的 best_model.pth
    # 不使用原来的 last_model.pth
    # --------------------------------------------------------

    best_model_path = os.path.join(
        config.CHECKPOINT_DIR,
        "best_model_prenorm.pth"
    )

    last_model_path = os.path.join(
        config.CHECKPOINT_DIR,
        "last_model_prenorm.pth"
    )

    os.makedirs(
        config.CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 初始状态
    # --------------------------------------------------------

    best_val_loss = float("inf")

    start_epoch = 1

    # --------------------------------------------------------
    # 如果之前已经训练过 Pre-Norm，
    # 可以从 last_model_prenorm.pth 继续
    # --------------------------------------------------------

    if os.path.exists(
        last_model_path
    ):

        print(
            f"Found checkpoint: "
            f"{last_model_path}"
        )

        checkpoint = torch.load(
            last_model_path,
            map_location=config.DEVICE,
            weights_only=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        loaded_epoch = checkpoint["epoch"]

        start_epoch = loaded_epoch + 1

        best_val_loss = checkpoint.get(
            "best_val_loss",
            float("inf")
        )

        print(
            f"Resuming from epoch: "
            f"{loaded_epoch}"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        start_epoch,
        config.NUM_EPOCHS + 1
    ):

        print()
        print(
            "=" * 70
        )

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scheduler,

            config.DEVICE,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss = evaluate(

            model,

            valid_loader,

            criterion,

            config.DEVICE,
        )

        # ----------------------------------------------------
        # 输出
        # ----------------------------------------------------

        print()

        print(
            f"Epoch {epoch}/{config.NUM_EPOCHS} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| lr={scheduler.get_lr():.8f}"
        )

        # ----------------------------------------------------
        # 是否是最佳模型
        # ----------------------------------------------------

        is_best = (
            val_loss < best_val_loss
        )

        if is_best:

            best_val_loss = val_loss

        # ----------------------------------------------------
        # 保存 last checkpoint
        # ----------------------------------------------------

        save_checkpoint(

            model,

            optimizer,

            scheduler,

            epoch,

            last_model_path,

            best_val_loss,
        )

        # ----------------------------------------------------
        # 保存 best checkpoint
        # ----------------------------------------------------

        if is_best:

            print(
                "New best model!"
            )

            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch,

                best_model_path,

                best_val_loss,
            )

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Pre-Norm training finished."
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint:"
        f" {best_model_path}"
    )

    print(
        f"Last checkpoint:"
        f" {last_model_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()