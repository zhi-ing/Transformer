# ============================================================
# Pre-Norm Transformer Ablation Training
#
# 支持：
#     baseline
#     no_pe
#     one_head
#     three_layers
#
# 使用：
#     python train_prenorm.py --ablation baseline
#     python train_prenorm.py --ablation no_pe
#     python train_prenorm.py --ablation one_head
#     python train_prenorm.py --ablation three_layers
#
# 每个实验使用独立 checkpoint：
#
# checkpoints/ablation/
# ├── baseline/
# │   ├── best.pth
# │   └── last.pth
# ├── no_pe/
# │   ├── best.pth
# │   └── last.pth
# ├── one_head/
# │   ├── best.pth
# │   └── last.pth
# └── three_layers/
#     ├── best.pth
#     └── last.pth
# ============================================================

import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.collate_fn import collate_fn as collate_batch
from dataset.tokenizer import Tokenizer
from dataset.translation_dataset import TranslationDataset

from models.transformer_prenorm import TransformerPreNorm

from utils import config
from utils.scheduler import NoamScheduler
from utils.seed import set_seed


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Train Pre-Norm Transformer with ablation experiments."
    )

    parser.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "no_pe",
            "one_head",
            "three_layers"
        ],
        help="Ablation experiment name."
    )

    return parser.parse_args()


# ============================================================
# 2. 数据路径解析
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
# 3. 创建 Tokenizer
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
# 4. 创建 Dataset
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
# 5. 创建 DataLoader
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
            tgt_pad_id
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
# 6. 创建 Transformer
# ============================================================

def create_model(
    src_vocab_size,
    tgt_vocab_size,
    src_pad_id,
    tgt_pad_id,
    ablation
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

        # 关键：
        # 将消融实验名称传给模型
        ablation=ablation
    )

    return model.to(config.DEVICE)


# ============================================================
# 7. Loss
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
# 8. Adam Optimizer
# ============================================================

def create_optimizer(model):

    return torch.optim.Adam(

        model.parameters(),

        lr=config.LEARNING_RATE,

        betas=config.BETAS,

        eps=config.EPS,
    )


# ============================================================
# 9. Noam Scheduler
# ============================================================

def create_scheduler(optimizer):

    return NoamScheduler(

        optimizer,

        d_model=config.D_MODEL,

        warmup_steps=config.WARMUP_STEPS,
    )


# ============================================================
# 10. 单个 Epoch 训练
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
# 11. Validation
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
# 12. 保存 Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path,
    best_val_loss,
    ablation
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

        # 保存实验名称
        "ablation":
            ablation,
    }

    torch.save(
        checkpoint,
        path
    )

    print(
        f"Checkpoint saved to: {path}"
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    args = parse_args()

    ablation = args.ablation

    # --------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------

    set_seed(config.SEED)

    print()
    print("=" * 70)
    print("Pre-Norm Transformer Ablation Training")
    print("=" * 70)

    print(
        f"Experiment: {ablation}"
    )

    print(
        f"Device: {config.DEVICE}"
    )

    print("=" * 70)

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

        ablation
    )

    print()
    print(
        "Pre-Norm Transformer created."
    )

    # --------------------------------------------------------
    # 打印实验配置
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("Ablation Configuration")
    print("-" * 70)

    print(
        f"Experiment       : {ablation}"
    )

    print(
        f"d_model          : {config.D_MODEL}"
    )

    print(
        f"num_layers       : {config.NUM_LAYERS}"
    )

    print(
        f"num_heads        : {config.NUM_HEADS}"
    )

    print(
        f"d_ff             : {config.D_FF}"
    )

    print(
        f"dropout          : {config.DROPOUT}"
    )

    print("-" * 70)

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
    # Checkpoint 路径
    #
    # 每个消融实验独立保存
    # --------------------------------------------------------

    checkpoint_dir = os.path.join(
        config.CHECKPOINT_DIR,
        "ablation",
        ablation
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        "best.pth"
    )

    last_model_path = os.path.join(
        checkpoint_dir,
        "last.pth"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    print()
    print(
        f"Checkpoint directory:"
    )

    print(
        f"  {checkpoint_dir}"
    )

    # --------------------------------------------------------
    # 初始状态
    # --------------------------------------------------------

    best_val_loss = float("inf")

    start_epoch = 1

    # --------------------------------------------------------
    # 如果之前训练过该消融实验
    # 则从该实验自己的 checkpoint 继续
    # --------------------------------------------------------

    if os.path.exists(
        last_model_path
    ):

        print()
        print(
            f"Found checkpoint:"
        )

        print(
            f"  {last_model_path}"
        )

        checkpoint = torch.load(
            last_model_path,
            map_location=config.DEVICE,
            weights_only=True,
        )

        # ----------------------------------------------------
        # 检查 checkpoint 是否属于当前实验
        # ----------------------------------------------------

        saved_ablation = checkpoint.get(
            "ablation",
            None
        )

        if saved_ablation != ablation:

            raise RuntimeError(
                "Checkpoint ablation mismatch!\n"
                f"Current experiment: {ablation}\n"
                f"Checkpoint experiment: {saved_ablation}"
            )

        # ----------------------------------------------------
        # 加载模型
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------------------
        # 加载 optimizer
        # ----------------------------------------------------

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # ----------------------------------------------------
        # 加载 scheduler
        # ----------------------------------------------------

        if "scheduler_state_dict" in checkpoint:

            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        # ----------------------------------------------------
        # epoch
        # ----------------------------------------------------

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
            f"Experiment: {ablation}"
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

            ablation
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

                ablation
            )

    # --------------------------------------------------------
    # Training finished
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "Pre-Norm ablation training finished."
    )

    print(
        f"Experiment: {ablation}"
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