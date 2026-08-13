#保存模型参数、优化器状态、学习率调度器状态和当前 epoch 到文件；加载时恢复所有状态，
# 支持从断点继续训练。
import os
import torch


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    path: str,
    best_val_loss: float | None = None,
):
    #创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # 保存模型所有可学习参数
        "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器状态
    }

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    #Scheduler可能为空
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)

    print(f"Checkpoint saved to: {path}")


def load_checkpoint(
    path: str,
    model,
    optimizer=None,
    scheduler=None,
    device="cpu",
):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    # Restore model
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Restore optimizer
    if (
        optimizer is not None and
        "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # Restore scheduler
    if (
        scheduler is not None and
        "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from: {path}")

    return epoch

