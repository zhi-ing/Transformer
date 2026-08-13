#实现了 Transformer 原论文中的 Noam 学习率调度器：先线性增加学习率（Warmup），
# 达到峰值后逐渐减小，最终学习率趋近于 0。
import math

class NoamScheduler:
    """
    Noam Learning Rate Scheduler

    Attention Is All You Need
    Section 5.3

    lr = d_model^(-0.5) *
         min(step^(-0.5),
             step * warmup_steps^(-1.5))
    """
    def __init__(
        self,
        optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        #当前训练步数
        self.current_step = 0

    def get_lr(self):
        step = max(self.current_step, 1)

        lr = (
            self.d_model ** (-0.5)
        ) * min(
            step ** (-0.5),
            step * (self.warmup_steps ** (-1.5))
        )

        return lr

    def step(self):
        self.current_step += 1

        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def state_dict(self):
        return {
            "current_step": self.current_step
        }

    def load_state_dict(
        self,
        state_dict
    ):
        self.current_step = state_dict["current_step"]