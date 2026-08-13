#同时设置 Python、NumPy、PyTorch（CPU/GPU）和 CUDA 的随机种子，
# 并禁用 cuDNN 的随机优化，确保所有随机操作结果一致。
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    #Python Random
    random.seed(seed)

    #Numpy
    np.random.seed(seed)

    #PyTorch CPU
    torch.manual_seed(seed)

    #PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    #保证卷积等运算可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
