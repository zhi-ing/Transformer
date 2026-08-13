#集中管理项目所需的所有参数：数据路径、模型架构、训练超参数、检查点保存路径等，
# 方便统一修改和调试。
import os

import torch

# Random Seed

SEED = 42

# Device

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Dataset

DATA_DIR = "./data"
CLEANED_DIR = f"{DATA_DIR}/cleaned"  # 新增

# Training set
TRAIN_SRC = f"{CLEANED_DIR}/train.zh"
TRAIN_TGT = f"{CLEANED_DIR}/train.en"

# Validation set
VALID_SRC = f"{CLEANED_DIR}/valid.zh"
VALID_TGT = f"{CLEANED_DIR}/valid.en"

# Test set
TEST_SRC = f"{CLEANED_DIR}/test.zh"
TEST_TGT = f"{CLEANED_DIR}/test.en"

# SentencePiece model
SRC_SP_MODEL = f"{DATA_DIR}/src.model"
TGT_SP_MODEL = f"{DATA_DIR}/tgt.model"


# Model

# Embedding dimension
D_MODEL = 512

# Number of Encoder / Decoder layers
NUM_LAYERS = 6

# Number of attention heads
NUM_HEADS = 8

# Feed Forward hidden dimension
D_FF = 2048

# Dropout probability
DROPOUT = 0.1

# Maximum sentence length
MAX_LEN = 5000


# Training

# Batch size
BATCH_SIZE = 64

# Number of epochs
NUM_EPOCHS = 20

# Adam optimizer
LEARNING_RATE = 1.0

BETAS = (0.9, 0.98)

EPS = 1e-9

# Noam Scheduler
WARMUP_STEPS = 4000

# Checkpoint

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")

# Evaluation

# Maximum decoding length
MAX_DECODE_LEN = 128

# Beam Search（以后可扩展）
BEAM_SIZE = 1