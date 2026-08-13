#使用 SentencePiece 的 BPE（Byte Pair Encoding）算法，
# 分别在清洗后的中文和英文训练数据上训练词汇表大小为 8000 的分词模型，
# 生成 .model 和 .vocab 文件。
import os
import sentencepiece as spm


DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned") 

SRC_INPUT = os.path.join(CLEANED_DIR, "train.zh")  # 从 cleaned 目录读取
TGT_INPUT = os.path.join(CLEANED_DIR, "train.en")  # 从 cleaned 目录读取

SRC_PREFIX = os.path.join(
    DATA_DIR,
    "src"
)

TGT_PREFIX = os.path.join(
    DATA_DIR,
    "tgt"
)

VOCAB_SIZE = 8000

CHARACTER_COVERAGE = 0.9995


def train_source_tokenizer():

    print("Training source tokenizer...")

    spm.SentencePieceTrainer.train(
        input=SRC_INPUT,
        model_prefix=SRC_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=CHARACTER_COVERAGE,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>"
    )

    print("Source tokenizer finished.")


def train_target_tokenizer():

    print("Training target tokenizer...")

    spm.SentencePieceTrainer.train(
        input=TGT_INPUT,
        model_prefix=TGT_PREFIX,
        vocab_size=VOCAB_SIZE,
        character_coverage=1.0,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>"
    )

    print("Target tokenizer finished.")


def main():

    if not os.path.exists(SRC_INPUT):
        raise FileNotFoundError(
            f"{SRC_INPUT} not found."
        )

    if not os.path.exists(TGT_INPUT):
        raise FileNotFoundError(
            f"{TGT_INPUT} not found."
        )

    train_source_tokenizer()

    train_target_tokenizer()

    print("SentencePiece models generated successfully.")


if __name__ == "__main__":
    main()