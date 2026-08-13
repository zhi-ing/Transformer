import os
import torch
import torch.nn as nn
from tqdm import tqdm
import sacrebleu

from dataset.tokenizer import Tokenizer
from dataset.translation_dataset import TranslationDataset
from models.transformer import Transformer
from utils import config


# ============================================================
# 1. 创建 Tokenizer
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
# 2. 创建 Transformer
# ============================================================

def create_model(
    src_tokenizer,
    tgt_tokenizer
):

    model = Transformer(

        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,

        src_pad_idx=src_tokenizer.pad_id,
        tgt_pad_idx=tgt_tokenizer.pad_id,

        d_model=config.D_MODEL,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_len=config.MAX_LEN,
    )

    return model.to(config.DEVICE)


# ============================================================
# 3. 加载 checkpoint
# ============================================================

def load_model(
    model,
    checkpoint_path,
    device
):

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    print()
    print("=" * 60)
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Epoch: {checkpoint['epoch']}"
    )

    if "best_val_loss" in checkpoint:

        print(
            f"Best validation loss: "
            f"{checkpoint['best_val_loss']:.4f}"
        )

    print("=" * 60)

    return model


# ============================================================
# 4. Greedy Decoding
# ============================================================

def greedy_decode(
    model,
    src,
    tgt_tokenizer,
    max_len
):

    device = src.device

    # --------------------------------------------------------
    # Decoder 从 <bos> 开始
    # --------------------------------------------------------

    tgt = torch.tensor(
        [
            [
                tgt_tokenizer.bos_id
            ]
        ],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # 自回归生成
    # --------------------------------------------------------

    for _ in range(max_len):

        logits, _, _, _ = model(
            src,
            tgt
        )

        # 取最后一个位置
        next_token = logits[:, -1, :].argmax(
            dim=-1
        )

        next_token = next_token.unsqueeze(1)

        # 拼接到 Decoder 输入
        tgt = torch.cat(
            [
                tgt,
                next_token
            ],
            dim=1
        )

        # 遇到 EOS 停止
        if next_token.item() == tgt_tokenizer.eos_id:
            break

    return tgt


# ============================================================
# 5. 单句翻译
# ============================================================

def translate(
    sentence,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device,
    max_len
):

    # --------------------------------------------------------
    # 中文 → token
    # --------------------------------------------------------

    src_tokens = src_tokenizer.encode(
        sentence
    )

    src = torch.tensor(
        [src_tokens],
        dtype=torch.long,
        device=device
    )

    # --------------------------------------------------------
    # Greedy decoding
    # --------------------------------------------------------

    with torch.no_grad():

        output = greedy_decode(
            model=model,
            src=src,
            tgt_tokenizer=tgt_tokenizer,
            max_len=max_len
        )

    # --------------------------------------------------------
    # Tensor → List
    # --------------------------------------------------------

    output_tokens = (
        output[0]
        .cpu()
        .tolist()
    )

    # --------------------------------------------------------
    # 删除 BOS，并在 EOS 处停止
    # --------------------------------------------------------

    result_tokens = []

    for token in output_tokens:

        if token == tgt_tokenizer.bos_id:
            continue

        if token == tgt_tokenizer.eos_id:
            break

        result_tokens.append(token)

    # --------------------------------------------------------
    # token → 英文
    # --------------------------------------------------------

    translation = tgt_tokenizer.decode(
        result_tokens
    )

    return translation


# ============================================================
# 6. 测试集 BLEU
# ============================================================

def evaluate_bleu(
    model,
    src_tokenizer,
    tgt_tokenizer,
    src_file,
    tgt_file,
    device,
    max_len
):

    # --------------------------------------------------------
    # 读取测试集
    # --------------------------------------------------------

    with open(
        src_file,
        "r",
        encoding="utf-8"
    ) as f:

        src_sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(
        tgt_file,
        "r",
        encoding="utf-8"
    ) as f:

        tgt_sentences = [
            line.strip()
            for line in f
            if line.strip()
        ]

    assert len(src_sentences) == len(tgt_sentences), (
        "测试集中文和英文句子数量不一致！"
    )

    print()
    print(
        f"Test samples: {len(src_sentences)}"
    )

    # --------------------------------------------------------
    # 保存模型预测
    # --------------------------------------------------------

    predictions = []

    references = []

    # --------------------------------------------------------
    # 遍历测试集
    # --------------------------------------------------------

    for src_sentence, tgt_sentence in tqdm(
        zip(src_sentences, tgt_sentences),
        total=len(src_sentences),
        desc="Evaluating"
    ):

        prediction = translate(
            sentence=src_sentence,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_len=max_len
        )

        predictions.append(
            prediction
        )

        references.append(
            tgt_sentence
        )

    # --------------------------------------------------------
    # BLEU
    # --------------------------------------------------------

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references]
    )

    return bleu


# ============================================================
# 7. 主程序
# ============================================================

def main():

    print(
        "DEVICE:",
        config.DEVICE
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    src_tokenizer, tgt_tokenizer = create_tokenizer()

    print(
        "Tokenizer loaded."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        src_tokenizer,
        tgt_tokenizer
    )

    print(
        "Transformer created."
    )

    # --------------------------------------------------------
    # 测试集路径
    # --------------------------------------------------------

    test_src = config.TEST_SRC
    test_tgt = config.TEST_TGT

    print()
    print(
        "Test source:",
        test_src
    )

    print(
        "Test target:",
        test_tgt
    )

    # ========================================================
    # Evaluate BEST MODEL
    # ========================================================

    model = load_model(
        model,
        config.BEST_MODEL_PATH,
        config.DEVICE
    )

    bleu = evaluate_bleu(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_file=test_src,
        tgt_file=test_tgt,
        device=config.DEVICE,
        max_len=config.MAX_LEN
    )

    print()
    print("=" * 60)
    print("BEST MODEL TEST RESULT")
    print("=" * 60)
    print(
        f"BLEU = {bleu.score:.4f}"
    )
    print("=" * 60)

    # ========================================================
    # Evaluate LAST MODEL
    # ========================================================

    model = load_model(
        model,
        config.LAST_MODEL_PATH,
        config.DEVICE
    )

    bleu = evaluate_bleu(
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_file=test_src,
        tgt_file=test_tgt,
        device=config.DEVICE,
        max_len=config.MAX_LEN
    )

    print()
    print("=" * 60)
    print("LAST MODEL TEST RESULT")
    print("=" * 60)
    print(
        f"BLEU = {bleu.score:.4f}"
    )
    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()