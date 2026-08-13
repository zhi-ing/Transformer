import torch

from  dataset.tokenizer import Tokenizer
from models.transformer import Transformer

import utils.config as config

#创建Tokenizer
def create_tokenizer():
    src_tokenizer = Tokenizer(
        config.SRC_SP_MODEL
    )

    tgt_tokenizer = Tokenizer(
        config.TGT_SP_MODEL
    )

    return src_tokenizer, tgt_tokenizer

#创建模型
def create_model(
        src_tokenizer,
        tgt_tokenizer
):
    model = Transformer(
        src_vocab_size = src_tokenizer.vocab_size,
        tgt_vocab_size = tgt_tokenizer.vocab_size,

        src_pad_idx=src_tokenizer.pad_id,
        tgt_pad_idx=tgt_tokenizer.pad_id,

        d_model = config.D_MODEL,
        num_layers = config.NUM_LAYERS,
        num_heads = config.NUM_HEADS,
        d_ff = config.D_FF,
        dropout = config.DROPOUT,
        max_len = config.MAX_LEN
    )

    return model.to(config.DEVICE)

#加载训练好的模型
def load_model(
        model,
        checkpoint_path,
        device
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.to(device)
    model.eval()

    print(
        f"Checkpoint loaded from: {checkpoint_path}"
    )

    print(
        f"Checkpoint epoch: {checkpoint['epoch']}"
    )

    if "best_val_loss" in checkpoint:
        print(
            f"Best validation loss: "
            f"{checkpoint['best_val_loss']:.4f}"
        )

    return model

#greed decoding
def greedy_decode(
        model,
        src,
        tgt_tokenizer,
        max_len=100
):

    """ 使用 Greedy Decoding 进行自回归翻译。 """
    device = src.device

    tgt = torch.tensor(
        [
            [
                tgt_tokenizer.bos_id
            ]
        ],
        dtype = torch.long,
        device = device
    )

    for _ in range(max_len):
        logits,_,_,_ = model(
            src,
            tgt
        )

        #取最后一个token概率
        next_token = logits[:, -1,:].argmax(
            dim = -1
        )
        print(
            "next_token:",
            next_token.item(),
            "eos_id:",
            tgt_tokenizer.eos_id
        )
        next_token = next_token.unsqueeze(1)

        #拼接
        tgt = torch.cat(
            [
                tgt,
                next_token
            ],
            dim = 1
        )

        #遇到eos停止
        if next_token.item() == tgt_tokenizer.eos_id:
            break

    return tgt

#translate
def translate(
        sentence,
        model,
        src_tokenizer,
        tgt_tokenizer,
        device,
        max_len = 100
):
    """
    中文 -> 英文
    """
    model.eval()

    #中文分词
    src_tokens = src_tokenizer.encode(
        sentence
    )

    src = torch.tensor(
        [src_tokens],
        dtype = torch.long,
        device = device
    )

    with torch.no_grad():
        output = greedy_decode(
            model=model,
            src = src,
            tgt_tokenizer=tgt_tokenizer,
            max_len = max_len
        )

    #tensor -> List
    output_tokens = (
        output[0]
        .cpu()
        .tolist()
    )

    #去掉bos/eos

    result_tokens = []
    for token in output_tokens:
        if token == tgt_tokenizer.bos_id:
            continue

        if token == tgt_tokenizer.eos_id:
            break

        result_tokens.append(token)

    #token->英文
    translation = tgt_tokenizer.decode(
        result_tokens
    )

    return translation


#main
def main():
    device = config.DEVICE

    print(
        "DEVICE:",
        device
    )

    #tokenizer
    src_tokenizer, tgt_tokenizer = create_tokenizer()
    print("Tokenizer loaded.")

    #model
    model = create_model(
        src_tokenizer,
        tgt_tokenizer
    )
    print("Transformer created.")

    #加载best_model
    model = load_model(
        model,
        config.BEST_MODEL_PATH,
        device
    )

    #开始交互式翻译
    print()
    print("=" * 60)
    print("Chinese -> English Translation")
    print("输入 q 退出")
    print("=" * 60)

    while(True):
        sentence = input(
            "\n中文输入(q退出):"
        ).strip()

        if sentence.lower() == "q":
            print("Exit.")
            break

        if not sentence:
            continue

        #翻译
        translation = translate(
            sentence=sentence,
            model=model,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
            max_len=config.MAX_LEN
        )
        print(f"英文翻译: {translation}")

#程序入口
if __name__ == "__main__":
    main()


