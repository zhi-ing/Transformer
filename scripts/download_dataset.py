#从 HuggingFace 下载 OPUS-100 英-中翻译数据集，将训练集、
# 验证集、测试集分别保存为 .zh（中文）和 .en（英文）两个文本文件，
# 每行一个句子，中英文一一对应。
import os

from datasets import load_dataset


DATASET_NAME = "Helsinki-NLP/opus-100"

CONFIG_NAME = "en-zh"

SAVE_DIR = "data"


def save_split(dataset, split_name):
    """
    Save one split to text files.

    Parameters
    ----------
    dataset
        HuggingFace Dataset

    split_name
        train / valid / test
    """

    zh_path = os.path.join(
        SAVE_DIR,
        f"{split_name}.zh"
    )

    en_path = os.path.join(
        SAVE_DIR,
        f"{split_name}.en"
    )

    with open(
        zh_path,
        "w",
        encoding="utf-8"
    ) as zh_file, open(
        en_path,
        "w",
        encoding="utf-8"
    ) as en_file:

        for sample in dataset:

            translation = sample["translation"]

            zh = translation["zh"].strip()
            en = translation["en"].strip()

            if zh == "" or en == "":
                continue

            zh_file.write(zh + "\n")
            en_file.write(en + "\n")


def main():

    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )

    print("Downloading OPUS-100 (en-zh)...")

    dataset = load_dataset(
        DATASET_NAME,
        CONFIG_NAME
    )

    print("Saving train set...")
    save_split(
        dataset["train"],
        "train"
    )

    print("Saving validation set...")
    save_split(
        dataset["validation"],
        "valid"
    )

    print("Saving test set...")
    save_split(
        dataset["test"],
        "test"
    )

    print("Done.")


if __name__ == "__main__":
    main()