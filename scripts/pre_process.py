#读取原始的中英文平行语料，去除空句、去重、过滤超长句子，
# 然后将清洗后的数据保存到 data/cleaned/ 目录，并输出清洗统计信息。
import os

DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")  # 新增清洗数据目录

MAX_SRC_LEN = 100
MAX_TGT_LEN = 100

def process_split(split_name):
    """Preprocess one dataset split."""

    # 创建清洗数据目录
    os.makedirs(CLEANED_DIR, exist_ok=True)

    # 输入路径（原始数据）
    src_path = os.path.join(DATA_DIR, f"{split_name}.zh")
    tgt_path = os.path.join(DATA_DIR, f"{split_name}.en")

    # 输出路径（清洗后的数据，保存到 cleaned 目录）
    out_src = os.path.join(CLEANED_DIR, f"{split_name}.zh")
    out_tgt = os.path.join(CLEANED_DIR, f"{split_name}.en")

    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = f.readlines()

    with open(tgt_path, "r", encoding="utf-8") as f:
        tgt_lines = f.readlines()

    assert len(src_lines) == len(tgt_lines)

    cleaned = []
    seen = set()
    total = len(src_lines)

    for src, tgt in zip(src_lines, tgt_lines):
        src = src.strip()
        tgt = tgt.strip()

        # 去除空句
        if src == "" or tgt == "":
            continue

        # 去除完全重复样本
        pair = (src, tgt)
        if pair in seen:
            continue
        seen.add(pair)

        # 长度过滤（字符数）
        if len(src) > MAX_SRC_LEN:
            continue
        if len(tgt) > MAX_TGT_LEN:
            continue

        cleaned.append(pair)

    with open(out_src, "w", encoding="utf-8") as f_src, \
         open(out_tgt, "w", encoding="utf-8") as f_tgt:
        for src, tgt in cleaned:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")

    print(f"{split_name}")
    print(f"Original : {total}")
    print(f"Remain   : {len(cleaned)}")
    print(f"Removed  : {total - len(cleaned)}")
    print("-" * 40)

def main():
    process_split("train")
    process_split("valid")
    process_split("test")
    print("Preprocessing Finished.")

if __name__ == "__main__":
    main()