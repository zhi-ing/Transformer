# 读取数据，并把每一对中英文句子转换成 Transformer 可以直接训练的数据格式。
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    中英文翻译数据集
    """

    def __init__(self, src_file: str, tgt_file: str, src_tokenizer, tgt_tokenizer):
        """
        Parameters
        ----------
        src_file : 源语言文件路径
        tgt_file : 目标语言文件路径
        src_tokenizer : 源语言 tokenizer
        tgt_tokenizer : 目标语言 tokenizer
        """
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        with open(src_file, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f if line.strip()]

        with open(tgt_file, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f if line.strip()]

        assert len(self.src_sentences) == len(self.tgt_sentences), "源语言和目标语言句子数量不一致！"

    def __len__(self):
        """
        数据集大小
        """
        return len(self.src_sentences)

    def __getitem__(self, idx):
        """
        获取一条训练样本
        """
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]

        src = self.src_tokenizer.encode(src_sentence)
        tgt = self.tgt_tokenizer.encode(tgt_sentence)

        tgt_input = [self.tgt_tokenizer.bos_id] + tgt
        tgt_output = tgt + [self.tgt_tokenizer.eos_id]

        return {
            "src": src,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
        }