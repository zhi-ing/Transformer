#封装 SentencePiece 分词模型，提供文本 ↔ Token ID 的编码/解码功能，
# 以及获取 pad/bos/eos/unk 等特殊 Token ID 的接口。

import sentencepiece as spm

class Tokenizer:
    """
    SentencePiece Tokenizer
    """
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)    
        #从 model_path 指定的 .model 文件中加载
        #已经训练好的分词模型（词表、子词规则等），
        # 加载完成后即可使用 self.sp.encode() 和 self.sp.decode() 等接口
        # 进行文本与 token 的相互转换。
    def encode(self, text: str):
        """
        文本 -> Token ID
        """
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        """
        Token ID -> 文本
        """
        return self.sp.decode(ids)

    @property   #调用函数时不用加括号
    def pad_id(self):
        return self.sp.pad_id()

    @property
    def unk_id(self):
        return self.sp.unk_id()

    @property
    def bos_id(self):
        return self.sp.bos_id()

    @property
    def eos_id(self):
        return self.sp.eos_id()

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()