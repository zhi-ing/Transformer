#使用 SacreBLEU 库，计算模型生成的翻译（hypotheses）与参考答案（references）之间的
#BLEU 分数，用于评估翻译质量。
import sacrebleu


def calculate_bleu(
    references,
    hypotheses
):

    if len(references) != len(hypotheses):
        raise ValueError(
            "The number of references and hypotheses must be the same."
        )

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references]
    )

    return bleu.score