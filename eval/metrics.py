"""eval/metrics.py —— 检索质量指标：Recall@k 与 MRR。

Recall@k：前 k 个结果里命中了多少相关文档（占应有相关文档的比例），
           衡量“该找到的有没有找到”。
MRR（平均倒数排名）：第一个相关结果出现在第几名，取 1/名次，
           衡量“相关结果排得靠前不靠前”。
"""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """前 k 条中覆盖的相关文档比例。retrieved 按名次排列。"""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """第一个相关条目的倒数排名；未命中为 0。"""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
