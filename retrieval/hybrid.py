"""retrieval/hybrid.py —— RRF（Reciprocal Rank Fusion）混合检索融合。


"""

from __future__ import annotations

from .vector_store import RetrievalHit


def rrf_fuse(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    rrf_k: int = 60,
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """融合两路结果，按 RRF 得分降序返回。

    - dense_hits：向量检索结果（已按相似度排序）
    - sparse_hits：BM25 结果（已按关键词得分排序）
    """
    merged: dict[str, RetrievalHit] = {}

    for hit_list in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hit_list, start=1):
            if hit.chunk_id in merged:
                merged[hit.chunk_id].rrf_score += 1.0 / (rrf_k + rank)
            else:
                hit.rrf_score = 1.0 / (rrf_k + rank)
                merged[hit.chunk_id] = hit

    fused = sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)
    if top_k is not None:
        fused = fused[:top_k]

    # 重新编号名次
    for rank, hit in enumerate(fused, start=1):
        hit.rank = rank
    return fused
