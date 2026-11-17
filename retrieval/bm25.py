"""retrieval/bm25.py —— BM25 稀疏检索（关键词召回）。

为什么要和向量检索并列？
  向量检索擅长“语义相近”，但对专有名词、型号、编号等
  “字面完全匹配”的词容易漏；BM25 正好补这块短板。
  两路召回再用 RRF 融合，就是“混合检索”。

分词：优先用 jieba 切中文；未安装时退化为按单字切分（仍可用，只是稍粗）。
"""

from __future__ import annotations

from ingestion.loaders import Chunk

from .vector_store import RetrievalHit


def _tokenize(text: str) -> list[str]:
    """中文分词：有 jieba 用 jieba，否则按单字兜底。"""
    try:
        import jieba

        return [w for w in jieba.lcut(text) if w.strip()]
    except ImportError:
        return list(text.strip())


class BM25Index:
    """内存版 BM25 索引：从块语料构建，按关键词打分召回。"""

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> None:
        """用全部块构建索引。"""
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        corpus_tokens = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        """按关键词召回 top_k 个块；索引未构建时返回空。"""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(self._chunks)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        hits: list[RetrievalHit] = []
        for rank, idx in enumerate(ranked, start=1):
            chunk = self._chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    title=chunk.title,
                    text=chunk.text,
                    score=float(scores[idx]),
                    rank=rank,
                    metadata=chunk.metadata,
                )
            )
        return hits
