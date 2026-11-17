"""retrieval 包：混合检索（稠密向量 + BM25）+ RRF 融合 + 可选 rerank。

对外主要接口：
    from retrieval.pipeline import Retriever
    retriever = Retriever(settings)
    hits = retriever.retrieve("问题")
"""

from .vector_store import ChromaVectorStore, RetrievalHit  # noqa: F401
