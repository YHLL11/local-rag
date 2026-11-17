"""retrieval/pipeline.py —— 检索编排：把向量库、BM25、RRF、rerank 串起来。

一条查询的完整链路：
    查询 -> 编码 -> 稠密召回(chroma) + 稀疏召回(BM25)
          -> RRF 融合 -> (可选) cross-encoder 重排 -> 最终块列表
"""

from __future__ import annotations

from config.settings import Settings
from ingestion.loaders import Chunk

from .bm25 import BM25Index
from .embeddings import EmbeddingService
from .hybrid import rrf_fuse
from .rerank import Reranker
from .vector_store import ChromaVectorStore, RetrievalHit


class Retriever:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding = EmbeddingService(settings.embedding_model, settings.device)
        self.store = ChromaVectorStore(settings.chroma_dir)
        self.bm25 = BM25Index()
        self.reranker = Reranker(
            model_name=settings.rerank_model,
            enabled=settings.rerank_enabled,
            threshold=settings.rerank_threshold,
            device=settings.device,
        )
        self._bm25_ready = False

    # ---------- 建库 ----------
    def index(self, chunks: list[Chunk]) -> None:
        """对切分好的块向量化、写入 ChromaDB，并重建 BM25 索引。

        入口（CLI / /ingest）都是对 data 目录全量重建，因此先清空旧库，
        避免已删除文档的块残留在向量库中。
        """
        if not chunks:
            return
        self.store.reset()
        print("  · 正在把文本向量化（encode）…")
        embeddings = self.embedding.encode([c.text for c in chunks])
        print("  · 正在写入 ChromaDB…")
        self.store.add(chunks, embeddings)
        # BM25 需要看到“全部块”，所以从库里取全量重建
        print("  · 正在重建 BM25 索引…")
        self.bm25.build(self.store.all_chunks())
        self._bm25_ready = True
        print("  · 入库索引构建完成")

    def _ensure_bm25(self) -> None:
        """进程重启后 BM25 在内存丢失，首次查询时从向量库重建。"""
        if not self._bm25_ready and self.store.count() > 0:
            self.bm25.build(self.store.all_chunks())
            self._bm25_ready = True

    # ---------- 查询 ----------
    def retrieve(self, query: str, use_rerank: bool | None = None) -> list[RetrievalHit]:
        """混合检索一条查询，返回重排后的最终块列表。

        use_rerank=None 跟随配置；显式传 True/False 可做消融对比（eval 用）。
        """
        self._ensure_bm25()

        # 1) 两路召回
        query_vec = self.embedding.encode_query(query)
        dense_hits = self.store.query(query_vec, self.settings.dense_top_k)
        sparse_hits = self.bm25.search(query, self.settings.sparse_top_k)

        # 2) RRF 融合（先多取一些候选交给 rerank）
        fused = rrf_fuse(
            dense_hits,
            sparse_hits,
            rrf_k=self.settings.rrf_k,
            top_k=self.settings.final_top_k * 2,
        )

        # 3) 可选重排
        do_rerank = self.settings.rerank_enabled if use_rerank is None else use_rerank
        if do_rerank:
            return self.reranker.rerank(query, fused, top_k=self.settings.final_top_k)
        return fused[: self.settings.final_top_k]
