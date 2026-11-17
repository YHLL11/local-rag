"""retrieval/vector_store.py —— ChromaDB 本地向量库封装。

职责单一：只负责“块的写入”和“按查询向量做稠密召回”。
混合检索、融合、重排都在别的模块，便于单独替换。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ingestion.loaders import Chunk


@dataclass
class RetrievalHit:
    """一次检索命中的块，带各种打分字段。"""

    chunk_id: str
    source: str
    title: str
    text: str
    score: float = 0.0            # 原始得分（距离 / BM25 / rerank），含义随来源不同
    rrf_score: float = 0.0        # RRF 融合后的得分
    rank: int = 0                 # 在最终结果中的名次（从 1 开始）
    metadata: dict = field(default_factory=dict)


class ChromaVectorStore:
    """ChromaDB 持久化向量库（无需单独启动数据库服务）。"""

    def __init__(self, persist_dir: str, collection_name: str = "local_rag") -> None:
        # 延迟导入：未安装 chromadb 时，只在真正用到时报错
        import chromadb

        self.chromadb = chromadb
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self._open()

    def _open(self) -> None:
        self.client = self.chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度
        )

    def _rebuild_empty(self) -> None:
        """旧索引损坏（多为中断写入所致）：删掉持久化目录，重建一个空库。"""
        import shutil

        shutil.rmtree(self.persist_dir, ignore_errors=True)
        self._open()

    def add(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 5000) -> None:
        """把块及其向量写入库（upsert：同 ID 覆盖旧块）。

        ChromaDB 单次 upsert 有批次上限（默认约 5461 条），块数多时必须分批，
        否则会报 "Batch size of N is greater than max batch size"。
        """
        if not chunks:
            return
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=[c.chunk_id for c in chunks[start:end]],
                documents=[c.text for c in chunks[start:end]],
                embeddings=embeddings[start:end],
                metadatas=[c.metadata for c in chunks[start:end]],
            )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievalHit]:
        """按查询向量做稠密召回，返回 top_k 个命中。"""
        try:
            n = self.collection.count()
            # 空库时不能向 chromadb 请求 n_results=0（会直接抛 TypeError），直接返回空
            if n == 0:
                return []
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, n),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # 索引损坏：重建空库并返回空命中（随后由 ingest 重新建库）
            self._rebuild_empty()
            return []

        hits: list[RetrievalHit] = []
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            # cosine 距离越小越相似；转成越大越好的分数，便于统一排序
            score = 1.0 / (1.0 + float(dist))
            hits.append(
                RetrievalHit(
                    chunk_id=cid,
                    source=meta.get("source", ""),
                    title=meta.get("title", ""),
                    text=doc,
                    score=score,
                    rank=rank,
                    metadata=meta,
                )
            )
        return hits

    def all_chunks(self) -> list[Chunk]:
        """取出库中全部块（用于在内存重建 BM25 索引）。"""
        data = self.collection.get(include=["documents", "metadatas"])
        chunks: list[Chunk] = []
        for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    source=meta.get("source", ""),
                    title=meta.get("title", ""),
                    text=doc,
                    chunk_index=int(meta.get("chunk_index", 0)),
                    metadata=meta,
                )
            )
        return chunks

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            # 索引损坏（中断写入所致）：自动重建空库，不让服务崩溃
            self._rebuild_empty()
            return 0

    def reset(self) -> None:
        """清空集合（重新 ingest 前调用）。"""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
