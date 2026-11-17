"""retrieval/rerank.py —— Cross-Encoder 重排（可选模块）。

它解决什么问题？
  向量/BM25 是“双塔”：查询和文档分别编码再算相似度，快但粗；
  rerank 是“交叉编码器”，把 (查询, 块) 拼在一起做深度交互打分，
  能从融合后的候选里把真正最相关的几个顶上去，提升准确率。

为什么做成可开关？
  rerank 模型也要占显存/算力，GTX 1650 Ti 4GB 紧张时可在
  config.yaml 把 rerank.enabled 设为 false，直接返回融合结果。
"""

from __future__ import annotations

from .vector_store import RetrievalHit


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-base",
        enabled: bool = True,
        threshold: float = 0.0,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.threshold = threshold
        self.device = device
        self._model = None
        self._available = True  # 模型能否加载；加载失败则自动降级为不重排

    def _resolve_device(self) -> str:
        if self.device in {"cuda", "cpu"}:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None or not self._available:
            return
        from pathlib import Path

        from sentence_transformers import CrossEncoder

        # 优先用项目内 models/reranker 本地目录（离线可用）；不存在则按模型名联网下载
        local_dir = Path("models") / "reranker"
        source = str(local_dir) if local_dir.exists() else self.model_name
        try:
            self._model = CrossEncoder(source, device=self._resolve_device())
        except Exception as exc:
            # 模型未下载到本地 / 离线 / 仓库不存在：不崩，降级为不重排
            self._available = False
            print(f"⚠️  rerank 模型不可用（{exc}），本次跳过重排，直接使用融合结果。")

    def rerank(
        self,
        query: str,
        hits: list[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        """对融合后的候选重排；关闭或模型不可用时原样返回。

        threshold > 0 时，低于阈值的块被过滤掉（不同模型分数尺度不同，
        默认 0.0 表示不过滤、只排序）。
        """
        if not hits:
            return []
        if not self.enabled or not self._available:
            return hits[:top_k] if top_k else hits

        self._load()
        if self._model is None:  # 加载失败后的兜底
            return hits[:top_k] if top_k else hits
        pairs = [(query, h.text) for h in hits]
        scores = self._model.predict(pairs)

        scored = list(zip(hits, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        results: list[RetrievalHit] = []
        for rank, (hit, score) in enumerate(scored, start=1):
            if self.threshold > 0.0 and float(score) < self.threshold:
                continue
            hit.score = float(score)
            hit.rank = rank
            results.append(hit)
            if top_k is not None and len(results) >= top_k:
                break
        return results
