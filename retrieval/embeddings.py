"""retrieval/embeddings.py —— 文本向量化服务（sentence-transformers）。

模型选中文小模型 BAAI/bge-small-zh-v1.5：
"""

from __future__ import annotations

import os

# 强制 HuggingFace Hub 离线：模型已在本地缓存，跳过所有联网检查/HEAD 请求，
# 避免国内网络下反复 SSL EOF 重试。若以后需要首次联网下载新模型，
# 可临时去掉下面这行。
# os.environ.setdefault("HF_HUB_OFFLINE", "1")


class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # 延迟加载

    def _resolve_device(self) -> str:
        """auto：有 CUDA 显卡用 cuda，否则退回 cpu。"""
        if self.device in {"cuda", "cpu"}:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.model_name,
            device=self._resolve_device(),
             local_files_only=True,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """批量把文本编码为向量；normalize 便于余弦相似度比较。"""
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [list(map(float, v)) for v in vectors]

    def encode_query(self, text: str) -> list[float]:
        """编码单条查询。bge 系列对查询加指令前缀效果更好，这里保持简单。"""
        return self.encode([text])[0]
