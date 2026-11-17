"""ingestion/splitter.py —— 把长文档切成可检索的小块（chunk）。

为什么要切分？
    大模型上下文有限，且语义相似度按“块”计算；块太大召回不准，
    块太小会丢失上下文。chunk_size / chunk_overlap 在 config.yaml 可调。

两种单位：
    char  ：按字符数（默认，稳定、无需额外依赖）
    token ：按 token 数（更贴近模型实际占用，需要 tiktoken）

切分策略：优先按段落（空行分隔）聚合，尽量不在句子中间硬切；
当单段超过 chunk_size 时才硬切，并保留 overlap 重叠，避免跨块语义断裂。
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from .loaders import Chunk, RawDocument, make_chunk_id


def _split_by_char(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按字符数切分，段落优先聚合，块间保留 overlap 重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # 单个段落本身超过 chunk_size：先硬切它
        while len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:chunk_size])
            # 尾部保留 overlap 个字符作为下一段的衔接
            para = para[chunk_size - overlap:] if overlap > 0 else para[chunk_size:]

        # 普通段落拼进当前块；放不下就先收束当前块
        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}" if current else para
        else:
            chunks.append(current)
            prefix = current[-overlap:] if overlap > 0 and current else ""
            current = f"{prefix}\n{para}" if prefix else para

    if current.strip():
        chunks.append(current)
    return chunks


def _split_by_token(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按 token 数切分（需要 tiktoken；未安装则自动退回字符切分）。"""
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "按 token 切分需要 tiktoken，请先 pip install tiktoken，或把 chunk_unit 改为 char"
        ) from exc

    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(encoder.decode(tokens[start:end]))
        start = end - overlap  # 步长 = chunk_size - overlap，形成重叠
    return chunks


def split_text(text: str, chunk_size: int = 500, overlap: int = 80, unit: str = "char") -> list[str]:
    """对外切分函数：把一整段文本切成若干小块字符串。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap 需满足 0 <= overlap < chunk_size")

    if unit == "token":
        return _split_by_token(text, chunk_size, overlap)
    return _split_by_char(text, chunk_size, overlap)


def _resolve_chunk_size(source: str, default: int, per_file: dict) -> int:
    """按文件名/路径匹配 per_file 规则，返回该文件使用的 chunk_size。

    per_file 的键是文件名匹配模式（fnmatch，如 "*.pdf"、"合同.docx"、"sample/*"），
    命中即用对应的值，否则用默认 chunk_size。
    """
    for pattern, size in per_file.items():
        if fnmatch.fnmatch(source, pattern) or fnmatch.fnmatch(Path(source).name, pattern):
            return int(size)
    return default


def build_chunks(
    documents: list[RawDocument],
    chunk_size: int = 500,
    overlap: int = 80,
    unit: str = "char",
    per_file: dict | None = None,
) -> list[Chunk]:
    """把一批 RawDocument 转成 Chunk 列表，附上溯源元数据。

    per_file：{文件名模式: chunk_size}，命中的文件单独使用该 chunk_size。
    """
    per_file = per_file or {}
    chunks: list[Chunk] = []
    for doc in documents:
        size = _resolve_chunk_size(doc.source, chunk_size, per_file)
        texts = split_text(doc.text, chunk_size=size, overlap=overlap, unit=unit)
        for idx, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.source, idx),
                    source=doc.source,
                    title=doc.title,
                    text=text,
                    chunk_index=idx,
                    metadata={
                        "source": doc.source,
                        "title": doc.title,
                        "chunk_index": idx,
                    },
                )
            )
    return chunks
