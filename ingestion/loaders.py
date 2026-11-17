"""ingestion/loaders.py —— 把各种格式的文档读成统一的纯文本。

支持：
    .pdf  （pypdf 抽取每页文本）
    .docx （python-docx 抽取所有段落）
    .md / .markdown / .txt （按 UTF-8 直接读）

设计：每个文件 -> 一个 RawDocument（原文），切分交给 splitter.py。
加载失败的文件不会让整个流程崩溃，而是记录到 errors 里返回。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

# 支持的扩展名 -> 加载方式
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}


@dataclass
class RawDocument:
    """一个文件读出来的原始文本（尚未切分）。"""

    source: str            # 文件相对路径（用于溯源展示）
    title: str             # 文档标题（默认取文件名）
    text: str              # 全文纯文本


@dataclass
class Chunk:
    """切分后的一个文本块，是检索 / 生成的最小单位。

    字段会被写进 ChromaDB 的 metadata，回答时用于“引用溯源”。
    """

    chunk_id: str          # 全局唯一 ID（重新入库时用于覆盖旧块）
    source: str           # 来源文件名 / 路径
    title: str             # 文档标题
    text: str              # 块文本
    chunk_index: int      # 该块在原文中的序号
    metadata: dict = field(default_factory=dict)


# ---------- 各格式加载器 ----------
def _load_pdf(path: Path) -> str:
    """用 pypdf 抽取 PDF 全部页面文本。"""
    from pypdf import PdfReader  # 延迟导入：未装依赖时只在用到时报错

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _load_docx(path: Path) -> str:
    """用 python-docx 抽取 Word 段落文本。"""
    from docx import Document

    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _load_plain_text(path: Path) -> str:
    """读取 Markdown / TXT 纯文本（UTF-8，容错乱码）。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------- 对外主接口 ----------
def load_document(path: Path | str) -> RawDocument:
    """读取单个文档，按扩展名分派到对应加载器。

    异常会向上抛出，由调用方（load_directory / CLI）统一收集处理。
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        text = _load_pdf(path)
    elif suffix == ".docx":
        text = _load_docx(path)
    elif suffix in {".md", ".markdown", ".txt"}:
        text = _load_plain_text(path)
    else:
        raise ValueError(f"不支持的文件类型：{path.name}（支持 {SUPPORTED_EXTENSIONS}）")

    return RawDocument(
        source=path.name,
        title=path.stem,
        text=text.strip(),
    )


def load_directory(dir_path: Path | str) -> tuple[list[RawDocument], list[str]]:
    """递归读取目录下所有支持的文档。

    返回 (文档列表, 错误信息列表)；单个文件失败只记录，不中断。
    """
    dir_path = Path(dir_path)
    documents: list[RawDocument] = []
    errors: list[str] = []

    if not dir_path.exists():
        return documents, [f"目录不存在：{dir_path}"]

    for file_path in sorted(dir_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            doc = load_document(file_path)
            # source 记录相对路径，方便溯源时知道文件位置
            doc.source = str(file_path.relative_to(dir_path.parent))
            if doc.text:
                documents.append(doc)
            else:
                errors.append(f"{file_path.name}：抽取到的文本为空（可能是扫描件 PDF）")
        except Exception as exc:  # noqa: BLE001 —— 这里要兜底所有加载异常
            errors.append(f"{file_path.name}：加载失败：{exc}")

    return documents, errors


def make_chunk_id(source: str, chunk_index: int) -> str:
    """生成稳定的块 ID：同一文件同一位置每次入库 ID 相同，便于覆盖更新。"""
    digest = hashlib.md5(f"{source}::{chunk_index}".encode("utf-8")).hexdigest()[:8]
    return f"{digest}-{chunk_index:04d}"
