"""ingestion 包：文档加载（PDF / Markdown / TXT / DOCX）与切分。

对外接口：
    from ingestion.loaders import load_document, load_directory, Chunk
    from ingestion.splitter import split_text, build_chunks
"""

from .loaders import Chunk, RawDocument, load_document, load_directory  # noqa: F401
from .splitter import build_chunks, split_text  # noqa: F401
