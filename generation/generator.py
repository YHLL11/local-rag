"""generation/generator.py —— 答案生成编排：检索 -> 组 Prompt -> 流式生成 -> 挂引用。

引用溯源（简历亮点）：
  每条命中块转成一条 Citation，回答结束后随答案一起返回，
  前端 / API 可以展示“这句话来自哪个文件的哪一段”。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.settings import Settings
from retrieval.pipeline import Retriever
from retrieval.vector_store import RetrievalHit

from .llm_client import LLMClient
from .prompt import build_messages, format_context


@dataclass
class Citation:
    """一条来源引用：编号 + 文件名 + 标题 + 原文片段。"""

    index: int
    source: str
    title: str
    snippet: str


@dataclass
class CitedAnswer:
    """最终答案：正文 + 引用列表。"""

    answer: str
    citations: list[Citation] = field(default_factory=list)


class AnswerGenerator:
    def __init__(self, retriever: Retriever, llm: LLMClient, settings: Settings) -> None:
        self.retriever = retriever
        self.llm = llm
        self.settings = settings

    # ---------- 流式 ----------
    def answer_stream(
        self,
        question: str,
        history: list[dict] | None = None,
        use_rerank: bool | None = None,
    ):
        """边检索边生成。

        产出事件：
            ("token", "文本片段")      逐段答案
            ("citations", [Citation])  答案结束后的引用列表
        """
        hits: list[RetrievalHit] = self.retriever.retrieve(question, use_rerank=use_rerank)

        # 知识库为空：不要让模型凭空回答，直接提示先入库
        if self.retriever.store.count() == 0:
            yield (
                "token",
                "知识库还是空的，还没有索引任何文档。请先运行 `python app.py ingest`"
                "（或在界面上传文档点“上传并重建索引”）把资料入库，再来提问。",
            )
            yield ("citations", [])
            return

        context = format_context(hits)
        messages = build_messages(question, context, history)

        citations = [
            Citation(index=h.rank, source=h.source, title=h.title, snippet=h.text)
            for h in hits
        ]

        for token in self.llm.chat_stream(messages):
            yield ("token", token)
        yield ("citations", citations)

    # ---------- 非流式 ----------
    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        use_rerank: bool | None = None,
    ) -> CitedAnswer:
        """一次性生成完整答案（eval / 命令行用）。"""
        answer_parts: list[str] = []
        citations: list[Citation] = []
        for event, payload in self.answer_stream(question, history, use_rerank):
            if event == "token":
                answer_parts.append(payload)
            else:
                citations = payload
        return CitedAnswer(answer="".join(answer_parts), citations=citations)
