"""api/main.py —— FastAPI 后端：文档入库 + 流式问答 + 会话历史。

接口：
    GET  /health            健康检查
    POST /ingest            上传文件 / 重新索引 data 目录
    POST /chat              流式问答（SSE），带会话历史

启动：python app.py api   （等价 uvicorn api.main:app）
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import Settings
from generation.generator import AnswerGenerator
from generation.llm_client import LLMClient
from ingestion.loaders import load_directory
from ingestion.splitter import build_chunks
from retrieval.pipeline import Retriever

app = FastAPI(title="Local-RAG", version="0.1.0")

# 会话历史：{session_id: [{"role","content"}, ...]}，内存保存，重启清空
_sessions: dict[str, list[dict]] = {}
_generator: AnswerGenerator | None = None


def get_generator() -> AnswerGenerator:
    """懒加载：首次请求时才初始化检索/生成组件（避免启动就下载模型）。"""
    global _generator
    if _generator is None:
        settings = Settings.load()
        settings.ensure_dirs()
        retriever = Retriever(settings)
        llm = LLMClient(settings)
        _generator = AnswerGenerator(retriever, llm, settings)
    return _generator


def _trim_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    """只保留最近 max_turns 轮对话，避免上下文无限膨胀。"""
    return history[-max_turns * 2 :]


class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str


@app.get("/health")
def health() -> dict:
    gen = get_generator()
    try:
        chunks_indexed = gen.retriever.store.count()
        index_ok = True
        note = ""
    except Exception as exc:  # 索引损坏/未建时不让健康检查直接 500
        chunks_indexed = -1
        index_ok = False
        note = f"索引读取失败（通常是旧库损坏，请删除 data/chroma 后重新 ingest）：{exc}"
         # 当前实际生效的模型：local 用本地模型，cloud 用云端模型
    active_model = (
        gen.settings.local_model
        if gen.settings.llm_provider == "local"
        else (gen.settings.cloud_model or "（未填写 CLOUD_MODEL）")
    )
    return {
        "status": "ok" if index_ok else "index_error",
        "provider": gen.settings.llm_provider,
        "active_model": active_model,
        "local_model": gen.settings.local_model,
        "chunks_indexed": chunks_indexed,
        "note": note,
    }


@app.post("/ingest")
async def ingest(file: UploadFile | None = File(default=None)) -> dict:
    """保存上传的文件，并对 data 目录全量重新索引。"""
    settings = get_generator().settings
    settings.ensure_dirs()

    if file is not None:
        # 只取文件名，避免上传路径中的分隔符逃逸 data 目录
        save_path = settings.data_dir / Path(file.filename).name
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)

    documents, errors = load_directory(settings.data_dir)
    chunks = build_chunks(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        unit=settings.chunk_unit,
        per_file=settings.chunk_per_file,
    )
    get_generator().retriever.index(chunks)

    return {
        "indexed_chunks": len(chunks),
        "documents": len(documents),
        "errors": errors,
    }


@app.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """流式问答：SSE 推送 token，结束推送 citations。"""
    history = _sessions.setdefault(req.session_id, [])
    generator = get_generator()

    def event_stream():
        answer_text = ""
        for event, payload in generator.answer_stream(
            req.message, history=_trim_history(history)
        ):
            if event == "token":
                answer_text += payload
                yield f"data: {json.dumps({'type': 'token', 'text': payload}, ensure_ascii=False)}\n\n"
            else:
                yield (
                    f"data: {json.dumps({'type': 'citations', 'citations': [asdict(c) for c in payload]}, ensure_ascii=False)}\n\n"
                )
        # 问答结束后把本轮对话写进历史
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": answer_text})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
