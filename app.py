#!/usr/bin/env python3
"""app.py —— 项目统一命令行入口。

子命令：
    python app.py ingest              # 把 data/ 下的文档入库（先跑这个）
    python app.py api                 # 启动 FastAPI 后端（默认 127.0.0.1:8000）
    python app.py web                 # 启动 Streamlit 问答界面
    python app.py eval                # 跑 Recall@k / MRR 基线对比
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_ingest(_args: argparse.Namespace) -> None:
    """读取 data 目录 -> 切分 -> 向量化 -> 写入 ChromaDB + 重建 BM25。"""
    from config.settings import Settings
    from ingestion.loaders import load_directory
    from ingestion.splitter import build_chunks
    from retrieval.pipeline import Retriever

    settings = Settings.load()
    settings.ensure_dirs()

    documents, errors = load_directory(settings.data_dir)
    total_chars = sum(len(d.text) for d in documents)
    print(f"已读取 {len(documents)} 个文档，共 {total_chars} 字符，开始切分…")
    chunks = build_chunks(
        documents,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        unit=settings.chunk_unit,
        per_file=settings.chunk_per_file,
    )
    print(f"切分为 {len(chunks)} 个块，开始向量化（CPU 上可能需要一两分钟，期间无进度条属正常）…")
    if len(chunks) > 2000:
        print(
            f"⚠️  块数异常多（{len(chunks)}）。请检查 data/ 是否误放了超大文件，"
            "或 config.yaml 的 chunk_size 是否被改成过小的值；"
            "块太多会让 CPU 向量化非常慢。"
        )

    retriever = Retriever(settings)
    retriever.index(chunks)

    print(f"✅ 入库完成：{len(documents)} 个文档 -> {len(chunks)} 个块")
    print(f"   向量库：{settings.chroma_dir}")
    for err in errors:
        print(f"⚠️  {err}")


def cmd_api(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_web(args: argparse.Namespace) -> None:
    """用 streamlit 启动前端界面。"""
    frontend = ROOT / "frontend" / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(frontend),
            "--server.port",
            str(args.port),
            "--server.headless",
            "true",
        ],
        check=True,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    from eval.run_eval import run

    run(dataset_path=args.dataset, top_k=args.k)


def main() -> None:
    parser = argparse.ArgumentParser(description="本地 RAG 知识库命令行")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="把 data/ 文档索引入库")
    p_ingest.set_defaults(func=cmd_ingest)

    p_api = sub.add_parser("api", help="启动 FastAPI 后端")
    p_api.add_argument("--host", default="127.0.0.1")
    p_api.add_argument("--port", type=int, default=8000)
    p_api.add_argument("--reload", action="store_true")
    p_api.set_defaults(func=cmd_api)

    p_web = sub.add_parser("web", help="启动 Streamlit 界面")
    p_web.add_argument("--port", type=int, default=8501)
    p_web.set_defaults(func=cmd_web)

    p_eval = sub.add_parser("eval", help="检索基线对比（Recall@k / MRR）")
    p_eval.add_argument("--dataset", default=None, help="自定义 JSONL 测试集路径")
    p_eval.add_argument("--k", type=int, default=5, help="Recall@k 的 k，默认 5")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
