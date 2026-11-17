"""eval/run_eval.py —— 检索基线对比脚本（无 rerank vs 混合检索+rerank）。

用法：
    python app.py eval
    python app.py eval --dataset eval/my_dataset.jsonl --k 5

数据集格式（JSONL，每行一条）：
    {"question": "问题", "answer": "参考答案(仅人工对照，可不填)",
     "relevant_docs": ["文件名.md"]}

输出：控制台对比表 + eval/report.md，方便写简历量化结果
（例如：混合检索+rerank 使 Recall@5 从 X% 提升到 Y%）。
"""

from __future__ import annotations

import json
from pathlib import Path

from config.settings import Settings
from retrieval.pipeline import Retriever
from retrieval.vector_store import RetrievalHit

from .metrics import average  # noqa: F401


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _evaluate_hits(hits: list[RetrievalHit], relevant_names: set[str], k: int) -> tuple[float, float]:
    """按“文档名”统计 Recall@k 与 MRR。"""
    matched_docs: set[str] = set()
    first_rank: int | None = None
    for rank, hit in enumerate(hits[:k], start=1):
        doc_name = Path(hit.source).name
        if doc_name in relevant_names:
            matched_docs.add(doc_name)
            if first_rank is None:
                first_rank = rank
    recall = len(matched_docs) / len(relevant_names) if relevant_names else 0.0
    mrr = 1.0 / first_rank if first_rank else 0.0
    return recall, mrr


def run(dataset_path: str | None = None, top_k: int = 5) -> dict:
    settings = Settings.load()
    settings.ensure_dirs()
    dataset_path = Path(dataset_path) if dataset_path else settings.eval_dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"测试集不存在：{dataset_path}")

    dataset = load_dataset(dataset_path)
    retriever = Retriever(settings)

    base_recalls, base_mrrs, enh_recalls, enh_mrrs = [], [], [], []
    rows: list[dict] = []

    for item in dataset:
        question = item["question"]
        relevant = set(item.get("relevant_docs", []))

        # 方案 A：仅混合检索（向量+BM25+RRF），关闭 rerank
        base_hits = retriever.retrieve(question, use_rerank=False)
        base_r, base_m = _evaluate_hits(base_hits, relevant, top_k)

        # 方案 B：混合检索 + cross-encoder rerank
        enh_hits = retriever.retrieve(question, use_rerank=True)
        enh_r, enh_m = _evaluate_hits(enh_hits, relevant, top_k)

        base_recalls.append(base_r)
        base_mrrs.append(base_m)
        enh_recalls.append(enh_r)
        enh_mrrs.append(enh_m)
        rows.append(
            {
                "question": question,
                "base_recall": base_r,
                "base_mrr": base_m,
                "enh_recall": enh_r,
                "enh_mrr": enh_m,
            }
        )

    result = {
        "count": len(dataset),
        "base_recall@k": average(base_recalls),
        "base_mrr": average(base_mrrs),
        "enh_recall@k": average(enh_recalls),
        "enh_mrr": average(enh_mrrs),
        "rows": rows,
        "k": top_k,
    }
    _print_report(result)
    _write_report(result)
    return result


def _print_report(r: dict) -> None:
    k = r["k"]
    print("\n===== 检索基线对比 =====")
    print(f"测试集条数：{r['count']}")
    print(f"{'方案':<24} Recall@{k}   MRR")
    print(f"{'混合检索(无rerank)':<22} {r['base_recall@k']:.4f}      {r['base_mrr']:.4f}")
    print(f"{'混合检索+rerank':<22} {r['enh_recall@k']:.4f}      {r['enh_mrr']:.4f}")
    print(
        f"Recall@{k} 提升：{(r['enh_recall@k'] - r['base_recall@k']) * 100:.2f} 个百分点\n"
    )


def _write_report(r: dict) -> None:
    k = r["k"]
    out = Path(__file__).parent / "report.md"
    lines = [
        "# 检索基线对比报告",
        "",
        f"- 测试集条数：{r['count']}",
        f"- 指标：Recall@{k}、MRR",
        "",
        "| 方案 | Recall@%d | MRR |" % k,
        "| --- | --- | --- |",
        f"| 混合检索（向量+BM25+RRF，无 rerank） | {r['base_recall@k']:.4f} | {r['base_mrr']:.4f} |",
        f"| 混合检索 + Cross-Encoder rerank | {r['enh_recall@k']:.4f} | {r['enh_mrr']:.4f} |",
        "",
        f"Recall@{k} 提升 {(r['enh_recall@k'] - r['base_recall@k']) * 100:.2f} 个百分点。",
        "",
        "## 逐题明细",
        "",
        "| 问题 | 无rerank Recall | 无rerank MRR | 有rerank Recall | 有rerank MRR |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in r["rows"]:
        lines.append(
            f"| {row['question']} | {row['base_recall']:.2f} | {row['base_mrr']:.2f} "
            f"| {row['enh_recall']:.2f} | {row['enh_mrr']:.2f} |"
        )
    out.write_text("\n".join(lines), encoding="utf-8")
