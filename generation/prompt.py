"""generation/prompt.py —— RAG Prompt 组装。

要点：
  - 把检索到的块编号 [1][2]…，让模型在回答中用编号引用；
  - 明确要求“只依据给定资料回答”，资料没有就直说不知道，避免幻觉；
  - 对话历史拼成多轮 messages，供多轮问答使用。
"""

from __future__ import annotations

from retrieval.vector_store import RetrievalHit

# 系统提示词：规定回答风格与引用规则
SYSTEM_PROMPT = (
    "你是一个严谨的个人知识库助手。请严格依据下面提供的【参考资料】回答用户问题：\n"
    "1. 只使用参考资料中的信息，不要编造；资料里没有答案时，直接说“根据现有资料无法回答”。\n"
    "2. 回答时用 [1]、[2] 这样的编号标注依据来自哪条资料，编号与资料列表对应。\n"
    "3. 语言简洁准确，先给结论再展开；不要复述资料原文大段摘抄。\n"
)


def format_context(hits: list[RetrievalHit]) -> str:
    """把命中块格式化成带编号的参考资料文本，供 Prompt 使用。"""
    blocks = []
    for hit in hits:
        snippet = hit.text.strip().replace("\n", " ")
        blocks.append(
            f"[{hit.rank}] 《{hit.title}》（来源：{hit.source}）\n{snippet}"
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """组装 OpenAI / Ollama 通用的 messages 结构。

    history: [{"role": "user"/"assistant", "content": ...}, ...]
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 注入历史对话（去掉过长的历史由调用方控制）
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})

    user_content = (
        f"【参考资料】\n{context}\n\n"
        f"【用户问题】\n{question}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages
