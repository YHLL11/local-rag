"""frontend/app.py —— Streamlit 问答界面（上传文档 + 提问 + 查看来源）。

它本身不做检索，而是把请求转发给 FastAPI 后端（默认 http://localhost:8000）。
启动：python app.py web 
"""

from __future__ import annotations

import json

import requests
import streamlit as st

st.set_page_config(page_title="本地 RAG 知识库", page_icon="📚", layout="wide")

BACKEND = st.sidebar.text_input("后端地址", "http://localhost:8000")

# 调本地后端时忽略系统代理：开着 Clash / v2ray 等代理时，requests 默认会把
# localhost 请求也送进本地代理导致 "Unable to connect to proxy"，故关闭环境代理读取。
_local_session = requests.Session()
_local_session.trust_env = False
_local_session.proxies = {"http": None, "https": None}

st.title("📚 本地 RAG 知识库问答")
st.caption("混合检索（向量 + BM25 + RRF）· Cross-Encoder 重排 · 答案带来源引用")

# ---------- 侧边栏：文档管理 ----------
with st.sidebar:
    st.header("文档管理")
    uploaded = st.file_uploader(
        "上传文档（PDF / Markdown / TXT / DOCX）",
        type=["pdf", "md", "txt", "docx"],
    )
    if st.button("上传并重建索引"):
        try:
            if uploaded is not None:
                resp = _local_session.post(
                    f"{BACKEND}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    timeout=300,
                )
            else:
                resp = _local_session.post(f"{BACKEND}/ingest", timeout=300)
            data = resp.json()
            st.success(
                f"已索引 {data['indexed_chunks']} 个块 / {data['documents']} 个文档"
            )
            if data.get("errors"):
                st.warning("；".join(data["errors"]))
        except Exception as exc:  # noqa: BLE001
            st.error(f"索引失败：{exc}")

    st.divider()
    try:
        health = _local_session.get(f"{BACKEND}/health", timeout=3).json()
        st.write(f"**提供方**：{health['provider']}")
        st.write(f"**模型**：{health.get('active_model', health.get('local_model'))}")
        st.write(f"**已索引块数**：{health['chunks_indexed']}")
    except Exception:  # noqa: BLE001
        st.warning("未连接到后端，请先运行 `python app.py api`")

# ---------- 聊天区 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("输入你的问题，回车发送…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        citations: list[dict] = []
        try:
            resp = _local_session.post(
                f"{BACKEND}/chat",
                json={"session_id": "default", "message": prompt},
                stream=True,
                timeout=300,
            )
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                if data["type"] == "token":
                    full_answer += data["text"]
                    answer_placeholder.markdown(full_answer)
                elif data["type"] == "citations":
                    citations = data["citations"]
        except Exception as exc:  # noqa: BLE001
            full_answer = f"请求失败：{exc}"
            answer_placeholder.markdown(full_answer)

        st.session_state.messages.append({"role": "assistant", "content": full_answer})

        if citations:
            with st.expander("📎 参考来源（点击展开）"):
                for cite in citations:
                    st.write(
                        f"[{cite['index']}] **{cite['title']}** —— {cite['source']}"
                    )
                    st.caption(cite["snippet"][:300] + "……")
