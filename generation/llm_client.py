"""generation/llm_client.py —— 统一的 LLM 调用客户端（本地 Ollama / 云端可切换）。

两条链路共用一套 messages 格式：
  - local：POST 到 Ollama 的 /api/chat（NDJSON 流式）
  - cloud：POST 到任意 OpenAI 兼容 /v1/chat/completions（SSE 流式）

切换方式：改 .env 的 LLM_PROVIDER，无需改代码。
"""

from __future__ import annotations

import json
from typing import Iterator

import requests

from config.settings import Settings


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ---------- 对外主接口 ----------
    def chat_stream(self, messages: list[dict]) -> Iterator[str]:
        """流式生成：逐段 yield 回答文本。"""
        if self.settings.using_local_llm:
            yield from self._ollama_stream(messages)
        else:
            yield from self._cloud_stream(messages)

    def chat(self, messages: list[dict]) -> str:
        """非流式：一次性返回完整回答。"""
        return "".join(self.chat_stream(messages))

    # ---------- 本地 Ollama ----------
    def _ollama_stream(self, messages: list[dict]):
        url = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.settings.local_model,
            "messages": messages,
            "stream": True,
        }
        try:
            # Ollama 是本地服务，必须绕过系统代理（否则开着 Clash 等代理时被劫持）；
            # 显式传空 proxies 表示不使用任何代理。
            resp = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=180,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "连不上 Ollama：请先启动 Ollama 桌面版，并确认已执行 "
                f"`ollama pull {self.settings.local_model}`。原始错误：{exc}"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Ollama 返回错误：{exc}") from exc

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                break

    # ---------- 云端 OpenAI 兼容 API ----------
    def _cloud_stream(self, messages: list[dict]):
        if not self.settings.cloud_api_key:
            raise RuntimeError("LLM_PROVIDER=cloud 但未在 .env 填写 CLOUD_API_KEY")

        url = f"{self.settings.cloud_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.cloud_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.cloud_model,
            "messages": messages,
            "stream": True,
        }
        try:
            # DeepSeek 等国内 API 直连即可，显式绕过系统代理，避免坏代理 ProxyError。
            # 若改用必须翻墙的境外服务（如 OpenAI），删掉下面的 proxies 参数即可走环境代理。
            resp = requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True,
                timeout=180,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"连不上云端 API：{exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"云端 API 返回错误：{exc}") from exc

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            data = json.loads(body)
            choices = data.get("choices") or [{}]
            delta = choices[0].get("delta", {}).get("content", "")
            if delta:
                yield delta
