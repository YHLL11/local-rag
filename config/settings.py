"""config/settings.py —— 项目唯一的配置入口。

设计思路（配置与代码分离）：
  1. 默认值写在本文件；
  2. 可调检索参数（chunk_size、top-k、rerank 阈值等）从 config/config.yaml 读；
  3. 密钥与环境开关（API Key、本地/云端切换、模型名）从 .env 读；
  4. .env 里的环境变量优先级高于 config.yaml，方便临时覆盖。

使用：
    from config.settings import Settings
    settings = Settings.load()
    print(settings.local_model)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 项目根目录（config/ 的上一级），所有相对路径都以它为基准
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- 环境变量类型转换小工具 ----------
def _env_str(name: str, default: str = "") -> str:
    """读取字符串环境变量，缺失则返回 default。"""
    value = os.environ.get(name)
    return value if value is not None and value != "" else default


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，非法或缺失时回退默认值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


@dataclass
class Settings:
    """集中保存所有运行参数。字段含义见 config.yaml / .env.example 注释。"""

    # --- LLM 提供方 ---
    llm_provider: str = "local"           # local | cloud
    ollama_base_url: str = "http://localhost:11434"
    local_model: str = "DeepSeek-R1-Distill-Qwen-1.5B"
    cloud_base_url: str = ""
    cloud_api_key: str = ""
    cloud_model: str = ""

    # --- Embedding / 设备 ---
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "auto"                  # auto | cuda | cpu

    # --- 文档切分 ---
    chunk_size: int = 1800
    chunk_overlap: int = 80
    chunk_unit: str = "char"              # char | token
    # 按文件名单独覆盖 chunk_size：{匹配模式: chunk_size}，命中的文件用各自的值
    chunk_per_file: dict = field(default_factory=dict)

    # --- 检索 ---
    dense_top_k: int = 10
    sparse_top_k: int = 10
    final_top_k: int = 5
    rrf_k: int = 60

    # --- 重排 ---
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-v2-mini"
    rerank_threshold: float = 0.0

    # --- 路径（相对项目根目录） ---
    data_dir: Path = PROJECT_ROOT / "data"
    chroma_dir: Path = PROJECT_ROOT / "data" / "chroma"
    eval_dataset: Path = PROJECT_ROOT / "eval" / "dataset.example.jsonl"

    # ---------- 加载 ----------
    @classmethod
    def load(
        cls,
        config_path: str | Path | None = None,
        env_path: str | Path | None = None,
    ) -> "Settings":
        """读取 config.yaml 与 .env，合并成一个 Settings 对象。

        顺序：默认值 → config.yaml → 环境变量(.env)，后者覆盖前者。
        """
        config_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
        env_path = Path(env_path) if env_path else PROJECT_ROOT / ".env"

        # 先加载 .env 到进程环境变量（.env 不存在也不报错）
        load_dotenv(env_path, override=False)

        # 读取 yaml（文件缺失时用空字典兜底，保证代码可独立运行）
        yaml_data: dict = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        chunking = yaml_data.get("chunking", {}) or {}
        retrieval = yaml_data.get("retrieval", {}) or {}
        rerank = yaml_data.get("rerank", {}) or {}
        paths = yaml_data.get("paths", {}) or {}

        settings = cls(
            # 环境变量（.env）优先
            llm_provider=_env_str("LLM_PROVIDER", "local"),
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            local_model=_env_str("LOCAL_MODEL", "DeepSeek-R1-Distill-Qwen-1.5B"),
            cloud_base_url=_env_str("CLOUD_BASE_URL"),
            cloud_api_key=_env_str("CLOUD_API_KEY"),
            cloud_model=_env_str("CLOUD_MODEL"),
            embedding_model=_env_str("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            device=_env_str("DEVICE", "auto"),
            # yaml 参数
            chunk_size=int(chunking.get("chunk_size", 500)),
            chunk_overlap=int(chunking.get("chunk_overlap", 80)),
            chunk_unit=str(chunking.get("chunk_unit", "char")),
            chunk_per_file={
                str(k): int(v) for k, v in (chunking.get("per_file") or {}).items()
            },
            dense_top_k=int(retrieval.get("dense_top_k", 10)),
            sparse_top_k=int(retrieval.get("sparse_top_k", 10)),
            final_top_k=int(retrieval.get("final_top_k", 5)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            rerank_enabled=bool(rerank.get("enabled", True)),
            rerank_model=str(rerank.get("model", "BAAI/bge-reranker-v2-mini")),
            rerank_threshold=float(rerank.get("threshold", 0.0)),
            # 路径
            data_dir=(PROJECT_ROOT / paths.get("data_dir", "data")),
            chroma_dir=(PROJECT_ROOT / paths.get("chroma_dir", "data/chroma")),
            eval_dataset=(PROJECT_ROOT / paths.get("eval_dataset", "eval/dataset.example.jsonl")),
        )
        return settings

    # ---------- 便捷方法 ----------
    def ensure_dirs(self) -> None:
        """创建运行所需目录（data、chroma 等），避免首次运行报错。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

    @property
    def using_local_llm(self) -> bool:
        return self.llm_provider.lower() == "local"
