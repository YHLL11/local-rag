"""generation 包：Prompt 组装、调用 LLM、带引用溯源的答案生成。

对外接口：
    from generation.generator import AnswerGenerator
    from generation.llm_client import LLMClient
"""

from .generator import AnswerGenerator, CitedAnswer  # noqa: F401
from .llm_client import LLMClient  # noqa: F401
