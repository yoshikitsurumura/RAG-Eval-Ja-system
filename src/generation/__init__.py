"""Generation module for LLM client and prompt templates"""

from .llm_client import (
    LLMClientBase,
    LLMResponse,
    MockLLMClient,
    OpenAIClient,
    get_llm_client,
)
from .prompt_templates import (
    PROMPT_REGISTRY,
    PromptTemplate,
    get_prompt,
)

__all__ = [
    # LLM Client
    "LLMClientBase",
    "OpenAIClient",
    "MockLLMClient",
    "LLMResponse",
    "get_llm_client",
    # Prompts
    "PromptTemplate",
    "PROMPT_REGISTRY",
    "get_prompt",
]
