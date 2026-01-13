"""
LLMクライアントモジュール

OpenAI APIを使用したテキスト生成
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLMレスポンス"""

    content: str
    model: str
    usage: dict
    metadata: dict


class LLMClientBase(ABC):
    """LLMクライアント基底クラス"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """テキスト生成"""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """チャット形式で生成"""
        pass


class OpenAIClient(LLMClientBase):
    """OpenAI APIクライアント"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        settings = get_settings()
        self.model = model or settings.llm_model
        self.api_key = api_key or settings.openai_api_key

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI Client with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """テキスト生成"""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.chat(messages, temperature=temperature, max_tokens=max_tokens)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """チャット形式で生成"""
        # gpt-5-mini などの新しいモデルは max_completion_tokens を使用
        completion_params = {
            "model": self.model,
            "messages": messages,
        }

        # gpt-5-mini は temperature=0.0 をサポートしていない（デフォルトの1のみ）
        if "gpt-5-mini" not in self.model:
            completion_params["temperature"] = temperature

        # モデルに応じてパラメータを切り替え
        if "gpt-5" in self.model or "gpt-4o" in self.model:
            completion_params["max_completion_tokens"] = max_tokens
        else:
            completion_params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**completion_params)

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            metadata={
                "finish_reason": choice.finish_reason,
            },
        )


class MockLLMClient(LLMClientBase):
    """
    モックLLMクライアント（開発・テスト用）

    API呼び出しを行わず、ダミーレスポンスを返す
    """

    def __init__(self):
        logger.warning("Using MockLLMClient - for development only!")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """ダミー生成"""
        return LLMResponse(
            content=f"[Mock Response] This is a mock response to: {prompt[:100]}...",
            model="mock-model",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            metadata={"mock": True},
        )

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """ダミーチャット"""
        last_message = messages[-1]["content"] if messages else ""
        return self.generate(last_message)


def get_llm_client(client_type: str = "openai", **kwargs) -> LLMClientBase:
    """LLMクライアントファクトリー"""
    clients = {
        "openai": OpenAIClient,
        "mock": MockLLMClient,
    }

    if client_type not in clients:
        raise ValueError(
            f"Unknown client type: {client_type}. Available: {list(clients.keys())}"
        )

    return clients[client_type](**kwargs)
