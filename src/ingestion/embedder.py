"""
埋め込み生成モジュール

OpenAI Embeddings APIを使用してテキストをベクトル化
"""

import logging
from abc import ABC, abstractmethod

from openai import OpenAI

from src.config import get_settings
from src.ingestion.text_splitter import TextChunk

logger = logging.getLogger(__name__)


class EmbedderBase(ABC):
    """埋め込み生成基底クラス"""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """単一テキストを埋め込み"""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括埋め込み"""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
        """チャンクを埋め込み"""
        pass


class OpenAIEmbedder(EmbedderBase):
    """OpenAI Embeddings API を使用した埋め込み生成"""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 100,
    ):
        settings = get_settings()
        self.model = model or settings.embedding_model
        self.api_key = api_key or settings.openai_api_key
        self.batch_size = batch_size

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAI Embedder with model: {self.model}")

    def embed_text(self, text: str) -> list[float]:
        """単一テキストを埋め込み"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括埋め込み（バッチ処理）"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(f"Embedding batch {i // self.batch_size + 1} ({len(batch)} texts)")

            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )

            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
        """チャンクを埋め込み"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts)

        return list(zip(chunks, embeddings, strict=True))


class MockEmbedder(EmbedderBase):
    """
    モック埋め込み生成（開発・テスト用）

    API呼び出しを行わず、ダミーベクトルを生成
    """

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        logger.warning("Using MockEmbedder - for development only!")

    def embed_text(self, text: str) -> list[float]:
        """ダミー埋め込みを生成"""
        import hashlib

        # テキストのハッシュから決定的なベクトルを生成
        hash_bytes = hashlib.sha256(text.encode()).digest()
        vector = []
        for i in range(self.dimension):
            byte_idx = i % len(hash_bytes)
            vector.append((hash_bytes[byte_idx] / 255.0) * 2 - 1)
        return vector

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """複数テキストのダミー埋め込み"""
        return [self.embed_text(text) for text in texts]

    def embed_chunks(self, chunks: list[TextChunk]) -> list[tuple[TextChunk, list[float]]]:
        """チャンクのダミー埋め込み"""
        return [(chunk, self.embed_text(chunk.content)) for chunk in chunks]


def get_embedder(embedder_type: str = "openai", **kwargs) -> EmbedderBase:
    """埋め込み生成器ファクトリー"""
    embedders = {
        "openai": OpenAIEmbedder,
        "mock": MockEmbedder,
    }

    if embedder_type not in embedders:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. Available: {list(embedders.keys())}"
        )

    return embedders[embedder_type](**kwargs)
