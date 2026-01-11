"""
RAG基底クラスモジュール

全てのRAG実装の抽象基底クラスを定義
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from src.retrieval.vector_store import SearchResult


@dataclass
class RAGResponse:
    """RAGレスポンス"""

    question: str
    answer: str
    sources: list[SearchResult]
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "content": s.content[:200] + "..." if len(s.content) > 200 else s.content,
                    "source_file": s.source_file,
                    "page_number": s.page_number,
                    "score": s.score,
                }
                for s in self.sources
            ],
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class RAGBase(ABC):
    """
    RAG基底クラス

    全てのRAG実装はこのクラスを継承する
    オブジェクト指向設計に基づき、共通インターフェースを定義
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """RAG実装の名前"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """RAG実装の説明"""
        pass

    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """
        質問に対して回答を生成

        Args:
            question: ユーザーの質問

        Returns:
            RAGResponse: 回答とソース情報を含むレスポンス
        """
        pass

    def _format_context(self, sources: list[SearchResult]) -> str:
        """
        検索結果をコンテキスト文字列にフォーマット

        Args:
            sources: 検索結果リスト

        Returns:
            フォーマットされたコンテキスト文字列
        """
        context_parts = []
        for i, source in enumerate(sources, start=1):
            context_parts.append(
                f"[{i}] 出典: {source.source_file} (p.{source.page_number})\n{source.content}"
            )
        return "\n\n".join(context_parts)
