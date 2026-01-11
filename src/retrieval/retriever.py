"""
リトリーバーモジュール

クエリに基づいて関連ドキュメントを検索
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.ingestion.embedder import EmbedderBase, get_embedder
from src.retrieval.vector_store import SearchResult, VectorStoreBase, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """検索結果"""

    query: str
    results: list[SearchResult]
    metadata: dict


class RetrieverBase(ABC):
    """リトリーバー基底クラス"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """クエリに関連するドキュメントを検索"""
        pass


class SimpleRetriever(RetrieverBase):
    """
    シンプルリトリーバー

    クエリを埋め込み → ベクトル検索
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        logger.info("Initialized SimpleRetriever")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """クエリに関連するドキュメントを検索"""
        # クエリを埋め込み
        query_embedding = self.embedder.embed_text(query)

        # ベクトル検索
        results = self.vector_store.search(query_embedding, top_k=top_k)

        logger.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")

        return RetrievalResult(
            query=query,
            results=results,
            metadata={"retriever": "simple", "top_k": top_k},
        )


class HybridRetriever(RetrieverBase):
    """
    ハイブリッドリトリーバー

    ベクトル検索 + キーワード検索（将来拡張用）
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
        alpha: float = 0.7,  # ベクトル検索の重み
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.alpha = alpha
        logger.info(f"Initialized HybridRetriever with alpha={alpha}")

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """ハイブリッド検索（現在はベクトル検索のみ）"""
        # ベクトル検索
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        # TODO: キーワード検索を追加し、スコアを統合

        return RetrievalResult(
            query=query,
            results=vector_results,
            metadata={"retriever": "hybrid", "top_k": top_k, "alpha": self.alpha},
        )


class MultiQueryRetriever(RetrieverBase):
    """
    マルチクエリリトリーバー

    元のクエリから複数の類似クエリを生成し、検索結果を統合
    AgenticRAGで使用
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
        num_queries: int = 3,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.num_queries = num_queries
        logger.info(f"Initialized MultiQueryRetriever with num_queries={num_queries}")

    def generate_queries(self, original_query: str) -> list[str]:
        """
        元のクエリから複数のクエリを生成

        注: 実際にはLLMを使用してクエリを生成するが、
        ここではシンプルな実装としてオリジナルのみ返す
        """
        # TODO: LLMを使用してクエリバリエーションを生成
        return [original_query]

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """マルチクエリ検索"""
        queries = self.generate_queries(query)

        all_results: dict[str, SearchResult] = {}

        for q in queries:
            query_embedding = self.embedder.embed_text(q)
            results = self.vector_store.search(query_embedding, top_k=top_k)

            for result in results:
                # 重複を除去（chunk_idで判定）、最高スコアを保持
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                elif result.score > all_results[result.chunk_id].score:
                    all_results[result.chunk_id] = result

        # スコア順にソート
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)

        return RetrievalResult(
            query=query,
            results=sorted_results[:top_k],
            metadata={
                "retriever": "multi_query",
                "num_queries": len(queries),
                "top_k": top_k,
            },
        )


def get_retriever(
    retriever_type: str = "simple",
    vector_store: VectorStoreBase | None = None,
    embedder: EmbedderBase | None = None,
    **kwargs,
) -> RetrieverBase:
    """リトリーバーファクトリー"""
    retrievers = {
        "simple": SimpleRetriever,
        "hybrid": HybridRetriever,
        "multi_query": MultiQueryRetriever,
    }

    if retriever_type not in retrievers:
        raise ValueError(
            f"Unknown retriever type: {retriever_type}. Available: {list(retrievers.keys())}"
        )

    return retrievers[retriever_type](
        vector_store=vector_store, embedder=embedder, **kwargs
    )
