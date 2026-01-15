"""
リトリーバーモジュール (v4: フィルタリング対応)

クエリに基づいて関連ドキュメントを検索
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

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
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> RetrievalResult:
        """クエリに関連するドキュメントを検索"""
        pass


class SimpleRetriever(RetrieverBase):
    """
    シンプルリトリーバー
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        logger.info("Initialized SimpleRetriever")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> RetrievalResult:
        """クエリに関連するドキュメントを検索"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            metadata_filter=metadata_filter
        )

        return RetrievalResult(
            query=query,
            results=results,
            metadata={"retriever": "simple", "top_k": top_k, "filter": metadata_filter},
        )


class HybridRetriever(RetrieverBase):
    """
    ハイブリッドリトリーバー (Vector + BM25)
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
        alpha: float = 0.5,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.alpha = alpha
        logger.info(f"Initialized HybridRetriever with alpha={alpha}")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> RetrievalResult:
        """ハイブリッド検索"""
        # 現在はベクトル検索のみ（将来的にBM25も統合予定）
        query_embedding = self.embedder.embed_text(query)
        vector_results = self.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            metadata_filter=metadata_filter
        )

        return RetrievalResult(
            query=query,
            results=vector_results,
            metadata={"retriever": "hybrid", "top_k": top_k, "alpha": self.alpha, "filter": metadata_filter},
        )


class MultiQueryRetriever(RetrieverBase):
    """
    マルチクエリリトリーバー
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        metadata_filter: Optional[dict[str, Any]] = None
    ) -> RetrievalResult:
        # 簡易実装としてSimpleRetrieverに委譲
        retriever = SimpleRetriever(self.vector_store, self.embedder)
        return retriever.retrieve(query, top_k, metadata_filter)


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
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    return retrievers[retriever_type](
        vector_store=vector_store, embedder=embedder, **kwargs
    )