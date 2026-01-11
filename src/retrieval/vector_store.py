"""
ベクトルストアモジュール

Qdrantを使用したベクトル検索
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from src.config import get_settings
from src.ingestion.text_splitter import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """検索結果"""

    chunk_id: str
    content: str
    score: float
    source_file: str
    page_number: int
    metadata: dict


class VectorStoreBase(ABC):
    """ベクトルストア基底クラス"""

    @abstractmethod
    def add_documents(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> int:
        """ドキュメントを追加"""
        pass

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """類似検索"""
        pass

    @abstractmethod
    def delete_collection(self) -> bool:
        """コレクションを削除"""
        pass


class QdrantVectorStore(VectorStoreBase):
    """Qdrantベクトルストア"""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection_name: str | None = None,
        embedding_dimension: int | None = None,
    ):
        settings = get_settings()
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_dimension = embedding_dimension or settings.embedding_dimension

        self.client = QdrantClient(host=self.host, port=self.port)
        self._ensure_collection()

        logger.info(
            f"Initialized Qdrant: {self.host}:{self.port}, collection: {self.collection_name}"
        )

    def _ensure_collection(self):
        """コレクションが存在しない場合は作成"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.embedding_dimension,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self.collection_name}")

    def add_documents(
        self, chunks: list[TextChunk], embeddings: list[list[float]]
    ) -> int:
        """ドキュメントを追加"""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = str(uuid4())
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        **chunk.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Added {len(points)} documents to {self.collection_name}")

        return len(points)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """類似検索"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        search_results = []
        for result in results:
            payload = result.payload or {}
            search_results.append(
                SearchResult(
                    chunk_id=payload.get("chunk_id", ""),
                    content=payload.get("content", ""),
                    score=result.score,
                    source_file=payload.get("source_file", ""),
                    page_number=payload.get("page_number", 0),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k not in ["chunk_id", "content", "source_file", "page_number"]
                    },
                )
            )

        return search_results

    def delete_collection(self) -> bool:
        """コレクションを削除"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_info(self) -> dict:
        """コレクション情報を取得"""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
        }


def get_vector_store(store_type: str = "qdrant", **kwargs) -> VectorStoreBase:
    """ベクトルストアファクトリー"""
    stores = {
        "qdrant": QdrantVectorStore,
    }

    if store_type not in stores:
        raise ValueError(f"Unknown store type: {store_type}. Available: {list(stores.keys())}")

    return stores[store_type](**kwargs)
