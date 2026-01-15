"""
リトリーバーモジュール

クエリに基づいて関連ドキュメントを検索
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.ingestion.embedder import EmbedderBase, get_embedder
from src.retrieval.reranker import Reranker
from src.retrieval.vector_store import SearchResult, VectorStoreBase, get_vector_store

logger = logging.getLogger(__name__)

# 日本語トークナイザー（軽量版）
try:
    from sudachipy import Dictionary, SplitMode

    SUDACHI_AVAILABLE = True
    _tokenizer_obj = Dictionary().create()
except ImportError:
    SUDACHI_AVAILABLE = False
    logger.warning("sudachipy not available, falling back to character-level tokenization")


def tokenize_japanese(text: str) -> list[str]:
    """日本語テキストをトークン化"""
    if SUDACHI_AVAILABLE:
        # Sudachiで形態素解析
        tokens = [m.surface() for m in _tokenizer_obj.tokenize(text, SplitMode.C)]
        return tokens
    else:
        # フォールバック: 文字単位 + 空白分割
        return list(text.replace(" ", ""))


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

    ベクトル検索（意味検索）+ BM25（キーワード検索）を組み合わせ、
    Reciprocal Rank Fusion (RRF)で統合
    """

    def __init__(
        self,
        vector_store: VectorStoreBase | None = None,
        embedder: EmbedderBase | None = None,
        alpha: float = 0.5,  # ベクトル検索の重み（0.0-1.0）
        rrf_k: int = 60,  # RRFパラメータ
        use_rerank: bool = True,  # リランクを使用するか
        reranker: Reranker | None = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.use_rerank = use_rerank
        self.reranker = reranker or (Reranker() if use_rerank else None)

        # BM25インデックス構築
        self._build_bm25_index()

        logger.info(
            f"Initialized HybridRetriever: alpha={alpha}, rrf_k={rrf_k}, use_rerank={use_rerank}"
        )

    def _build_bm25_index(self):
        """全チャンクをBM25インデックスに登録"""
        logger.info("Building BM25 index from vector store...")

        # Qdrantから全チャンク取得
        all_chunks = self.vector_store.get_all_chunks()

        if not all_chunks:
            logger.warning("No chunks found in vector store for BM25 index")
            self.bm25 = None
            self.bm25_chunks = []
            self.bm25_chunk_map = {}
            return

        # チャンク内容をトークン化
        self.bm25_chunks = all_chunks
        tokenized_corpus = [tokenize_japanese(chunk.content) for chunk in all_chunks]

        # BM25インデックス構築
        self.bm25 = BM25Okapi(tokenized_corpus)

        # chunk_id -> SearchResultのマッピング
        self.bm25_chunk_map = {chunk.chunk_id: chunk for chunk in all_chunks}

        logger.info(f"BM25 index built with {len(all_chunks)} chunks")

    def _vector_search(self, query: str, top_k: int) -> list[SearchResult]:
        """ベクトル検索"""
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.debug(f"Vector search returned {len(results)} results")
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[SearchResult]:
        """BM25検索"""
        if self.bm25 is None:
            logger.warning("BM25 index not available, returning empty results")
            return []

        # クエリをトークン化
        query_tokens = tokenize_japanese(query)

        # BM25スコア計算
        bm25_scores = self.bm25.get_scores(query_tokens)

        # スコア順にソートしてtop_k件取得
        top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]

        # SearchResultに変換
        results = []
        for idx in top_indices:
            chunk = self.bm25_chunks[idx]
            # BM25スコアを正規化（0-1範囲に）
            score = float(bm25_scores[idx]) / (bm25_scores[idx] + 1.0)
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=score,
                    source_file=chunk.source_file,
                    page_number=chunk.page_number,
                    metadata=chunk.metadata,
                )
            )

        logger.debug(f"BM25 search returned {len(results)} results")
        return results

    def _reciprocal_rank_fusion(
        self,
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF)で2つのランキングを統合

        RRFスコア = Σ 1 / (k + rank)
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        # ベクトル検索のスコア
        for rank, result in enumerate(vector_results):
            rrf_score = self.alpha * (1.0 / (self.rrf_k + rank + 1))
            scores[result.chunk_id] = rrf_score
            chunk_map[result.chunk_id] = result

        # BM25検索のスコアを加算
        for rank, result in enumerate(bm25_results):
            rrf_score = (1.0 - self.alpha) * (1.0 / (self.rrf_k + rank + 1))
            if result.chunk_id in scores:
                scores[result.chunk_id] += rrf_score
            else:
                scores[result.chunk_id] = rrf_score
                chunk_map[result.chunk_id] = result

        # スコア順にソート
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # 上位top_k件を返す（スコアを更新）
        merged_results = []
        for chunk_id in sorted_ids[:top_k]:
            result = chunk_map[chunk_id]
            # RRFスコアを新しいスコアとして設定
            merged_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=scores[chunk_id],
                    source_file=result.source_file,
                    page_number=result.page_number,
                    metadata=result.metadata,
                )
            )

        logger.debug(f"RRF merged {len(vector_results)} + {len(bm25_results)} results")
        return merged_results

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        ハイブリッド検索 + リランク

        1. ベクトル検索でtop_k*2件取得
        2. BM25検索でtop_k*2件取得
        3. RRFで統合してtop_k*4件に絞る
        4. リランクでtop_k件に絞る（use_rerank=Trueの場合）
        """
        # リランクを使う場合は多めに取得
        fetch_multiplier = 4 if self.use_rerank else 2

        # 1. ベクトル検索
        vector_results = self._vector_search(query, top_k * 2)

        # 2. BM25検索
        bm25_results = self._bm25_search(query, top_k * 2)

        # 3. RRFで統合
        merged_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            top_k * fetch_multiplier,
        )

        # 4. リランク（オプション）
        if self.use_rerank and self.reranker:
            final_results = self.reranker.rerank(query, merged_results, top_k)
        else:
            final_results = merged_results[:top_k]

        logger.info(
            f"Hybrid search complete: {len(final_results)} documents (reranked={self.use_rerank})"
        )

        return RetrievalResult(
            query=query,
            results=final_results,
            metadata={
                "retriever": "hybrid",
                "top_k": top_k,
                "alpha": self.alpha,
                "use_rerank": self.use_rerank,
            },
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
