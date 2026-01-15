"""
リランカーモジュール

Cross-Encoderを使用して検索結果を再スコアリング
"""

import logging

from sentence_transformers import CrossEncoder

from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """
    検索結果のリランク

    Cross-Encoderで質問とチャンクのペアを直接評価し、
    関連性スコアを算出して上位k件に絞る
    """

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-reranker-small",
        device: str = "cpu",
    ):
        """
        Args:
            model_name: Cross-Encoderモデル名（デフォルト: 日本語対応モデル）
            device: 実行デバイス ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        logger.info("Cross-Encoder loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        候補をリランク

        Args:
            query: 質問文
            candidates: 検索結果（多めに取得した候補）
            top_k: 最終的に返す件数

        Returns:
            リランク後の上位k件
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            logger.info(
                f"Candidate count ({len(candidates)}) <= top_k ({top_k}), skipping rerank"
            )
            return candidates

        # クエリとドキュメントのペアを作成
        pairs = [(query, c.content) for c in candidates]

        # Cross-Encoderでスコア算出
        logger.info(f"Reranking {len(candidates)} candidates with Cross-Encoder")
        scores = self.model.predict(pairs)

        # スコア順にソート
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 上位k件を返す
        reranked_results = [c for c, _ in ranked[:top_k]]

        logger.info(
            f"Reranking complete: {len(candidates)} -> {len(reranked_results)} documents"
        )

        return reranked_results
