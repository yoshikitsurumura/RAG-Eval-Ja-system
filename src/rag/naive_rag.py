"""
Naive RAG実装

課題要件: 「単純なテキスト分割、ベクトル検索のみ」

最もシンプルなRAG実装:
1. クエリを埋め込み
2. ベクトル検索でtop_k件を取得
3. コンテキストとして結合
4. LLMで回答生成
"""

import logging
from typing import Optional

from src.config import get_settings
from src.generation.llm_client import LLMClientBase, get_llm_client
from src.generation.prompt_templates import get_prompt
from src.ingestion.embedder import EmbedderBase, get_embedder
from src.rag.base import RAGBase, RAGResponse
from src.retrieval.retriever import RetrieverBase, get_retriever
from src.retrieval.vector_store import VectorStoreBase, get_vector_store

logger = logging.getLogger(__name__)


class NaiveRAG(RAGBase):
    """
    ナイーブRAG

    最もシンプルなRAG実装。
    単純なベクトル検索と回答生成のみを行う。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreBase] = None,
        embedder: Optional[EmbedderBase] = None,
        llm_client: Optional[LLMClientBase] = None,
        retriever: Optional[RetrieverBase] = None,
        top_k: int = 10,
    ):
        """
        初期化

        Args:
            vector_store: ベクトルストア（省略時はデフォルト）
            embedder: 埋め込み生成器（省略時はデフォルト）
            llm_client: LLMクライアント（省略時はデフォルト）
            retriever: リトリーバー（省略時は設定のretriever_typeを使用）
            top_k: 検索結果の取得数
        """
        settings = get_settings()
        self.top_k = top_k or settings.retrieval_top_k

        # 依存コンポーネントの初期化
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.llm_client = llm_client or get_llm_client()

        # リトリーバーの初期化（設定から読み込み）
        if retriever:
            self.retriever = retriever
        else:
            retriever_type = settings.retriever_type
            retriever_kwargs = {
                "vector_store": self.vector_store,
                "embedder": self.embedder,
            }

            # HybridRetrieverの場合は追加パラメータを設定
            if retriever_type == "hybrid":
                retriever_kwargs.update(
                    {
                        "alpha": settings.hybrid_alpha,
                        "rrf_k": settings.rrf_k,
                        "use_rerank": settings.use_rerank,
                    }
                )

            self.retriever = get_retriever(
                retriever_type=retriever_type,
                **retriever_kwargs,
            )

        # プロンプトテンプレート
        self.system_prompt = get_prompt("naive_rag_system")
        self.user_prompt = get_prompt("naive_rag_user")

        logger.info(f"Initialized NaiveRAG with top_k={self.top_k}")

    @property
    def name(self) -> str:
        return "NaiveRAG"

    @property
    def description(self) -> str:
        return "単純なベクトル検索と回答生成を行うベースラインRAG"

    def query(self, question: str) -> RAGResponse:
        """
        質問に対して回答を生成

        処理フロー:
        1. リトリーバーで関連ドキュメントを検索
        2. コンテキストをフォーマット
        3. LLMで回答生成
        """
        logger.info(f"NaiveRAG query: {question[:50]}...")

        # 1. 検索
        retrieval_result = self.retriever.retrieve(question, top_k=self.top_k)
        sources = retrieval_result.results

        if not sources:
            logger.warning("No documents retrieved")
            return RAGResponse(
                question=question,
                answer="関連するドキュメントが見つかりませんでした。",
                sources=[],
                metadata={"rag_type": self.name, "retrieval_count": 0},
            )

        # 2. コンテキストフォーマット
        context = self._format_context(sources)

        # 3. 回答生成
        prompt = self.user_prompt.format(context=context, question=question)
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt.template,
            max_tokens=8192,  # gpt-5-miniは推論トークンも含むため大きめに設定
        )

        logger.info(f"Generated answer ({response.usage.get('total_tokens', 0)} tokens)")

        return RAGResponse(
            question=question,
            answer=response.content,
            sources=sources,
            metadata={
                "rag_type": self.name,
                "retrieval_count": len(sources),
                "model": response.model,
                "usage": response.usage,
            },
        )
