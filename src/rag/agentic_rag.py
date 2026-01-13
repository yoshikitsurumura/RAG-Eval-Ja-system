"""
Agentic RAG実装

課題要件: 「AgenticRAGの定義は世の中的にも確立されていないため、
調査した上で独自に定義・実装してください」

========================================
独自定義: Agentic RAG
========================================

「LLMエージェントが検索・推論・生成プロセスを自律的に制御し、
クエリの複雑さに応じて動的に戦略を変更できるRAGシステム」

コア機能:
1. Query Analysis - クエリの複雑さを分析し、戦略を決定
2. Adaptive Retrieval - クエリ書き換え、複数回検索
3. Self-Reflection - 回答品質を評価し、必要に応じて再生成
4. Multi-step Reasoning - 複雑な質問を分解して段階的に回答

========================================
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.config import get_settings
from src.generation.llm_client import LLMClientBase, get_llm_client
from src.generation.prompt_templates import get_prompt
from src.ingestion.embedder import EmbedderBase, get_embedder
from src.rag.base import RAGBase, RAGResponse
from src.retrieval.retriever import RetrieverBase, get_retriever
from src.retrieval.vector_store import SearchResult, VectorStoreBase, get_vector_store

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """クエリタイプ"""

    SIMPLE = "simple"  # 単純なクエリ - 直接検索で回答可能
    COMPLEX = "complex"  # 複雑なクエリ - 分解が必要


class SearchStrategy(Enum):
    """検索戦略"""

    DIRECT = "direct"  # 直接検索
    DECOMPOSE = "decompose"  # クエリ分解
    ITERATIVE = "iterative"  # 反復的検索


@dataclass
class QueryAnalysis:
    """クエリ分析結果"""

    query_type: QueryType
    search_strategy: SearchStrategy
    sub_queries: list[str]
    reasoning: str


@dataclass
class ReflectionResult:
    """リフレクション結果"""

    relevance: int
    accuracy: int
    completeness: int
    needs_improvement: bool
    improvement_suggestions: str


class AgenticRAG(RAGBase):
    """
    Agentic RAG

    LLMエージェントによる自律的なRAGシステム。
    クエリの複雑さに応じて検索戦略を動的に変更し、
    回答品質を自己評価して改善する。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreBase] = None,
        embedder: Optional[EmbedderBase] = None,
        llm_client: Optional[LLMClientBase] = None,
        retriever: Optional[RetrieverBase] = None,
        top_k: int = 3,
        max_iterations: int = 1,
        quality_threshold: int = 4,
    ):
        """
        初期化

        Args:
            vector_store: ベクトルストア
            embedder: 埋め込み生成器
            llm_client: LLMクライアント
            retriever: リトリーバー
            top_k: 検索結果の取得数
            max_iterations: リフレクションの最大反復回数
            quality_threshold: 品質閾値（1-5）
        """
        settings = get_settings()
        self.top_k = top_k or settings.retrieval_top_k
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

        # 依存コンポーネント
        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.llm_client = llm_client or get_llm_client()
        self.retriever = retriever or get_retriever(
            retriever_type="simple",
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

        # プロンプトテンプレート
        self.query_analyzer_prompt = get_prompt("query_analyzer")
        self.query_rewriter_prompt = get_prompt("query_rewriter")
        self.answer_generator_prompt = get_prompt("answer_generator")
        self.reflection_prompt = get_prompt("reflection")
        self.synthesis_prompt = get_prompt("synthesis")

        logger.info(
            f"Initialized AgenticRAG with top_k={self.top_k}, "
            f"max_iterations={max_iterations}, quality_threshold={quality_threshold}"
        )

    @property
    def name(self) -> str:
        return "AgenticRAG"

    @property
    def description(self) -> str:
        return (
            "LLMエージェントが検索・推論・生成プロセスを自律的に制御し、"
            "クエリの複雑さに応じて動的に戦略を変更できるRAGシステム"
        )

    def query(self, question: str) -> RAGResponse:
        """
        質問に対して回答を生成

        処理フロー:
        1. クエリ分析 - 複雑さと戦略を決定
        2. 適応的検索 - 戦略に応じた検索を実行
        3. 回答生成
        4. 自己評価 - 品質が不足していれば再生成
        """
        logger.info(f"AgenticRAG query: {question[:50]}...")

        # 1. クエリ分析
        analysis = self._analyze_query(question)
        logger.info(
            f"Query analysis: type={analysis.query_type.value}, "
            f"strategy={analysis.search_strategy.value}"
        )

        # 2. 戦略に応じた検索と回答生成
        if analysis.search_strategy == SearchStrategy.DECOMPOSE:
            # 複雑なクエリ: 分解して個別に回答、統合
            response = self._handle_complex_query(question, analysis)
        else:
            # シンプルなクエリ: 直接検索と回答
            response = self._handle_simple_query(question)

        # 3. 自己評価と改善
        final_response = self._reflect_and_improve(response)

        return final_response

    def _analyze_query(self, question: str) -> QueryAnalysis:
        """クエリを分析して戦略を決定"""
        prompt = self.query_analyzer_prompt.format(question=question)
        response = self.llm_client.generate(prompt, temperature=0.0)

        try:
            result = json.loads(response.content)
            return QueryAnalysis(
                query_type=QueryType(result.get("query_type", "simple")),
                search_strategy=SearchStrategy(result.get("search_strategy", "direct")),
                sub_queries=result.get("sub_queries", []),
                reasoning=result.get("reasoning", ""),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse query analysis: {e}")
            return QueryAnalysis(
                query_type=QueryType.SIMPLE,
                search_strategy=SearchStrategy.DIRECT,
                sub_queries=[],
                reasoning="Parse error, defaulting to simple",
            )

    def _rewrite_query(self, original_query: str) -> list[str]:
        """クエリを複数のバリエーションに書き換え"""
        prompt = self.query_rewriter_prompt.format(original_query=original_query)
        response = self.llm_client.generate(prompt, temperature=0.3)

        try:
            result = json.loads(response.content)
            queries = result.get("rewritten_queries", [])
            # オリジナルも含める
            return [original_query] + queries
        except json.JSONDecodeError:
            return [original_query]

    def _handle_simple_query(self, question: str) -> RAGResponse:
        """シンプルなクエリを処理"""
        # クエリ書き換えで検索精度を向上
        queries = self._rewrite_query(question)

        # 複数クエリで検索、結果をマージ
        all_sources: dict[str, SearchResult] = {}
        for q in queries:
            retrieval_result = self.retriever.retrieve(q, top_k=self.top_k)
            for source in retrieval_result.results:
                if source.chunk_id not in all_sources:
                    all_sources[source.chunk_id] = source
                elif source.score > all_sources[source.chunk_id].score:
                    all_sources[source.chunk_id] = source

        # スコア順でtop_kを選択
        sources = sorted(all_sources.values(), key=lambda x: x.score, reverse=True)[
            : self.top_k
        ]

        if not sources:
            return RAGResponse(
                question=question,
                answer="関連するドキュメントが見つかりませんでした。",
                sources=[],
                metadata={"rag_type": self.name, "strategy": "simple"},
            )

        # 回答生成
        context = self._format_context(sources)
        prompt = self.answer_generator_prompt.format(context=context, question=question)
        response = self.llm_client.generate(prompt, temperature=0.0, max_tokens=8192)

        return RAGResponse(
            question=question,
            answer=response.content,
            sources=sources,
            metadata={
                "rag_type": self.name,
                "strategy": "simple",
                "queries_used": len(queries),
                "usage": response.usage,
            },
        )

    def _handle_complex_query(
        self, question: str, analysis: QueryAnalysis
    ) -> RAGResponse:
        """複雑なクエリを分解して処理"""
        sub_queries = analysis.sub_queries or [question]
        sub_answers = []
        all_sources: list[SearchResult] = []

        for sub_query in sub_queries:
            # 各サブクエリに対して検索と回答生成
            retrieval_result = self.retriever.retrieve(sub_query, top_k=self.top_k)
            sources = retrieval_result.results
            all_sources.extend(sources)

            if sources:
                context = self._format_context(sources)
                prompt = self.answer_generator_prompt.format(
                    context=context, question=sub_query
                )
                response = self.llm_client.generate(prompt, temperature=0.0)
                sub_answers.append(f"Q: {sub_query}\nA: {response.content}")

        # 統合
        if sub_answers:
            sub_answers_text = "\n\n".join(sub_answers)
            synthesis_prompt = self.synthesis_prompt.format(
                original_question=question, sub_answers=sub_answers_text
            )
            synthesis_response = self.llm_client.generate(
                synthesis_prompt, temperature=0.0
            )
            final_answer = synthesis_response.content
        else:
            final_answer = "関連するドキュメントが見つかりませんでした。"

        # ソースの重複を除去
        unique_sources = list({s.chunk_id: s for s in all_sources}.values())

        return RAGResponse(
            question=question,
            answer=final_answer,
            sources=unique_sources[: self.top_k * 2],
            metadata={
                "rag_type": self.name,
                "strategy": "decompose",
                "sub_queries": sub_queries,
                "sub_answers_count": len(sub_answers),
            },
        )

    def _reflect_and_improve(self, response: RAGResponse) -> RAGResponse:
        """回答品質を評価し、必要に応じて改善"""
        for iteration in range(self.max_iterations):
            reflection = self._evaluate_response(response)

            logger.info(
                f"Reflection (iteration {iteration + 1}): "
                f"relevance={reflection.relevance}, accuracy={reflection.accuracy}, "
                f"completeness={reflection.completeness}"
            )

            # 品質が十分ならそのまま返す
            avg_score = (
                reflection.relevance + reflection.accuracy + reflection.completeness
            ) / 3
            if avg_score >= self.quality_threshold or not reflection.needs_improvement:
                response.metadata["reflection"] = {
                    "iterations": iteration + 1,
                    "final_scores": {
                        "relevance": reflection.relevance,
                        "accuracy": reflection.accuracy,
                        "completeness": reflection.completeness,
                    },
                }
                return response

            # 改善を試みる
            logger.info(f"Attempting improvement: {reflection.improvement_suggestions}")
            response = self._improve_response(
                response, reflection.improvement_suggestions
            )

        return response

    def _evaluate_response(self, response: RAGResponse) -> ReflectionResult:
        """回答品質を評価"""
        context = self._format_context(response.sources) if response.sources else ""
        prompt = self.reflection_prompt.format(
            question=response.question,
            context=context,
            answer=response.answer,
        )
        llm_response = self.llm_client.generate(prompt, temperature=0.0)

        try:
            result = json.loads(llm_response.content)
            return ReflectionResult(
                relevance=result.get("relevance", 3),
                accuracy=result.get("accuracy", 3),
                completeness=result.get("completeness", 3),
                needs_improvement=result.get("needs_improvement", False),
                improvement_suggestions=result.get("improvement_suggestions", ""),
            )
        except json.JSONDecodeError:
            return ReflectionResult(
                relevance=3,
                accuracy=3,
                completeness=3,
                needs_improvement=False,
                improvement_suggestions="",
            )

    def _improve_response(
        self, response: RAGResponse, suggestions: str
    ) -> RAGResponse:
        """改善提案に基づいて回答を改善"""
        context = self._format_context(response.sources) if response.sources else ""
        improve_prompt = f"""以下の回答を改善してください。

## 元の質問
{response.question}

## コンテキスト
{context}

## 現在の回答
{response.answer}

## 改善提案
{suggestions}

## 改善した回答"""

        llm_response = self.llm_client.generate(improve_prompt, temperature=0.0)

        return RAGResponse(
            question=response.question,
            answer=llm_response.content,
            sources=response.sources,
            metadata={
                **response.metadata,
                "improved": True,
            },
        )
