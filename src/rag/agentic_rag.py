"""
Agentic RAG実装 (v4: 司令塔AI搭載版)

LLMエージェントが検索範囲（カテゴリ）を自動判定し、
画像キャプションと統合して回答を生成する。
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any

from src.config import get_settings
from src.generation.llm_client import LLMClientBase, get_llm_client
from src.generation.prompt_templates import get_prompt
from src.ingestion.embedder import EmbedderBase, get_embedder
from src.ingestion.document_classifier import CATEGORIES
from src.rag.base import RAGBase, RAGResponse
from src.retrieval.retriever import RetrieverBase, get_retriever
from src.retrieval.vector_store import SearchResult, VectorStoreBase, get_vector_store

logger = logging.getLogger(__name__)


class QueryType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"


class SearchStrategy(Enum):
    DIRECT = "direct"
    DECOMPOSE = "decompose"
    ITERATIVE = "iterative"


@dataclass
class QueryAnalysis:
    """クエリ分析結果 (司令塔AIの判断)"""

    query_type: QueryType
    search_strategy: SearchStrategy
    sub_queries: list[str]
    category_id: Optional[int]  # どのカテゴリを検索すべきか
    reasoning: str


@dataclass
class ReflectionResult:
    relevance: int
    accuracy: int
    completeness: int
    needs_improvement: bool
    improvement_suggestions: str


class AgenticRAG(RAGBase):
    """
    Agentic RAG v4

    司令塔AIがドキュメントカテゴリを判定して検索を最適化し、
    画像解析結果を含めた高度な回答を生成する。
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreBase] = None,
        embedder: Optional[EmbedderBase] = None,
        llm_client: Optional[LLMClientBase] = None,
        retriever: Optional[RetrieverBase] = None,
        top_k: int = 5,
        max_iterations: int = 1,
        quality_threshold: int = 4,
    ):
        settings = get_settings()
        self.top_k = top_k or settings.retrieval_top_k
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

        self.vector_store = vector_store or get_vector_store()
        self.embedder = embedder or get_embedder()
        self.llm_client = llm_client or get_llm_client()
        
        # ハイブリッドリトリーバーを使用
        self.retriever = retriever or get_retriever(
            retriever_type="hybrid",
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

        # プロンプト (v4用にカテゴリ情報を注入)
        self.query_analyzer_prompt = self._build_analyzer_prompt()
        self.answer_generator_prompt = get_prompt("answer_generator")
        self.reflection_prompt = get_prompt("reflection")
        self.synthesis_prompt = get_prompt("synthesis")

        logger.info(f"Initialized AgenticRAG v4 (Commander AI Enabled)")

    def _build_analyzer_prompt(self) -> str:
        """司令塔AI用のシステムプロンプトを構築"""
        cat_desc = "\n".join([f"{cid}: {desc}" for cid, desc in CATEGORIES.items()])
        return (
            "あなたはRAGシステムの司令塔AIです。ユーザーの質問を分析し、最適な検索戦略を決定してください。\n"
            "また、質問の内容から、以下のどのドキュメントカテゴリを検索すべきか判断してください。\n"
            "不明な場合はカテゴリIDをnullにしてください。\n\n"
            "【カテゴリ一覧】\n" + cat_desc + "\n\n"
            "回答は必ず以下のJSON形式で行ってください:\n"
            '{"query_type": "simple|complex", "search_strategy": "direct|decompose", "sub_queries": [], "category_id": int|null, "reasoning": "理由"}'
        )

    @property
    def name(self) -> str:
        return "AgenticRAG_v4"

    @property
    def description(self) -> str:
        return "司令塔AIによるカテゴリ絞り込みと画像解析結果を統合した高度なRAG"

    def query(self, question: str) -> RAGResponse:
        """質問に対して回答を生成"""
        logger.info(f"AgenticRAG v4 query: {question[:50]}...")

        # 1. 司令塔AIによる分析
        analysis = self._analyze_query(question)
        logger.info(
            f"Commander Analysis: category_id={analysis.category_id}, "
            f"strategy={analysis.search_strategy.value}"
        )

        # 2. フィルタ条件の構築
        metadata_filter = None
        if analysis.category_id:
            metadata_filter = {"category_id": analysis.category_id}

        # 3. 戦略に応じた検索と回答生成
        if analysis.search_strategy == SearchStrategy.DECOMPOSE:
            response = self._handle_complex_query(question, analysis, metadata_filter)
        else:
            response = self._handle_simple_query(question, metadata_filter)

        # 4. 自己評価と改善
        final_response = self._reflect_and_improve(response)

        return final_response

    def _analyze_query(self, question: str) -> QueryAnalysis:
        """司令塔AIが戦略とカテゴリを決定"""
        response = self.llm_client.generate(
            prompt=f"質問: {question}",
            system_prompt=self.query_analyzer_prompt,
            temperature=0.0
        )

        try:
            # コードブロック等を除去してパース
            content = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return QueryAnalysis(
                query_type=QueryType(result.get("query_type", "simple")),
                search_strategy=SearchStrategy(result.get("search_strategy", "direct")),
                sub_queries=result.get("sub_queries", []),
                category_id=result.get("category_id"),
                reasoning=result.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse analysis: {e}. Defaulting to simple search.")
            return QueryAnalysis(
                query_type=QueryType.SIMPLE,
                search_strategy=SearchStrategy.DIRECT,
                sub_queries=[],
                category_id=None,
                reasoning="Parse error fallback",
            )

    def _handle_simple_query(self, question: str, metadata_filter: Optional[dict] = None) -> RAGResponse:
        """シンプルなクエリを処理 (フィルタ適用)"""
        retrieval_result = self.retriever.retrieve(
            question, 
            top_k=self.top_k, 
            metadata_filter=metadata_filter
        )
        sources = retrieval_result.results

        if not sources:
            return RAGResponse(
                question=question,
                answer="関連するドキュメントが見つかりませんでした。絞り込み条件（カテゴリ）が不適切だった可能性があります。",
                sources=[],
                metadata={"rag_type": self.name, "filter_used": metadata_filter},
            )

        # 回答生成
        context = self._format_context(sources)
        prompt = self.answer_generator_prompt.format(context=context, question=question)
        
        # システムプロンプトで画像解析結果への注目を促す
        system_msg = (
            "あなたは提供されたコンテキストに基づいて正確に回答するアシスタントです。\n"
            "コンテキストには[画像nの説明]という形式で図表やグラフの解析結果が含まれている場合があります。\n"
            "これらを本物の図表と同様に扱い、数値や傾向を回答に反映させてください。"
        )
        
        response = self.llm_client.generate(
            prompt=prompt, 
            system_prompt=system_msg,
            temperature=0.0, 
            max_tokens=8192
        )

        return RAGResponse(
            question=question,
            answer=response.content,
            sources=sources,
            metadata={
                "rag_type": self.name,
                "category_id": metadata_filter.get("category_id") if metadata_filter else None,
                "usage": response.usage,
            },
        )

    def _handle_complex_query(
        self,
        question: str,
        analysis: QueryAnalysis,
        metadata_filter: Optional[dict] = None,
    ) -> RAGResponse:
        """複雑なクエリを分解して処理 (各サブクエリにフィルタ適用)"""
        sub_queries = analysis.sub_queries or [question]
        sub_answers = []
        all_sources: list[SearchResult] = []

        for sub_query in sub_queries:
            retrieval_result = self.retriever.retrieve(
                sub_query, 
                top_k=self.top_k, 
                metadata_filter=metadata_filter
            )
            sources = retrieval_result.results
            all_sources.extend(sources)

            if sources:
                context = self._format_context(sources)
                prompt = self.answer_generator_prompt.format(context=context, question=sub_query)
                response = self.llm_client.generate(prompt, temperature=0.0)
                sub_answers.append(f"Q: {sub_query}\nA: {response.content}")

        # 統合
        if sub_answers:
            sub_answers_text = "\n\n".join(sub_answers)
            synthesis_prompt = self.synthesis_prompt.format(
                original_question=question, sub_answers=sub_answers_text
            )
            synthesis_response = self.llm_client.generate(synthesis_prompt, temperature=0.0)
            final_answer = synthesis_response.content
        else:
            final_answer = "関連するドキュメントが見つかりませんでした。"

        unique_sources = list({s.chunk_id: s for s in all_sources}.values())

        return RAGResponse(
            question=question,
            answer=final_answer,
            sources=unique_sources[: self.top_k * 2],
            metadata={
                "rag_type": self.name,
                "strategy": "decompose",
                "category_id": metadata_filter.get("category_id") if metadata_filter else None,
            },
        )

    def _reflect_and_improve(self, response: RAGResponse) -> RAGResponse:
        """回答品質を評価し、必要に応じて改善"""
        for iteration in range(self.max_iterations):
            reflection = self._evaluate_response(response)
            avg_score = (reflection.relevance + reflection.accuracy + reflection.completeness) / 3
            
            if avg_score >= self.quality_threshold or not reflection.needs_improvement:
                return response

            # 改善を試みる
            response = self._improve_response(response, reflection.improvement_suggestions)

        return response

    def _evaluate_response(self, response: RAGResponse) -> ReflectionResult:
        context = self._format_context(response.sources) if response.sources else ""
        prompt = self.reflection_prompt.format(
            question=response.question,
            context=context,
            answer=response.answer,
        )
        llm_response = self.llm_client.generate(prompt, temperature=0.0)

        try:
            content = llm_response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return ReflectionResult(
                relevance=result.get("relevance", 3),
                accuracy=result.get("accuracy", 3),
                completeness=result.get("completeness", 3),
                needs_improvement=result.get("needs_improvement", False),
                improvement_suggestions=result.get("improvement_suggestions", ""),
            )
        except:
            return ReflectionResult(3, 3, 3, False, "")

    def _improve_response(self, response: RAGResponse, suggestions: str) -> RAGResponse:
        context = self._format_context(response.sources) if response.sources else ""
        improve_prompt = (
            f"元の質問: {response.question}\n"
            f"コンテキスト: {context}\n"
            f"現在の回答: {response.answer}\n"
            f"改善提案: {suggestions}\n\n"
            f"改善した回答:"
        )
        llm_response = self.llm_client.generate(improve_prompt, temperature=0.0)
        return RAGResponse(
            question=response.question,
            answer=llm_response.content,
            sources=response.sources,
            metadata={{**response.metadata, "improved": True}},
        )