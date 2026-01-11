"""
FastAPI アプリケーションサーバー

RAGシステムのAPIエンドポイントを提供
"""

import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag import AgenticRAG, NaiveRAG, RAGResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# グローバル変数でRAGインスタンスを保持
rag_instances: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    logger.info("Starting RAG API Server...")

    # RAGインスタンスの初期化は遅延実行（必要時に作成）
    yield

    logger.info("Shutting down RAG API Server...")
    rag_instances.clear()


app = FastAPI(
    title="Laboro RAG System API",
    description="日本語RAG評価データセットを使用したRAGシステムAPI",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================================
# リクエスト/レスポンスモデル
# ========================================


class RAGType(str, Enum):
    """RAGタイプ"""

    NAIVE = "naive"
    AGENTIC = "agentic"


class QueryRequest(BaseModel):
    """クエリリクエスト"""

    question: str = Field(..., min_length=1, description="質問文")
    rag_type: RAGType = Field(default=RAGType.NAIVE, description="RAGタイプ")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="検索結果数")


class SourceInfo(BaseModel):
    """ソース情報"""

    content: str
    source_file: str
    page_number: int
    score: float


class QueryResponse(BaseModel):
    """クエリレスポンス"""

    question: str
    answer: str
    sources: list[SourceInfo]
    rag_type: str
    metadata: dict


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""

    status: str
    version: str


class SystemInfoResponse(BaseModel):
    """システム情報レスポンス"""

    rag_types: list[str]
    settings: dict


# ========================================
# ヘルパー関数
# ========================================


def get_rag_instance(rag_type: RAGType, top_k: int = 5):
    """RAGインスタンスを取得（遅延初期化）"""
    key = f"{rag_type.value}_{top_k}"

    if key not in rag_instances:
        logger.info(f"Initializing RAG instance: {key}")
        if rag_type == RAGType.NAIVE:
            rag_instances[key] = NaiveRAG(top_k=top_k)
        else:
            rag_instances[key] = AgenticRAG(top_k=top_k)

    return rag_instances[key]


def rag_response_to_api_response(
    response: RAGResponse, rag_type: str
) -> QueryResponse:
    """RAGResponseをAPIレスポンスに変換"""
    sources = [
        SourceInfo(
            content=s.content[:500] if len(s.content) > 500 else s.content,
            source_file=s.source_file,
            page_number=s.page_number,
            score=s.score,
        )
        for s in response.sources
    ]

    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=sources,
        rag_type=rag_type,
        metadata=response.metadata,
    )


# ========================================
# エンドポイント
# ========================================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェック"""
    return HealthResponse(status="healthy", version="0.1.0")


@app.get("/info", response_model=SystemInfoResponse)
async def system_info():
    """システム情報を取得"""
    settings = get_settings()
    return SystemInfoResponse(
        rag_types=[t.value for t in RAGType],
        settings={
            "llm_model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "retrieval_top_k": settings.retrieval_top_k,
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAGクエリを実行

    - **question**: 質問文
    - **rag_type**: RAGタイプ（naive または agentic）
    - **top_k**: 検索結果数
    """
    try:
        logger.info(
            f"Query request: rag_type={request.rag_type.value}, "
            f"question={request.question[:50]}..."
        )

        rag = get_rag_instance(request.rag_type, request.top_k or 5)
        response = rag.query(request.question)

        return rag_response_to_api_response(response, request.rag_type.value)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/naive", response_model=QueryResponse)
async def query_naive(request: QueryRequest):
    """Naive RAGでクエリを実行"""
    request.rag_type = RAGType.NAIVE
    return await query(request)


@app.post("/query/agentic", response_model=QueryResponse)
async def query_agentic(request: QueryRequest):
    """Agentic RAGでクエリを実行"""
    request.rag_type = RAGType.AGENTIC
    return await query(request)


@app.get("/rag/types")
async def get_rag_types():
    """利用可能なRAGタイプを取得"""
    return {
        "types": [
            {
                "name": "naive",
                "description": "単純なベクトル検索と回答生成を行うベースラインRAG",
            },
            {
                "name": "agentic",
                "description": (
                    "LLMエージェントが検索・推論・生成プロセスを自律的に制御し、"
                    "クエリの複雑さに応じて動的に戦略を変更できるRAGシステム"
                ),
            },
        ]
    }
