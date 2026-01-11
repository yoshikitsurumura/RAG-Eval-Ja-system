"""RAG module containing Naive RAG and Agentic RAG implementations"""

from .agentic_rag import AgenticRAG, QueryAnalysis, QueryType, SearchStrategy
from .base import RAGBase, RAGResponse
from .naive_rag import NaiveRAG

__all__ = [
    # Base
    "RAGBase",
    "RAGResponse",
    # Naive RAG
    "NaiveRAG",
    # Agentic RAG
    "AgenticRAG",
    "QueryType",
    "SearchStrategy",
    "QueryAnalysis",
]


def get_rag(rag_type: str = "naive", **kwargs) -> RAGBase:
    """RAGファクトリー"""
    rags = {
        "naive": NaiveRAG,
        "agentic": AgenticRAG,
    }

    if rag_type not in rags:
        raise ValueError(f"Unknown RAG type: {rag_type}. Available: {list(rags.keys())}")

    return rags[rag_type](**kwargs)
