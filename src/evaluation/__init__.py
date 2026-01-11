"""Evaluation module for RAG system evaluation"""

from .evaluator import EvaluationReport, EvaluationResult, RAGEvaluator
from .metrics import (
    AnswerCorrectnessMetric,
    ContainsAnswerMetric,
    EvaluationScore,
    ExactMatchMetric,
    LLMJudgeMetric,
    MetricBase,
    get_metric,
)

__all__ = [
    # Metrics
    "MetricBase",
    "ExactMatchMetric",
    "ContainsAnswerMetric",
    "LLMJudgeMetric",
    "AnswerCorrectnessMetric",
    "EvaluationScore",
    "get_metric",
    # Evaluator
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationReport",
]
