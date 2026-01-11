"""
評価実行モジュール

RAGシステムの精度評価パイプライン
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.evaluation.metrics import EvaluationScore, MetricBase, get_metric
from src.rag.base import RAGBase, RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """単一評価結果"""

    question_id: int
    question: str
    reference_answer: str
    generated_answer: str
    domain: str
    context_type: str
    scores: list[EvaluationScore]
    rag_response: RAGResponse
    is_correct: bool = False


@dataclass
class EvaluationReport:
    """評価レポート"""

    rag_name: str
    total_questions: int
    results: list[EvaluationResult]
    metrics_summary: dict = field(default_factory=dict)
    domain_breakdown: dict = field(default_factory=dict)
    type_breakdown: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """辞書形式に変換"""
        return {
            "rag_name": self.rag_name,
            "total_questions": self.total_questions,
            "metrics_summary": self.metrics_summary,
            "domain_breakdown": self.domain_breakdown,
            "type_breakdown": self.type_breakdown,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """DataFrameに変換"""
        rows = []
        for result in self.results:
            row = {
                "question_id": result.question_id,
                "question": result.question,
                "reference_answer": result.reference_answer,
                "generated_answer": result.generated_answer,
                "domain": result.domain,
                "context_type": result.context_type,
                "is_correct": result.is_correct,
            }
            for score in result.scores:
                row[f"score_{score.metric_name}"] = score.score
            rows.append(row)
        return pd.DataFrame(rows)


class RAGEvaluator:
    """RAG評価器"""

    def __init__(
        self,
        rag: RAGBase,
        metrics: Optional[list[str]] = None,
    ):
        """
        初期化

        Args:
            rag: 評価対象のRAGシステム
            metrics: 使用するメトリクス名のリスト
        """
        self.rag = rag
        self.metric_names = metrics or ["answer_correctness", "llm_judge"]
        self.metrics: list[MetricBase] = [get_metric(m) for m in self.metric_names]

        logger.info(f"Initialized RAGEvaluator for {rag.name} with metrics: {self.metric_names}")

    def evaluate_single(
        self,
        question_id: int,
        question: str,
        reference_answer: str,
        domain: str = "",
        context_type: str = "",
    ) -> EvaluationResult:
        """単一の質問を評価"""
        logger.info(f"Evaluating question {question_id}: {question[:50]}...")

        # RAGで回答生成
        rag_response = self.rag.query(question)

        # 各メトリクスで評価
        scores = []
        for metric in self.metrics:
            score = metric.evaluate(
                question=question,
                generated_answer=rag_response.answer,
                reference_answer=reference_answer,
            )
            scores.append(score)

        # 正解判定（answer_correctnessがあれば使用、なければllm_judgeのスコア）
        is_correct = False
        for score in scores:
            if score.metric_name == "answer_correctness":
                is_correct = score.score >= 0.5
                break
            elif score.metric_name == "llm_judge":
                is_correct = score.score >= 0.6  # 3/5以上

        return EvaluationResult(
            question_id=question_id,
            question=question,
            reference_answer=reference_answer,
            generated_answer=rag_response.answer,
            domain=domain,
            context_type=context_type,
            scores=scores,
            rag_response=rag_response,
            is_correct=is_correct,
        )

    def evaluate_dataset(
        self,
        dataset: pd.DataFrame,
        question_col: str = "question",
        answer_col: str = "target_answer",
        domain_col: str = "domain",
        type_col: str = "type",
        limit: Optional[int] = None,
    ) -> EvaluationReport:
        """データセット全体を評価"""
        logger.info(f"Starting evaluation of {len(dataset)} questions...")

        if limit:
            dataset = dataset.head(limit)

        results = []
        for idx, row in dataset.iterrows():
            result = self.evaluate_single(
                question_id=idx,
                question=row[question_col],
                reference_answer=row[answer_col],
                domain=row.get(domain_col, ""),
                context_type=row.get(type_col, ""),
            )
            results.append(result)

            # 進捗表示
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(dataset)}")

        # レポート作成
        report = self._create_report(results)
        return report

    def _create_report(self, results: list[EvaluationResult]) -> EvaluationReport:
        """評価レポートを作成"""
        # メトリクスサマリー
        metrics_summary = {}
        for metric_name in self.metric_names:
            scores = [
                score.score
                for result in results
                for score in result.scores
                if score.metric_name == metric_name
            ]
            if scores:
                metrics_summary[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                }

        # 正解率
        correct_count = sum(1 for r in results if r.is_correct)
        metrics_summary["accuracy"] = correct_count / len(results) if results else 0

        # ドメイン別集計
        domain_breakdown = {}
        for result in results:
            domain = result.domain or "unknown"
            if domain not in domain_breakdown:
                domain_breakdown[domain] = {"total": 0, "correct": 0}
            domain_breakdown[domain]["total"] += 1
            if result.is_correct:
                domain_breakdown[domain]["correct"] += 1

        for domain in domain_breakdown:
            total = domain_breakdown[domain]["total"]
            correct = domain_breakdown[domain]["correct"]
            domain_breakdown[domain]["accuracy"] = correct / total if total else 0

        # タイプ別集計
        type_breakdown = {}
        for result in results:
            ctx_type = result.context_type or "unknown"
            if ctx_type not in type_breakdown:
                type_breakdown[ctx_type] = {"total": 0, "correct": 0}
            type_breakdown[ctx_type]["total"] += 1
            if result.is_correct:
                type_breakdown[ctx_type]["correct"] += 1

        for ctx_type in type_breakdown:
            total = type_breakdown[ctx_type]["total"]
            correct = type_breakdown[ctx_type]["correct"]
            type_breakdown[ctx_type]["accuracy"] = correct / total if total else 0

        return EvaluationReport(
            rag_name=self.rag.name,
            total_questions=len(results),
            results=results,
            metrics_summary=metrics_summary,
            domain_breakdown=domain_breakdown,
            type_breakdown=type_breakdown,
        )

    def save_report(self, report: EvaluationReport, output_dir: Path):
        """レポートを保存"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # サマリーJSON
        summary_path = output_dir / f"evaluation_summary_{report.rag_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        # 詳細CSV
        detail_path = output_dir / f"evaluation_detail_{report.rag_name}.csv"
        report.to_dataframe().to_csv(detail_path, index=False, encoding="utf-8-sig")

        logger.info(f"Saved report to {output_dir}")

        return summary_path, detail_path
