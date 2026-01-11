"""
評価実行スクリプト

Naive RAGとAgentic RAGの精度評価を実行
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import RAGEvaluator
from src.rag import AgenticRAG, NaiveRAG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_evaluation(
    dataset_path: Path,
    output_dir: Path,
    rag_types: list[str] = None,
    limit: int = None,
    metrics: list[str] = None,
):
    """
    評価を実行

    Args:
        dataset_path: QAデータセットのパス
        output_dir: 出力ディレクトリ
        rag_types: 評価するRAGタイプ
        limit: 評価する質問数の上限
        metrics: 使用するメトリクス
    """
    rag_types = rag_types or ["naive", "agentic"]
    metrics = metrics or ["answer_correctness", "llm_judge"]

    # データセット読み込み
    logger.info(f"Loading dataset from {dataset_path}")
    if dataset_path.suffix == ".parquet":
        dataset = pd.read_parquet(dataset_path)
    else:
        dataset = pd.read_csv(dataset_path)

    logger.info(f"Dataset size: {len(dataset)} questions")

    # 各RAGタイプで評価
    reports = {}
    for rag_type in rag_types:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {rag_type.upper()} RAG")
        logger.info(f"{'='*50}")

        # RAGインスタンス作成
        if rag_type == "naive":
            rag = NaiveRAG()
        elif rag_type == "agentic":
            rag = AgenticRAG()
        else:
            logger.error(f"Unknown RAG type: {rag_type}")
            continue

        # 評価実行
        evaluator = RAGEvaluator(rag=rag, metrics=metrics)
        report = evaluator.evaluate_dataset(dataset, limit=limit)

        # レポート保存
        evaluator.save_report(report, output_dir)
        reports[rag_type] = report

        # 結果表示
        logger.info(f"\n{rag_type.upper()} RAG Results:")
        logger.info(f"  Accuracy: {report.metrics_summary.get('accuracy', 0):.2%}")
        for metric, values in report.metrics_summary.items():
            if isinstance(values, dict):
                logger.info(f"  {metric}: mean={values['mean']:.3f}")

        logger.info("\n  Domain Breakdown:")
        for domain, values in report.domain_breakdown.items():
            logger.info(
                f"    {domain}: {values['correct']}/{values['total']} "
                f"({values['accuracy']:.2%})"
            )

        logger.info("\n  Context Type Breakdown:")
        for ctx_type, values in report.type_breakdown.items():
            logger.info(
                f"    {ctx_type}: {values['correct']}/{values['total']} "
                f"({values['accuracy']:.2%})"
            )

    # 比較サマリー
    if len(reports) > 1:
        logger.info(f"\n{'='*50}")
        logger.info("COMPARISON SUMMARY")
        logger.info(f"{'='*50}")

        for rag_type, report in reports.items():
            acc = report.metrics_summary.get("accuracy", 0)
            logger.info(f"  {rag_type.upper()}: {acc:.2%} accuracy")


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (default: data/evaluation/qa_dataset.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/evaluation/results)",
    )
    parser.add_argument(
        "--rag-types",
        type=str,
        nargs="+",
        default=["naive", "agentic"],
        help="RAG types to evaluate (default: naive agentic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["answer_correctness", "llm_judge"],
        help="Evaluation metrics (default: answer_correctness llm_judge)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    dataset_path = (
        Path(args.dataset) if args.dataset else project_root / "data" / "evaluation" / "qa_dataset.parquet"
    )
    output_dir = (
        Path(args.output) if args.output else project_root / "data" / "evaluation" / "results"
    )

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Run download_dataset.py first to download the dataset")
        sys.exit(1)

    run_evaluation(
        dataset_path=dataset_path,
        output_dir=output_dir,
        rag_types=args.rag_types,
        limit=args.limit,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
