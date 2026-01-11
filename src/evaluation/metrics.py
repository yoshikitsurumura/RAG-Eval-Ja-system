"""
評価メトリクスモジュール

RAGシステムの精度評価に使用するメトリクス
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.generation.llm_client import LLMClientBase, get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class EvaluationScore:
    """評価スコア"""

    metric_name: str
    score: float  # 0.0 - 1.0 または 1 - 5
    reasoning: str
    metadata: dict


class MetricBase(ABC):
    """評価メトリクス基底クラス"""

    @property
    @abstractmethod
    def name(self) -> str:
        """メトリクス名"""
        pass

    @abstractmethod
    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> EvaluationScore:
        """評価を実行"""
        pass


class ExactMatchMetric(MetricBase):
    """完全一致メトリクス"""

    @property
    def name(self) -> str:
        return "exact_match"

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> EvaluationScore:
        """完全一致を評価"""
        # 正規化して比較
        gen_normalized = generated_answer.strip().lower()
        ref_normalized = reference_answer.strip().lower()

        is_match = gen_normalized == ref_normalized
        score = 1.0 if is_match else 0.0

        return EvaluationScore(
            metric_name=self.name,
            score=score,
            reasoning="Exact match" if is_match else "No match",
            metadata={},
        )


class ContainsAnswerMetric(MetricBase):
    """正解を含むかどうかのメトリクス"""

    @property
    def name(self) -> str:
        return "contains_answer"

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> EvaluationScore:
        """生成回答が正解を含むか評価"""
        gen_normalized = generated_answer.strip().lower()
        ref_normalized = reference_answer.strip().lower()

        # 参照回答の主要なキーワードを抽出（簡易版）
        ref_keywords = set(ref_normalized.split())
        gen_words = set(gen_normalized.split())

        # 共通するキーワードの割合
        if ref_keywords:
            overlap = len(ref_keywords & gen_words) / len(ref_keywords)
        else:
            overlap = 0.0

        return EvaluationScore(
            metric_name=self.name,
            score=overlap,
            reasoning=f"Keyword overlap: {overlap:.2%}",
            metadata={"keyword_overlap": overlap},
        )


class LLMJudgeMetric(MetricBase):
    """LLMによる評価メトリクス（LLM-as-Judge）"""

    def __init__(self, llm_client: LLMClientBase | None = None):
        self.llm_client = llm_client or get_llm_client()

    @property
    def name(self) -> str:
        return "llm_judge"

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> EvaluationScore:
        """LLMで回答品質を評価"""
        prompt = f"""あなたは回答品質を評価する専門家です。
以下の質問に対する生成回答を、正解と比較して評価してください。

## 質問
{question}

## 正解
{reference_answer}

## 生成回答
{generated_answer}

以下の観点で1-5のスコアをつけてください：
1. **正確性**: 生成回答は正解と同じ内容を述べているか
2. **完全性**: 必要な情報が含まれているか
3. **関連性**: 質問に適切に回答しているか

最終スコア（1-5）と理由を以下の形式で回答してください：
スコア: [1-5の数字]
理由: [評価理由]"""

        response = self.llm_client.generate(prompt, temperature=0.0)

        # レスポンスをパース
        try:
            lines = response.content.strip().split("\n")
            score_line = [l for l in lines if l.startswith("スコア:")][0]
            score = int(score_line.replace("スコア:", "").strip())
            score = max(1, min(5, score))  # 1-5に制限

            reason_line = [l for l in lines if l.startswith("理由:")][0]
            reasoning = reason_line.replace("理由:", "").strip()
        except (IndexError, ValueError):
            score = 3
            reasoning = "Parse error"

        return EvaluationScore(
            metric_name=self.name,
            score=score / 5.0,  # 0-1に正規化
            reasoning=reasoning,
            metadata={"raw_score": score, "scale": "1-5"},
        )


class AnswerCorrectnessMetric(MetricBase):
    """回答正確性メトリクス（O/X判定）"""

    def __init__(self, llm_client: LLMClientBase | None = None):
        self.llm_client = llm_client or get_llm_client()

    @property
    def name(self) -> str:
        return "answer_correctness"

    def evaluate(
        self,
        question: str,
        generated_answer: str,
        reference_answer: str,
        context: str = "",
    ) -> EvaluationScore:
        """回答が正解かどうかをO/Xで評価"""
        prompt = f"""以下の質問に対する回答が正解かどうかを判定してください。

## 質問
{question}

## 正解
{reference_answer}

## 評価対象の回答
{generated_answer}

## 判定基準
- 回答が正解と同じ意味・内容を含んでいれば「O」（正解）
- 回答が間違っているか、重要な情報が欠けていれば「X」（不正解）

回答: O または X（1文字のみ）"""

        response = self.llm_client.generate(prompt, temperature=0.0)

        result = response.content.strip().upper()
        is_correct = result.startswith("O")

        return EvaluationScore(
            metric_name=self.name,
            score=1.0 if is_correct else 0.0,
            reasoning="Correct" if is_correct else "Incorrect",
            metadata={"judgment": "O" if is_correct else "X"},
        )


# メトリクスレジストリ
METRIC_REGISTRY = {
    "exact_match": ExactMatchMetric,
    "contains_answer": ContainsAnswerMetric,
    "llm_judge": LLMJudgeMetric,
    "answer_correctness": AnswerCorrectnessMetric,
}


def get_metric(metric_name: str, **kwargs) -> MetricBase:
    """メトリクスを取得"""
    if metric_name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {metric_name}. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[metric_name](**kwargs)
