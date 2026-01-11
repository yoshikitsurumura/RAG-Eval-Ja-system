"""
プロンプトテンプレートモジュール

RAGシステムで使用する各種プロンプトテンプレート
"""

from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """プロンプトテンプレート"""

    name: str
    template: str
    description: str

    def format(self, **kwargs) -> str:
        """テンプレートをフォーマット"""
        return self.template.format(**kwargs)


# ========================================
# Naive RAG用プロンプト
# ========================================

NAIVE_RAG_SYSTEM_PROMPT = PromptTemplate(
    name="naive_rag_system",
    template="""あなたは、与えられたコンテキスト情報に基づいて質問に回答するAIアシスタントです。

以下のルールに従って回答してください：
1. コンテキストに含まれる情報のみを使用して回答してください
2. コンテキストに情報がない場合は「提供された情報からは回答できません」と答えてください
3. 回答は簡潔かつ正確にしてください
4. 推測や憶測は避けてください""",
    description="Naive RAG用システムプロンプト",
)

NAIVE_RAG_USER_PROMPT = PromptTemplate(
    name="naive_rag_user",
    template="""以下のコンテキスト情報を参考に、質問に回答してください。

## コンテキスト情報
{context}

## 質問
{question}

## 回答""",
    description="Naive RAG用ユーザープロンプト",
)


# ========================================
# Agentic RAG用プロンプト
# ========================================

QUERY_ANALYZER_PROMPT = PromptTemplate(
    name="query_analyzer",
    template="""あなたはクエリ分析エージェントです。
ユーザーの質問を分析し、最適な検索戦略を決定してください。

## 質問
{question}

以下のJSON形式で回答してください：
{{
    "query_type": "simple" または "complex",
    "reasoning": "分析の理由",
    "sub_queries": ["複雑な場合のサブクエリリスト"],
    "search_strategy": "direct" または "decompose" または "iterative"
}}

- simple: 単一の検索で回答可能な質問
- complex: 複数の情報を組み合わせる必要がある質問

回答:""",
    description="クエリ分析用プロンプト",
)

QUERY_REWRITER_PROMPT = PromptTemplate(
    name="query_rewriter",
    template="""あなたはクエリ書き換えエージェントです。
元のクエリをより検索に適した形式に書き換えてください。

## 元のクエリ
{original_query}

## 検索対象のコンテキスト
日本の官公庁・公的機関が発行した文書（金融、IT、製造業、公共、小売業界）

以下の観点で3つの異なるクエリを生成してください：
1. 同義語や関連語を使用したバリエーション
2. より具体的な表現
3. より一般的な表現

JSON形式で回答:
{{
    "rewritten_queries": ["クエリ1", "クエリ2", "クエリ3"]
}}""",
    description="クエリ書き換え用プロンプト",
)

ANSWER_GENERATOR_PROMPT = PromptTemplate(
    name="answer_generator",
    template="""あなたは質問回答エージェントです。
与えられたコンテキスト情報に基づいて、質問に正確に回答してください。

## コンテキスト情報
{context}

## 質問
{question}

## 回答ガイドライン
- コンテキストに基づいた事実のみを述べてください
- 情報源のページ番号があれば言及してください
- 不確かな情報は含めないでください

回答:""",
    description="回答生成用プロンプト",
)

REFLECTION_PROMPT = PromptTemplate(
    name="reflection",
    template="""あなたは回答品質評価エージェントです。
生成された回答の品質を評価し、改善が必要かどうか判断してください。

## 質問
{question}

## コンテキスト
{context}

## 生成された回答
{answer}

以下の観点で評価し、JSON形式で回答してください：

1. relevance (1-5): 質問との関連性
2. accuracy (1-5): コンテキストとの整合性
3. completeness (1-5): 回答の完全性
4. needs_improvement: true/false
5. improvement_suggestions: 改善提案（必要な場合）

{{
    "relevance": 4,
    "accuracy": 5,
    "completeness": 3,
    "needs_improvement": true,
    "improvement_suggestions": "具体的な改善提案"
}}""",
    description="回答品質評価用プロンプト",
)

SYNTHESIS_PROMPT = PromptTemplate(
    name="synthesis",
    template="""あなたは情報統合エージェントです。
複数のサブ質問への回答を統合して、元の質問に対する包括的な回答を生成してください。

## 元の質問
{original_question}

## サブ質問と回答
{sub_answers}

## 統合ガイドライン
- 重複する情報は1回だけ述べてください
- 矛盾する情報がある場合は、両方の視点を提示してください
- 論理的な流れで回答を構成してください

統合回答:""",
    description="情報統合用プロンプト",
)


# ========================================
# プロンプトレジストリ
# ========================================

PROMPT_REGISTRY = {
    # Naive RAG
    "naive_rag_system": NAIVE_RAG_SYSTEM_PROMPT,
    "naive_rag_user": NAIVE_RAG_USER_PROMPT,
    # Agentic RAG
    "query_analyzer": QUERY_ANALYZER_PROMPT,
    "query_rewriter": QUERY_REWRITER_PROMPT,
    "answer_generator": ANSWER_GENERATOR_PROMPT,
    "reflection": REFLECTION_PROMPT,
    "synthesis": SYNTHESIS_PROMPT,
}


def get_prompt(name: str) -> PromptTemplate:
    """プロンプトを取得"""
    if name not in PROMPT_REGISTRY:
        raise ValueError(
            f"Unknown prompt: {name}. Available: {list(PROMPT_REGISTRY.keys())}"
        )
    return PROMPT_REGISTRY[name]
