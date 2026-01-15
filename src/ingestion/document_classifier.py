"""
ドキュメント分類モジュール

PDFの内容（特に冒頭ページ）を分析し、適切なカテゴリタグを自動付与する
"""

import json
import logging
from enum import IntEnum
from typing import Optional

from src.generation.llm_client import LLMClientBase, get_llm_client

logger = logging.getLogger(__name__)


class CategoryID(IntEnum):
    """カテゴリID定義"""
    FINANCE = 1      # 金融・経済
    MANUFACTURING = 2 # 製造・技術
    FOOD = 3         # 食品・生活
    IT = 4           # IT・セキュリティ
    GOVERNMENT = 5   # 行政・政策
    OTHER = 99       # その他


CATEGORIES = {
    CategoryID.FINANCE: "金融・経済 (予算, 決算, 税金, 市場動向)",
    CategoryID.MANUFACTURING: "製造・技術 (インフラ, 建築, ガイドライン, 仕様書)",
    CategoryID.FOOD: "食品・生活 (食品表示, 安全基準, 消費者情報, 医療)",
    CategoryID.IT: "IT・セキュリティ (DX, サイバーセキュリティ, クラウド, AI)",
    CategoryID.GOVERNMENT: "行政・政策 (白書, 計画, 地方自治, 環境, 教育)",
}


class DocumentClassifier:
    """ドキュメント分類クラス"""

    def __init__(self, llm_client: Optional[LLMClientBase] = None):
        self.llm_client = llm_client or get_llm_client()
        self.system_prompt = (
            "あなたはドキュメント分類の専門家です。\n"
            "与えられたテキスト（ドキュメントの冒頭部分）に基づいて、"
            "最も適切なカテゴリを選択し、JSON形式で出力してください。\n"
            "カテゴリ一覧:\n" + 
            "\n".join([f"{cat.value}: {desc}" for cat, desc in CATEGORIES.items()])
        )

    def classify(self, text: str, filename: str) -> dict:
        """
        ドキュメントを分類する

        Args:
            text: ドキュメントのテキスト（冒頭1000文字程度でOK）
            filename: ファイル名（分類のヒントになる）

        Returns:
            dict: {"category_id": int, "category_name": str, "reasoning": str}
        """
        # 入力情報を作成
        input_text = f"ファイル名: {filename}\n\n本文抜粋:\n{text[:2000]}"
        
        prompt = (
            f"以下のドキュメントを適切なカテゴリに分類してください。\n"
            f"必ず以下のJSON形式のみを出力してください。\n"
            f'{{"category_id": int, "reasoning": "分類理由"}}\n\n' 
            f"{input_text}"
        )

        try:
            # LLMに分類させる（Temperature=0.0で安定化）
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.0
            )
            
            # JSONパース（Markdownのコードブロック除去）
            content = response.content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            
            category_id = int(result.get("category_id", 99))
            
            # 定義されていないIDならその他にする
            if category_id not in [c.value for c in CategoryID]:
                category_id = CategoryID.OTHER.value

            return {
                "category_id": category_id,
                "category_name": CATEGORIES.get(category_id, "その他"),
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            logger.error(f"Failed to classify document {filename}: {e}")
            return {
                "category_id": CategoryID.OTHER.value,
                "category_name": "その他 (エラー)",
                "reasoning": f"Classification failed: {e}"
            }


def get_document_classifier(llm_client: Optional[LLMClientBase] = None) -> DocumentClassifier:
    """分類器ファクトリー"""
    return DocumentClassifier(llm_client=llm_client)
