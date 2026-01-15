"""
画像処理モジュール

PDFから抽出された画像に対してLLMを使用して説明文（キャプション）を生成
"""

import logging
from typing import Optional

from src.generation.llm_client import LLMClientBase, get_llm_client
from src.ingestion.pdf_parser import ParsedPage


logger = logging.getLogger(__name__)


class ImageProcessor:
    """画像説明生成クラス"""

    def __init__(self, llm_client: Optional[LLMClientBase] = None):
        self.llm_client = llm_client or get_llm_client()

    def process_page_images(self, page: ParsedPage) -> str:
        """
        ページ内の画像すべてに対して説明を生成し、結合して返す
        """
        if not page.images:
            return ""

        logger.info(f"Processing {len(page.images)} images on page {page.page_number}")
        
        descriptions = []
        for i, img_bytes in enumerate(page.images):
            try:
                # 専門的な図表が含まれる可能性があるため、詳細な説明を求める
                prompt = (
                    "この画像（PDFの抜粋）の内容を詳細に日本語で説明してください。\n"
                    "図表、グラフ、図解、イラストが含まれる場合は、そこに記載されている数値や項目、"
                    "示されている関係性（矢印など）を漏れなくテキスト化してください。\n"
                    "これはRAGシステムの検索用インデックスとして使用されます。"
                )
                
                response = self.llm_client.describe_image(img_bytes, prompt=prompt)
                desc = f"[画像{i+1}の説明]: {response.content}"
                descriptions.append(desc)
                
                logger.debug(f"  - Generated description for image {i+1} ({len(response.content)} chars)")
            except Exception as e:
                logger.error(f"Failed to describe image {i+1} on page {page.page_number}: {e}")
                descriptions.append(f"[画像{i+1}の説明生成に失敗しました]")

        return "\n\n".join(descriptions)


def get_image_processor(llm_client: Optional[LLMClientBase] = None) -> ImageProcessor:
    """画像プロセッサファクトリー"""
    return ImageProcessor(llm_client=llm_client)
