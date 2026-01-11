"""
データセットダウンロードスクリプト

Hugging FaceからRAG-Evaluation-Dataset-JAをダウンロードし、
関連するPDFドキュメントも取得する
"""

import asyncio
import csv
import logging
import sys
from pathlib import Path

import httpx
from datasets import load_dataset

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATASET_NAME = "allganize/RAG-Evaluation-Dataset-JA"
DOCUMENTS_CSV_URL = (
    "https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-JA/raw/main/documents.csv"
)


async def download_pdf(
    client: httpx.AsyncClient,
    url: str,
    filepath: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    """PDFファイルを非同期でダウンロード"""
    async with semaphore:
        try:
            if filepath.exists():
                logger.info(f"Skip (exists): {filepath.name}")
                return True

            logger.info(f"Downloading: {filepath.name}")
            response = await client.get(url, follow_redirects=True, timeout=60.0)
            response.raise_for_status()

            filepath.write_bytes(response.content)
            logger.info(f"Downloaded: {filepath.name} ({len(response.content) / 1024:.1f} KB)")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False


async def download_all_pdfs(documents: list[dict], output_dir: Path, max_concurrent: int = 5):
    """全PDFを並列ダウンロード"""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max_concurrent)

    async with httpx.AsyncClient() as client:
        tasks = []
        for doc in documents:
            url = doc["url"]
            filename = doc["file_name"]
            filepath = output_dir / filename
            tasks.append(download_pdf(client, url, filepath, semaphore))

        results = await asyncio.gather(*tasks)

    success_count = sum(results)
    logger.info(f"Downloaded {success_count}/{len(documents)} PDFs")
    return success_count


def download_documents_csv(output_path: Path) -> list[dict]:
    """documents.csvをダウンロードしてパース"""
    logger.info("Downloading documents.csv...")

    response = httpx.get(DOCUMENTS_CSV_URL, follow_redirects=True)
    response.raise_for_status()

    # CSVを保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")

    # パース
    documents = []
    reader = csv.DictReader(response.text.splitlines())
    for row in reader:
        documents.append(row)

    logger.info(f"Found {len(documents)} documents in CSV")
    return documents


def download_qa_dataset(output_dir: Path) -> int:
    """QAデータセットをダウンロード"""
    logger.info(f"Downloading QA dataset from {DATASET_NAME}...")

    dataset = load_dataset(DATASET_NAME, split="test")

    # Parquetとして保存
    output_path = output_dir / "qa_dataset.parquet"
    dataset.to_parquet(output_path)
    logger.info(f"Saved QA dataset to {output_path} ({len(dataset)} rows)")

    return len(dataset)


def main():
    """メイン処理"""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    evaluation_dir = data_dir / "evaluation"

    # 1. documents.csvダウンロード
    csv_path = raw_dir / "documents.csv"
    documents = download_documents_csv(csv_path)

    # 2. QAデータセットダウンロード
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    qa_count = download_qa_dataset(evaluation_dir)

    # 3. PDFダウンロード
    pdf_dir = raw_dir / "pdfs"
    logger.info(f"Starting PDF download to {pdf_dir}...")
    success_count = asyncio.run(download_all_pdfs(documents, pdf_dir))

    # サマリー
    logger.info("=" * 50)
    logger.info("Download Summary:")
    logger.info(f"  - Documents CSV: {csv_path}")
    logger.info(f"  - QA Dataset: {qa_count} questions")
    logger.info(f"  - PDFs: {success_count}/{len(documents)} files")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
