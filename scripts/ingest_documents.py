"""
ドキュメント取り込みスクリプト

PDFをパースしてベクトルDBに格納
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.ingestion import get_embedder, get_parser, get_text_splitter
from src.retrieval import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdfs(
    pdf_dir: Path,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    parser_type: str = "hybrid",
    use_mock_embedder: bool = False,
):
    """
    PDFディレクトリ内のファイルをベクトルDBに取り込む

    Args:
        pdf_dir: PDFファイルのディレクトリ
        chunk_size: チャンクサイズ
        chunk_overlap: オーバーラップ
        parser_type: パーサータイプ
        use_mock_embedder: モック埋め込みを使用（開発用）
    """
    settings = get_settings()

    # コンポーネント初期化
    parser = get_parser(parser_type)
    splitter = get_text_splitter("table_aware", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = get_embedder("mock" if use_mock_embedder else "openai")
    vector_store = get_vector_store()

    # PDFファイル一覧
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing: {pdf_path.name}")

            # 1. PDFパース
            doc = parser.parse(pdf_path)
            logger.info(f"  - Parsed {doc.total_pages} pages")

            # 2. チャンク分割
            all_chunks = []
            for page in doc.pages:
                chunks = splitter.split(
                    text=page.text,
                    source_file=doc.file_name,
                    page_number=page.page_number,
                    tables=page.tables if hasattr(splitter, "split") else None,
                )
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning(f"  - No chunks generated for {pdf_path.name}")
                continue

            logger.info(f"  - Generated {len(all_chunks)} chunks")

            # 3. 埋め込み生成
            embedded_chunks = embedder.embed_chunks(all_chunks)
            logger.info(f"  - Generated embeddings")

            # 4. ベクトルDBに格納
            chunks_list = [c for c, _ in embedded_chunks]
            embeddings_list = [e for _, e in embedded_chunks]
            count = vector_store.add_documents(chunks_list, embeddings_list)
            total_chunks += count

            logger.info(f"  - Added {count} documents to vector store")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            continue

    # サマリー
    logger.info("=" * 50)
    logger.info("Ingestion Summary:")
    logger.info(f"  - PDFs processed: {len(pdf_files)}")
    logger.info(f"  - Total chunks: {total_chunks}")
    logger.info(f"  - Vector store: {vector_store.get_collection_info()}")
    logger.info("=" * 50)


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF documents into vector store")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="PDF directory path (default: data/raw/pdfs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk size (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Chunk overlap (default: 100)",
    )
    parser.add_argument(
        "--parser",
        type=str,
        default="hybrid",
        choices=["pymupdf", "pdfplumber", "hybrid"],
        help="PDF parser type (default: hybrid)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock embedder (for development)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else project_root / "data" / "raw" / "pdfs"

    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        logger.info("Run download_dataset.py first to download PDFs")
        sys.exit(1)

    ingest_pdfs(
        pdf_dir=pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        parser_type=args.parser,
        use_mock_embedder=args.mock,
    )


if __name__ == "__main__":
    main()
