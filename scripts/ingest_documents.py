"""
ドキュメント取り込みスクリプト (v4: Vision + Auto-Tagging)

PDFをパースし、図表解析結果と自動カテゴリタグを付与してベクトルDBに格納
"""

import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.ingestion import get_embedder, get_parser, get_text_splitter
from src.ingestion.image_processor import get_image_processor
from src.ingestion.document_classifier import get_document_classifier
from src.retrieval import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdfs(
    pdf_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    parser_type: str = "hybrid",
    use_vision: bool = True,
    use_mock_embedder: bool = False,
):
    """
    PDFディレクトリ内のファイルをベクトルDBに取り込む
    """
    settings = get_settings()

    # コンポーネント初期化
    parser = get_parser(parser_type)
    splitter = get_text_splitter("table_aware", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedder = get_embedder("mock" if use_mock_embedder else "openai")
    vector_store = get_vector_store()
    
    # Vision & Classifier
    image_processor = get_image_processor() if use_vision else None
    classifier = get_document_classifier()

    # PDFファイル一覧
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    total_chunks = 0
    for pdf_path in pdf_files:
        try:
            logger.info(f"Processing: {pdf_path.name}")

            # 1. PDFパース (テキスト + 表 + 画像)
            doc = parser.parse(pdf_path)
            logger.info(f"  - Parsed {doc.total_pages} pages")

            # 2. 自動分類 (Auto-Tagging)
            # 冒頭のテキストを使ってカテゴリを判定
            first_page_text = doc.pages[0].text if doc.pages else ""
            classification = classifier.classify(first_page_text, pdf_path.name)
            logger.info(f"  - Classified as: {classification['category_name']}")
            
            # ドキュメント全体の共通メタデータ
            doc_metadata = {
                "category_id": classification["category_id"],
                "category_name": classification["category_name"],
                "category_reasoning": classification["reasoning"]
            }

            # 3. ページごとに処理
            all_chunks = []
            for page in doc.pages:
                page_text = page.text
                
                # 画像キャプションの生成と結合
                if image_processor and page.images:
                    image_descriptions = image_processor.process_page_images(page)
                    if image_descriptions:
                        page_text += "\n\n" + image_descriptions
                        logger.info(f"  - Page {page.page_number}: Added image descriptions")

                # チャンク分割
                chunks = splitter.split(
                    text=page_text,
                    source_file=doc.file_name,
                    page_number=page.page_number,
                    tables=page.tables,
                )
                
                # 分類タグをチャンクのメタデータに追加
                for chunk in chunks:
                    chunk.metadata.update(doc_metadata)
                    
                all_chunks.extend(chunks)

            if not all_chunks:
                logger.warning(f"  - No chunks generated for {pdf_path.name}")
                continue

            logger.info(f"  - Generated {len(all_chunks)} chunks (with vision: {use_vision})")

            # 4. 埋め込み生成
            embedded_chunks = embedder.embed_chunks(all_chunks)
            logger.info(f"  - Generated embeddings")

            # 5. ベクトルDBに格納
            chunks_list = [c for c, _ in embedded_chunks]
            embeddings_list = [e for _, e in embedded_chunks]
            count = vector_store.add_documents(chunks_list, embeddings_list)
            total_chunks += count

            logger.info(f"  - Added {count} documents to vector store")

        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # サマリー
    logger.info("=" * 50)
    logger.info("Ingestion Summary (v4 Vision + Auto-Tagging):")
    logger.info(f"  - PDFs processed: {len(pdf_files)}")
    logger.info(f"  - Total chunks: {total_chunks}")
    logger.info(f"  - Vector store: {vector_store.get_collection_info()}")
    logger.info("=" * 50)


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest PDF documents with Vision & Auto-Tagging")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="PDF directory path (default: data/raw/pdfs)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap (default: 200)",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable vision processing",
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
        sys.exit(1)

    ingest_pdfs(
        pdf_dir=pdf_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_vision=not args.no_vision,
        use_mock_embedder=args.mock,
    )


if __name__ == "__main__":
    main()
