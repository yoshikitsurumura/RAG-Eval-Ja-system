"""
PDFパーサーモジュール

PyMuPDFとpdfplumberを使用してPDFからテキストと表を抽出
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """パース済みページデータ"""

    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """パース済みドキュメントデータ"""

    file_path: str
    file_name: str
    total_pages: int
    pages: list[ParsedPage]
    metadata: dict = field(default_factory=dict)

    def get_full_text(self) -> str:
        """全テキストを結合して返す"""
        return "\n\n".join(page.text for page in self.pages if page.text)


class PDFParserBase(ABC):
    """PDFパーサー基底クラス"""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """PDFをパースしてドキュメントを返す"""
        pass


class PyMuPDFParser(PDFParserBase):
    """PyMuPDFを使用したパーサー（高速、基本的なテキスト抽出）"""

    def parse(self, file_path: Path) -> ParsedDocument:
        """PDFをパース"""
        logger.info(f"Parsing with PyMuPDF: {file_path.name}")

        pages = []
        with fitz.open(file_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                pages.append(
                    ParsedPage(
                        page_number=page_num,
                        text=text.strip(),
                        metadata={"method": "pymupdf"},
                    )
                )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            total_pages=len(pages),
            pages=pages,
            metadata={"parser": "PyMuPDF"},
        )


class PdfPlumberParser(PDFParserBase):
    """pdfplumberを使用したパーサー（表抽出に強い）"""

    def parse(self, file_path: Path) -> ParsedDocument:
        """PDFをパース（表も抽出）"""
        logger.info(f"Parsing with pdfplumber: {file_path.name}")

        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # テキスト抽出
                text = page.extract_text() or ""

                # 表抽出
                tables = []
                extracted_tables = page.extract_tables()
                if extracted_tables:
                    tables = extracted_tables

                pages.append(
                    ParsedPage(
                        page_number=page_num,
                        text=text.strip(),
                        tables=tables,
                        metadata={"method": "pdfplumber", "has_tables": len(tables) > 0},
                    )
                )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            total_pages=len(pages),
            pages=pages,
            metadata={"parser": "pdfplumber"},
        )


class HybridPDFParser(PDFParserBase):
    """
    ハイブリッドパーサー

    - PyMuPDF: 基本テキスト抽出（高速）
    - pdfplumber: 表抽出
    """

    def __init__(self):
        self.pymupdf_parser = PyMuPDFParser()
        self.pdfplumber_parser = PdfPlumberParser()

    def parse(self, file_path: Path) -> ParsedDocument:
        """ハイブリッドパース"""
        logger.info(f"Parsing with Hybrid method: {file_path.name}")

        # PyMuPDFで基本テキスト
        pymupdf_doc = self.pymupdf_parser.parse(file_path)

        # pdfplumberで表を抽出
        pdfplumber_doc = self.pdfplumber_parser.parse(file_path)

        # マージ
        merged_pages = []
        for pymupdf_page, pdfplumber_page in zip(
            pymupdf_doc.pages, pdfplumber_doc.pages, strict=False
        ):
            # テキストはPyMuPDFを優先（通常より正確）
            # 表はpdfplumberから取得
            merged_pages.append(
                ParsedPage(
                    page_number=pymupdf_page.page_number,
                    text=pymupdf_page.text,
                    tables=pdfplumber_page.tables,
                    metadata={
                        "method": "hybrid",
                        "has_tables": len(pdfplumber_page.tables) > 0,
                    },
                )
            )

        return ParsedDocument(
            file_path=str(file_path),
            file_name=file_path.name,
            total_pages=len(merged_pages),
            pages=merged_pages,
            metadata={"parser": "Hybrid (PyMuPDF + pdfplumber)"},
        )


def get_parser(parser_type: str = "hybrid") -> PDFParserBase:
    """パーサーファクトリー"""
    parsers = {
        "pymupdf": PyMuPDFParser,
        "pdfplumber": PdfPlumberParser,
        "hybrid": HybridPDFParser,
    }

    if parser_type not in parsers:
        raise ValueError(f"Unknown parser type: {parser_type}. Available: {list(parsers.keys())}")

    return parsers[parser_type]()
