"""
テキスト分割モジュール

RAGのためのチャンク分割を行う
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """テキストチャンク"""

    content: str
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.content)


class TextSplitterBase(ABC):
    """テキスト分割基底クラス"""

    @abstractmethod
    def split(self, text: str, source_file: str, page_number: int) -> list[TextChunk]:
        """テキストをチャンクに分割"""
        pass


class RecursiveTextSplitter(TextSplitterBase):
    """
    再帰的テキスト分割

    LangChainのRecursiveCharacterTextSplitterをラップ
    日本語に対応したセパレーターを使用
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 日本語対応セパレーター
        self.separators = [
            "\n\n",  # 段落
            "\n",  # 改行
            "。",  # 句点
            "、",  # 読点
            " ",  # スペース
            "",  # 文字単位
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split(self, text: str, source_file: str, page_number: int) -> list[TextChunk]:
        """テキストを分割"""
        if not text.strip():
            return []

        split_texts = self._splitter.split_text(text)

        chunks = []
        for idx, chunk_text in enumerate(split_texts):
            chunk_id = f"{source_file}_p{page_number}_c{idx}"
            chunks.append(
                TextChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=idx,
                    metadata={
                        "chunk_size": len(chunk_text),
                        "splitter": "recursive",
                    },
                )
            )

        logger.debug(f"Split page {page_number} into {len(chunks)} chunks")
        return chunks


class TableAwareTextSplitter(TextSplitterBase):
    """
    表を考慮したテキスト分割

    表は分割せずに1チャンクとして保持
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)

    def split(
        self,
        text: str,
        source_file: str,
        page_number: int,
        tables: list[list[list[str]]] | None = None,
    ) -> list[TextChunk]:
        """テキストと表を分割"""
        chunks = []

        # テキストを分割
        text_chunks = self.text_splitter.split(text, source_file, page_number)
        chunks.extend(text_chunks)

        # 表を追加（分割しない）
        if tables:
            for table_idx, table in enumerate(tables):
                table_text = self._table_to_text(table)
                if table_text.strip():
                    chunk_id = f"{source_file}_p{page_number}_table{table_idx}"
                    chunks.append(
                        TextChunk(
                            content=table_text,
                            chunk_id=chunk_id,
                            source_file=source_file,
                            page_number=page_number,
                            chunk_index=len(text_chunks) + table_idx,
                            metadata={
                                "chunk_size": len(table_text),
                                "splitter": "table_aware",
                                "is_table": True,
                            },
                        )
                    )

        return chunks

    def _table_to_text(self, table: list[list[str]]) -> str:
        """表をテキストに変換"""
        rows = []
        for row in table:
            # Noneを空文字に変換
            cleaned_row = [str(cell) if cell is not None else "" for cell in row]
            rows.append(" | ".join(cleaned_row))
        return "\n".join(rows)


def get_text_splitter(
    splitter_type: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> TextSplitterBase:
    """テキスト分割器ファクトリー"""
    splitters = {
        "recursive": RecursiveTextSplitter,
        "table_aware": TableAwareTextSplitter,
    }

    if splitter_type not in splitters:
        raise ValueError(
            f"Unknown splitter type: {splitter_type}. Available: {list(splitters.keys())}"
        )

    return splitters[splitter_type](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
