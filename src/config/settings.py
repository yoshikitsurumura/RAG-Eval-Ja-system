"""
設定管理モジュール

Pydantic Settingsを使用して環境変数から設定を読み込む
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """アプリケーション設定"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI API
    openai_api_key: str = Field(..., description="OpenAI API Key")
    llm_model: str = Field(default="gpt-5-mini", description="LLM Model Name")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding Model Name"
    )
    embedding_dimension: int = Field(default=1536, description="Embedding Dimension")

    # Qdrant
    qdrant_host: str = Field(default="localhost", description="Qdrant Host")
    qdrant_port: int = Field(default=6333, description="Qdrant Port")
    qdrant_collection: str = Field(default="laboro_rag", description="Qdrant Collection Name")

    # API Server
    api_host: str = Field(default="0.0.0.0", description="API Host")
    api_port: int = Field(default=8000, description="API Port")

    # UI Server
    ui_port: int = Field(default=8501, description="UI Port")

    # Logging
    log_level: str = Field(default="INFO", description="Log Level")

    # RAG Configuration
    chunk_size: int = Field(default=500, description="Text chunk size")
    chunk_overlap: int = Field(default=100, description="Chunk overlap")
    retrieval_top_k: int = Field(default=5, description="Number of documents to retrieve")

    # Paths
    @property
    def project_root(self) -> Path:
        """プロジェクトルートパス"""
        return Path(__file__).parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """データディレクトリパス"""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """生データディレクトリパス"""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """処理済みデータディレクトリパス"""
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    """設定のシングルトンインスタンスを取得"""
    return Settings()
