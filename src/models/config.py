"""Configuration models for the Confluence RAG system."""

from typing import Any

from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    type: str = Field(default=..., description="Vector store type (chroma, faiss, qdrant, etc.)")
    config: dict[str, Any] = Field(default_factory=dict, description="Store-specific configuration")


class ProcessingConfig(BaseModel):
    """Configuration for document processing."""

    chunk_size: int = Field(
        default=1000, ge=500, le=2000, description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=500, description="Overlap between chunks in tokens"
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )


class ConfluenceConfig(BaseModel):
    """Configuration for Confluence connection."""

    base_url: HttpUrl = Field(default=..., description="Confluence instance URL")
    auth_token: str = Field(default=..., description="API authentication token")
    space_key: str = Field(default=..., description="Space key to sync")
    cloud: bool = Field(default=True, description="True for Cloud, False for Server/Data Center")


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    json_logs: bool = Field(
        default=True,
        description="If True, output JSON logs. If False, use console format.",
    )
    log_file: str | None = Field(
        default=None,
        description="Optional path to log file. If None, logs only to stdout.",
    )


class AppConfig(BaseSettings):
    """Main application configuration.

    This class uses pydantic-settings to load configuration from environment
    variables with the APP_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    confluence: ConfluenceConfig
    processing: ProcessingConfig
    vector_store: VectorStoreConfig
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    top_k_results: int = Field(
        default=10, ge=1, le=100, description="Number of search results to return"
    )
