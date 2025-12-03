"""Pydantic models for Confluence pages and document chunks."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Page(BaseModel):
    """Represents a Confluence page with metadata."""

    id: str = Field(default=..., description="Unique page identifier")
    title: str = Field(default=..., description="Page title")
    space_key: str = Field(default=..., description="Confluence space key")
    content: str = Field(default=..., description="Page content in storage format")
    author: str = Field(default=..., description="Page author username")
    created_date: datetime = Field(default=..., description="Page creation timestamp")
    modified_date: datetime = Field(default=..., description="Last modification timestamp")
    url: HttpUrl = Field(default=..., description="Full URL to the page")
    version: int = Field(default=..., ge=1, description="Page version number")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "123456",
                "title": "Getting Started",
                "space_key": "DOCS",
                "content": "<p>Welcome to our documentation</p>",
                "author": "john.doe",
                "created_date": "2024-01-01T10:00:00Z",
                "modified_date": "2024-01-15T14:30:00Z",
                "url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                "version": 3,
            }
        }
    }


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata."""

    chunk_id: str = Field(
        default=..., description="Unique chunk identifier (format: {page_id}_{chunk_index})"
    )
    page_id: str = Field(default=..., description="Parent page identifier")
    content: str = Field(default=..., min_length=1, description="Chunk text content")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (title, url, author, dates)",
    )
    chunk_index: int = Field(default=..., ge=0, description="Zero-based chunk position within page")

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id_format(cls, v: str, _) -> str:
        """Validate that chunk_id follows the format {page_id}_{chunk_index}."""
        if "_" not in v:
            raise ValueError("chunk_id must follow format {page_id}_{chunk_index}")
        return v

    @property
    def page_title(self) -> str | None:
        """Extract page title from metadata."""
        return self.metadata.get("page_title")

    @property
    def page_url(self) -> str | None:
        """Extract page URL from metadata."""
        return self.metadata.get("page_url")

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": "123456_0",
                "page_id": "123456",
                "content": "This is the first chunk of content...",
                "metadata": {
                    "page_title": "Getting Started",
                    "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                    "author": "john.doe",
                    "modified_date": "2024-01-15T14:30:00Z",
                },
                "chunk_index": 0,
            }
        }
    }


class SearchResult(BaseModel):
    """Represents a search result with relevance score."""

    chunk_id: str = Field(default=..., description="Unique chunk identifier")
    page_id: str = Field(default=..., description="Parent page identifier")
    page_title: str = Field(default=..., description="Page title")
    page_url: HttpUrl = Field(default=..., description="Full URL to the source page")
    content: str = Field(default=..., description="Chunk text content")
    similarity_score: float = Field(
        default=..., ge=0.0, le=1.0, description="Cosine similarity score"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": "123456_0",
                "page_id": "123456",
                "page_title": "Getting Started",
                "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                "content": "This is the first chunk of content...",
                "similarity_score": 0.87,
                "metadata": {
                    "author": "john.doe",
                    "modified_date": "2024-01-15T14:30:00Z",
                },
            }
        }
    }


class SyncState(BaseModel):
    """Tracks synchronization state for a space."""

    space_key: str = Field(default=..., description="Confluence space key")
    last_sync_timestamp: datetime = Field(default=..., description="Last successful sync timestamp")
    page_count: int = Field(default=..., ge=0, description="Total number of pages synced")
    chunk_count: int = Field(default=..., ge=0, description="Total number of chunks created")

    model_config = {
        "json_schema_extra": {
            "example": {
                "space_key": "DOCS",
                "last_sync_timestamp": "2024-01-15T14:30:00Z",
                "page_count": 150,
                "chunk_count": 1250,
            }
        }
    }
