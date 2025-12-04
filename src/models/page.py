"""Pydantic models for Confluence pages and document chunks."""

from datetime import datetime
from typing import Any

from langchain_core.documents import Document
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


# Document Mapping Functions


def to_langchain_document(chunk: DocumentChunk) -> Document:
    """Convert DocumentChunk to LangChain Document.
    
    Maps the DocumentChunk model to LangChain's Document format, preserving
    all metadata fields including chunk_id, page_id, and chunk_index.
    
    Args:
        chunk: DocumentChunk instance to convert
        
    Returns:
        LangChain Document with content and metadata
    """
    # Create a copy of metadata and add chunk-specific fields
    metadata = chunk.metadata.copy()
    metadata["chunk_id"] = chunk.chunk_id
    metadata["page_id"] = chunk.page_id
    metadata["chunk_index"] = chunk.chunk_index
    
    return Document(
        page_content=chunk.content,
        metadata=metadata
    )


def from_langchain_document(doc: Document, chunk_id: str, page_id: str) -> DocumentChunk:
    """Convert LangChain Document to DocumentChunk.
    
    Maps LangChain's Document format back to the DocumentChunk model,
    extracting chunk-specific fields from metadata.
    
    Args:
        doc: LangChain Document instance to convert
        chunk_id: Unique chunk identifier (format: {page_id}_{chunk_index})
        page_id: Parent page identifier
        
    Returns:
        DocumentChunk instance with content and metadata
    """
    # Extract chunk_index from metadata or derive from chunk_id
    metadata = doc.metadata.copy()
    chunk_index = metadata.pop("chunk_index", None)
    
    # If chunk_index not in metadata, try to extract from chunk_id
    if chunk_index is None:
        try:
            chunk_index = int(chunk_id.split("_")[-1])
        except (ValueError, IndexError):
            chunk_index = 0
    
    # Remove chunk_id and page_id from metadata if present (they're separate fields)
    metadata.pop("chunk_id", None)
    metadata.pop("page_id", None)
    
    return DocumentChunk(
        chunk_id=chunk_id,
        page_id=page_id,
        content=doc.page_content,
        metadata=metadata,
        chunk_index=chunk_index
    )


def to_langchain_documents(chunks: list[DocumentChunk]) -> list[Document]:
    """Convert a list of DocumentChunks to LangChain Documents.
    
    Batch conversion helper for processing multiple chunks at once.
    
    Args:
        chunks: List of DocumentChunk instances to convert
        
    Returns:
        List of LangChain Documents
    """
    return [to_langchain_document(chunk) for chunk in chunks]


def from_langchain_documents(
    docs: list[Document],
    chunk_ids: list[str],
    page_ids: list[str]
) -> list[DocumentChunk]:
    """Convert a list of LangChain Documents to DocumentChunks.
    
    Batch conversion helper for processing multiple documents at once.
    
    Args:
        docs: List of LangChain Document instances to convert
        chunk_ids: List of chunk identifiers (must match length of docs)
        page_ids: List of page identifiers (must match length of docs)
        
    Returns:
        List of DocumentChunk instances
        
    Raises:
        ValueError: If lengths of docs, chunk_ids, and page_ids don't match
    """
    if not (len(docs) == len(chunk_ids) == len(page_ids)):
        raise ValueError(
            f"Length mismatch: docs={len(docs)}, chunk_ids={len(chunk_ids)}, page_ids={len(page_ids)}"
        )
    
    return [
        from_langchain_document(doc, chunk_id, page_id)
        for doc, chunk_id, page_id in zip(docs, chunk_ids, page_ids)
    ]
