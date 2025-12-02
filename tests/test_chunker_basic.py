"""Basic tests for DocumentChunker functionality."""

from datetime import datetime

import pytest

from src.processing.chunker import DocumentChunker
from src.models.page import Page


def test_chunker_initialization():
    """Test that DocumentChunker initializes correctly."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    assert chunker.chunk_size == 1000
    assert chunker.chunk_overlap == 200


def test_chunker_validates_chunk_size():
    """Test that DocumentChunker validates chunk_size bounds."""
    with pytest.raises(ValueError, match="chunk_size must be between 500 and 2000"):
        DocumentChunker(chunk_size=400, chunk_overlap=200)
    
    with pytest.raises(ValueError, match="chunk_size must be between 500 and 2000"):
        DocumentChunker(chunk_size=2100, chunk_overlap=200)


def test_chunker_validates_chunk_overlap():
    """Test that DocumentChunker validates chunk_overlap bounds."""
    with pytest.raises(ValueError, match="chunk_overlap must be between 0 and 500"):
        DocumentChunker(chunk_size=1000, chunk_overlap=-1)
    
    with pytest.raises(ValueError, match="chunk_overlap must be between 0 and 500"):
        DocumentChunker(chunk_size=1000, chunk_overlap=600)


def test_chunk_document_basic():
    """Test basic document chunking."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
    
    page = Page(
        id="12345",
        title="Test Page",
        space_key="TEST",
        content="<p>This is a test page with some content. " * 10 + "</p>",
        author="test.user",
        created_date=datetime(2024, 1, 1, 10, 0, 0),
        modified_date=datetime(2024, 1, 15, 14, 30, 0),
        url="https://example.atlassian.net/wiki/spaces/TEST/pages/12345",
        version=1,
    )
    
    chunks = chunker.chunk_document(page)
    
    # Should create multiple chunks
    assert len(chunks) > 0
    
    # Each chunk should have correct structure
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"12345_{idx}"
        assert chunk.page_id == "12345"
        assert chunk.chunk_index == idx
        assert len(chunk.content) > 0
        assert chunk.metadata["page_title"] == "Test Page"
        assert chunk.metadata["page_url"] == "https://example.atlassian.net/wiki/spaces/TEST/pages/12345"
        assert chunk.metadata["author"] == "test.user"


def test_html_cleaning():
    """Test that HTML is properly cleaned from content."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    
    page = Page(
        id="12345",
        title="Test Page",
        space_key="TEST",
        content="<p>Paragraph 1</p><p>Paragraph 2</p><script>alert('bad')</script>",
        author="test.user",
        created_date=datetime(2024, 1, 1, 10, 0, 0),
        modified_date=datetime(2024, 1, 15, 14, 30, 0),
        url="https://example.atlassian.net/wiki/spaces/TEST/pages/12345",
        version=1,
    )
    
    chunks = chunker.chunk_document(page)
    
    assert len(chunks) > 0
    # Script tags should be removed
    assert "alert" not in chunks[0].content
    assert "script" not in chunks[0].content
    # Content should be preserved
    assert "Paragraph 1" in chunks[0].content
    assert "Paragraph 2" in chunks[0].content


def test_empty_content_handling():
    """Test handling of empty content after HTML cleaning."""
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    
    page = Page(
        id="12345",
        title="Empty Page",
        space_key="TEST",
        content="<script>alert('only script')</script>",
        author="test.user",
        created_date=datetime(2024, 1, 1, 10, 0, 0),
        modified_date=datetime(2024, 1, 15, 14, 30, 0),
        url="https://example.atlassian.net/wiki/spaces/TEST/pages/12345",
        version=1,
    )
    
    chunks = chunker.chunk_document(page)
    
    # Should return empty list for empty content
    assert len(chunks) == 0
