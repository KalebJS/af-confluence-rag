"""Unit tests for MetadataEnricher class."""

from datetime import datetime, timezone

import pytest

from src.models.page import Page
from src.processing.metadata_enricher import MetadataEnricher


@pytest.fixture
def sample_page():
    """Create a sample Page for testing."""
    return Page(
        id="123456",
        title="Test Page",
        space_key="TEST",
        content="<p>Test content</p>",
        author="test.user",
        created_date=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        modified_date=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        url="https://example.atlassian.net/wiki/spaces/TEST/pages/123456",
        version=3,
    )


def test_enrich_chunk_creates_correct_chunk_id(sample_page):
    """Test that enrich_chunk generates chunk_id in correct format."""
    enricher = MetadataEnricher()

    chunk = enricher.enrich_chunk(sample_page, "Test chunk content", 0)

    assert chunk.chunk_id == "123456_0"
    assert chunk.page_id == "123456"
    assert chunk.chunk_index == 0


def test_enrich_chunk_preserves_content(sample_page):
    """Test that enrich_chunk preserves the chunk text content."""
    enricher = MetadataEnricher()
    chunk_text = "This is the chunk content"

    chunk = enricher.enrich_chunk(sample_page, chunk_text, 0)

    assert chunk.content == chunk_text


def test_enrich_chunk_includes_all_metadata(sample_page):
    """Test that enrich_chunk includes all required metadata fields."""
    enricher = MetadataEnricher()

    chunk = enricher.enrich_chunk(sample_page, "Test content", 0)

    assert chunk.metadata["page_title"] == "Test Page"
    assert (
        chunk.metadata["page_url"] == "https://example.atlassian.net/wiki/spaces/TEST/pages/123456"
    )
    assert chunk.metadata["author"] == "test.user"
    assert chunk.metadata["created_date"] == "2024-01-01T10:00:00+00:00"
    assert chunk.metadata["modified_date"] == "2024-01-15T14:30:00+00:00"
    assert chunk.metadata["space_key"] == "TEST"
    assert chunk.metadata["version"] == 3


def test_enrich_chunk_generates_unique_ids_for_different_indices(sample_page):
    """Test that different chunk indices generate unique chunk_ids."""
    enricher = MetadataEnricher()

    chunk0 = enricher.enrich_chunk(sample_page, "First chunk", 0)
    chunk1 = enricher.enrich_chunk(sample_page, "Second chunk", 1)
    chunk2 = enricher.enrich_chunk(sample_page, "Third chunk", 2)

    assert chunk0.chunk_id == "123456_0"
    assert chunk1.chunk_id == "123456_1"
    assert chunk2.chunk_id == "123456_2"

    # All chunk_ids should be unique
    chunk_ids = {chunk0.chunk_id, chunk1.chunk_id, chunk2.chunk_id}
    assert len(chunk_ids) == 3


def test_enrich_chunks_processes_multiple_chunks(sample_page):
    """Test that enrich_chunks processes multiple text chunks correctly."""
    enricher = MetadataEnricher()
    chunk_texts = ["First chunk", "Second chunk", "Third chunk"]

    chunks = enricher.enrich_chunks(sample_page, chunk_texts)

    assert len(chunks) == 3
    assert chunks[0].content == "First chunk"
    assert chunks[1].content == "Second chunk"
    assert chunks[2].content == "Third chunk"

    # Verify chunk indices
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2

    # Verify chunk_ids
    assert chunks[0].chunk_id == "123456_0"
    assert chunks[1].chunk_id == "123456_1"
    assert chunks[2].chunk_id == "123456_2"


def test_enrich_chunks_with_empty_list(sample_page):
    """Test that enrich_chunks handles empty list correctly."""
    enricher = MetadataEnricher()

    chunks = enricher.enrich_chunks(sample_page, [])

    assert len(chunks) == 0
    assert chunks == []


def test_enrich_chunks_preserves_metadata_for_all_chunks(sample_page):
    """Test that all chunks get the same page metadata."""
    enricher = MetadataEnricher()
    chunk_texts = ["Chunk 1", "Chunk 2", "Chunk 3"]

    chunks = enricher.enrich_chunks(sample_page, chunk_texts)

    # All chunks should have the same metadata
    for chunk in chunks:
        assert chunk.metadata["page_title"] == "Test Page"
        assert chunk.metadata["author"] == "test.user"
        assert chunk.metadata["space_key"] == "TEST"
        assert chunk.page_id == "123456"
