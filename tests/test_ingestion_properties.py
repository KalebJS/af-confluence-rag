"""Property-based tests for IngestionService.

**Feature: confluence-rag-system**
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.ingestion.ingestion_service import IngestionService
from src.models.page import DocumentChunk, Page

# Strategy for generating valid page IDs
page_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=48, max_codepoint=122),
    min_size=1,
    max_size=20,
)

# Strategy for generating page titles
title_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=100,
)

# Strategy for generating page content
content_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=10,
    max_size=500,
)

# Strategy for generating space keys
space_key_strategy = st.text(
    alphabet=st.characters(min_codepoint=65, max_codepoint=90),  # A-Z
    min_size=2,
    max_size=10,
)


def create_mock_page(page_id: str, title: str, content: str, space_key: str) -> Page:
    """Create a mock Page object for testing."""
    return Page(
        id=page_id,
        title=title,
        space_key=space_key,
        content=f"<p>{content}</p>",
        author="test_user",
        created_date=datetime.now(),
        modified_date=datetime.now(),
        url=f"https://example.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}",
        version=1,
    )


@given(
    page_id=page_id_strategy,
    title=title_strategy,
    content=content_strategy,
    space_key=space_key_strategy,
)
@settings(max_examples=100, deadline=None)
def test_property_33_graceful_error_recovery(
    page_id: str,
    title: str,
    content: str,
    space_key: str,
):
    """Property 33: Graceful error recovery

    *For any* invalid document encountered during batch processing, the system
    should log the error and continue processing remaining documents.

    **Validates: Requirements 9.3**
    **Feature: confluence-rag-system, Property 33: Graceful error recovery**
    """
    # Create fresh mocks for each test run
    confluence_client = Mock()
    chunker = Mock()
    embedder = Mock()
    vector_store = Mock()

    # Create valid and invalid pages
    valid_page = create_mock_page(page_id, title, content, space_key)
    invalid_page = create_mock_page(
        f"{page_id}_invalid",
        f"{title}_invalid",
        "",  # Empty content to trigger error
        space_key,
    )

    # Mock confluence client to return both pages
    confluence_client.get_space_pages.return_value = iter([valid_page, invalid_page])

    # Mock chunker to succeed for valid page, fail for invalid
    def chunker_side_effect(page):
        if page.id == valid_page.id:
            return [
                DocumentChunk(
                    chunk_id=f"{page.id}_0",
                    page_id=page.id,
                    content=content,
                    metadata={
                        "page_title": page.title,
                        "page_url": str(page.url),
                    },
                    chunk_index=0,
                )
            ]
        else:
            raise ValueError("Invalid page content")

    chunker.chunk_document.side_effect = chunker_side_effect

    # Mock embedder
    embedder.generate_batch_embeddings.return_value = [[0.1] * 384]

    # Mock vector store
    vector_store.add_documents.return_value = None
    vector_store.get_document_metadata.return_value = None

    # Create ingestion service
    with patch("src.ingestion.ingestion_service.SyncCoordinator"):
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        # Perform full ingestion (not incremental to test batch processing)
        result = service.ingest_space(space_key, incremental=False)

        # Verify that processing continued despite error
        # At least one page should have been processed successfully
        assert result["pages_processed"] >= 1, (
            f"Expected at least 1 page processed, got {result['pages_processed']}"
        )

        # Verify that errors were logged
        assert len(result["errors"]) >= 1, (
            f"Expected at least 1 error logged, got {len(result['errors'])}"
        )

        # Verify that the result indicates partial success
        # (success=False because there were errors, but pages_processed > 0)
        assert result["pages_processed"] > 0 or len(result["errors"]) > 0, (
            "Expected either pages processed or errors logged"
        )


@given(
    space_key=space_key_strategy,
    page_count=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50, deadline=None)
def test_property_35_completion_logging(
    space_key: str,
    page_count: int,
):
    """Property 35: Completion logging

    *For any* completed ingestion operation, the log should contain summary
    statistics including document_count and duration_seconds.

    **Validates: Requirements 9.5**
    **Feature: confluence-rag-system, Property 35: Completion logging**
    """
    # Create fresh mocks for each test run
    confluence_client = Mock()
    chunker = Mock()
    embedder = Mock()
    vector_store = Mock()

    # Create mock pages
    pages = [
        create_mock_page(
            f"page_{i}",
            f"Title {i}",
            f"Content {i}",
            space_key,
        )
        for i in range(page_count)
    ]

    # Mock confluence client
    confluence_client.get_space_pages.return_value = iter(pages)

    # Mock chunker to return one chunk per page
    def chunker_side_effect(page):
        return [
            DocumentChunk(
                chunk_id=f"{page.id}_0",
                page_id=page.id,
                content=f"Content for {page.title}",
                metadata={
                    "page_title": page.title,
                    "page_url": str(page.url),
                },
                chunk_index=0,
            )
        ]

    chunker.chunk_document.side_effect = chunker_side_effect

    # Mock embedder
    embedder.generate_batch_embeddings.return_value = [[0.1] * 384]

    # Mock vector store
    vector_store.add_documents.return_value = None
    vector_store.get_document_metadata.return_value = None

    # Create ingestion service
    with patch("src.ingestion.ingestion_service.SyncCoordinator"):
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        # Perform full ingestion
        result = service.ingest_space(space_key, incremental=False)

        # Verify that result contains required statistics
        assert "pages_processed" in result, "Result missing 'pages_processed'"
        assert "duration_seconds" in result, "Result missing 'duration_seconds'"
        assert "chunks_created" in result, "Result missing 'chunks_created'"
        assert "success" in result, "Result missing 'success'"

        # Verify that pages_processed matches expected count
        assert result["pages_processed"] == page_count, (
            f"Expected {page_count} pages processed, got {result['pages_processed']}"
        )

        # Verify that duration is positive
        assert result["duration_seconds"] > 0, (
            f"Expected positive duration, got {result['duration_seconds']}"
        )

        # Verify that chunks were created
        assert result["chunks_created"] == page_count, (
            f"Expected {page_count} chunks created, got {result['chunks_created']}"
        )

        # Verify success flag
        assert result["success"] is True, f"Expected success=True, got {result['success']}"


@given(space_key=space_key_strategy)
@settings(max_examples=50, deadline=None)
def test_database_unavailability_handling(
    space_key: str,
):
    """Test that database unavailability is handled gracefully.

    This test verifies that when the vector database is unavailable,
    the system logs the error and returns a clear error message.

    **Validates: Requirements 9.4**
    """
    # Create fresh mocks for each test run
    confluence_client = Mock()
    chunker = Mock()
    embedder = Mock()
    vector_store = Mock()

    # Mock vector store to raise error on health check
    vector_store.get_document_metadata.side_effect = RuntimeError("Connection refused")

    # Create ingestion service
    with patch("src.ingestion.ingestion_service.SyncCoordinator"):
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        # Attempt ingestion
        result = service.ingest_space(space_key, incremental=False)

        # Verify that the operation failed gracefully
        assert result["success"] is False, f"Expected success=False, got {result['success']}"

        # Verify that error message mentions database unavailability
        assert len(result["errors"]) > 0, "Expected error messages"
        error_msg = result["errors"][0].lower()
        assert "database" in error_msg or "unavailable" in error_msg, (
            f"Expected error message about database unavailability, got: {error_msg}"
        )

        # Verify that no pages were processed
        assert result["pages_processed"] == 0, (
            f"Expected 0 pages processed, got {result['pages_processed']}"
        )
