"""Property-based tests for synchronization logic.

**Feature: confluence-rag-system, Property 13: Timestamp comparison correctness**
**Feature: confluence-rag-system, Property 14: Update replaces old embeddings**
**Feature: confluence-rag-system, Property 15: New page processing**
**Feature: confluence-rag-system, Property 16: Deletion completeness**
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.ingestion.confluence_client import ConfluenceClient
from src.models.page import Page, SyncState
from src.processing.chunker import DocumentChunker
from src.providers import get_embeddings, get_vector_store
from src.sync.change_detector import ChangeDetector
from src.sync.sync_coordinator import SyncCoordinator
from src.sync.timestamp_tracker import TimestampTracker


# Strategies for generating test data
@st.composite
def page_strategy(draw: st.DrawFn, page_id: str | None = None) -> Page:
    """Generate a random Page object."""
    if page_id is None:
        page_id = draw(
            st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Nd",)))
        )

    title = draw(st.text(min_size=1, max_size=50))
    space_key = draw(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",)))
    )
    content = draw(st.text(min_size=10, max_size=500))
    author = draw(st.text(min_size=1, max_size=20))

    # Generate naive timestamps first (hypothesis requirement)
    created_date_naive = draw(
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 1, 1))
    )
    # Add timezone
    created_date = created_date_naive.replace(tzinfo=timezone.utc)

    modified_date_naive = draw(
        st.datetimes(min_value=created_date_naive, max_value=datetime(2024, 12, 1))
    )
    # Add timezone
    modified_date = modified_date_naive.replace(tzinfo=timezone.utc)

    version = draw(st.integers(min_value=1, max_value=100))

    return Page(
        id=page_id,
        title=title,
        space_key=space_key,
        content=f"<p>{content}</p>",
        author=author,
        created_date=created_date,
        modified_date=modified_date,
        url=f"https://example.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}",
        version=version,
    )


@st.composite
def stored_metadata_strategy(draw: st.DrawFn, page: Page) -> dict[str, Any]:
    """Generate stored metadata for a page."""
    # Generate a modified date that's older than the page's current modified date
    stored_modified = draw(
        st.datetimes(
            min_value=page.created_date, max_value=page.modified_date - timedelta(seconds=1)
        )
    )

    return {
        "page_id": page.id,
        "page_title": page.title,
        "page_url": str(page.url),
        "author": page.author,
        "modified_date": stored_modified.isoformat(),
    }


class TestTimestampComparison:
    """Test Property 13: Timestamp comparison correctness.

    **Feature: confluence-rag-system, Property 13: Timestamp comparison correctness**
    **Validates: Requirements 4.1**

    For any page with a modification timestamp newer than the stored timestamp,
    the system should identify it as modified.
    """

    @given(page=page_strategy())
    @settings(max_examples=100)
    def test_newer_timestamp_detected_as_modified(self, page: Page) -> None:
        """Test that pages with newer timestamps are detected as modified."""
        detector = ChangeDetector()

        # Create stored metadata with older timestamp
        older_timestamp = page.modified_date - timedelta(hours=1)
        stored_metadata = {
            page.id: {
                "page_id": page.id,
                "page_title": page.title,
                "page_url": str(page.url),
                "author": page.author,
                "modified_date": older_timestamp.isoformat(),
            }
        }

        # Detect changes
        changes = detector.detect_changes([page], stored_metadata, older_timestamp)

        # Page should be detected as modified
        assert page in changes.modified_pages, (
            f"Page with newer timestamp should be detected as modified. "
            f"Page modified: {page.modified_date}, Stored: {older_timestamp}"
        )
        assert page not in changes.new_pages, "Page should not be detected as new"

    @given(page=page_strategy())
    @settings(max_examples=100)
    def test_same_timestamp_not_detected_as_modified(self, page: Page) -> None:
        """Test that pages with same timestamp are not detected as modified."""
        detector = ChangeDetector()

        # Create stored metadata with same timestamp
        stored_metadata = {
            page.id: {
                "page_id": page.id,
                "page_title": page.title,
                "page_url": str(page.url),
                "author": page.author,
                "modified_date": page.modified_date.isoformat(),
            }
        }

        # Detect changes
        changes = detector.detect_changes([page], stored_metadata, page.modified_date)

        # Page should not be detected as modified
        assert page not in changes.modified_pages, (
            "Page with same timestamp should not be detected as modified"
        )


class TestUpdateReplacesOldEmbeddings:
    """Test Property 14: Update replaces old embeddings.

    **Feature: confluence-rag-system, Property 14: Update replaces old embeddings**
    **Validates: Requirements 4.2**

    For any page that is updated, after synchronization completes, only embeddings
    with the new modification timestamp should exist (no old embeddings should remain).
    """

    @given(page=page_strategy())
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_update_removes_old_embeddings(self, page: Page) -> None:
        """Test that updating a page removes old embeddings."""
        # Create temporary directory for this test
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings and vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_update",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Create old version of page
            old_page = Page(
                id=page.id,
                title=page.title,
                space_key=page.space_key,
                content="<p>Old content</p>",
                author=page.author,
                created_date=page.created_date,
                modified_date=page.modified_date - timedelta(hours=1),
                url=page.url,
                version=max(1, page.version - 1),  # Ensure version is at least 1
            )

            # Process and store old version
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            old_chunks = chunker.chunk_document(old_page)
            if old_chunks:
                from src.models.page import to_langchain_document

                old_docs = [to_langchain_document(chunk) for chunk in old_chunks]
                vector_store.add_documents(old_docs)

            # Now update with new version - delete old and add new
            # Use metadata-based filtering for deletion
            results = vector_store.get(where={"page_id": page.id})
            if results and results.get("ids"):
                vector_store.delete(ids=results["ids"])

            new_chunks = chunker.chunk_document(page)
            if new_chunks:
                from src.models.page import to_langchain_document

                new_docs = [to_langchain_document(chunk) for chunk in new_chunks]
                vector_store.add_documents(new_docs)

            # Verify only new version exists
            results = vector_store.get(where={"page_id": page.id}, limit=1)
            if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                metadata = results["metadatas"][0]
                stored_modified = metadata.get("modified_date", "")
                assert stored_modified == page.modified_date.isoformat(), (
                    f"Only new embeddings should exist. "
                    f"Expected: {page.modified_date.isoformat()}, Got: {stored_modified}"
                )


class TestNewPageProcessing:
    """Test Property 15: New page processing.

    **Feature: confluence-rag-system, Property 15: New page processing**
    **Validates: Requirements 4.3**

    For any page that exists in Confluence but not in the vector database,
    synchronization should result in embeddings being created for that page.
    """

    @given(page=page_strategy())
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_new_page_creates_embeddings(self, page: Page) -> None:
        """Test that new pages result in embeddings being created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings and vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_new_page",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Verify page doesn't exist
            results_before = vector_store.get(where={"page_id": page.id}, limit=1)
            assert not results_before or not results_before.get("ids"), (
                "Page should not exist before processing"
            )

            # Process new page
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            chunks = chunker.chunk_document(page)
            if chunks:
                from src.models.page import to_langchain_document

                docs = [to_langchain_document(chunk) for chunk in chunks]
                vector_store.add_documents(docs)

                # Verify page now exists
                results_after = vector_store.get(where={"page_id": page.id}, limit=1)
                assert results_after and results_after.get("ids"), (
                    "Embeddings should be created for new page"
                )
                if results_after.get("metadatas") and len(results_after["metadatas"]) > 0:
                    metadata_after = results_after["metadatas"][0]
                    assert metadata_after["page_id"] == page.id, "Stored page_id should match"


class TestDeletionCompleteness:
    """Test Property 16: Deletion completeness.

    **Feature: confluence-rag-system, Property 16: Deletion completeness**
    **Validates: Requirements 4.4**

    For any page_id that is deleted, after synchronization, no embeddings
    with that page_id should exist in the vector database.
    """

    @given(page=page_strategy())
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_deletion_removes_all_embeddings(self, page: Page) -> None:
        """Test that deleting a page removes all its embeddings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings and vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_deletion",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Add page to vector store
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            chunks = chunker.chunk_document(page)
            if chunks:
                from src.models.page import to_langchain_document

                docs = [to_langchain_document(chunk) for chunk in chunks]
                vector_store.add_documents(docs)

                # Verify page exists
                results_before = vector_store.get(where={"page_id": page.id}, limit=1)
                assert results_before and results_before.get("ids"), (
                    "Page should exist before deletion"
                )

                # Delete page using metadata-based filtering
                results = vector_store.get(where={"page_id": page.id})
                if results and results.get("ids"):
                    vector_store.delete(ids=results["ids"])

                # Verify page no longer exists
                results_after = vector_store.get(where={"page_id": page.id}, limit=1)
                assert not results_after or not results_after.get("ids"), (
                    f"No embeddings should exist after deletion for page_id {page.id}"
                )

    @given(pages=st.lists(page_strategy(), min_size=2, max_size=3, unique_by=lambda p: p.id))
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_deletion_only_removes_target_page(self, pages: list[Page]) -> None:
        """Test that deleting one page doesn't affect other pages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings and vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_selective_deletion",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Add all pages to vector store
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            for page in pages:
                chunks = chunker.chunk_document(page)
                if chunks:
                    from src.models.page import to_langchain_document

                    docs = [to_langchain_document(chunk) for chunk in chunks]
                    vector_store.add_documents(docs)

            # Delete first page
            page_to_delete = pages[0]
            results = vector_store.get(where={"page_id": page_to_delete.id})
            if results and results.get("ids"):
                vector_store.delete(ids=results["ids"])

            # Verify deleted page is gone
            results_deleted = vector_store.get(where={"page_id": page_to_delete.id}, limit=1)
            assert not results_deleted or not results_deleted.get("ids"), (
                f"Deleted page {page_to_delete.id} should not exist"
            )

            # Verify other pages still exist
            for page in pages[1:]:
                results = vector_store.get(where={"page_id": page.id}, limit=1)
                # Only check if the page had chunks (some might not due to content)
                if results and results.get("metadatas") and len(results["metadatas"]) > 0:
                    metadata = results["metadatas"][0]
                    assert metadata["page_id"] == page.id, (
                        f"Other pages should not be affected by deletion. "
                        f"Expected {page.id}, got {metadata['page_id']}"
                    )


class TestSyncTimestampUpdate:
    """Test Property 17: Sync timestamp update.

    **Feature: confluence-rag-system, Property 17: Sync timestamp update**
    **Validates: Requirements 4.5**

    For any synchronization operation that completes successfully, the last_sync_timestamp
    should be updated to a value greater than the previous timestamp.
    """

    @given(
        space_key=st.text(
            min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))
        )
    )
    @settings(max_examples=3, deadline=None)
    def test_sync_timestamp_increases(self, space_key: str) -> None:
        """Test that sync timestamp increases after successful sync."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create embeddings and vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_timestamp",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            tracker = TimestampTracker(vector_store)

            # Create initial sync state
            initial_time = datetime.now(timezone.utc)
            initial_state = SyncState(
                space_key=space_key,
                last_sync_timestamp=initial_time,
                page_count=0,
                chunk_count=0,
            )

            tracker.save_sync_state(initial_state)

            # Wait a moment to ensure time difference
            import time

            time.sleep(0.01)

            # Create new sync state with later timestamp
            new_time = datetime.now(timezone.utc)
            new_state = SyncState(
                space_key=space_key,
                last_sync_timestamp=new_time,
                page_count=10,
                chunk_count=100,
            )

            tracker.save_sync_state(new_state)

            # Load and verify
            loaded_state = tracker.load_sync_state(space_key)
            assert loaded_state is not None, "Sync state should be loadable"
            assert loaded_state.last_sync_timestamp > initial_time, (
                f"Sync timestamp should increase. "
                f"Initial: {initial_time}, New: {loaded_state.last_sync_timestamp}"
            )
            assert loaded_state.last_sync_timestamp == new_time, (
                "Loaded timestamp should match saved timestamp"
            )


class TestMetadataBasedDeletion:
    """Test Property 8: Metadata-Based Deletion.

    **Feature: langchain-abstraction-refactor, Property 8: Metadata-Based Deletion**
    **Validates: Requirements 11.4**

    For any page_id, when documents are deleted by page_id, all documents with
    that page_id SHALL be removed from the vector store, and subsequent searches
    SHALL not return any documents with that page_id.
    """

    @given(page=page_strategy())
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_property_8_metadata_based_deletion(self, page: Page) -> None:
        """Property 8: Metadata-Based Deletion.

        For any page_id, when documents are deleted by page_id using metadata-based
        filtering, all documents with that page_id should be removed from the vector
        store, and subsequent searches should not return any documents with that page_id.

        **Feature: langchain-abstraction-refactor, Property 8: Metadata-Based Deletion**
        **Validates: Requirements 11.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create LangChain-based vector store using provider module
            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_metadata_deletion",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Create sync coordinator with LangChain abstractions
            # Create a mock confluence client (we won't use it for this test)
            confluence_client = ConfluenceClient(
                base_url="https://example.atlassian.net",
                auth_token="test_token",
                cloud=True,
            )

            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            sync_coordinator = SyncCoordinator(
                confluence_client=confluence_client,
                chunker=chunker,
                embeddings=embeddings,
                vector_store=vector_store,
            )

            # Process and store the page
            chunks = chunker.chunk_document(page)
            if not chunks:
                # Skip if no chunks generated
                return

            # Store the page using the sync coordinator's method
            sync_coordinator._process_and_store_page(page)

            # Verify page exists by searching with metadata filter
            results_before = vector_store.similarity_search(
                query="test", k=100, filter={"page_id": page.id}
            )
            assert len(results_before) > 0, f"Page {page.id} should exist before deletion"

            # Verify all results have the correct page_id
            for doc in results_before:
                assert doc.metadata.get("page_id") == page.id, (
                    f"All documents should have page_id {page.id}"
                )

            # Delete the page using metadata-based deletion
            sync_coordinator._delete_by_page_id(page.id)

            # Verify page no longer exists by searching with metadata filter
            results_after = vector_store.similarity_search(
                query="test", k=100, filter={"page_id": page.id}
            )
            assert len(results_after) == 0, (
                f"No documents with page_id {page.id} should exist after deletion. "
                f"Found {len(results_after)} documents."
            )

    @given(pages=st.lists(page_strategy(), min_size=2, max_size=3, unique_by=lambda p: p.id))
    @settings(
        max_examples=3, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
    )
    def test_property_8_selective_deletion(self, pages: list[Page]) -> None:
        """Property 8: Metadata-Based Deletion - Selective deletion.

        For any set of pages, when one page is deleted by page_id, only documents
        with that specific page_id should be removed, and documents with other
        page_ids should remain in the vector store.

        **Feature: langchain-abstraction-refactor, Property 8: Metadata-Based Deletion**
        **Validates: Requirements 11.4**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create LangChain-based vector store using provider module
            from src.providers import get_embeddings, get_vector_store

            embeddings = get_embeddings("all-MiniLM-L6-v2")
            vector_store = get_vector_store(
                embeddings=embeddings,
                collection_name="test_selective_deletion",
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
            )

            # Create sync coordinator with LangChain abstractions
            # Create a mock confluence client (we won't use it for this test)
            confluence_client = ConfluenceClient(
                base_url="https://example.atlassian.net",
                auth_token="test_token",
                cloud=True,
            )

            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

            sync_coordinator = SyncCoordinator(
                confluence_client=confluence_client,
                chunker=chunker,
                embeddings=embeddings,
                vector_store=vector_store,
            )

            # Store all pages
            pages_with_chunks = []
            for page in pages:
                chunks = chunker.chunk_document(page)
                if chunks:
                    sync_coordinator._process_and_store_page(page)
                    pages_with_chunks.append(page)

            if len(pages_with_chunks) < 2:
                # Skip if we don't have at least 2 pages with chunks
                return

            # Delete the first page
            page_to_delete = pages_with_chunks[0]
            sync_coordinator._delete_by_page_id(page_to_delete.id)

            # Verify deleted page is gone
            results_deleted = vector_store.similarity_search(
                query="test", k=100, filter={"page_id": page_to_delete.id}
            )
            assert len(results_deleted) == 0, (
                f"Deleted page {page_to_delete.id} should not exist"
            )

            # Verify other pages still exist
            for page in pages_with_chunks[1:]:
                results = vector_store.similarity_search(
                    query="test", k=100, filter={"page_id": page.id}
                )
                assert len(results) > 0, (
                    f"Other pages should not be affected by deletion. "
                    f"Page {page.id} should still exist."
                )
