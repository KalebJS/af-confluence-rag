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
from hypothesis import given, settings, strategies as st, HealthCheck

from src.ingestion.confluence_client import ConfluenceClient
from src.models.page import Page, SyncState
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import ChromaStore
from src.sync.change_detector import ChangeDetector
from src.sync.sync_coordinator import SyncCoordinator
from src.sync.timestamp_tracker import TimestampTracker


# Strategies for generating test data
@st.composite
def page_strategy(draw: st.DrawFn, page_id: str | None = None) -> Page:
    """Generate a random Page object."""
    if page_id is None:
        page_id = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Nd",))))
    
    title = draw(st.text(min_size=1, max_size=50))
    space_key = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))))
    content = draw(st.text(min_size=10, max_size=500))
    author = draw(st.text(min_size=1, max_size=20))
    
    # Generate naive timestamps first (hypothesis requirement)
    created_date_naive = draw(st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 1, 1)
    ))
    # Add timezone
    created_date = created_date_naive.replace(tzinfo=timezone.utc)
    
    modified_date_naive = draw(st.datetimes(
        min_value=created_date_naive,
        max_value=datetime(2024, 12, 1)
    ))
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
    stored_modified = draw(st.datetimes(
        min_value=page.created_date,
        max_value=page.modified_date - timedelta(seconds=1)
    ))
    
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
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
    def test_update_removes_old_embeddings(
        self, page: Page
    ) -> None:
        """Test that updating a page removes old embeddings."""
        # Create temporary directory for this test
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create vector store
            vector_store = ChromaStore(
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
                collection_name="test_update",
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
            embedder = EmbeddingGenerator()
            
            old_chunks = chunker.chunk_document(old_page)
            if old_chunks:
                old_embeddings = embedder.generate_batch_embeddings(
                    [chunk.content for chunk in old_chunks]
                )
                vector_store.add_documents(old_chunks, old_embeddings)
        
            # Now update with new version
            vector_store.delete_by_page_id(page.id)
            new_chunks = chunker.chunk_document(page)
            if new_chunks:
                new_embeddings = embedder.generate_batch_embeddings(
                    [chunk.content for chunk in new_chunks]
                )
                vector_store.add_documents(new_chunks, new_embeddings)
        
            # Verify only new version exists
            metadata = vector_store.get_document_metadata(page.id)
            if metadata:
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
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
    def test_new_page_creates_embeddings(
        self, page: Page
    ) -> None:
        """Test that new pages result in embeddings being created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create vector store
            vector_store = ChromaStore(
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
                collection_name="test_new_page",
            )
            
            # Verify page doesn't exist
            metadata_before = vector_store.get_document_metadata(page.id)
            assert metadata_before is None, "Page should not exist before processing"
            
            # Process new page
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            embedder = EmbeddingGenerator()
            
            chunks = chunker.chunk_document(page)
            if chunks:
                embeddings = embedder.generate_batch_embeddings(
                    [chunk.content for chunk in chunks]
                )
                vector_store.add_documents(chunks, embeddings)
                
                # Verify page now exists
                metadata_after = vector_store.get_document_metadata(page.id)
                assert metadata_after is not None, (
                    "Embeddings should be created for new page"
                )
                assert metadata_after["page_id"] == page.id, (
                    "Stored page_id should match"
                )


class TestDeletionCompleteness:
    """Test Property 16: Deletion completeness.
    
    **Feature: confluence-rag-system, Property 16: Deletion completeness**
    **Validates: Requirements 4.4**
    
    For any page_id that is deleted, after synchronization, no embeddings
    with that page_id should exist in the vector database.
    """

    @given(page=page_strategy())
    @settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
    def test_deletion_removes_all_embeddings(
        self, page: Page
    ) -> None:
        """Test that deleting a page removes all its embeddings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create vector store
            vector_store = ChromaStore(
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
                collection_name="test_deletion",
            )
            
            # Add page to vector store
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            embedder = EmbeddingGenerator()
            
            chunks = chunker.chunk_document(page)
            if chunks:
                embeddings = embedder.generate_batch_embeddings(
                    [chunk.content for chunk in chunks]
                )
                vector_store.add_documents(chunks, embeddings)
                
                # Verify page exists
                metadata_before = vector_store.get_document_metadata(page.id)
                assert metadata_before is not None, "Page should exist before deletion"
                
                # Delete page
                vector_store.delete_by_page_id(page.id)
                
                # Verify page no longer exists
                metadata_after = vector_store.get_document_metadata(page.id)
                assert metadata_after is None, (
                    f"No embeddings should exist after deletion for page_id {page.id}"
                )

    @given(
        pages=st.lists(
            page_strategy(),
            min_size=2,
            max_size=5,
            unique_by=lambda p: p.id
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=5000)
    def test_deletion_only_removes_target_page(
        self, pages: list[Page]
    ) -> None:
        """Test that deleting one page doesn't affect other pages."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create vector store
            vector_store = ChromaStore(
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
                collection_name="test_selective_deletion",
            )
            
            # Add all pages to vector store
            chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
            embedder = EmbeddingGenerator()
            
            for page in pages:
                chunks = chunker.chunk_document(page)
                if chunks:
                    embeddings = embedder.generate_batch_embeddings(
                        [chunk.content for chunk in chunks]
                    )
                    vector_store.add_documents(chunks, embeddings)
            
            # Delete first page
            page_to_delete = pages[0]
            vector_store.delete_by_page_id(page_to_delete.id)
            
            # Verify deleted page is gone
            metadata_deleted = vector_store.get_document_metadata(page_to_delete.id)
            assert metadata_deleted is None, (
                f"Deleted page {page_to_delete.id} should not exist"
            )
            
            # Verify other pages still exist
            for page in pages[1:]:
                metadata = vector_store.get_document_metadata(page.id)
                # Only check if the page had chunks (some might not due to content)
                if metadata is not None:
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

    @given(space_key=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))))
    @settings(max_examples=20, deadline=5000)
    def test_sync_timestamp_increases(self, space_key: str) -> None:
        """Test that sync timestamp increases after successful sync."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create vector store and timestamp tracker
            vector_store = ChromaStore(
                persist_directory=str(Path(tmp_dir) / "chroma_test"),
                collection_name="test_timestamp",
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
