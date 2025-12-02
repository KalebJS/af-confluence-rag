"""Synchronization coordinator for orchestrating incremental updates."""

from datetime import datetime
from typing import Any

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.models.page import Page, SyncState
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import VectorStoreInterface
from src.sync.change_detector import ChangeDetector
from src.sync.models import ChangeSet, SyncReport
from src.sync.timestamp_tracker import TimestampTracker

log = structlog.stdlib.get_logger()


class SyncCoordinator:
    """Orchestrates synchronization between Confluence and vector store."""

    def __init__(
        self,
        confluence_client: ConfluenceClient,
        vector_store: VectorStoreInterface,
        chunker: DocumentChunker,
        embedder: EmbeddingGenerator,
    ):
        """
        Initialize sync coordinator.

        Args:
            confluence_client: Client for Confluence API
            vector_store: Vector store for document storage
            chunker: Document chunker for text processing
            embedder: Embedding generator for vectorization
        """
        self._confluence_client: ConfluenceClient = confluence_client
        self._vector_store: VectorStoreInterface = vector_store
        self._chunker: DocumentChunker = chunker
        self._embedder: EmbeddingGenerator = embedder
        self._change_detector: ChangeDetector = ChangeDetector()
        self._timestamp_tracker: TimestampTracker = TimestampTracker(vector_store)

        log.info("sync_coordinator_initialized")

    def sync_space(self, space_key: str) -> SyncReport:
        """
        Perform full space synchronization with incremental updates.

        This method:
        1. Retrieves all pages from Confluence
        2. Detects changes (new, modified, deleted)
        3. Applies changes to vector store
        4. Updates sync state

        Args:
            space_key: Confluence space key to synchronize

        Returns:
            SyncReport with synchronization results

        Raises:
            RuntimeError: If synchronization fails
        """
        start_time = datetime.now()
        log.info("sync_space_started", space_key=space_key, start_time=start_time)

        errors: list[str] = []

        try:
            # Load last sync state
            sync_state = self._timestamp_tracker.load_sync_state(space_key)
            last_sync_timestamp = sync_state.last_sync_timestamp if sync_state else None

            log.info(
                "loaded_sync_state",
                space_key=space_key,
                last_sync_timestamp=last_sync_timestamp,
            )

            # Retrieve all pages from Confluence
            confluence_pages = list(self._confluence_client.get_space_pages(space_key))

            log.info(
                "retrieved_confluence_pages",
                space_key=space_key,
                page_count=len(confluence_pages),
            )

            # Detect changes
            changes = self.detect_changes(
                space_key, confluence_pages, last_sync_timestamp
            )

            log.info(
                "changes_detected",
                space_key=space_key,
                new_pages=len(changes.new_pages),
                modified_pages=len(changes.modified_pages),
                deleted_pages=len(changes.deleted_page_ids),
            )

            # Apply changes
            pages_added, pages_updated, pages_deleted, apply_errors = self.apply_changes(
                changes
            )

            errors.extend(apply_errors)

            # Update sync state
            total_chunks = self._count_total_chunks(space_key)
            new_sync_state = SyncState(
                space_key=space_key,
                last_sync_timestamp=datetime.now(),
                page_count=len(confluence_pages),
                chunk_count=total_chunks,
            )

            self._timestamp_tracker.save_sync_state(new_sync_state)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            sync_report = SyncReport(
                space_key=space_key,
                pages_added=pages_added,
                pages_updated=pages_updated,
                pages_deleted=pages_deleted,
                duration_seconds=duration,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
            )

            log.info(
                "sync_space_completed",
                space_key=space_key,
                pages_added=pages_added,
                pages_updated=pages_updated,
                pages_deleted=pages_deleted,
                duration_seconds=duration,
                success=sync_report.success,
            )

            return sync_report

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = f"Sync failed: {str(e)}"
            errors.append(error_msg)

            log.error(
                "sync_space_failed",
                space_key=space_key,
                error=str(e),
                duration_seconds=duration,
            )

            return SyncReport(
                space_key=space_key,
                pages_added=0,
                pages_updated=0,
                pages_deleted=0,
                duration_seconds=duration,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
            )

    def detect_changes(
        self,
        space_key: str,
        confluence_pages: list[Page],
        last_sync_timestamp: datetime | None = None,
    ) -> ChangeSet:
        """
        Detect changes between Confluence and vector store.

        Args:
            space_key: Confluence space key
            confluence_pages: Current pages from Confluence
            last_sync_timestamp: Timestamp of last successful sync

        Returns:
            ChangeSet containing detected changes
        """
        log.info(
            "detecting_changes",
            space_key=space_key,
            confluence_page_count=len(confluence_pages),
            last_sync_timestamp=last_sync_timestamp,
        )

        # Get stored page metadata
        stored_page_metadata = self._get_stored_page_metadata(confluence_pages)

        # Detect changes
        changes = self._change_detector.detect_changes(
            confluence_pages, stored_page_metadata, last_sync_timestamp
        )

        return changes

    def apply_changes(self, changes: ChangeSet) -> tuple[int, int, int, list[str]]:
        """
        Apply detected changes to vector store.

        Args:
            changes: ChangeSet containing changes to apply

        Returns:
            Tuple of (pages_added, pages_updated, pages_deleted, errors)
        """
        log.info(
            "applying_changes",
            new_pages=len(changes.new_pages),
            modified_pages=len(changes.modified_pages),
            deleted_pages=len(changes.deleted_page_ids),
        )

        errors: list[str] = []
        pages_added = 0
        pages_updated = 0
        pages_deleted = 0

        # Process deleted pages
        for page_id in changes.deleted_page_ids:
            try:
                self._vector_store.delete_by_page_id(page_id)
                pages_deleted += 1
            except Exception as e:
                error_msg = f"Failed to delete page {page_id}: {str(e)}"
                errors.append(error_msg)
                log.error("failed_to_delete_page", page_id=page_id, error=str(e))

        # Process modified pages (delete old, add new)
        for page in changes.modified_pages:
            try:
                # Delete old version
                self._vector_store.delete_by_page_id(page.id)

                # Add new version
                self._process_and_store_page(page)
                pages_updated += 1
            except Exception as e:
                error_msg = f"Failed to update page {page.id} ({page.title}): {str(e)}"
                errors.append(error_msg)
                log.error(
                    "failed_to_update_page",
                    page_id=page.id,
                    page_title=page.title,
                    error=str(e),
                )

        # Process new pages
        for page in changes.new_pages:
            try:
                self._process_and_store_page(page)
                pages_added += 1
            except Exception as e:
                error_msg = f"Failed to add page {page.id} ({page.title}): {str(e)}"
                errors.append(error_msg)
                log.error(
                    "failed_to_add_page",
                    page_id=page.id,
                    page_title=page.title,
                    error=str(e),
                )

        log.info(
            "changes_applied",
            pages_added=pages_added,
            pages_updated=pages_updated,
            pages_deleted=pages_deleted,
            errors=len(errors),
        )

        return pages_added, pages_updated, pages_deleted, errors

    def _get_stored_page_metadata(self, confluence_pages: list[Page]) -> dict[str, dict[str, Any]]:
        """
        Get metadata for pages currently stored in vector store.

        Args:
            confluence_pages: Pages from Confluence to check

        Returns:
            Dictionary mapping page_id to metadata
        """
        stored_metadata: dict[str, dict[str, Any]] = {}

        for page in confluence_pages:
            try:
                metadata = self._vector_store.get_document_metadata(page.id)
                if metadata:
                    stored_metadata[page.id] = metadata
            except Exception as e:
                log.warning(
                    "failed_to_get_stored_metadata",
                    page_id=page.id,
                    error=str(e),
                )

        return stored_metadata

    def _process_and_store_page(self, page: Page) -> None:
        """
        Process a page and store it in the vector store.

        Args:
            page: Page to process and store

        Raises:
            Exception: If processing or storage fails
        """
        # Chunk the document
        chunks = self._chunker.chunk_document(page)

        if not chunks:
            log.warning("no_chunks_generated", page_id=page.id, page_title=page.title)
            return

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedder.generate_batch_embeddings(texts)

        # Store in vector database
        self._vector_store.add_documents(chunks, embeddings)

        log.info(
            "page_processed_and_stored",
            page_id=page.id,
            page_title=page.title,
            chunk_count=len(chunks),
        )

    def _count_total_chunks(self, space_key: str) -> int:  # noqa: ARG002
        """
        Count total chunks in vector store for a space.

        This is a simplified implementation. In production, you might want
        to query the vector store for an accurate count.

        Args:
            space_key: Confluence space key

        Returns:
            Estimated chunk count
        """
        # For now, return 0 as we don't have a direct way to count
        # This could be improved by adding a count method to VectorStoreInterface
        return 0
