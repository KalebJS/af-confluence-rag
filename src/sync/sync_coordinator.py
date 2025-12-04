"""Synchronization coordinator for orchestrating incremental updates."""

from datetime import datetime
from typing import Any

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.ingestion.confluence_client import ConfluenceClient
from src.models.config import ProcessingConfig, VectorStoreConfig
from src.models.page import Page, SyncState, to_langchain_documents
from src.processing.chunker import DocumentChunker
from src.providers import get_embeddings, get_vector_store
from src.sync.change_detector import ChangeDetector
from src.sync.models import ChangeSet, SyncReport
from src.sync.timestamp_tracker import TimestampTracker

log = structlog.stdlib.get_logger()


class SyncCoordinator:
    """Orchestrates synchronization between Confluence and vector store."""

    def __init__(
        self,
        confluence_client: ConfluenceClient,
        chunker: DocumentChunker,
        embeddings: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        processing_config: ProcessingConfig | None = None,
        vector_store_config: VectorStoreConfig | None = None,
    ):
        """
        Initialize sync coordinator.

        Args:
            confluence_client: Client for Confluence API
            chunker: Document chunker for text processing
            embeddings: Optional embeddings instance (uses provider module if None)
            vector_store: Optional vector store instance (uses provider module if None)
            processing_config: Optional processing config (required if embeddings is None)
            vector_store_config: Optional vector store config (required if vector_store is None)
        """
        self._confluence_client: ConfluenceClient = confluence_client
        self._chunker: DocumentChunker = chunker
        
        # Use provided instances or create from config via provider module
        if embeddings is not None:
            self._embeddings: Embeddings = embeddings
        else:
            if processing_config is None:
                raise ValueError("processing_config is required when embeddings is not provided")
            self._embeddings = get_embeddings(processing_config.embedding_model)
        
        if vector_store is not None:
            self._vector_store: VectorStore = vector_store
        else:
            if vector_store_config is None:
                raise ValueError("vector_store_config is required when vector_store is not provided")
            self._vector_store = get_vector_store(
                self._embeddings,
                vector_store_config.collection_name,
                vector_store_config.persist_directory,
            )
        
        self._change_detector: ChangeDetector = ChangeDetector()
        self._timestamp_tracker: TimestampTracker = TimestampTracker(self._vector_store)

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
            changes = self.detect_changes(space_key, confluence_pages, last_sync_timestamp)

            log.info(
                "changes_detected",
                space_key=space_key,
                new_pages=len(changes.new_pages),
                modified_pages=len(changes.modified_pages),
                deleted_pages=len(changes.deleted_page_ids),
            )

            # Apply changes
            pages_added, pages_updated, pages_deleted, apply_errors = self.apply_changes(changes)

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
                self._delete_by_page_id(page_id)
                pages_deleted += 1
            except Exception as e:
                error_msg = f"Failed to delete page {page_id}: {str(e)}"
                errors.append(error_msg)
                log.error("failed_to_delete_page", page_id=page_id, error=str(e))

        # Process modified pages (delete old, add new)
        for page in changes.modified_pages:
            try:
                # Delete old version
                self._delete_by_page_id(page.id)

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
                # Use similarity_search with metadata filter to check if page exists
                # Search for documents with this page_id
                results = self._vector_store.similarity_search(
                    query="",  # Empty query, we only care about metadata filter
                    k=1,
                    filter={"page_id": page.id}
                )
                
                if results:
                    # Extract metadata from first result
                    metadata_dict: dict[str, Any] = dict(results[0].metadata)
                    stored_metadata[page.id] = metadata_dict
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

        # Convert to LangChain Documents
        langchain_docs = to_langchain_documents(chunks)

        # Store in vector database (embeddings are generated automatically)
        self._vector_store.add_documents(langchain_docs)

        log.info(
            "page_processed_and_stored",
            page_id=page.id,
            page_title=page.title,
            chunk_count=len(chunks),
        )
    
    def _delete_by_page_id(self, page_id: str) -> None:
        """
        Delete all chunks associated with a page using metadata-based filtering.
        
        This method handles deletion using LangChain's VectorStore interface.
        For Chroma, we use the get() method to retrieve document IDs by metadata filter,
        then delete those IDs.
        
        Args:
            page_id: Unique identifier of the page to delete
            
        Raises:
            RuntimeError: If deletion operation fails
        """
        try:
            # For Chroma, we need to use the underlying collection's get method
            # to retrieve document IDs by metadata filter
            if hasattr(self._vector_store, '_collection'):
                # Access Chroma collection directly
                collection = self._vector_store._collection
                
                # Get documents with this page_id using metadata filter
                results = collection.get(
                    where={"page_id": page_id},
                    include=[]  # We only need IDs, not embeddings or documents
                )
                
                ids_to_delete = results.get('ids', [])
                
                if ids_to_delete:
                    # Delete using the collection's delete method
                    collection.delete(ids=ids_to_delete)
                    log.info(
                        "page_deleted_from_vector_store",
                        page_id=page_id,
                        chunks_deleted=len(ids_to_delete)
                    )
                else:
                    log.info("page_not_found_for_deletion", page_id=page_id)
            else:
                # Fallback for other vector stores: try similarity_search + delete
                results = self._vector_store.similarity_search(
                    query="",  # Empty query, we only care about metadata filter
                    k=1000,  # Large number to get all chunks
                    filter={"page_id": page_id}
                )
                
                if not results:
                    log.info("page_not_found_for_deletion", page_id=page_id)
                    return
                
                # Extract IDs from results - try chunk_id from metadata
                ids_to_delete: list[str] = []
                for doc in results:
                    metadata_dict: dict[str, Any] = dict(doc.metadata)
                    doc_id = metadata_dict.get("chunk_id")
                    if doc_id and isinstance(doc_id, str):
                        ids_to_delete.append(doc_id)
                
                if ids_to_delete:
                    try:
                        _ = self._vector_store.delete(ids_to_delete)
                        log.info(
                            "page_deleted_from_vector_store",
                            page_id=page_id,
                            chunks_deleted=len(ids_to_delete)
                        )
                    except (AttributeError, NotImplementedError) as e:
                        log.warning(
                            "vector_store_delete_not_supported",
                            page_id=page_id,
                            error=str(e),
                            message="Vector store does not support deletion. Documents may remain in store."
                        )
                else:
                    log.warning(
                        "no_document_ids_found_for_deletion",
                        page_id=page_id,
                        results_count=len(results)
                    )
                
        except Exception as e:
            log.error("delete_by_page_id_failed", page_id=page_id, error=str(e))
            raise RuntimeError(f"Failed to delete page from vector store: {e}") from e

    def _count_total_chunks(self, space_key: str) -> int:
        """
        Count total chunks in vector store for a space.

        This is a simplified implementation. In production, you might want
        to query the vector store for an accurate count.

        Args:
            space_key: Confluence space key

        Returns:
            Estimated chunk count
        """
        _ = space_key  # Unused parameter, kept for interface compatibility
        # For now, return 0 as we don't have a direct way to count
        # This could be improved by adding a count method to VectorStore
        return 0
