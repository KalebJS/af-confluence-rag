"""Ingestion service for orchestrating the full ingestion pipeline."""

from datetime import datetime

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.models.page import Page
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import VectorStoreInterface
from src.sync.sync_coordinator import SyncCoordinator

log = structlog.stdlib.get_logger()


class IngestionService:
    """Orchestrates the full ingestion pipeline.

    This service wires together all components needed for ingesting Confluence
    content into the vector store: client, chunker, embedder, and storage.
    """

    def __init__(
        self,
        confluence_client: ConfluenceClient,
        chunker: DocumentChunker,
        embedder: EmbeddingGenerator,
        vector_store: VectorStoreInterface,
    ):
        """
        Initialize the ingestion service.

        Args:
            confluence_client: Client for Confluence API
            chunker: Document chunker for text processing
            embedder: Embedding generator for vectorization
            vector_store: Vector store for document storage
        """
        self._confluence_client: ConfluenceClient = confluence_client
        self._chunker: DocumentChunker = chunker
        self._embedder: EmbeddingGenerator = embedder
        self._vector_store: VectorStoreInterface = vector_store

        # Create sync coordinator for incremental updates
        self._sync_coordinator: SyncCoordinator = SyncCoordinator(
            confluence_client=confluence_client,
            vector_store=vector_store,
            chunker=chunker,
            embedder=embedder,
        )

        log.info("ingestion_service_initialized")

    def ingest_space(
        self, space_key: str, incremental: bool = True
    ) -> dict[str, int | float | bool]:
        """
        Ingest all pages from a Confluence space.

        This method performs either a full ingestion or an incremental sync
        based on the incremental parameter. It includes comprehensive error
        handling for database unavailability and invalid content.

        Args:
            space_key: Confluence space key to ingest
            incremental: If True, perform incremental sync. If False, full re-ingestion.

        Returns:
            Dictionary with ingestion statistics:
                - pages_processed: Number of pages processed
                - chunks_created: Number of chunks created
                - duration_seconds: Time taken for ingestion
                - success: Whether ingestion completed successfully
                - errors: List of error messages (if any)

        Raises:
            RuntimeError: If ingestion fails critically
        """
        start_time = datetime.now()
        log.info(
            "ingest_space_started",
            space_key=space_key,
            incremental=incremental,
            start_time=start_time,
        )

        try:
            # Check database availability before starting
            self._check_database_availability()

            if incremental:
                # Use sync coordinator for incremental updates
                sync_report = self._sync_coordinator.sync_space(space_key)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                result = {
                    "pages_processed": sync_report.pages_added + sync_report.pages_updated,
                    "pages_added": sync_report.pages_added,
                    "pages_updated": sync_report.pages_updated,
                    "pages_deleted": sync_report.pages_deleted,
                    "chunks_created": 0,  # Not tracked in sync report
                    "duration_seconds": duration,
                    "success": sync_report.success,
                    "errors": sync_report.errors,
                }

                log.info(
                    "ingest_space_completed_incremental",
                    space_key=space_key,
                    **result,
                )

                return result
            else:
                # Full re-ingestion
                return self._full_ingest_space(space_key, start_time)

        except RuntimeError as e:
            # Database unavailability or critical errors
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = f"Database unavailable or critical error: {str(e)}"
            log.error(
                "ingest_space_failed_database_unavailable",
                space_key=space_key,
                error=str(e),
                duration_seconds=duration,
            )

            return {
                "pages_processed": 0,
                "chunks_created": 0,
                "duration_seconds": duration,
                "success": False,
                "errors": [error_msg],
            }
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = f"Space ingestion failed: {str(e)}"
            log.error(
                "ingest_space_failed",
                space_key=space_key,
                error=str(e),
                duration_seconds=duration,
            )

            return {
                "pages_processed": 0,
                "chunks_created": 0,
                "duration_seconds": duration,
                "success": False,
                "errors": [error_msg],
            }

    def _full_ingest_space(
        self, space_key: str, start_time: datetime
    ) -> dict[str, int | float | bool]:
        """
        Perform full re-ingestion of a space.

        Args:
            space_key: Confluence space key to ingest
            start_time: Start time of ingestion

        Returns:
            Dictionary with ingestion statistics
        """
        pages_processed = 0
        chunks_created = 0
        errors: list[str] = []

        try:
            # Retrieve all pages from Confluence
            log.info("retrieving_all_pages", space_key=space_key)

            for page in self._confluence_client.get_space_pages(space_key):
                try:
                    # Process and store the page
                    chunks_count = self._process_and_store_page(page)
                    pages_processed += 1
                    chunks_created += chunks_count

                    # Log progress every 10 pages
                    if pages_processed % 10 == 0:
                        log.info(
                            "ingestion_progress",
                            space_key=space_key,
                            pages_processed=pages_processed,
                            chunks_created=chunks_created,
                        )

                except Exception as e:
                    error_msg = f"Failed to process page {page.id} ({page.title}): {str(e)}"
                    errors.append(error_msg)
                    log.error(
                        "page_processing_failed",
                        page_id=page.id,
                        page_title=page.title,
                        error=str(e),
                    )
                    # Continue processing other pages

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Calculate summary statistics
            avg_chunks_per_page = chunks_created / pages_processed if pages_processed > 0 else 0
            pages_per_second = pages_processed / duration if duration > 0 else 0

            result = {
                "pages_processed": pages_processed,
                "chunks_created": chunks_created,
                "duration_seconds": duration,
                "success": len(errors) == 0,
                "errors": errors,
                "avg_chunks_per_page": round(avg_chunks_per_page, 2),
                "pages_per_second": round(pages_per_second, 2),
            }

            log.info(
                "ingest_space_completed_full",
                space_key=space_key,
                pages_processed=pages_processed,
                chunks_created=chunks_created,
                duration_seconds=duration,
                avg_chunks_per_page=round(avg_chunks_per_page, 2),
                pages_per_second=round(pages_per_second, 2),
                error_count=len(errors),
                success=len(errors) == 0,
            )

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            error_msg = f"Full space ingestion failed: {str(e)}"
            errors.append(error_msg)

            log.error(
                "full_ingest_space_failed",
                space_key=space_key,
                error=str(e),
                duration_seconds=duration,
            )

            return {
                "pages_processed": pages_processed,
                "chunks_created": chunks_created,
                "duration_seconds": duration,
                "success": False,
                "errors": errors,
            }

    def ingest_page(self, page_id: str) -> dict[str, int | bool]:
        """
        Ingest a single Confluence page.

        This method is useful for testing or processing individual pages.

        Args:
            page_id: Confluence page ID to ingest

        Returns:
            Dictionary with ingestion statistics:
                - chunks_created: Number of chunks created
                - success: Whether ingestion completed successfully
                - error: Error message (if any)

        Raises:
            RuntimeError: If page retrieval fails
        """
        log.info("ingest_page_started", page_id=page_id)

        try:
            # Retrieve the page
            page = self._confluence_client.get_page_content(page_id)

            # Delete existing chunks for this page (if any)
            try:
                self._vector_store.delete_by_page_id(page_id)
                log.info("deleted_existing_chunks", page_id=page_id)
            except Exception as e:
                log.warning(
                    "failed_to_delete_existing_chunks",
                    page_id=page_id,
                    error=str(e),
                )

            # Process and store the page
            chunks_count = self._process_and_store_page(page)

            result = {
                "chunks_created": chunks_count,
                "success": True,
                "error": None,
            }

            log.info(
                "ingest_page_completed",
                page_id=page_id,
                page_title=page.title,
                chunks_created=chunks_count,
            )

            return result

        except Exception as e:
            error_msg = f"Page ingestion failed: {str(e)}"
            log.error("ingest_page_failed", page_id=page_id, error=str(e))

            return {
                "chunks_created": 0,
                "success": False,
                "error": error_msg,
            }

    def _check_database_availability(self) -> None:
        """
        Check if the vector database is available.

        This method attempts a simple operation to verify database connectivity.

        Raises:
            RuntimeError: If database is unavailable
        """
        try:
            # Try to get metadata for a non-existent page as a health check
            # This should return None without raising an exception
            self._vector_store.get_document_metadata("__health_check__")
            log.debug("database_availability_check_passed")
        except Exception as e:
            log.error(
                "database_unavailable",
                error=str(e),
            )
            raise RuntimeError(
                f"Vector database is unavailable: {str(e)}. "
                "Please check database connection and try again."
            ) from e

    def _process_and_store_page(self, page: Page) -> int:
        """
        Process a page and store it in the vector store.

        This method includes error recovery for invalid content by continuing
        processing even if individual pages fail.

        Args:
            page: Page to process and store

        Returns:
            Number of chunks created

        Raises:
            Exception: If processing or storage fails
        """
        log.info(
            "processing_page",
            page_id=page.id,
            page_title=page.title,
            content_length=len(page.content),
        )

        try:
            # Validate page content
            if not page.content or not page.content.strip():
                log.warning(
                    "empty_page_content",
                    page_id=page.id,
                    page_title=page.title,
                )
                return 0

            # Chunk the document
            chunks = self._chunker.chunk_document(page)

            if not chunks:
                log.warning(
                    "no_chunks_generated",
                    page_id=page.id,
                    page_title=page.title,
                )
                return 0

            # Generate embeddings
            texts = [chunk.content for chunk in chunks]
            embeddings = self._embedder.generate_batch_embeddings(texts)

            # Validate embeddings were generated
            if len(embeddings) != len(chunks):
                log.error(
                    "embedding_count_mismatch",
                    page_id=page.id,
                    page_title=page.title,
                    chunks=len(chunks),
                    embeddings=len(embeddings),
                )
                raise ValueError(
                    f"Embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
                )

            # Store in vector database
            self._vector_store.add_documents(chunks, embeddings)

            log.info(
                "page_processed_and_stored",
                page_id=page.id,
                page_title=page.title,
                chunk_count=len(chunks),
            )

            return len(chunks)

        except ValueError as e:
            # Invalid content errors - log and re-raise
            log.error(
                "invalid_page_content",
                page_id=page.id,
                page_title=page.title,
                error=str(e),
            )
            raise
        except Exception as e:
            # Unexpected errors - log with full context
            log.error(
                "page_processing_error",
                page_id=page.id,
                page_title=page.title,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
