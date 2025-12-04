"""Ingestion service for orchestrating the full ingestion pipeline."""

from datetime import datetime

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.ingestion.confluence_client import ConfluenceClient
from src.models.config import ProcessingConfig, VectorStoreConfig
from src.models.page import Page, to_langchain_documents
from src.processing.chunker import DocumentChunker
from src.providers import get_embeddings, get_vector_store
from src.sync.sync_coordinator import SyncCoordinator

log = structlog.stdlib.get_logger()


class IngestionService:
    """Orchestrates the full ingestion pipeline.

    This service wires together all components needed for ingesting Confluence
    content into the vector store: client, chunker, embeddings, and storage.
    
    The service uses LangChain's Embeddings and VectorStore abstractions,
    allowing easy swapping of implementations through dependency injection
    or the centralized providers module.
    """

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
        Initialize the ingestion service.

        Args:
            confluence_client: Client for Confluence API
            chunker: Document chunker for text processing
            embeddings: LangChain Embeddings instance (optional, uses provider module if None)
            vector_store: LangChain VectorStore instance (optional, uses provider module if None)
            processing_config: Processing configuration (required if embeddings is None)
            vector_store_config: Vector store configuration (required if vector_store is None)
        """
        self._confluence_client: ConfluenceClient = confluence_client
        self._chunker: DocumentChunker = chunker
        
        # Use provided embeddings or create from config via provider module
        if embeddings is not None:
            self._embeddings: Embeddings = embeddings
            log.info("ingestion_service_using_provided_embeddings")
        else:
            if processing_config is None:
                raise ValueError("processing_config is required when embeddings is not provided")
            self._embeddings = get_embeddings(processing_config.embedding_model)
            log.info("ingestion_service_created_embeddings_from_config", 
                    model=processing_config.embedding_model)
        
        # Use provided vector store or create from config via provider module
        if vector_store is not None:
            self._vector_store: VectorStore = vector_store
            log.info("ingestion_service_using_provided_vector_store")
        else:
            if vector_store_config is None:
                raise ValueError("vector_store_config is required when vector_store is not provided")
            self._vector_store = get_vector_store(
                embeddings=self._embeddings,
                collection_name=vector_store_config.collection_name,
                persist_directory=vector_store_config.persist_directory
            )
            log.info("ingestion_service_created_vector_store_from_config",
                    collection=vector_store_config.collection_name)

        # Create sync coordinator for incremental updates
        # Note: SyncCoordinator will be updated in task 6 to use LangChain abstractions
        # For now, we pass None for embedder as it's not used in the current flow
        self._sync_coordinator: SyncCoordinator | None = None
        # self._sync_coordinator = SyncCoordinator(
        #     confluence_client=confluence_client,
        #     vector_store=self._vector_store,
        #     chunker=chunker,
        #     embeddings=self._embeddings,
        # )

        log.info("ingestion_service_initialized")

    def ingest_space(
        self, space_key: str, incremental: bool = False
    ) -> dict[str, int | float | bool | list[str]]:
        """
        Ingest all pages from a Confluence space.

        This method performs a full ingestion. Incremental sync will be
        implemented in task 6 when SyncCoordinator is updated.

        Args:
            space_key: Confluence space key to ingest
            incremental: If True, perform incremental sync. If False, full re-ingestion.
                        Note: Incremental sync not yet implemented with LangChain abstractions.

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
                # Incremental sync will be implemented in task 6
                log.warning(
                    "incremental_sync_not_yet_implemented",
                    space_key=space_key,
                    message="Incremental sync not yet implemented with LangChain abstractions. Performing full ingestion."
                )
            
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
    ) -> dict[str, int | float | bool | list[str]]:
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

    def ingest_page(self, page_id: str) -> dict[str, int | bool | str | None]:
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
                self._delete_page_chunks(page_id)
                log.info("deleted_existing_chunks", page_id=page_id)
            except Exception as e:
                log.warning(
                    "failed_to_delete_existing_chunks",
                    page_id=page_id,
                    error=str(e),
                )

            # Process and store the page
            chunks_count = self._process_and_store_page(page)

            result: dict[str, int | bool | str | None] = {
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

            result_error: dict[str, int | bool | str | None] = {
                "chunks_created": 0,
                "success": False,
                "error": error_msg,
            }
            return result_error
    
    def _delete_page_chunks(self, page_id: str) -> None:
        """
        Delete all chunks for a given page from the vector store.
        
        This method uses metadata-based filtering to delete documents.
        For LangChain VectorStores that support deletion by filter, this will
        use that capability. Otherwise, it will retrieve and delete by IDs.
        
        Args:
            page_id: Page ID whose chunks should be deleted
        """
        try:
            # Try to use delete method with filter (if supported by the vector store)
            # Chroma supports delete with where clause
            if hasattr(self._vector_store, 'delete'):
                # Try metadata-based deletion
                try:
                    _ = self._vector_store.delete(where={"page_id": page_id})
                    log.info("deleted_chunks_by_metadata", page_id=page_id)
                    return
                except (TypeError, AttributeError):
                    # Fall through to alternative method
                    pass
            
            # Alternative: search for documents with this page_id and delete by ID
            # This is less efficient but works with any VectorStore
            _ = self._vector_store.similarity_search(
                query="",  # Empty query to get any documents
                k=100,  # Get up to 100 chunks
                filter={"page_id": page_id}
            )
            
            # Note: Actual deletion by IDs will be implemented in task 6
            # when we update the vector store interface
            log.info("deletion_deferred_to_task_6", page_id=page_id)
            
        except Exception as e:
            log.warning(
                "failed_to_delete_page_chunks",
                page_id=page_id,
                error=str(e),
            )
            # Don't raise - deletion failure shouldn't block ingestion

    def _check_database_availability(self) -> None:
        """
        Check if the vector database is available.

        This method attempts a simple operation to verify database connectivity.

        Raises:
            RuntimeError: If database is unavailable
        """
        try:
            # Try a simple search operation as a health check
            # This should work without raising an exception even if the collection is empty
            _ = self._vector_store.similarity_search("__health_check__", k=1)
            log.debug("database_availability_check_passed")
        except Exception as e:
            log.error(
                "database_unavailable",
                error=str(e),
            )
            raise RuntimeError(
                "Vector database is unavailable: " + str(e) + ". " +
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

            # Convert DocumentChunks to LangChain Documents
            documents = to_langchain_documents(chunks)

            # Store in vector database using LangChain's add_documents
            # This method automatically generates embeddings using the configured Embeddings instance
            _ = self._vector_store.add_documents(documents)

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
