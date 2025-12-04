"""Query processing functionality for semantic search."""

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.models.config import ProcessingConfig, VectorStoreConfig
from src.models.page import SearchResult, from_langchain_document
from src.providers import get_embeddings, get_vector_store

log = structlog.stdlib.get_logger()


class QueryProcessor:
    """Handles query processing and semantic search operations.

    This class coordinates query embedding generation and vector similarity
    search to retrieve relevant documents from the vector store.
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        vector_store: VectorStore | None = None,
        processing_config: ProcessingConfig | None = None,
        vector_store_config: VectorStoreConfig | None = None,
    ) -> None:
        """Initialize the query processor.

        Args:
            embeddings: Optional Embeddings instance. If not provided, will be created
                from processing_config using the provider module.
            vector_store: Optional VectorStore instance. If not provided, will be created
                from vector_store_config using the provider module.
            processing_config: Configuration for creating embeddings (required if embeddings not provided)
            vector_store_config: Configuration for creating vector store (required if vector_store not provided)

        Raises:
            ValueError: If neither embeddings nor processing_config is provided, or
                if neither vector_store nor vector_store_config is provided
        """
        # Handle embeddings - either use provided instance or create from config
        if embeddings is not None:
            self.embeddings = embeddings
            log.info("query_processor_using_provided_embeddings")
        elif processing_config is not None:
            self.embeddings = get_embeddings(processing_config.embedding_model)
            log.info(
                "query_processor_created_embeddings_from_config",
                embedding_model=processing_config.embedding_model,
            )
        else:
            raise ValueError(
                "Either embeddings instance or processing_config must be provided"
            )

        # Handle vector store - either use provided instance or create from config
        if vector_store is not None:
            self.vector_store = vector_store
            log.info("query_processor_using_provided_vector_store")
        elif vector_store_config is not None:
            self.vector_store = get_vector_store(
                embeddings=self.embeddings,
                collection_name=vector_store_config.collection_name,
                persist_directory=vector_store_config.persist_directory,
            )
            log.info(
                "query_processor_created_vector_store_from_config",
                collection_name=vector_store_config.collection_name,
            )
        else:
            raise ValueError(
                "Either vector_store instance or vector_store_config must be provided"
            )

        log.info("query_processor_initialized")

    def process_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Process a search query and return relevant results.

        This method:
        1. Uses LangChain's VectorStore to perform similarity search with scores
        2. Converts LangChain Documents to SearchResult objects
        3. Filters and deduplicates results

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return (default: 10)

        Returns:
            List of SearchResult objects ordered by similarity score (descending)

        Raises:
            ValueError: If query is empty or top_k is invalid
            RuntimeError: If search operation fails
        """
        # Validate inputs
        if not query or not query.strip():
            log.warning("empty_query_provided")
            raise ValueError("Query cannot be empty")

        if top_k < 1:
            log.warning("invalid_top_k", top_k=top_k)
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        log.info("processing_query", query_length=len(query), top_k=top_k)

        try:
            # Use LangChain's similarity_search_with_score method
            # This handles embedding generation and search internally
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
            )

            log.debug(
                "vector_search_completed",
                results_count=len(docs_with_scores),
            )

            # Convert LangChain Documents to SearchResult objects
            results = self._convert_to_search_results(docs_with_scores)

            # Filter and deduplicate results
            filtered_results = self._filter_and_deduplicate(results)

            log.info(
                "query_processed_successfully",
                query_length=len(query),
                results_count=len(filtered_results),
                top_k=top_k,
            )

            return filtered_results

        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            log.error(
                "query_processing_failed",
                query_length=len(query),
                top_k=top_k,
                error=str(e),
            )
            raise RuntimeError(f"Failed to process query: {e}") from e

    def _convert_to_search_results(
        self,
        docs_with_scores: list[tuple[any, float]],
    ) -> list[SearchResult]:
        """Convert LangChain Documents with scores to SearchResult objects.

        Args:
            docs_with_scores: List of (Document, score) tuples from vector store

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If document metadata is missing required fields
        """
        results: list[SearchResult] = []

        for doc, score in docs_with_scores:
            try:
                # Extract required fields from metadata
                chunk_id = doc.metadata.get("chunk_id")
                page_id = doc.metadata.get("page_id")
                page_title = doc.metadata.get("page_title")
                page_url = doc.metadata.get("page_url")

                if not all([chunk_id, page_id, page_title, page_url]):
                    log.warning(
                        "document_missing_required_metadata",
                        chunk_id=chunk_id,
                        page_id=page_id,
                        has_title=bool(page_title),
                        has_url=bool(page_url),
                    )
                    continue

                # Create a copy of metadata without the fields we're extracting
                metadata = doc.metadata.copy()
                metadata.pop("chunk_id", None)
                metadata.pop("page_id", None)
                metadata.pop("page_title", None)
                metadata.pop("page_url", None)
                metadata.pop("chunk_index", None)

                result = SearchResult(
                    chunk_id=chunk_id,
                    page_id=page_id,
                    page_title=page_title,
                    page_url=page_url,
                    content=doc.page_content,
                    similarity_score=score,
                    metadata=metadata,
                )
                results.append(result)

            except Exception as e:
                log.warning(
                    "failed_to_convert_document_to_search_result",
                    error=str(e),
                    metadata=doc.metadata,
                )
                continue

        return results

    def _filter_and_deduplicate(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        """Filter and deduplicate search results.

        This method:
        - Removes duplicate results (same page_id)
        - Keeps the highest-scoring chunk from each page
        - Maintains descending similarity score order

        Args:
            results: List of search results to filter

        Returns:
            Filtered and deduplicated list of search results
        """
        if not results:
            return results

        # Track seen page IDs to deduplicate
        seen_pages: set[str] = set()
        deduplicated: list[SearchResult] = []

        # Results are already sorted by similarity score (descending)
        # So we keep the first (highest-scoring) chunk from each page
        for result in results:
            if result.page_id not in seen_pages:
                seen_pages.add(result.page_id)
                deduplicated.append(result)

        if len(deduplicated) < len(results):
            log.debug(
                "results_deduplicated",
                original_count=len(results),
                deduplicated_count=len(deduplicated),
            )

        return deduplicated
