"""Query processing functionality for semantic search."""

import structlog

from src.models.page import SearchResult
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import VectorStoreInterface

log = structlog.stdlib.get_logger()


class QueryProcessor:
    """Handles query processing and semantic search operations.
    
    This class coordinates query embedding generation and vector similarity
    search to retrieve relevant documents from the vector store.
    """

    def __init__(
        self,
        embedder: EmbeddingGenerator,
        vector_store: VectorStoreInterface,
    ) -> None:
        """Initialize the query processor.
        
        Args:
            embedder: Embedding generator for converting queries to vectors
            vector_store: Vector store interface for similarity search
        """
        self.embedder = embedder
        self.vector_store = vector_store
        
        log.info(
            "query_processor_initialized",
            embedding_model=embedder.model_name,
            embedding_dimension=embedder.get_embedding_dimension(),
        )

    def process_query(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Process a search query and return relevant results.
        
        This method:
        1. Converts the query text to a vector embedding
        2. Searches the vector store for similar documents
        3. Ranks results by similarity score (descending)
        4. Filters and deduplicates results
        
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
            # Step 1: Generate query embedding
            query_embedding = self.embedder.generate_embedding(query)
            
            log.debug(
                "query_embedding_generated",
                embedding_shape=query_embedding.shape,
            )
            
            # Step 2: Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
            )
            
            # Step 3: Results are already ranked by similarity score (descending)
            # from the vector store implementation
            
            # Step 4: Filter and deduplicate results
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
