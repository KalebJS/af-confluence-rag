"""Vector store interface and implementations for document storage and retrieval."""

from abc import ABC, abstractmethod
from typing import List, Optional

import chromadb
import numpy as np
import structlog

from src.models.page import DocumentChunk, SearchResult

log = structlog.stdlib.get_logger()


class VectorStoreInterface(ABC):
    """Abstract interface for vector database operations.

    This interface defines the contract that all vector store implementations
    must follow, enabling pluggable vector database backends.
    """

    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]) -> None:
        """Add documents with embeddings to the vector store.

        Args:
            chunks: List of document chunks with metadata
            embeddings: List of embedding vectors corresponding to chunks

        Raises:
            ValueError: If chunks and embeddings lists have different lengths
            RuntimeError: If storage operation fails
        """
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search for similar documents using vector similarity.

        Args:
            query_embedding: Query vector to search for
            top_k: Maximum number of results to return

        Returns:
            List of search results ordered by similarity score (descending)

        Raises:
            ValueError: If query_embedding has incorrect dimensions
            RuntimeError: If search operation fails
        """
        pass

    @abstractmethod
    def delete_by_page_id(self, page_id: str) -> None:
        """Delete all chunks associated with a page.

        This is used during incremental updates to remove old versions
        of a page before adding the updated content.

        Args:
            page_id: Unique identifier of the page to delete

        Raises:
            RuntimeError: If deletion operation fails
        """
        pass

    @abstractmethod
    def get_document_metadata(self, page_id: str) -> Optional[dict]:
        """Retrieve metadata for a document.

        Args:
            page_id: Unique identifier of the page

        Returns:
            Dictionary containing document metadata, or None if not found

        Raises:
            RuntimeError: If retrieval operation fails
        """
        pass


class ChromaStore(VectorStoreInterface):
    """Chroma implementation of vector store.

    This implementation uses ChromaDB for persistent vector storage with
    support for metadata filtering and similarity search.
    """

    def __init__(self, persist_directory: str, collection_name: str = "confluence_docs"):
        """Initialize Chroma vector store.

        Args:
            persist_directory: Directory path for persistent storage
            collection_name: Name of the Chroma collection

        Raises:
            RuntimeError: If Chroma client initialization fails
        """
        try:
            self._client = chromadb.PersistentClient(path=persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
            )
            log.info(
                "chroma_store_initialized",
                persist_directory=persist_directory,
                collection_name=collection_name,
            )
        except Exception as e:
            log.error(
                "chroma_store_initialization_failed",
                persist_directory=persist_directory,
                collection_name=collection_name,
                error=str(e),
            )
            raise RuntimeError(f"Failed to initialize Chroma store: {e}") from e

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]) -> None:
        """Add documents with embeddings to the vector store.

        Args:
            chunks: List of document chunks with metadata
            embeddings: List of embedding vectors corresponding to chunks

        Raises:
            ValueError: If chunks and embeddings lists have different lengths
            RuntimeError: If storage operation fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings must have same length: {len(chunks)} != {len(embeddings)}"
            )

        if not chunks:
            log.warning("add_documents_called_with_empty_list")
            return

        try:
            # Prepare data for Chroma
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = []

            for chunk in chunks:
                # Flatten metadata for Chroma storage
                # Start with standard fields
                metadata = {
                    "page_id": chunk.page_id,
                    "chunk_index": chunk.chunk_index,
                    "page_title": chunk.metadata.get("page_title", ""),
                    "page_url": chunk.metadata.get("page_url", ""),
                    "author": chunk.metadata.get("author", ""),
                    "modified_date": chunk.metadata.get("modified_date", ""),
                }
                # Add any additional metadata fields (for special cases like sync state)
                for key, value in chunk.metadata.items():
                    if key not in metadata:
                        metadata[key] = value
                metadatas.append(metadata)

            # Convert embeddings to list format
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Add to collection (upsert to handle duplicates)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )

            log.info("documents_added_to_chroma", count=len(chunks))

        except Exception as e:
            log.error(
                "add_documents_failed",
                chunk_count=len(chunks),
                error=str(e),
            )
            raise RuntimeError(f"Failed to add documents to Chroma: {e}") from e

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search for similar documents using vector similarity.

        Args:
            query_embedding: Query vector to search for
            top_k: Maximum number of results to return

        Returns:
            List of search results ordered by similarity score (descending)

        Raises:
            ValueError: If query_embedding has incorrect dimensions
            RuntimeError: If search operation fails
        """
        try:
            # Query the collection
            results = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Convert Chroma results to SearchResult objects
            search_results = []

            if not results["ids"] or not results["ids"][0]:
                log.info("search_returned_no_results", top_k=top_k)
                return search_results

            # Chroma returns results for each query (we only have one)
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i, chunk_id in enumerate(ids):
                metadata = metadatas[i]
                # Convert distance to similarity score (cosine distance to similarity)
                # Chroma returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity: similarity = 1 - (distance / 2)
                similarity_score = 1.0 - (distances[i] / 2.0)
                # Clamp to [0, 1] range
                similarity_score = max(0.0, min(1.0, similarity_score))

                search_result = SearchResult(
                    chunk_id=chunk_id,
                    page_id=metadata["page_id"],
                    page_title=metadata["page_title"],
                    page_url=metadata["page_url"],
                    content=documents[i],
                    similarity_score=similarity_score,
                    metadata={
                        "author": metadata.get("author", ""),
                        "modified_date": metadata.get("modified_date", ""),
                        "chunk_index": metadata.get("chunk_index", 0),
                    },
                )
                search_results.append(search_result)

            log.info("search_completed", results_count=len(search_results), top_k=top_k)
            return search_results

        except Exception as e:
            log.error("search_failed", top_k=top_k, error=str(e))
            raise RuntimeError(f"Failed to search Chroma: {e}") from e

    def delete_by_page_id(self, page_id: str) -> None:
        """Delete all chunks associated with a page.

        This is used during incremental updates to remove old versions
        of a page before adding the updated content.

        Args:
            page_id: Unique identifier of the page to delete

        Raises:
            RuntimeError: If deletion operation fails
        """
        try:
            # Query for all chunks with this page_id
            results = self._collection.get(
                where={"page_id": page_id},
                include=[],  # We only need IDs
            )

            if results["ids"]:
                self._collection.delete(ids=results["ids"])
                log.info(
                    "page_deleted_from_chroma", page_id=page_id, chunks_deleted=len(results["ids"])
                )
            else:
                log.info("page_not_found_for_deletion", page_id=page_id)

        except Exception as e:
            log.error("delete_by_page_id_failed", page_id=page_id, error=str(e))
            raise RuntimeError(f"Failed to delete page from Chroma: {e}") from e

    def get_document_metadata(self, page_id: str) -> Optional[dict]:
        """Retrieve metadata for a document.

        Args:
            page_id: Unique identifier of the page

        Returns:
            Dictionary containing document metadata, or None if not found

        Raises:
            RuntimeError: If retrieval operation fails
        """
        try:
            # Get the first chunk for this page to retrieve metadata
            results = self._collection.get(
                where={"page_id": page_id}, limit=1, include=["metadatas"]
            )

            if results["ids"] and results["metadatas"]:
                metadata = results["metadatas"][0]
                log.info("document_metadata_retrieved", page_id=page_id)
                return metadata
            else:
                log.info("document_metadata_not_found", page_id=page_id)
                return None

        except Exception as e:
            log.error("get_document_metadata_failed", page_id=page_id, error=str(e))
            raise RuntimeError(f"Failed to retrieve metadata from Chroma: {e}") from e



