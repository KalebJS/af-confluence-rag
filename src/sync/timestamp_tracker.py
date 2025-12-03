"""Timestamp tracking for maintaining synchronization state."""

from datetime import datetime

import structlog

from src.models.page import SyncState
from src.storage.vector_store import VectorStoreInterface

log = structlog.stdlib.get_logger()


class TimestampTracker:
    """Manages sync state timestamps in vector database metadata."""

    # Special metadata key for storing sync state
    SYNC_STATE_PAGE_ID: str = "__sync_state__"

    def __init__(self, vector_store: VectorStoreInterface):
        """
        Initialize timestamp tracker.

        Args:
            vector_store: Vector store interface for persistence
        """
        self._vector_store: VectorStoreInterface = vector_store
        log.info("timestamp_tracker_initialized")

    def save_sync_state(self, sync_state: SyncState) -> None:
        """
        Save synchronization state to vector database.

        The sync state is stored as metadata in the vector database using
        a special page_id to distinguish it from regular documents.

        Args:
            sync_state: Sync state to save

        Raises:
            RuntimeError: If save operation fails
        """
        log.info(
            "saving_sync_state",
            space_key=sync_state.space_key,
            last_sync_timestamp=sync_state.last_sync_timestamp,
            page_count=sync_state.page_count,
            chunk_count=sync_state.chunk_count,
        )

        try:
            # Store sync state as metadata
            # We use the vector store's internal storage mechanism
            # Since we can't add documents without embeddings, we'll use
            # a workaround: store it as metadata on a special document
            import numpy as np

            from src.models.page import DocumentChunk
            from src.storage.vector_store import ChromaStore

            chunk_id = f"{self.SYNC_STATE_PAGE_ID}_{sync_state.space_key}"

            # Delete existing sync state for this space if it exists
            # Chroma's upsert behavior with add_documents might not update metadata correctly
            if isinstance(self._vector_store, ChromaStore):
                try:
                    self._vector_store._collection.delete(ids=[chunk_id])
                except Exception:
                    # It's okay if it doesn't exist
                    pass

            # Create a special chunk to hold sync state
            sync_chunk = DocumentChunk(
                chunk_id=chunk_id,
                page_id=self.SYNC_STATE_PAGE_ID,
                content=f"Sync state for space {sync_state.space_key}",
                metadata={
                    "space_key": sync_state.space_key,
                    "last_sync_timestamp": sync_state.last_sync_timestamp.isoformat(),
                    "page_count": sync_state.page_count,
                    "chunk_count": sync_state.chunk_count,
                    "is_sync_state": True,
                },
                chunk_index=0,
            )

            # Create a zero embedding (we don't care about the embedding for metadata)
            # Get embedding dimension from a test query or use a standard dimension
            zero_embedding = np.zeros(384)  # all-MiniLM-L6-v2 has 384 dimensions

            self._vector_store.add_documents([sync_chunk], [zero_embedding])

            log.info("sync_state_saved", space_key=sync_state.space_key)

        except Exception as e:
            log.error(
                "failed_to_save_sync_state",
                space_key=sync_state.space_key,
                error=str(e),
            )
            raise RuntimeError(f"Failed to save sync state: {e}") from e

    def load_sync_state(self, space_key: str) -> SyncState | None:
        """
        Load synchronization state from vector database.

        Args:
            space_key: Confluence space key

        Returns:
            SyncState if found, None otherwise

        Raises:
            RuntimeError: If load operation fails
        """
        log.info("loading_sync_state", space_key=space_key)

        try:
            # We need to search for the specific sync state by chunk_id
            # since multiple spaces can have sync states
            chunk_id = f"{self.SYNC_STATE_PAGE_ID}_{space_key}"

            # Access the underlying collection to query by chunk_id
            from src.storage.vector_store import ChromaStore

            if isinstance(self._vector_store, ChromaStore):
                # Use Chroma's get method to retrieve by ID
                results = self._vector_store._collection.get(ids=[chunk_id], include=["metadatas"])

                if not results["ids"] or not results["metadatas"]:
                    log.info("no_sync_state_found", space_key=space_key)
                    return None

                metadata = results["metadatas"][0]
            else:
                # Fallback for other vector store implementations
                metadata = self._vector_store.get_document_metadata(self.SYNC_STATE_PAGE_ID)

                if not metadata:
                    log.info("no_sync_state_found", space_key=space_key)
                    return None

                # Verify it's for the correct space
                if metadata.get("space_key") != space_key:
                    log.info("no_sync_state_found", space_key=space_key)
                    return None

            # Check if this is actually sync state metadata
            if not metadata.get("is_sync_state"):
                log.warning("invalid_sync_state_metadata", space_key=space_key, metadata=metadata)
                return None

            # Parse the sync state
            last_sync_str = metadata.get("last_sync_timestamp")
            if not last_sync_str:
                log.warning("missing_last_sync_timestamp", space_key=space_key)
                return None

            last_sync_timestamp = datetime.fromisoformat(last_sync_str.replace("Z", "+00:00"))

            sync_state = SyncState(
                space_key=metadata.get("space_key", space_key),
                last_sync_timestamp=last_sync_timestamp,
                page_count=metadata.get("page_count", 0),
                chunk_count=metadata.get("chunk_count", 0),
            )

            log.info(
                "sync_state_loaded",
                space_key=space_key,
                last_sync_timestamp=sync_state.last_sync_timestamp,
            )

            return sync_state

        except Exception as e:
            log.error(
                "failed_to_load_sync_state",
                space_key=space_key,
                error=str(e),
            )
            raise RuntimeError(f"Failed to load sync state: {e}") from e
