"""Timestamp tracking for maintaining synchronization state."""

from datetime import datetime

import structlog
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from src.models.page import SyncState

log = structlog.stdlib.get_logger()


class TimestampTracker:
    """Manages sync state timestamps in vector database metadata."""

    # Special metadata key for storing sync state
    SYNC_STATE_PAGE_ID: str = "__sync_state__"

    def __init__(self, vector_store: VectorStore):
        """
        Initialize timestamp tracker.

        Args:
            vector_store: LangChain VectorStore instance for persistence
        """
        self._vector_store: VectorStore = vector_store
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
            chunk_id = f"{self.SYNC_STATE_PAGE_ID}_{sync_state.space_key}"

            # Delete existing sync state for this space if it exists
            try:
                # Try to delete using the vector store's delete method
                self._vector_store.delete([chunk_id])
            except (AttributeError, NotImplementedError, Exception):
                # It's okay if delete is not supported or document doesn't exist
                pass

            # Create a LangChain Document to hold sync state
            sync_doc = Document(
                page_content=f"Sync state for space {sync_state.space_key}",
                metadata={
                    "chunk_id": chunk_id,
                    "page_id": self.SYNC_STATE_PAGE_ID,
                    "space_key": sync_state.space_key,
                    "last_sync_timestamp": sync_state.last_sync_timestamp.isoformat(),
                    "page_count": sync_state.page_count,
                    "chunk_count": sync_state.chunk_count,
                    "is_sync_state": True,
                    "chunk_index": 0,
                },
            )

            # Add document to vector store (embeddings generated automatically)
            self._vector_store.add_documents([sync_doc], ids=[chunk_id])

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
            # Search for sync state using metadata filter
            # Chroma requires $and operator for multiple conditions
            results = self._vector_store.similarity_search(
                query="",  # Empty query, we only care about metadata filter
                k=1,
                filter={
                    "$and": [
                        {"page_id": self.SYNC_STATE_PAGE_ID},
                        {"space_key": space_key},
                        {"is_sync_state": True}
                    ]
                }
            )

            if not results:
                log.info("no_sync_state_found", space_key=space_key)
                return None

            metadata = results[0].metadata

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
