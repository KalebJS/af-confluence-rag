"""Change detection for identifying new, modified, and deleted pages."""

from datetime import datetime
from typing import Any

import structlog

from src.models.page import Page
from src.sync.models import ChangeSet

log = structlog.stdlib.get_logger()


class ChangeDetector:
    """Detects changes between Confluence and local vector store."""

    def detect_changes(
        self,
        confluence_pages: list[Page],
        stored_page_metadata: dict[str, dict[str, Any]],
        last_sync_timestamp: datetime | None = None,
    ) -> ChangeSet:
        """
        Detect new, modified, and deleted pages.

        Args:
            confluence_pages: Current pages from Confluence
            stored_page_metadata: Metadata of pages currently in vector store
                                 (keyed by page_id)
            last_sync_timestamp: Timestamp of last successful sync

        Returns:
            ChangeSet containing new, modified, and deleted pages
        """
        log.info(
            "detecting_changes",
            confluence_page_count=len(confluence_pages),
            stored_page_count=len(stored_page_metadata),
            last_sync_timestamp=last_sync_timestamp,
        )

        # Build sets for efficient lookup
        confluence_page_ids: set[str] = {page.id for page in confluence_pages}
        stored_page_ids: set[str] = set(stored_page_metadata.keys())

        # Detect new pages
        new_page_ids = confluence_page_ids - stored_page_ids
        new_pages = [page for page in confluence_pages if page.id in new_page_ids]

        # Detect modified pages
        modified_pages = self._detect_modified_pages(
            confluence_pages, stored_page_metadata, last_sync_timestamp
        )

        # Detect deleted pages
        deleted_page_ids = list(stored_page_ids - confluence_page_ids)

        change_set = ChangeSet(
            new_pages=new_pages,
            modified_pages=modified_pages,
            deleted_page_ids=deleted_page_ids,
        )

        log.info(
            "changes_detected",
            new_pages=len(new_pages),
            modified_pages=len(modified_pages),
            deleted_pages=len(deleted_page_ids),
            total_changes=change_set.total_changes,
        )

        return change_set

    def detect_new_pages(
        self, confluence_pages: list[Page], stored_page_ids: set[str]
    ) -> list[Page]:
        """
        Detect pages that exist in Confluence but not in vector store.

        Args:
            confluence_pages: Current pages from Confluence
            stored_page_ids: Set of page IDs currently in vector store

        Returns:
            List of new pages
        """
        new_pages = [page for page in confluence_pages if page.id not in stored_page_ids]

        log.info("new_pages_detected", count=len(new_pages))
        return new_pages

    def detect_modified_pages(
        self,
        confluence_pages: list[Page],
        stored_page_metadata: dict[str, dict[str, Any]],
        last_sync_timestamp: datetime | None = None,
    ) -> list[Page]:
        """
        Detect pages that have been modified since last sync.

        Args:
            confluence_pages: Current pages from Confluence
            stored_page_metadata: Metadata of pages currently in vector store
            last_sync_timestamp: Timestamp of last successful sync

        Returns:
            List of modified pages
        """
        return self._detect_modified_pages(
            confluence_pages, stored_page_metadata, last_sync_timestamp
        )

    def detect_deleted_pages(
        self, confluence_page_ids: set[str], stored_page_ids: set[str]
    ) -> list[str]:
        """
        Detect pages that exist in vector store but not in Confluence.

        Args:
            confluence_page_ids: Set of current page IDs from Confluence
            stored_page_ids: Set of page IDs currently in vector store

        Returns:
            List of deleted page IDs
        """
        deleted_page_ids = list(stored_page_ids - confluence_page_ids)

        log.info("deleted_pages_detected", count=len(deleted_page_ids))
        return deleted_page_ids

    def _detect_modified_pages(
        self,
        confluence_pages: list[Page],
        stored_page_metadata: dict[str, dict[str, Any]],
        last_sync_timestamp: datetime | None = None,
    ) -> list[Page]:
        """
        Internal method to detect modified pages using timestamp comparison.

        Args:
            confluence_pages: Current pages from Confluence
            stored_page_metadata: Metadata of pages currently in vector store
            last_sync_timestamp: Timestamp of last successful sync

        Returns:
            List of modified pages
        """
        modified_pages = []

        for page in confluence_pages:
            # Skip if page is not in store (it's new, not modified)
            if page.id not in stored_page_metadata:
                continue

            stored_metadata = stored_page_metadata[page.id]

            # Check if page has been modified
            if self._is_page_modified(page, stored_metadata, last_sync_timestamp):
                modified_pages.append(page)

        log.info("modified_pages_detected", count=len(modified_pages))
        return modified_pages

    def _is_page_modified(
        self,
        page: Page,
        stored_metadata: dict[str, Any],
        last_sync_timestamp: datetime | None = None,
    ) -> bool:
        """
        Check if a page has been modified based on timestamp comparison.

        Args:
            page: Current page from Confluence
            stored_metadata: Stored metadata for the page
            last_sync_timestamp: Timestamp of last successful sync

        Returns:
            True if page has been modified, False otherwise
        """
        # Get stored modified date
        stored_modified_str = stored_metadata.get("modified_date", "")

        if not stored_modified_str:
            # If no stored date, consider it modified to be safe
            return True

        try:
            # Parse stored date (it's stored as ISO string in metadata)
            if isinstance(stored_modified_str, str):
                stored_modified = datetime.fromisoformat(
                    stored_modified_str.replace("Z", "+00:00")
                )
            else:
                stored_modified = stored_modified_str

            # Compare timestamps
            # Page is modified if Confluence timestamp is newer than stored timestamp
            is_modified = page.modified_date > stored_modified

            if is_modified:
                log.debug(
                    "page_modified_detected",
                    page_id=page.id,
                    page_title=page.title,
                    confluence_modified=page.modified_date,
                    stored_modified=stored_modified,
                )

            return is_modified

        except (ValueError, AttributeError) as e:
            log.warning(
                "failed_to_parse_stored_date",
                page_id=page.id,
                stored_date=stored_modified_str,
                error=str(e),
            )
            # If we can't parse the date, consider it modified to be safe
            return True
