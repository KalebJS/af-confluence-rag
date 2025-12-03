"""Confluence client wrapper for API interactions."""

from typing import Generator, List, Optional

import structlog
from atlassian import Confluence
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

from src.models.page import Page
from src.utils.retry import exponential_backoff_retry

log = structlog.stdlib.get_logger()


class ConfluenceClient:
    """Wrapper around atlassian-python-api Confluence client."""

    def __init__(self, base_url: str, auth_token: str, cloud: bool = True):
        """
        Initialize Confluence client.

        Args:
            base_url: Confluence instance URL
            auth_token: API token for authentication
            cloud: True for Confluence Cloud, False for Server/Data Center
        """
        self._client = Confluence(url=base_url, token=auth_token, cloud=cloud)
        self._base_url = base_url
        log.info(
            "confluence_client_initialized",
            base_url=base_url,
            cloud=cloud,
        )

    @exponential_backoff_retry(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(RequestException, HTTPError, Timeout, ConnectionError),
    )
    def get_space_pages(self, space_key: str) -> Generator[Page, None, None]:
        """
        Get all pages from a space using generator for memory efficiency.

        Args:
            space_key: The Confluence space key

        Yields:
            Page objects with full metadata

        Raises:
            Exception: If API call fails after retries
        """
        log.info("fetching_space_pages", space_key=space_key)

        try:
            # Use the generator method from atlassian-python-api for memory efficiency
            page_generator = self._client.get_all_pages_from_space(
                space=space_key,
                start=0,
                limit=100,
                expand="body.storage,version,history",
            )

            page_count = 0
            for confluence_page in page_generator:
                try:
                    page = self._convert_to_page_model(confluence_page, space_key)
                    page_count += 1
                    yield page
                except Exception as e:
                    log.warning(
                        "failed_to_convert_page",
                        page_id=confluence_page.get("id"),
                        error=str(e),
                    )
                    continue

            log.info("space_pages_fetched", space_key=space_key, page_count=page_count)

        except Exception as e:
            log.error("failed_to_fetch_space_pages", space_key=space_key, error=str(e))
            raise

    @exponential_backoff_retry(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(RequestException, HTTPError, Timeout, ConnectionError),
    )
    def get_page_content(self, page_id: str) -> Page:
        """
        Get page content with metadata.

        Args:
            page_id: The Confluence page ID

        Returns:
            Page object with full metadata

        Raises:
            Exception: If API call fails after retries or page not found
        """
        log.info("fetching_page_content", page_id=page_id)

        try:
            confluence_page = self._client.get_page_by_id(
                page_id=page_id,
                expand="body.storage,version,history",
            )

            page = self._convert_to_page_model(
                confluence_page,
                confluence_page.get("space", {}).get("key", ""),
            )

            log.info("page_content_fetched", page_id=page_id, title=page.title)
            return page

        except Exception as e:
            log.error("failed_to_fetch_page_content", page_id=page_id, error=str(e))
            raise

    @exponential_backoff_retry(
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(RequestException, HTTPError, Timeout, ConnectionError),
    )
    def get_page_by_title(self, space_key: str, title: str) -> Optional[Page]:
        """
        Get a specific page by title.

        Args:
            space_key: The Confluence space key
            title: The page title

        Returns:
            Page object if found, None otherwise

        Raises:
            Exception: If API call fails after retries
        """
        log.info("fetching_page_by_title", space_key=space_key, title=title)

        try:
            confluence_page = self._client.get_page_by_title(
                space=space_key,
                title=title,
                expand="body.storage,version,history",
            )

            if not confluence_page:
                log.info("page_not_found", space_key=space_key, title=title)
                return None

            page = self._convert_to_page_model(confluence_page, space_key)
            log.info("page_found_by_title", page_id=page.id, title=title)
            return page

        except Exception as e:
            log.error(
                "failed_to_fetch_page_by_title",
                space_key=space_key,
                title=title,
                error=str(e),
            )
            raise

    def _convert_to_page_model(self, confluence_page: dict, space_key: str) -> Page:
        """
        Convert Confluence API response to Page model.

        Args:
            confluence_page: Raw page data from Confluence API
            space_key: The space key (may be passed separately)

        Returns:
            Page model instance

        Raises:
            ValueError: If required fields are missing
        """
        try:
            # Extract content from body.storage
            content = confluence_page.get("body", {}).get("storage", {}).get("value", "")

            # Extract version information
            version_info = confluence_page.get("version", {})
            version_number = version_info.get("number", 1)

            # Extract history information
            history = confluence_page.get("history", {})
            created_by = history.get("createdBy", {}).get("username", "unknown")

            # Extract dates
            created_date = history.get("createdDate", version_info.get("when"))
            modified_date = version_info.get("when", created_date)

            # Construct URL
            page_id = confluence_page["id"]
            url = f"{self._base_url}/wiki/spaces/{space_key}/pages/{page_id}"

            return Page(
                id=page_id,
                title=confluence_page["title"],
                space_key=space_key,
                content=content,
                author=created_by,
                created_date=created_date,
                modified_date=modified_date,
                url=url,
                version=version_number,
            )

        except KeyError as e:
            log.error(
                "missing_required_field",
                field=str(e),
                page_id=confluence_page.get("id"),
            )
            raise ValueError(f"Missing required field in Confluence page: {e}")
