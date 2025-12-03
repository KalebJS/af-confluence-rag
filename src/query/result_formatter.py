"""Result formatting functionality for displaying search results."""

from typing import Any
from urllib.parse import urlparse

import structlog

from src.models.page import SearchResult

log = structlog.stdlib.get_logger()


class ResultFormatter:
    """Formats search results for display in the user interface.

    This class provides methods to format search results into various
    display formats with proper metadata and URL validation.
    """

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the result formatter.

        Args:
            base_url: Base URL for Confluence instance (used for URL validation)
        """
        self.base_url = base_url
        log.info("result_formatter_initialized", base_url=base_url)

    def format_results(self, results: list[SearchResult]) -> str:
        """Format search results as a readable string.

        This method creates a formatted text representation of search results
        suitable for display in console or text-based interfaces.

        Args:
            results: List of search results to format

        Returns:
            Formatted string containing all results
        """
        if not results:
            return "No results found."

        formatted_lines = [f"Found {len(results)} result(s):\n"]

        for i, result in enumerate(results, 1):
            formatted_lines.append(f"\n{i}. {result.page_title}")
            formatted_lines.append(f"   Score: {result.similarity_score:.3f}")
            formatted_lines.append(f"   URL: {result.page_url}")

            # Truncate content for display
            content_preview = result.content[:200]
            if len(result.content) > 200:
                content_preview += "..."
            formatted_lines.append(f"   Content: {content_preview}")

        log.debug("results_formatted", result_count=len(results))
        return "\n".join(formatted_lines)

    def create_result_card(self, result: SearchResult) -> dict[str, Any]:
        """Create a result card dictionary with metadata.

        This method creates a structured dictionary representation of a search
        result suitable for rendering in web interfaces (e.g., Streamlit).

        Args:
            result: Search result to format

        Returns:
            Dictionary containing formatted result data with keys:
            - title: Page title
            - url: Validated page URL
            - content: Content excerpt
            - score: Similarity score
            - metadata: Additional metadata (author, date, etc.)

        Raises:
            ValueError: If result URL is invalid
        """
        # Validate URL
        validated_url = self._validate_url(result.page_url)

        # Create content excerpt
        content_excerpt = self._create_excerpt(result.content, max_length=300)

        # Build result card
        card = {
            "title": result.page_title,
            "url": validated_url,
            "content": content_excerpt,
            "score": round(result.similarity_score, 3),
            "metadata": {
                "page_id": result.page_id,
                "chunk_id": result.chunk_id,
                "author": result.metadata.get("author", "Unknown"),
                "modified_date": result.metadata.get("modified_date", "Unknown"),
            },
        }

        log.debug(
            "result_card_created",
            page_id=result.page_id,
            title=result.page_title,
        )

        return card

    def create_result_cards(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Create result cards for multiple search results.

        Args:
            results: List of search results to format

        Returns:
            List of result card dictionaries
        """
        cards = [self.create_result_card(result) for result in results]
        log.info("result_cards_created", count=len(cards))
        return cards

    def _validate_url(self, url: str) -> str:
        """Validate and format a URL.

        Args:
            url: URL to validate

        Returns:
            Validated URL string

        Raises:
            ValueError: If URL is invalid or doesn't match base_url
        """
        try:
            parsed = urlparse(str(url))

            # Check that URL has scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL format: {url}")

            # If base_url is set, validate that result URL matches
            if self.base_url:
                base_parsed = urlparse(self.base_url)
                if parsed.netloc != base_parsed.netloc:
                    log.warning(
                        "url_domain_mismatch",
                        expected_domain=base_parsed.netloc,
                        actual_domain=parsed.netloc,
                        url=url,
                    )

            return str(url)

        except Exception as e:
            log.error("url_validation_failed", url=url, error=str(e))
            raise ValueError(f"Invalid URL: {url}") from e

    def _create_excerpt(self, content: str, max_length: int = 300) -> str:
        """Create a content excerpt with ellipsis if needed.

        Args:
            content: Full content text
            max_length: Maximum length of excerpt

        Returns:
            Content excerpt string
        """
        if len(content) <= max_length:
            return content

        # Try to break at a sentence boundary
        excerpt = content[:max_length]

        # Look for last sentence ending
        last_period = excerpt.rfind(". ")
        last_question = excerpt.rfind("? ")
        last_exclamation = excerpt.rfind("! ")

        last_sentence_end = max(last_period, last_question, last_exclamation)

        if last_sentence_end > max_length * 0.5:  # At least 50% of max_length
            excerpt = excerpt[: last_sentence_end + 1]
        else:
            # Break at last space
            last_space = excerpt.rfind(" ")
            if last_space > 0:
                excerpt = excerpt[:last_space]
            excerpt += "..."

        return excerpt
