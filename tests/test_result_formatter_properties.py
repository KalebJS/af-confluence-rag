"""Property-based tests for result formatting.

**Feature: confluence-rag-system**
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.page import SearchResult
from src.query.result_formatter import ResultFormatter

# Strategies for generating test data
page_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=48, max_codepoint=122),
    min_size=1,
    max_size=20,
)

page_title_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=100,
)

content_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=500,
)


# Strategy for generating valid URLs with the expected base
def url_strategy(base_url: str = "https://example.atlassian.net"):
    """Generate valid Confluence URLs."""
    return st.from_regex(rf"{base_url}/wiki/spaces/[A-Z]+/pages/\d+", fullmatch=True)


def generate_search_result(
    page_id: str,
    page_title: str,
    page_url: str,
    content: str,
    similarity_score: float,
) -> SearchResult:
    """Generate a SearchResult with valid data."""
    return SearchResult(
        chunk_id=f"{page_id}_0",
        page_id=page_id,
        page_title=page_title,
        page_url=page_url,
        content=content,
        similarity_score=similarity_score,
        metadata={
            "author": "test_author",
            "modified_date": "2024-01-01",
        },
    )


@given(
    page_id=page_id_strategy,
    page_title=page_title_strategy,
    content=content_strategy,
    similarity_score=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None, max_examples=100)
def test_property_22_valid_result_urls(
    page_id: str,
    page_title: str,
    content: str,
    similarity_score: float,
):
    """Property 22: Valid result URLs

    *For any* search result, the page_url field should be a valid URL starting
    with the configured Confluence base URL.

    **Validates: Requirements 5.5**
    **Feature: confluence-rag-system, Property 22: Valid result URLs**
    """
    base_url = "https://example.atlassian.net"
    formatter = ResultFormatter(base_url=base_url)

    # Generate a valid URL for this test
    page_url = f"{base_url}/wiki/spaces/DOCS/pages/{abs(hash(page_id)) % 1000000}"

    result = generate_search_result(
        page_id=page_id,
        page_title=page_title,
        page_url=page_url,
        content=content,
        similarity_score=similarity_score,
    )

    # Create result card (this validates the URL)
    card = formatter.create_result_card(result)

    # Property: URL should be valid and start with base URL
    assert card["url"].startswith(base_url), (
        f"Result URL should start with base URL {base_url}, but got {card['url']}"
    )

    # URL should be properly formatted
    assert "://" in card["url"], "URL should contain scheme separator"
    assert card["url"].startswith("http"), "URL should start with http/https"


@given(
    page_id=page_id_strategy,
    page_title=page_title_strategy,
    content=content_strategy,
    similarity_score=st.floats(min_value=0.0, max_value=1.0),
)
@settings(deadline=None, max_examples=100)
def test_property_23_result_metadata_display(
    page_id: str,
    page_title: str,
    content: str,
    similarity_score: float,
):
    """Property 23: Result metadata display

    *For any* rendered search result in the UI, the output should contain
    the page_title and similarity_score.

    **Validates: Requirements 6.4**
    **Feature: confluence-rag-system, Property 23: Result metadata display**
    """
    formatter = ResultFormatter(base_url="https://example.atlassian.net")

    # Generate a valid URL for this test
    page_url = (
        f"https://example.atlassian.net/wiki/spaces/DOCS/pages/{abs(hash(page_id)) % 1000000}"
    )

    result = generate_search_result(
        page_id=page_id,
        page_title=page_title,
        page_url=page_url,
        content=content,
        similarity_score=similarity_score,
    )

    # Test format_results (string output)
    formatted_string = formatter.format_results([result])

    # Property: formatted string should contain page_title and score
    assert page_title in formatted_string, (
        f"Formatted result should contain page_title '{page_title}'"
    )
    assert "Score:" in formatted_string, "Formatted result should display similarity score"

    # Test create_result_card (structured output)
    card = formatter.create_result_card(result)

    # Property: card should contain page_title and score
    assert card["title"] == page_title, (
        f"Result card should contain page_title '{page_title}', but got '{card['title']}'"
    )
    assert "score" in card, "Result card should contain score field"
    assert 0.0 <= card["score"] <= 1.0, f"Score should be in [0, 1], got {card['score']}"


@given(
    results_count=st.integers(min_value=0, max_value=10),
)
@settings(deadline=None, max_examples=50)
def test_result_formatter_handles_empty_results(results_count: int):
    """Test that formatter handles empty and non-empty result lists correctly."""
    formatter = ResultFormatter(base_url="https://example.atlassian.net")

    # Generate results
    results = []
    for i in range(results_count):
        result = generate_search_result(
            page_id=f"page_{i}",
            page_title=f"Page {i}",
            page_url=f"https://example.atlassian.net/wiki/spaces/DOCS/pages/{i}",
            content=f"Content {i}",
            similarity_score=0.9 - (i * 0.1),
        )
        results.append(result)

    # Test format_results
    formatted = formatter.format_results(results)

    if results_count == 0:
        assert "No results found" in formatted
    else:
        assert f"Found {results_count} result(s)" in formatted
        # All titles should be present
        for result in results:
            assert result.page_title in formatted


@given(
    content_length=st.integers(min_value=1, max_value=1000),
)
@settings(deadline=None, max_examples=50)
def test_result_formatter_creates_excerpts(content_length: int):
    """Test that formatter creates appropriate content excerpts."""
    formatter = ResultFormatter(base_url="https://example.atlassian.net")

    # Generate content of specified length
    content = "a" * content_length

    result = generate_search_result(
        page_id="test_page",
        page_title="Test Page",
        page_url="https://example.atlassian.net/wiki/spaces/DOCS/pages/123",
        content=content,
        similarity_score=0.85,
    )

    card = formatter.create_result_card(result)

    # Property: excerpt should not exceed max length (300 + ellipsis)
    assert len(card["content"]) <= 303, (
        f"Content excerpt should not exceed 303 chars, got {len(card['content'])}"
    )

    # If original content was longer than 300, excerpt should be truncated with ellipsis
    if content_length > 300:
        assert card["content"].endswith("..."), "Long content should be truncated with ellipsis"
        assert len(card["content"]) <= 303, "Truncated content should not exceed 303 chars"
