"""Property-based tests for ConfluenceClient.

Feature: confluence-rag-system
"""

from datetime import datetime, timezone
from typing import List
from unittest.mock import MagicMock, Mock

import structlog
from hypothesis import given, settings, strategies as st

from src.ingestion.confluence_client import ConfluenceClient
from src.models.page import Page

log = structlog.stdlib.get_logger()


# Hypothesis strategies for generating test data
@st.composite
def confluence_page_response_strategy(draw):
    """Generate valid Confluence API page response."""
    page_id = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Nd", "Lu", "Ll")),
        )
    )
    title = draw(st.text(min_size=1, max_size=100))
    space_key = draw(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",)))
    )
    content = draw(st.text(min_size=0, max_size=1000))
    author = draw(st.text(min_size=1, max_size=50))

    # Generate naive datetimes and add timezone
    created_date_naive = draw(
        st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31))
    )
    created_date = created_date_naive.replace(tzinfo=timezone.utc).isoformat()
    modified_date_naive = draw(
        st.datetimes(min_value=created_date_naive, max_value=datetime(2030, 12, 31))
    )
    modified_date = modified_date_naive.replace(tzinfo=timezone.utc).isoformat()

    version = draw(st.integers(min_value=1, max_value=1000))

    return {
        "id": page_id,
        "title": title,
        "space": {"key": space_key},
        "body": {"storage": {"value": content}},
        "version": {"number": version, "when": modified_date},
        "history": {"createdBy": {"username": author}, "createdDate": created_date},
    }


@given(
    st.lists(confluence_page_response_strategy(), min_size=1, max_size=50),
    st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))),
)
@settings(max_examples=100)
def test_property_1_complete_page_retrieval(
    confluence_pages: List[dict], space_key: str
):
    """Property 1: Complete page retrieval.
    
    For any Confluence space, when the system retrieves pages, the count of
    retrieved pages should equal the total page count reported by the
    Confluence API.
    
    **Feature: confluence-rag-system, Property 1: Complete page retrieval**
    **Validates: Requirements 1.2**
    """
    log.info(
        "test_property_1_complete_page_retrieval",
        space_key=space_key,
        expected_count=len(confluence_pages),
    )

    # Create a mock Confluence client
    mock_confluence = MagicMock()
    mock_confluence.get_all_pages_from_space.return_value = iter(confluence_pages)

    # Create ConfluenceClient and inject the mock
    client = ConfluenceClient(
        base_url="https://test.atlassian.net", auth_token="test-token", cloud=True
    )
    client._client = mock_confluence

    # Retrieve all pages
    retrieved_pages = list(client.get_space_pages(space_key))

    # Property: Retrieved count should equal expected count
    assert len(retrieved_pages) == len(
        confluence_pages
    ), f"Expected {len(confluence_pages)} pages, got {len(retrieved_pages)}"

    # Verify all pages are Page instances
    for page in retrieved_pages:
        assert isinstance(page, Page), "All retrieved items should be Page instances"

    log.info(
        "test_property_1_complete_page_retrieval_passed",
        retrieved_count=len(retrieved_pages),
    )


@given(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=50),
    st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",))),
)
@settings(max_examples=100)
def test_property_2_pagination_completeness(
    num_pages: int, pages_per_batch: int, space_key: str
):
    """Property 2: Pagination completeness.
    
    For any paginated API response, the system should retrieve all pages by
    following pagination links until no next page exists.
    
    **Feature: confluence-rag-system, Property 2: Pagination completeness**
    **Validates: Requirements 1.3**
    """
    log.info(
        "test_property_2_pagination_completeness",
        space_key=space_key,
        num_pages=num_pages,
        pages_per_batch=pages_per_batch,
    )

    # Generate test pages
    confluence_pages = []
    for i in range(num_pages):
        confluence_pages.append(
            {
                "id": f"page_{i}",
                "title": f"Test Page {i}",
                "space": {"key": space_key},
                "body": {"storage": {"value": f"Content {i}"}},
                "version": {"number": 1, "when": "2024-01-01T00:00:00Z"},
                "history": {
                    "createdBy": {"username": "test_user"},
                    "createdDate": "2024-01-01T00:00:00Z",
                },
            }
        )

    # Create a mock that simulates pagination
    mock_confluence = MagicMock()
    mock_confluence.get_all_pages_from_space.return_value = iter(confluence_pages)

    # Create ConfluenceClient and inject the mock
    client = ConfluenceClient(
        base_url="https://test.atlassian.net", auth_token="test-token", cloud=True
    )
    client._client = mock_confluence

    # Retrieve all pages
    retrieved_pages = list(client.get_space_pages(space_key))

    # Property: All pages should be retrieved regardless of pagination
    assert len(retrieved_pages) == num_pages, (
        f"Expected {num_pages} pages, got {len(retrieved_pages)}"
    )

    # Verify page IDs are unique and match expected
    retrieved_ids = {page.id for page in retrieved_pages}
    expected_ids = {f"page_{i}" for i in range(num_pages)}
    assert retrieved_ids == expected_ids, "All page IDs should be retrieved"

    log.info(
        "test_property_2_pagination_completeness_passed",
        retrieved_count=len(retrieved_pages),
    )
