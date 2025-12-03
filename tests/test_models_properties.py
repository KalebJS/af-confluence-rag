"""Property-based tests for Pydantic models.

Feature: confluence-rag-system
"""

from datetime import datetime, timezone

import structlog
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.models import DocumentChunk, Page, SearchResult, SyncState

log = structlog.stdlib.get_logger()


# Hypothesis strategies for generating test data
@st.composite
def page_strategy(draw):
    """Generate valid Page instances."""
    page_id = draw(
        st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Nd", "Lu", "Ll"))
        )
    )
    title = draw(st.text(min_size=1, max_size=100))
    space_key = draw(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",)))
    )
    content = draw(st.text(min_size=1, max_size=1000))
    author = draw(st.text(min_size=1, max_size=50))

    # Generate naive datetimes and add timezone
    created_date_naive = draw(
        st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31))
    )
    created_date = created_date_naive.replace(tzinfo=timezone.utc)
    modified_date_naive = draw(
        st.datetimes(min_value=created_date_naive, max_value=datetime(2030, 12, 31))
    )
    modified_date = modified_date_naive.replace(tzinfo=timezone.utc)

    version = draw(st.integers(min_value=1, max_value=1000))

    # Generate a valid ASCII domain name (lowercase letters and hyphens only)
    domain = draw(st.text(min_size=3, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"))
    url = f"https://{domain}.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}"

    return Page(
        id=page_id,
        title=title,
        space_key=space_key,
        content=content,
        author=author,
        created_date=created_date,
        modified_date=modified_date,
        url=url,
        version=version,
    )


@st.composite
def document_chunk_strategy(draw):
    """Generate valid DocumentChunk instances."""
    page_id = draw(
        st.text(
            min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Nd", "Lu", "Ll"))
        )
    )
    chunk_index = draw(st.integers(min_value=0, max_value=100))
    chunk_id = f"{page_id}_{chunk_index}"
    content = draw(st.text(min_size=1, max_size=1000))

    # Generate metadata
    page_title = draw(st.text(min_size=1, max_size=100))
    # Generate a valid ASCII domain name (lowercase letters only)
    domain = draw(st.text(min_size=3, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"))
    space_key = draw(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("Lu",)))
    )
    page_url = f"https://{domain}.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}"
    author = draw(st.text(min_size=1, max_size=50))

    # Generate naive datetime and add timezone
    modified_date_naive = draw(
        st.datetimes(min_value=datetime(2000, 1, 1), max_value=datetime(2030, 12, 31))
    )
    modified_date = modified_date_naive.replace(tzinfo=timezone.utc)

    metadata = {
        "page_title": page_title,
        "page_url": page_url,
        "author": author,
        "modified_date": modified_date.isoformat(),
    }

    return DocumentChunk(
        chunk_id=chunk_id,
        page_id=page_id,
        content=content,
        metadata=metadata,
        chunk_index=chunk_index,
    )


@given(page_strategy())
def test_property_3_metadata_completeness(page: Page):
    """Property 3: Metadata completeness.

    For any retrieved Confluence page, the extracted data should contain
    non-empty values for title, author, creation date, modified date, and URL.

    **Feature: confluence-rag-system, Property 3: Metadata completeness**
    **Validates: Requirements 1.4**
    """
    log.info("test_property_3_metadata_completeness", page_id=page.id)

    # All required fields should be non-empty
    assert page.title, "Page title should be non-empty"
    assert page.author, "Page author should be non-empty"
    assert page.created_date, "Page created_date should be non-empty"
    assert page.modified_date, "Page modified_date should be non-empty"
    assert page.url, "Page URL should be non-empty"

    # Additional validation: modified_date should be >= created_date
    assert page.modified_date >= page.created_date, "Modified date should be >= created date"


@given(document_chunk_strategy())
def test_property_10_metadata_storage_completeness(chunk: DocumentChunk):
    """Property 10: Metadata storage completeness.

    For any vector stored in the database, all required metadata fields
    (page_id, page_title, page_url, chunk_index, content) should be present
    and non-empty.

    **Feature: confluence-rag-system, Property 10: Metadata storage completeness**
    **Validates: Requirements 3.2**
    """
    log.info("test_property_10_metadata_storage_completeness", chunk_id=chunk.chunk_id)

    # All required fields should be non-empty
    assert chunk.page_id, "page_id should be non-empty"
    assert chunk.content, "content should be non-empty"
    assert chunk.chunk_index >= 0, "chunk_index should be >= 0"

    # Metadata should contain required fields
    assert "page_title" in chunk.metadata, "metadata should contain page_title"
    assert "page_url" in chunk.metadata, "metadata should contain page_url"
    assert chunk.metadata["page_title"], "page_title in metadata should be non-empty"
    assert chunk.metadata["page_url"], "page_url in metadata should be non-empty"

    # Properties should work correctly
    assert chunk.page_title == chunk.metadata["page_title"]
    assert chunk.page_url == chunk.metadata["page_url"]
