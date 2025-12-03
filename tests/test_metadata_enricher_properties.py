"""Property-based tests for MetadataEnricher.

**Feature: confluence-rag-system**
"""

from datetime import datetime, timezone

import structlog
from hypothesis import assume, given
from hypothesis import strategies as st

from src.models.page import Page
from src.processing.metadata_enricher import MetadataEnricher

log = structlog.stdlib.get_logger()


# Strategy for generating valid Page objects
@st.composite
def page_strategy(draw):
    """Generate valid Page objects for testing."""
    page_id = draw(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        )
    )
    title = draw(st.text(min_size=1, max_size=100))
    space_key = draw(
        st.text(
            min_size=1,
            max_size=10,
            alphabet=st.characters(min_codepoint=65, max_codepoint=90),
        )
    )
    content = draw(st.text(min_size=1, max_size=1000))
    author = draw(st.text(min_size=1, max_size=50))

    # Generate valid datetime objects (naive, then add timezone)
    created_date_naive = draw(
        st.datetimes(min_value=datetime(2020, 1, 1), max_value=datetime(2024, 12, 31))
    )
    created_date = created_date_naive.replace(tzinfo=timezone.utc)

    modified_date_naive = draw(
        st.datetimes(min_value=created_date_naive, max_value=datetime(2024, 12, 31))
    )
    modified_date = modified_date_naive.replace(tzinfo=timezone.utc)

    version = draw(st.integers(min_value=1, max_value=100))

    # Generate a valid ASCII domain name (lowercase letters only)
    domain = draw(st.text(min_size=3, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz"))

    return Page(
        id=page_id,
        title=title,
        space_key=space_key,
        content=content,
        author=author,
        created_date=created_date,
        modified_date=modified_date,
        url=f"https://{domain}.atlassian.net/wiki/spaces/{space_key}/pages/{page_id}",
        version=version,
    )


# Strategy for generating chunk text
chunk_text_strategy = st.text(min_size=1, max_size=1000)

# Strategy for generating chunk indices
chunk_index_strategy = st.integers(min_value=0, max_value=100)


@given(page=page_strategy(), chunk_text=chunk_text_strategy, chunk_index=chunk_index_strategy)
def test_property_8_metadata_preservation_in_embeddings(
    page: Page, chunk_text: str, chunk_index: int
):
    """Property 8: Metadata preservation in embeddings.

    *For any* generated embedding, the associated metadata should contain
    page_title, page_url, and chunk_index fields.

    **Feature: confluence-rag-system, Property 8: Metadata preservation in embeddings**
    **Validates: Requirements 2.4**
    """
    log.info(
        "test_property_8_metadata_preservation_in_embeddings",
        page_id=page.id,
        chunk_index=chunk_index,
    )

    enricher = MetadataEnricher()
    chunk = enricher.enrich_chunk(page, chunk_text, chunk_index)

    # Verify that all required metadata fields are present
    assert "page_title" in chunk.metadata, "metadata should contain page_title"
    assert "page_url" in chunk.metadata, "metadata should contain page_url"
    assert chunk.chunk_index == chunk_index, "chunk_index should be preserved"

    # Verify that metadata values are non-empty
    assert chunk.metadata["page_title"], "page_title in metadata should be non-empty"
    assert chunk.metadata["page_url"], "page_url in metadata should be non-empty"

    # Verify that metadata values match the source page
    assert chunk.metadata["page_title"] == page.title, "page_title should match source page"
    assert chunk.metadata["page_url"] == str(page.url), "page_url should match source page"

    # Verify additional metadata fields are present
    assert "author" in chunk.metadata, "metadata should contain author"
    assert "created_date" in chunk.metadata, "metadata should contain created_date"
    assert "modified_date" in chunk.metadata, "metadata should contain modified_date"
    assert "space_key" in chunk.metadata, "metadata should contain space_key"
    assert "version" in chunk.metadata, "metadata should contain version"

    # Verify that the chunk content is preserved
    assert chunk.content == chunk_text, "chunk content should be preserved"
    assert chunk.page_id == page.id, "page_id should match source page"


@given(page=page_strategy(), num_chunks=st.integers(min_value=1, max_value=20))
def test_property_11_unique_identifier_generation(page: Page, num_chunks: int):
    """Property 11: Unique identifier generation.

    *For any* set of stored chunks, all chunk_ids should be unique and follow
    the format {page_id}_{chunk_index}.

    **Feature: confluence-rag-system, Property 11: Unique identifier generation**
    **Validates: Requirements 3.3**
    """
    log.info(
        "test_property_11_unique_identifier_generation",
        page_id=page.id,
        num_chunks=num_chunks,
    )

    enricher = MetadataEnricher()

    # Generate multiple chunks
    chunk_texts = [f"Chunk {i} content" for i in range(num_chunks)]
    chunks = enricher.enrich_chunks(page, chunk_texts)

    # Verify that we got the expected number of chunks
    assert len(chunks) == num_chunks, f"Expected {num_chunks} chunks, got {len(chunks)}"

    # Collect all chunk_ids
    chunk_ids = [chunk.chunk_id for chunk in chunks]

    # Verify that all chunk_ids are unique
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk_ids should be unique"

    # Verify that each chunk_id follows the format {page_id}_{chunk_index}
    for idx, chunk in enumerate(chunks):
        expected_chunk_id = f"{page.id}_{idx}"
        assert chunk.chunk_id == expected_chunk_id, (
            f"chunk_id should be {expected_chunk_id}, got {chunk.chunk_id}"
        )

        # Verify the format by checking for underscore
        assert "_" in chunk.chunk_id, "chunk_id should contain underscore"

        # Verify that the chunk_id can be split into page_id and chunk_index
        parts = chunk.chunk_id.split("_")
        assert len(parts) >= 2, "chunk_id should have at least 2 parts when split by underscore"

        # The first part(s) should reconstruct the page_id
        # (page_id might contain underscores)
        reconstructed_page_id = "_".join(parts[:-1])
        chunk_index_str = parts[-1]

        assert reconstructed_page_id == page.id, (
            f"page_id in chunk_id should be {page.id}, got {reconstructed_page_id}"
        )
        assert chunk_index_str == str(idx), (
            f"chunk_index in chunk_id should be {idx}, got {chunk_index_str}"
        )


@given(
    page=page_strategy(),
    chunk_texts=st.lists(chunk_text_strategy, min_size=1, max_size=10),
)
def test_property_8_metadata_preservation_multiple_chunks(page: Page, chunk_texts: list[str]):
    """Property 8: Metadata preservation in embeddings (multiple chunks).

    *For any* set of chunks from the same page, all chunks should preserve
    the same page metadata.

    **Feature: confluence-rag-system, Property 8: Metadata preservation in embeddings**
    **Validates: Requirements 2.4**
    """
    log.info(
        "test_property_8_metadata_preservation_multiple_chunks",
        page_id=page.id,
        num_chunks=len(chunk_texts),
    )

    enricher = MetadataEnricher()
    chunks = enricher.enrich_chunks(page, chunk_texts)

    # Verify that we got the expected number of chunks
    assert len(chunks) == len(chunk_texts), "Number of chunks should match input"

    # Verify that all chunks have the same page metadata
    for idx, chunk in enumerate(chunks):
        # Verify required metadata fields
        assert "page_title" in chunk.metadata, f"Chunk {idx}: metadata should contain page_title"
        assert "page_url" in chunk.metadata, f"Chunk {idx}: metadata should contain page_url"
        assert "author" in chunk.metadata, f"Chunk {idx}: metadata should contain author"
        assert "created_date" in chunk.metadata, (
            f"Chunk {idx}: metadata should contain created_date"
        )
        assert "modified_date" in chunk.metadata, (
            f"Chunk {idx}: metadata should contain modified_date"
        )
        assert "space_key" in chunk.metadata, f"Chunk {idx}: metadata should contain space_key"
        assert "version" in chunk.metadata, f"Chunk {idx}: metadata should contain version"

        # Verify metadata values match the source page
        assert chunk.metadata["page_title"] == page.title, f"Chunk {idx}: page_title should match"
        assert chunk.metadata["page_url"] == str(page.url), f"Chunk {idx}: page_url should match"
        assert chunk.metadata["author"] == page.author, f"Chunk {idx}: author should match"
        assert chunk.metadata["created_date"] == page.created_date.isoformat(), (
            f"Chunk {idx}: created_date should match"
        )
        assert chunk.metadata["modified_date"] == page.modified_date.isoformat(), (
            f"Chunk {idx}: modified_date should match"
        )
        assert chunk.metadata["space_key"] == page.space_key, f"Chunk {idx}: space_key should match"
        assert chunk.metadata["version"] == page.version, f"Chunk {idx}: version should match"

        # Verify chunk-specific fields
        assert chunk.page_id == page.id, f"Chunk {idx}: page_id should match"
        assert chunk.chunk_index == idx, f"Chunk {idx}: chunk_index should be {idx}"
        assert chunk.content == chunk_texts[idx], f"Chunk {idx}: content should be preserved"
