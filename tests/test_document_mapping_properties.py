"""Property-based tests for document mapping functions.

Feature: langchain-abstraction-refactor
"""

from datetime import datetime, timezone

import structlog
from hypothesis import given
from hypothesis import strategies as st

from src.models.page import (
    DocumentChunk,
    from_langchain_document,
    from_langchain_documents,
    to_langchain_document,
    to_langchain_documents,
)

log = structlog.stdlib.get_logger()


# Hypothesis strategies for generating test data
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


@given(document_chunk_strategy())
def test_property_7_document_mapping_round_trip(chunk: DocumentChunk):
    """Property 7: Document Mapping Round-Trip.

    For any DocumentChunk, converting it to a LangChain Document and back
    SHALL produce an equivalent DocumentChunk with the same content, metadata,
    chunk_id, page_id, and chunk_index.

    **Feature: langchain-abstraction-refactor, Property 7: Document Mapping Round-Trip**
    **Validates: Requirements 11.3**
    """
    log.info("test_property_7_document_mapping_round_trip", chunk_id=chunk.chunk_id)

    # Convert to LangChain Document
    langchain_doc = to_langchain_document(chunk)

    # Verify LangChain Document has correct structure
    assert langchain_doc.page_content == chunk.content, "Content should be preserved"
    assert "chunk_id" in langchain_doc.metadata, "chunk_id should be in metadata"
    assert "page_id" in langchain_doc.metadata, "page_id should be in metadata"
    assert "chunk_index" in langchain_doc.metadata, "chunk_index should be in metadata"

    # Convert back to DocumentChunk
    restored_chunk = from_langchain_document(langchain_doc, chunk.chunk_id, chunk.page_id)

    # Verify all fields are preserved
    assert restored_chunk.chunk_id == chunk.chunk_id, "chunk_id should be preserved"
    assert restored_chunk.page_id == chunk.page_id, "page_id should be preserved"
    assert restored_chunk.content == chunk.content, "content should be preserved"
    assert restored_chunk.chunk_index == chunk.chunk_index, "chunk_index should be preserved"

    # Verify metadata is preserved (excluding chunk_id, page_id, chunk_index which are separate fields)
    for key, value in chunk.metadata.items():
        assert key in restored_chunk.metadata, f"metadata key '{key}' should be preserved"
        assert restored_chunk.metadata[key] == value, f"metadata value for '{key}' should be preserved"

    # Verify no extra metadata was added
    for key in restored_chunk.metadata:
        assert key in chunk.metadata, f"No extra metadata key '{key}' should be added"


@given(st.lists(document_chunk_strategy(), min_size=1, max_size=10))
def test_property_7_batch_document_mapping_round_trip(chunks: list[DocumentChunk]):
    """Property 7: Batch Document Mapping Round-Trip.

    For any list of DocumentChunks, batch converting them to LangChain Documents
    and back SHALL produce equivalent DocumentChunks with all fields preserved.

    **Feature: langchain-abstraction-refactor, Property 7: Document Mapping Round-Trip**
    **Validates: Requirements 11.3**
    """
    log.info("test_property_7_batch_document_mapping_round_trip", chunk_count=len(chunks))

    # Convert to LangChain Documents
    langchain_docs = to_langchain_documents(chunks)

    # Verify we got the same number of documents
    assert len(langchain_docs) == len(chunks), "Number of documents should be preserved"

    # Extract chunk_ids and page_ids for conversion back
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    page_ids = [chunk.page_id for chunk in chunks]

    # Convert back to DocumentChunks
    restored_chunks = from_langchain_documents(langchain_docs, chunk_ids, page_ids)

    # Verify we got the same number of chunks
    assert len(restored_chunks) == len(chunks), "Number of chunks should be preserved"

    # Verify each chunk is preserved
    for original, restored in zip(chunks, restored_chunks):
        assert restored.chunk_id == original.chunk_id, "chunk_id should be preserved"
        assert restored.page_id == original.page_id, "page_id should be preserved"
        assert restored.content == original.content, "content should be preserved"
        assert restored.chunk_index == original.chunk_index, "chunk_index should be preserved"

        # Verify metadata is preserved
        for key, value in original.metadata.items():
            assert key in restored.metadata, f"metadata key '{key}' should be preserved"
            assert restored.metadata[key] == value, f"metadata value for '{key}' should be preserved"


@given(document_chunk_strategy())
def test_metadata_fields_in_langchain_document(chunk: DocumentChunk):
    """Verify that all metadata fields are included in LangChain Document.

    For any DocumentChunk, the converted LangChain Document should contain
    all original metadata plus chunk_id, page_id, and chunk_index.

    **Feature: langchain-abstraction-refactor, Property 7: Document Mapping Round-Trip**
    **Validates: Requirements 11.3**
    """
    log.info("test_metadata_fields_in_langchain_document", chunk_id=chunk.chunk_id)

    # Convert to LangChain Document
    langchain_doc = to_langchain_document(chunk)

    # Verify all original metadata is present
    for key, value in chunk.metadata.items():
        assert key in langchain_doc.metadata, f"Original metadata key '{key}' should be present"
        assert langchain_doc.metadata[key] == value, f"Original metadata value for '{key}' should be preserved"

    # Verify chunk-specific fields are added
    assert langchain_doc.metadata["chunk_id"] == chunk.chunk_id
    assert langchain_doc.metadata["page_id"] == chunk.page_id
    assert langchain_doc.metadata["chunk_index"] == chunk.chunk_index


@given(st.lists(document_chunk_strategy(), min_size=0, max_size=5))
def test_batch_conversion_length_mismatch_error(chunks: list[DocumentChunk]):
    """Verify that batch conversion raises error on length mismatch.

    For any list of DocumentChunks, if the lengths of docs, chunk_ids, and
    page_ids don't match, from_langchain_documents should raise ValueError.

    **Feature: langchain-abstraction-refactor, Property 7: Document Mapping Round-Trip**
    **Validates: Requirements 11.3**
    """
    log.info("test_batch_conversion_length_mismatch_error", chunk_count=len(chunks))

    if len(chunks) == 0:
        # Skip test for empty list
        return

    # Convert to LangChain Documents
    langchain_docs = to_langchain_documents(chunks)

    # Test with mismatched chunk_ids length
    try:
        from_langchain_documents(langchain_docs, [], [chunk.page_id for chunk in chunks])
        assert False, "Should raise ValueError for mismatched chunk_ids length"
    except ValueError as e:
        assert "Length mismatch" in str(e)

    # Test with mismatched page_ids length
    try:
        from_langchain_documents(langchain_docs, [chunk.chunk_id for chunk in chunks], [])
        assert False, "Should raise ValueError for mismatched page_ids length"
    except ValueError as e:
        assert "Length mismatch" in str(e)
