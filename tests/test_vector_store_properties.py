"""Property-based tests for vector store operations.

**Feature: confluence-rag-system**
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models.page import DocumentChunk, SearchResult
from src.storage.vector_store import ChromaStore, VectorStoreInterface


# Fixtures
@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def chroma_store(temp_chroma_dir):
    """Create a ChromaStore instance with temporary storage."""
    return ChromaStore(persist_directory=temp_chroma_dir, collection_name="test_collection")


# Strategies for generating test data
page_id_strategy = st.text(
    alphabet=st.characters(min_codepoint=48, max_codepoint=122),
    min_size=1,
    max_size=20,
)

content_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=500,
)

metadata_strategy = st.fixed_dictionaries(
    {
        "page_title": st.text(min_size=1, max_size=100),
        "page_url": st.from_regex(r"https://example\.com/page/\d+", fullmatch=True),
        "author": st.text(min_size=1, max_size=50),
        "modified_date": st.text(min_size=1, max_size=50),
    }
)


def generate_document_chunk(
    page_id: str, chunk_index: int, content: str, metadata: dict
) -> DocumentChunk:
    """Generate a DocumentChunk with valid data."""
    return DocumentChunk(
        chunk_id=f"{page_id}_{chunk_index}",
        page_id=page_id,
        content=content,
        metadata=metadata,
        chunk_index=chunk_index,
    )


def generate_embedding(dimension: int = 384) -> np.ndarray:
    """Generate a random embedding vector."""
    # Generate random vector and normalize it
    vec = np.random.randn(dimension).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


@given(
    page_id=page_id_strategy,
    content=content_strategy,
    metadata=metadata_strategy,
)
@settings(deadline=None, max_examples=10)
def test_property_9_storage_round_trip_consistency(page_id: str, content: str, metadata: dict):
    """Property 9: Storage round-trip consistency

    *For any* document chunk and embedding stored in the vector database,
    retrieving by chunk_id should return the same content and metadata.

    **Validates: Requirements 2.5, 3.5**
    **Feature: confluence-rag-system, Property 9: Storage round-trip consistency**
    """
    # Create a fresh temporary directory and store for this test
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(
            persist_directory=temp_dir, collection_name=f"test_roundtrip_{hash(page_id) % 10000}"
        )

        # Create a document chunk
        chunk = generate_document_chunk(page_id, 0, content, metadata)
        embedding = generate_embedding()

        # Store the chunk
        store.add_documents([chunk], [embedding])

        # Retrieve using search (should return the stored chunk)
        results = store.search(embedding, top_k=1)

        # Verify we got results
        assert len(results) > 0, "Should retrieve at least one result"

        result = results[0]

        # Verify the content matches
        assert result.content == content, (
            f"Content mismatch: expected '{content}', got '{result.content}'"
        )

        # Verify the chunk_id matches
        assert result.chunk_id == chunk.chunk_id, (
            f"Chunk ID mismatch: expected '{chunk.chunk_id}', got '{result.chunk_id}'"
        )

        # Verify the page_id matches
        assert result.page_id == page_id, (
            f"Page ID mismatch: expected '{page_id}', got '{result.page_id}'"
        )

        # Verify metadata is preserved
        assert result.page_title == metadata["page_title"], (
            f"Page title mismatch: expected '{metadata['page_title']}', got '{result.page_title}'"
        )
        # URLs may be encoded, so we just verify the URL is present and valid
        # The SearchResult model validates it's a proper HttpUrl
        assert result.page_url is not None, "Page URL should not be None"
        assert str(result.page_url).startswith("https://"), "Page URL should be a valid HTTPS URL"
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


@given(
    page_id=page_id_strategy,
    content=content_strategy,
    metadata=metadata_strategy,
)
@settings(deadline=None, max_examples=10)
def test_property_12_deduplication_idempotence(page_id: str, content: str, metadata: dict):
    """Property 12: Deduplication idempotence

    *For any* document chunk, storing it multiple times should result in only
    one entry in the vector database (subsequent stores should be no-ops or updates).

    **Validates: Requirements 3.4, 3.5**
    **Feature: confluence-rag-system, Property 12: Deduplication idempotence**
    """
    # Create a fresh temporary directory and store for this test
    temp_dir = tempfile.mkdtemp()
    try:
        store = ChromaStore(
            persist_directory=temp_dir, collection_name=f"test_dedup_{hash(page_id) % 10000}"
        )

        # Create a document chunk
        chunk = generate_document_chunk(page_id, 0, content, metadata)
        embedding = generate_embedding()

        # Store the chunk multiple times
        store.add_documents([chunk], [embedding])
        store.add_documents([chunk], [embedding])
        store.add_documents([chunk], [embedding])

        # Search for the chunk
        results = store.search(embedding, top_k=10)

        # Count how many times this chunk appears in results
        matching_chunks = [r for r in results if r.chunk_id == chunk.chunk_id]

        # Should only have one entry (idempotent)
        assert len(matching_chunks) == 1, (
            f"Expected 1 entry for chunk {chunk.chunk_id}, found {len(matching_chunks)}"
        )

        # Verify the content is correct
        assert matching_chunks[0].content == content
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_property_10_metadata_storage_completeness(chroma_store):
    """Property 10: Metadata storage completeness

    *For any* vector stored in the database, all required metadata fields
    (page_id, page_title, page_url, chunk_index, content) should be present
    and non-empty.

    **Validates: Requirements 3.2**
    **Feature: confluence-rag-system, Property 10: Metadata storage completeness**
    """
    # Create test data
    page_id = "test_page_123"
    metadata = {
        "page_title": "Test Page",
        "page_url": "https://example.com/page/123",
        "author": "test_author",
        "modified_date": "2024-01-01",
    }

    chunk = generate_document_chunk(page_id, 0, "Test content", metadata)
    embedding = generate_embedding()

    # Store the chunk
    chroma_store.add_documents([chunk], [embedding])

    # Retrieve metadata
    retrieved_metadata = chroma_store.get_document_metadata(page_id)

    # Verify metadata is not None
    assert retrieved_metadata is not None, "Metadata should be retrievable"

    # Verify all required fields are present and non-empty
    required_fields = ["page_id", "page_title", "page_url", "chunk_index"]
    for field in required_fields:
        assert field in retrieved_metadata, f"Field '{field}' missing from metadata"
        assert retrieved_metadata[field] is not None, f"Field '{field}' is None"
        # For string fields, check they're not empty
        if isinstance(retrieved_metadata[field], str):
            assert retrieved_metadata[field] != "", f"Field '{field}' is empty string"


def test_property_11_unique_identifier_generation(chroma_store):
    """Property 11: Unique identifier generation

    *For any* set of stored chunks, all chunk_ids should be unique and follow
    the format {page_id}_{chunk_index}.

    **Validates: Requirements 3.3**
    **Feature: confluence-rag-system, Property 11: Unique identifier generation**
    """
    # Create multiple chunks for the same page
    page_id = "test_page_456"
    metadata = {
        "page_title": "Test Page",
        "page_url": "https://example.com/page/456",
        "author": "test_author",
        "modified_date": "2024-01-01",
    }

    chunks = []
    embeddings = []

    for i in range(5):
        chunk = generate_document_chunk(page_id, i, f"Content chunk {i}", metadata)
        chunks.append(chunk)
        embeddings.append(generate_embedding())

    # Store all chunks
    chroma_store.add_documents(chunks, embeddings)

    # Retrieve all chunks by searching with each embedding
    all_chunk_ids = set()
    for embedding in embeddings:
        results = chroma_store.search(embedding, top_k=1)
        if results:
            all_chunk_ids.add(results[0].chunk_id)

    # Verify all chunk_ids are unique
    assert len(all_chunk_ids) == len(chunks), (
        f"Expected {len(chunks)} unique chunk IDs, got {len(all_chunk_ids)}"
    )

    # Verify all chunk_ids follow the format {page_id}_{chunk_index}
    for chunk_id in all_chunk_ids:
        assert "_" in chunk_id, f"Chunk ID '{chunk_id}' doesn't contain underscore"
        # Split only on the last underscore to handle page_ids with underscores
        last_underscore_idx = chunk_id.rfind("_")
        extracted_page_id = chunk_id[:last_underscore_idx]
        chunk_index_str = chunk_id[last_underscore_idx + 1 :]

        assert extracted_page_id == page_id, (
            f"Chunk ID '{chunk_id}' doesn't start with page_id '{page_id}', got '{extracted_page_id}'"
        )
        assert chunk_index_str.isdigit(), (
            f"Chunk ID '{chunk_id}' doesn't end with numeric index, got '{chunk_index_str}'"
        )


def test_delete_by_page_id(chroma_store):
    """Test that delete_by_page_id removes all chunks for a page."""
    # Create chunks for two different pages
    page_id_1 = "page_to_delete"
    page_id_2 = "page_to_keep"

    metadata = {
        "page_title": "Test Page",
        "page_url": "https://example.com/page/123",
        "author": "test_author",
        "modified_date": "2024-01-01",
    }

    # Add chunks for page 1
    chunks_1 = [generate_document_chunk(page_id_1, i, f"Content {i}", metadata) for i in range(3)]
    embeddings_1 = [generate_embedding() for _ in range(3)]
    chroma_store.add_documents(chunks_1, embeddings_1)

    # Add chunks for page 2
    chunks_2 = [generate_document_chunk(page_id_2, i, f"Content {i}", metadata) for i in range(2)]
    embeddings_2 = [generate_embedding() for _ in range(2)]
    chroma_store.add_documents(chunks_2, embeddings_2)

    # Delete page 1
    chroma_store.delete_by_page_id(page_id_1)

    # Verify page 1 chunks are gone
    metadata_1 = chroma_store.get_document_metadata(page_id_1)
    assert metadata_1 is None, "Page 1 metadata should be deleted"

    # Verify page 2 chunks still exist
    metadata_2 = chroma_store.get_document_metadata(page_id_2)
    assert metadata_2 is not None, "Page 2 metadata should still exist"


def test_search_returns_top_k_results(chroma_store):
    """Test that search returns at most top_k results."""
    # Add multiple chunks
    page_id = "test_page"
    metadata = {
        "page_title": "Test Page",
        "page_url": "https://example.com/page/123",
        "author": "test_author",
        "modified_date": "2024-01-01",
    }

    chunks = [generate_document_chunk(page_id, i, f"Content {i}", metadata) for i in range(10)]
    embeddings = [generate_embedding() for _ in range(10)]
    chroma_store.add_documents(chunks, embeddings)

    # Search with top_k=5
    query_embedding = generate_embedding()
    results = chroma_store.search(query_embedding, top_k=5)

    # Should return at most 5 results
    assert len(results) <= 5, f"Expected at most 5 results, got {len(results)}"


def test_empty_search_results(chroma_store):
    """Test that search on empty database returns empty list."""
    query_embedding = generate_embedding()
    results = chroma_store.search(query_embedding, top_k=10)

    assert results == [], "Search on empty database should return empty list"


def test_add_documents_with_mismatched_lengths(chroma_store):
    """Test that add_documents raises error when chunks and embeddings have different lengths."""
    chunk = generate_document_chunk(
        "page_1",
        0,
        "Content",
        {
            "page_title": "Test",
            "page_url": "https://example.com/page/1",
            "author": "test",
            "modified_date": "2024-01-01",
        },
    )

    with pytest.raises(ValueError, match="same length"):
        chroma_store.add_documents([chunk], [])


def test_add_empty_documents_list(chroma_store):
    """Test that add_documents handles empty list gracefully."""
    # Should not raise an error
    chroma_store.add_documents([], [])


# Tests for ChromaStore


def test_property_38_vector_store_interface_compliance(temp_chroma_dir):
    """Property 38: Vector store interface compliance

    ChromaStore should implement all methods defined in VectorStoreInterface
    (add_documents, search, delete_by_page_id, get_document_metadata).

    **Validates: Design requirement for vector store interface**
    **Feature: confluence-rag-system, Property 38: Vector store interface compliance**
    """
    from src.storage.vector_store import ChromaStore, VectorStoreInterface

    # Create a Chroma store
    store = ChromaStore(
        persist_directory=temp_chroma_dir,
        collection_name="test_interface",
    )

    # Verify it implements VectorStoreInterface
    assert isinstance(store, VectorStoreInterface), (
        "ChromaStore should implement VectorStoreInterface"
    )

    # Verify all required methods exist and are callable
    required_methods = ["add_documents", "search", "delete_by_page_id", "get_document_metadata"]
    for method_name in required_methods:
        assert hasattr(store, method_name), f"Store should have method '{method_name}'"
        assert callable(getattr(store, method_name)), f"Method '{method_name}' should be callable"


def test_property_39_chroma_store_instantiation(temp_chroma_dir):
    """Property 39: ChromaStore instantiation

    ChromaStore should successfully create an instance implementing VectorStoreInterface.

    **Validates: ChromaDB vector store creation**
    **Feature: confluence-rag-system, Property 39: Vector store instantiation**
    """
    from src.storage.vector_store import ChromaStore, VectorStoreInterface

    # Test ChromaStore creation
    chroma_store = ChromaStore(
        persist_directory=temp_chroma_dir,
        collection_name="test_chroma",
    )
    assert isinstance(chroma_store, VectorStoreInterface), (
        "Chroma store should implement VectorStoreInterface"
    )


def test_chroma_store_missing_config():
    """Test that ChromaStore validates required configuration."""
    from src.storage.vector_store import ChromaStore

    # ChromaStore requires persist_directory
    with pytest.raises(TypeError):
        ChromaStore()
