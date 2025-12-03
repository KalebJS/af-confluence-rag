"""Property-based tests for query processing.

**Feature: confluence-rag-system**
"""

import shutil
import tempfile

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.models.page import DocumentChunk, SearchResult
from src.processing.embedder import EmbeddingGenerator
from src.query.query_processor import QueryProcessor
from src.storage.vector_store import ChromaStore


# Fixtures
@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma storage."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embedder():
    """Create an EmbeddingGenerator instance."""
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")


@pytest.fixture
def chroma_store(temp_chroma_dir):
    """Create a ChromaStore instance with temporary storage."""
    return ChromaStore(persist_directory=temp_chroma_dir, collection_name="test_query_collection")


@pytest.fixture
def query_processor(embedder, chroma_store):
    """Create a QueryProcessor instance."""
    return QueryProcessor(embedder=embedder, vector_store=chroma_store)


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

query_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=200,
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


@given(
    query=query_strategy,
    top_k=st.integers(min_value=1, max_value=10),
    num_docs=st.integers(min_value=0, max_value=15),
)
@settings(deadline=None, max_examples=20)
def test_property_19_result_count_correctness(
    query: str,
    top_k: int,
    num_docs: int,
):
    """Property 19: Result count correctness

    *For any* search query with parameter top_k, the number of returned results
    should be min(top_k, total_documents_in_database).

    **Validates: Requirements 5.2**
    **Feature: confluence-rag-system, Property 19: Result count correctness**
    """
    # Create a fresh temporary directory and components for this test
    temp_dir = tempfile.mkdtemp()
    try:
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        store = ChromaStore(persist_directory=temp_dir, collection_name="test_result_count")
        processor = QueryProcessor(embedder=embedder, vector_store=store)

        # Add documents to the store
        if num_docs > 0:
            chunks = []
            embeddings = []

            for i in range(num_docs):
                chunk = generate_document_chunk(
                    page_id=f"page_{i}",
                    chunk_index=0,
                    content=f"Test content {i}",
                    metadata={
                        "page_title": f"Page {i}",
                        "page_url": f"https://example.com/page/{i}",
                        "author": "test_author",
                        "modified_date": "2024-01-01",
                    },
                )
                chunks.append(chunk)
                embeddings.append(embedder.generate_embedding(chunk.content))

            store.add_documents(chunks, embeddings)

        # Process query
        results = processor.process_query(query, top_k=top_k)

        # Property: result count should be min(top_k, num_docs)
        expected_count = min(top_k, num_docs)
        assert len(results) == expected_count, (
            f"Expected {expected_count} results (min({top_k}, {num_docs})), but got {len(results)}"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@given(
    query=query_strategy,
    num_docs=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None, max_examples=20)
def test_property_20_search_result_completeness(
    query: str,
    num_docs: int,
):
    """Property 20: Search result completeness

    *For any* search result, it should contain non-empty values for content,
    page_title, page_url, and similarity_score.

    **Validates: Requirements 5.3**
    **Feature: confluence-rag-system, Property 20: Search result completeness**
    """
    # Create a fresh temporary directory and components for this test
    temp_dir = tempfile.mkdtemp()
    try:
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        store = ChromaStore(persist_directory=temp_dir, collection_name="test_completeness")
        processor = QueryProcessor(embedder=embedder, vector_store=store)

        # Add documents to the store
        chunks = []
        embeddings = []

        for i in range(num_docs):
            chunk = generate_document_chunk(
                page_id=f"page_{i}",
                chunk_index=0,
                content=f"Test content for document {i}",
                metadata={
                    "page_title": f"Document Title {i}",
                    "page_url": f"https://example.com/page/{i}",
                    "author": "test_author",
                    "modified_date": "2024-01-01",
                },
            )
            chunks.append(chunk)
            embeddings.append(embedder.generate_embedding(chunk.content))

        store.add_documents(chunks, embeddings)

        # Process query
        results = processor.process_query(query, top_k=10)

        # Property: all results should have complete data
        for result in results:
            assert result.content, "Result content should not be empty"
            assert result.page_title, "Result page_title should not be empty"
            assert result.page_url, "Result page_url should not be empty"
            assert 0.0 <= result.similarity_score <= 1.0, (
                f"Similarity score should be in [0, 1], got {result.similarity_score}"
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@given(
    query=query_strategy,
    num_docs=st.integers(min_value=2, max_value=10),
)
@settings(deadline=None, max_examples=20)
def test_property_21_result_ranking_order(
    query: str,
    num_docs: int,
):
    """Property 21: Result ranking order

    *For any* list of search results, each result's similarity_score should be
    greater than or equal to the next result's similarity_score (descending order).

    **Validates: Requirements 5.4**
    **Feature: confluence-rag-system, Property 21: Result ranking order**
    """
    # Create a fresh temporary directory and components for this test
    temp_dir = tempfile.mkdtemp()
    try:
        embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        store = ChromaStore(persist_directory=temp_dir, collection_name="test_ranking")
        processor = QueryProcessor(embedder=embedder, vector_store=store)

        # Add documents to the store
        chunks = []
        embeddings = []

        for i in range(num_docs):
            chunk = generate_document_chunk(
                page_id=f"page_{i}",
                chunk_index=0,
                content=f"Test content for document {i}",
                metadata={
                    "page_title": f"Document Title {i}",
                    "page_url": f"https://example.com/page/{i}",
                    "author": "test_author",
                    "modified_date": "2024-01-01",
                },
            )
            chunks.append(chunk)
            embeddings.append(embedder.generate_embedding(chunk.content))

        store.add_documents(chunks, embeddings)

        # Process query
        results = processor.process_query(query, top_k=num_docs)

        # Property: results should be in descending order by similarity score
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i + 1].similarity_score, (
                f"Results not in descending order: "
                f"result[{i}].score={results[i].similarity_score} < "
                f"result[{i + 1}].score={results[i + 1].similarity_score}"
            )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
