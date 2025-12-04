"""Property-based tests for QueryProcessor.

Feature: langchain-abstraction-refactor
"""

import tempfile
from pathlib import Path

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.models.config import ProcessingConfig, VectorStoreConfig
from src.query.query_processor import QueryProcessor

log = structlog.stdlib.get_logger()


class MockEmbeddings(Embeddings):
    """Mock embeddings implementation for testing dependency injection."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.embed_query_called = False
        self.embed_documents_called = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for documents."""
        self.embed_documents_called = True
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        """Generate mock embedding for query."""
        self.embed_query_called = True
        return [0.1] * self.dimension


class MockVectorStore(VectorStore):
    """Mock vector store implementation for testing dependency injection."""

    def __init__(self, embeddings: Embeddings):
        self._embeddings = embeddings
        self.add_documents_called = False
        self.similarity_search_called = False
        self.similarity_search_with_score_called = False
        self._documents: list[Document] = []

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None, **kwargs
    ) -> list[str]:
        """Add texts to mock store."""
        self.add_documents_called = True
        return [f"id_{i}" for i in range(len(texts))]

    def add_documents(self, documents: list[Document], **kwargs) -> list[str]:
        """Add documents to mock store."""
        self.add_documents_called = True
        self._documents.extend(documents)
        return [f"id_{i}" for i in range(len(documents))]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        """Mock similarity search."""
        self.similarity_search_called = True
        # Return mock documents with required metadata
        return [
            Document(
                page_content=f"Mock result {i}",
                metadata={
                    "chunk_id": f"page1_{i}",
                    "page_id": "page1",
                    "page_title": "Test Page",
                    "page_url": "https://example.com/page1",
                    "chunk_index": i,
                },
            )
            for i in range(min(k, 3))
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs
    ) -> list[tuple[Document, float]]:
        """Mock similarity search with scores."""
        self.similarity_search_with_score_called = True
        docs = self.similarity_search(query, k, **kwargs)
        return [(doc, 0.9 - i * 0.1) for i, doc in enumerate(docs)]

    @classmethod
    def from_texts(
        cls, texts: list[str], embedding: Embeddings, metadatas: list[dict] | None = None, **kwargs
    ):
        """Create mock vector store from texts."""
        instance = cls(embedding)
        instance.add_texts(texts, metadatas, **kwargs)
        return instance


@settings(deadline=None, max_examples=10)
@given(st.integers(min_value=128, max_value=768))
def test_property_4_dependency_injection_override_embeddings(embedding_dimension: int):
    """Property 4: Dependency Injection Override (Embeddings).

    For any custom Embeddings implementation provided via constructor, the
    QueryProcessor SHALL use the provided instance instead of creating one
    from configuration.

    **Feature: langchain-abstraction-refactor, Property 4: Dependency Injection Override**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    log.info("test_property_4_embeddings_override", embedding_dimension=embedding_dimension)

    # Create custom embeddings instance
    custom_embeddings = MockEmbeddings(dimension=embedding_dimension)

    # Create mock vector store
    custom_vector_store = MockVectorStore(custom_embeddings)

    # Create QueryProcessor with custom implementations
    processor = QueryProcessor(
        embeddings=custom_embeddings,
        vector_store=custom_vector_store,
    )

    # Verify the processor is using the custom embeddings
    assert processor.embeddings is custom_embeddings, (
        "QueryProcessor should use provided embeddings instance"
    )
    assert processor.vector_store is custom_vector_store, (
        "QueryProcessor should use provided vector store instance"
    )

    # Process a query to verify the custom implementations are actually used
    results = processor.process_query("test query", top_k=5)

    # Verify the custom vector store's method was called
    assert custom_vector_store.similarity_search_with_score_called, (
        "Custom vector store's similarity_search_with_score should be called"
    )

    log.info(
        "embeddings_override_verified",
        embedding_dimension=embedding_dimension,
        results_count=len(results),
    )


def test_property_4_dependency_injection_override_vector_store():
    """Property 4: Dependency Injection Override (VectorStore).

    For any custom VectorStore implementation provided via constructor, the
    QueryProcessor SHALL use the provided instance instead of creating one
    from configuration.

    **Feature: langchain-abstraction-refactor, Property 4: Dependency Injection Override**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    log.info("test_property_4_vector_store_override")

    # Create custom implementations
    custom_embeddings = MockEmbeddings()
    custom_vector_store = MockVectorStore(custom_embeddings)

    # Create QueryProcessor with custom implementations
    processor = QueryProcessor(
        embeddings=custom_embeddings,
        vector_store=custom_vector_store,
    )

    # Verify the processor is using the custom vector store
    assert processor.vector_store is custom_vector_store, (
        "QueryProcessor should use provided vector store instance"
    )

    # Process a query to verify the custom vector store is actually used
    results = processor.process_query("test query", top_k=3)

    # Verify the custom vector store's method was called
    assert custom_vector_store.similarity_search_with_score_called, (
        "Custom vector store's similarity_search_with_score should be called"
    )

    # Verify results were returned
    assert len(results) > 0, "Should return results from custom vector store"

    log.info("vector_store_override_verified", results_count=len(results))


def test_property_4_dependency_injection_backward_compatibility_with_config():
    """Property 4: Dependency Injection Backward Compatibility.

    When no custom implementations are provided, the QueryProcessor SHALL
    create instances from configuration using the provider module, maintaining
    backward compatibility.

    **Feature: langchain-abstraction-refactor, Property 4: Dependency Injection Override**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    log.info("test_property_4_backward_compatibility")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configurations
        processing_config = ProcessingConfig(embedding_model="all-MiniLM-L6-v2")
        vector_store_config = VectorStoreConfig(
            collection_name="test_collection", persist_directory=temp_dir
        )

        # Create QueryProcessor using configuration (no custom implementations)
        processor = QueryProcessor(
            processing_config=processing_config,
            vector_store_config=vector_store_config,
        )

        # Verify instances were created
        assert processor.embeddings is not None, "Embeddings should be created from config"
        assert processor.vector_store is not None, "VectorStore should be created from config"

        # Verify they are the correct types
        assert isinstance(processor.embeddings, Embeddings), (
            "Should create Embeddings instance from config"
        )
        assert isinstance(processor.vector_store, VectorStore), (
            "Should create VectorStore instance from config"
        )

        log.info("backward_compatibility_verified")


def test_property_4_dependency_injection_requires_either_instance_or_config():
    """Property 4: Dependency Injection Validation.

    The QueryProcessor SHALL raise ValueError if neither custom instance nor
    configuration is provided for embeddings or vector store.

    **Feature: langchain-abstraction-refactor, Property 4: Dependency Injection Override**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    log.info("test_property_4_validation")

    # Test missing embeddings
    try:
        QueryProcessor(
            vector_store=MockVectorStore(MockEmbeddings()),
        )
        assert False, "Should raise ValueError when embeddings not provided"
    except ValueError as e:
        assert "embeddings" in str(e).lower() or "processing_config" in str(e).lower(), (
            f"Error should mention embeddings or processing_config: {e}"
        )
        log.info("embeddings_validation_passed", error=str(e))

    # Test missing vector store
    try:
        QueryProcessor(
            embeddings=MockEmbeddings(),
        )
        assert False, "Should raise ValueError when vector_store not provided"
    except ValueError as e:
        assert "vector_store" in str(e).lower() or "vector_store_config" in str(e).lower(), (
            f"Error should mention vector_store or vector_store_config: {e}"
        )
        log.info("vector_store_validation_passed", error=str(e))


@settings(deadline=None, max_examples=5)
@given(st.integers(min_value=1, max_value=10))
def test_property_4_dependency_injection_custom_implementations_work_correctly(top_k: int):
    """Property 4: Dependency Injection Functional Verification.

    For any custom implementations provided via dependency injection, the
    QueryProcessor SHALL successfully process queries and return results.

    **Feature: langchain-abstraction-refactor, Property 4: Dependency Injection Override**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """
    log.info("test_property_4_functional_verification", top_k=top_k)

    # Create custom implementations
    custom_embeddings = MockEmbeddings()
    custom_vector_store = MockVectorStore(custom_embeddings)

    # Create QueryProcessor with custom implementations
    processor = QueryProcessor(
        embeddings=custom_embeddings,
        vector_store=custom_vector_store,
    )

    # Process a query
    results = processor.process_query("test query", top_k=top_k)

    # Verify results
    assert isinstance(results, list), "Should return a list of results"
    assert len(results) <= top_k, f"Should return at most {top_k} results"
    assert all(hasattr(r, "chunk_id") for r in results), "Results should have chunk_id"
    assert all(hasattr(r, "page_id") for r in results), "Results should have page_id"
    assert all(hasattr(r, "similarity_score") for r in results), (
        "Results should have similarity_score"
    )

    log.info("functional_verification_passed", results_count=len(results), top_k=top_k)
