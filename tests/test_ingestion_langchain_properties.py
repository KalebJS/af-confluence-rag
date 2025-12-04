"""Property-based tests for IngestionService with LangChain abstractions.

**Feature: langchain-abstraction-refactor**
"""

import tempfile
from unittest.mock import Mock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.ingestion.ingestion_service import IngestionService
from src.models.config import ProcessingConfig, VectorStoreConfig
from src.processing.chunker import DocumentChunker
from src.providers import get_embeddings, get_vector_store


# Strategies
model_name_strategy = st.sampled_from([
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
])


@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_property_1_embeddings_interface_compliance(model_name: str):
    """Property 1: Embeddings Interface Compliance

    *For any* text input, when the system generates embeddings, the embeddings
    instance SHALL be of type `langchain_core.embeddings.Embeddings` and SHALL
    support both `embed_documents` and `embed_query` methods.

    **Validates: Requirements 1.1, 1.4**
    **Feature: langchain-abstraction-refactor, Property 1: Embeddings Interface Compliance**
    """
    # Create embeddings instance via provider module
    embeddings = get_embeddings(model_name)
    
    # Verify instance type
    assert isinstance(embeddings, Embeddings), (
        f"Expected Embeddings instance, got {type(embeddings)}"
    )
    
    # Verify required methods exist and are callable
    assert hasattr(embeddings, 'embed_documents'), (
        "Embeddings instance must have embed_documents method"
    )
    assert callable(embeddings.embed_documents), (
        "embed_documents must be callable"
    )
    
    assert hasattr(embeddings, 'embed_query'), (
        "Embeddings instance must have embed_query method"
    )
    assert callable(embeddings.embed_query), (
        "embed_query must be callable"
    )
    
    # Test that methods work with sample text
    test_texts = ["Hello world", "Test document"]
    test_query = "Sample query"
    
    # Test embed_documents
    doc_embeddings = embeddings.embed_documents(test_texts)
    assert isinstance(doc_embeddings, list), (
        "embed_documents should return a list"
    )
    assert len(doc_embeddings) == len(test_texts), (
        f"Expected {len(test_texts)} embeddings, got {len(doc_embeddings)}"
    )
    assert all(isinstance(emb, list) for emb in doc_embeddings), (
        "Each embedding should be a list of floats"
    )
    
    # Test embed_query
    query_embedding = embeddings.embed_query(test_query)
    assert isinstance(query_embedding, list), (
        "embed_query should return a list"
    )
    assert len(query_embedding) > 0, (
        "Query embedding should not be empty"
    )
    assert all(isinstance(val, float) for val in query_embedding), (
        "Query embedding should contain floats"
    )
    
    # Verify dimensionality consistency
    assert len(query_embedding) == len(doc_embeddings[0]), (
        "Query and document embeddings should have same dimensionality"
    )


@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_ingestion_service_uses_embeddings_interface(model_name: str):
    """Test that IngestionService properly uses Embeddings interface.
    
    This test verifies that when IngestionService is created with config,
    it uses the Embeddings interface from the provider module.
    
    **Validates: Requirements 1.1, 1.4**
    """
    # Create mocks
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    # Create configs
    processing_config = ProcessingConfig(embedding_model=model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_config = VectorStoreConfig(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Create IngestionService with configs (should use provider module)
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            processing_config=processing_config,
            vector_store_config=vector_store_config
        )
        
        # Verify that the service has embeddings instance
        assert hasattr(service, '_embeddings'), (
            "IngestionService should have _embeddings attribute"
        )
        
        # Verify it's an Embeddings instance
        assert isinstance(service._embeddings, Embeddings), (
            f"Expected Embeddings instance, got {type(service._embeddings)}"
        )
        
        # Verify required methods are available
        assert hasattr(service._embeddings, 'embed_documents'), (
            "Embeddings instance must have embed_documents method"
        )
        assert hasattr(service._embeddings, 'embed_query'), (
            "Embeddings instance must have embed_query method"
        )


@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_ingestion_service_accepts_custom_embeddings(model_name: str):
    """Test that IngestionService accepts custom Embeddings via dependency injection.
    
    This test verifies that when a custom Embeddings instance is provided,
    the service uses it instead of creating one from config.
    
    **Validates: Requirements 4.1, 4.2**
    """
    # Create mocks
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    # Create custom embeddings instance
    custom_embeddings = get_embeddings(model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create custom vector store
        custom_vector_store = get_vector_store(
            embeddings=custom_embeddings,
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Create IngestionService with custom instances (no configs needed)
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embeddings=custom_embeddings,
            vector_store=custom_vector_store
        )
        
        # Verify that the service uses the provided embeddings
        assert service._embeddings is custom_embeddings, (
            "IngestionService should use provided embeddings instance"
        )
        
        # Verify it's an Embeddings instance
        assert isinstance(service._embeddings, Embeddings), (
            f"Expected Embeddings instance, got {type(service._embeddings)}"
        )


def test_ingestion_service_requires_config_when_no_embeddings():
    """Test that IngestionService raises error when neither embeddings nor config provided."""
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_config = VectorStoreConfig(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Should raise ValueError when embeddings is None and processing_config is None
        with pytest.raises(ValueError, match="processing_config is required"):
            IngestionService(
                confluence_client=confluence_client,
                chunker=chunker,
                embeddings=None,
                vector_store_config=vector_store_config,
                processing_config=None
            )



@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_property_2_vector_store_interface_compliance(model_name: str):
    """Property 2: VectorStore Interface Compliance

    *For any* vector operation, the vector store instance SHALL be of type
    `langchain_core.vectorstores.VectorStore` and SHALL support `add_documents`,
    `similarity_search`, and deletion methods.

    **Validates: Requirements 2.1, 2.3, 2.4, 2.5**
    **Feature: langchain-abstraction-refactor, Property 2: VectorStore Interface Compliance**
    """
    # Create embeddings and vector store via provider module
    embeddings = get_embeddings(model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = get_vector_store(
            embeddings=embeddings,
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Verify instance type
        assert isinstance(vector_store, VectorStore), (
            f"Expected VectorStore instance, got {type(vector_store)}"
        )
        
        # Verify required methods exist and are callable
        required_methods = [
            'add_documents',
            'add_texts',
            'similarity_search',
            'similarity_search_with_score'
        ]
        
        for method_name in required_methods:
            assert hasattr(vector_store, method_name), (
                f"VectorStore instance must have {method_name} method"
            )
            assert callable(getattr(vector_store, method_name)), (
                f"{method_name} must be callable"
            )
        
        # Test add_documents method
        from langchain_core.documents import Document
        
        test_docs = [
            Document(page_content="Test document 1", metadata={"id": "1"}),
            Document(page_content="Test document 2", metadata={"id": "2"})
        ]
        
        doc_ids = vector_store.add_documents(test_docs)
        assert isinstance(doc_ids, list), (
            "add_documents should return a list of IDs"
        )
        assert len(doc_ids) == len(test_docs), (
            f"Expected {len(test_docs)} IDs, got {len(doc_ids)}"
        )
        
        # Test similarity_search method
        results = vector_store.similarity_search("Test query", k=2)
        assert isinstance(results, list), (
            "similarity_search should return a list"
        )
        assert all(isinstance(doc, Document) for doc in results), (
            "similarity_search should return Document instances"
        )
        
        # Test similarity_search_with_score method
        results_with_scores = vector_store.similarity_search_with_score("Test query", k=2)
        assert isinstance(results_with_scores, list), (
            "similarity_search_with_score should return a list"
        )
        assert all(
            isinstance(item, tuple) and len(item) == 2
            for item in results_with_scores
        ), (
            "similarity_search_with_score should return list of (Document, score) tuples"
        )
        
        # Verify deletion method exists (may not be supported by all stores)
        assert hasattr(vector_store, 'delete'), (
            "VectorStore should have delete method"
        )


@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_ingestion_service_uses_vector_store_interface(model_name: str):
    """Test that IngestionService properly uses VectorStore interface.
    
    This test verifies that when IngestionService is created with config,
    it uses the VectorStore interface from the provider module.
    
    **Validates: Requirements 2.1, 2.3**
    """
    # Create mocks
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    # Create configs
    processing_config = ProcessingConfig(embedding_model=model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_config = VectorStoreConfig(
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Create IngestionService with configs (should use provider module)
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            processing_config=processing_config,
            vector_store_config=vector_store_config
        )
        
        # Verify that the service has vector_store instance
        assert hasattr(service, '_vector_store'), (
            "IngestionService should have _vector_store attribute"
        )
        
        # Verify it's a VectorStore instance
        assert isinstance(service._vector_store, VectorStore), (
            f"Expected VectorStore instance, got {type(service._vector_store)}"
        )
        
        # Verify required methods are available
        required_methods = [
            'add_documents',
            'similarity_search',
            'similarity_search_with_score'
        ]
        
        for method_name in required_methods:
            assert hasattr(service._vector_store, method_name), (
                f"VectorStore instance must have {method_name} method"
            )


@given(model_name=model_name_strategy)
@settings(max_examples=10, deadline=None)
def test_ingestion_service_accepts_custom_vector_store(model_name: str):
    """Test that IngestionService accepts custom VectorStore via dependency injection.
    
    This test verifies that when a custom VectorStore instance is provided,
    the service uses it instead of creating one from config.
    
    **Validates: Requirements 4.1, 4.3**
    """
    # Create mocks
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    # Create custom embeddings and vector store
    custom_embeddings = get_embeddings(model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        custom_vector_store = get_vector_store(
            embeddings=custom_embeddings,
            collection_name="test_collection",
            persist_directory=temp_dir
        )
        
        # Create IngestionService with custom instances (no configs needed)
        service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embeddings=custom_embeddings,
            vector_store=custom_vector_store
        )
        
        # Verify that the service uses the provided vector store
        assert service._vector_store is custom_vector_store, (
            "IngestionService should use provided vector store instance"
        )
        
        # Verify it's a VectorStore instance
        assert isinstance(service._vector_store, VectorStore), (
            f"Expected VectorStore instance, got {type(service._vector_store)}"
        )


def test_ingestion_service_requires_config_when_no_vector_store():
    """Test that IngestionService raises error when neither vector_store nor config provided."""
    confluence_client = Mock()
    chunker = Mock(spec=DocumentChunker)
    
    processing_config = ProcessingConfig(embedding_model="all-MiniLM-L6-v2")
    
    # Should raise ValueError when vector_store is None and vector_store_config is None
    with pytest.raises(ValueError, match="vector_store_config is required"):
        IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            processing_config=processing_config,
            embeddings=None,
            vector_store=None,
            vector_store_config=None
        )



# Strategies for metadata testing
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
    min_size=10,
    max_size=500,
)


@given(
    model_name=model_name_strategy,
    page_id=page_id_strategy,
    page_title=page_title_strategy,
    content=content_strategy,
)
@settings(max_examples=10, deadline=None)
def test_property_6_metadata_preservation_through_vector_operations(
    model_name: str,
    page_id: str,
    page_title: str,
    content: str,
):
    """Property 6: Metadata Preservation Through Vector Operations

    *For any* DocumentChunk with metadata fields (page_id, page_title, page_url,
    author, modified_date), when added to the vector store and retrieved via search,
    all metadata fields SHALL be preserved and accessible.

    **Validates: Requirements 11.1, 11.2**
    **Feature: langchain-abstraction-refactor, Property 6: Metadata Preservation Through Vector Operations**
    """
    from langchain_core.documents import Document
    from src.models.page import DocumentChunk, to_langchain_document
    
    # Create embeddings and vector store
    embeddings = get_embeddings(model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = get_vector_store(
            embeddings=embeddings,
            collection_name="test_metadata_preservation",
            persist_directory=temp_dir
        )
        
        # Create a DocumentChunk with full metadata
        chunk = DocumentChunk(
            chunk_id=f"{page_id}_0",
            page_id=page_id,
            content=content,
            metadata={
                "page_title": page_title,
                "page_url": f"https://example.com/page/{page_id}",
                "author": "test_author",
                "modified_date": "2024-01-01T00:00:00Z",
            },
            chunk_index=0
        )
        
        # Convert to LangChain Document
        doc = to_langchain_document(chunk)
        
        # Verify conversion preserves metadata
        assert doc.page_content == content, (
            "Document content should match chunk content"
        )
        assert "page_id" in doc.metadata, (
            "Document metadata should contain page_id"
        )
        assert doc.metadata["page_id"] == page_id, (
            f"page_id should be {page_id}, got {doc.metadata.get('page_id')}"
        )
        assert "page_title" in doc.metadata, (
            "Document metadata should contain page_title"
        )
        assert doc.metadata["page_title"] == page_title, (
            f"page_title should be {page_title}, got {doc.metadata.get('page_title')}"
        )
        assert "chunk_id" in doc.metadata, (
            "Document metadata should contain chunk_id"
        )
        assert "chunk_index" in doc.metadata, (
            "Document metadata should contain chunk_index"
        )
        
        # Add to vector store
        doc_ids = vector_store.add_documents([doc])
        assert len(doc_ids) == 1, (
            f"Expected 1 document ID, got {len(doc_ids)}"
        )
        
        # Search for the document
        results = vector_store.similarity_search(content[:50], k=1)
        
        # Verify we got results
        assert len(results) > 0, (
            "Should retrieve at least one result"
        )
        
        result_doc = results[0]
        
        # Verify metadata is preserved
        assert "page_id" in result_doc.metadata, (
            "Retrieved document should have page_id in metadata"
        )
        assert result_doc.metadata["page_id"] == page_id, (
            f"Retrieved page_id should be {page_id}, got {result_doc.metadata.get('page_id')}"
        )
        
        assert "page_title" in result_doc.metadata, (
            "Retrieved document should have page_title in metadata"
        )
        assert result_doc.metadata["page_title"] == page_title, (
            f"Retrieved page_title should be {page_title}, got {result_doc.metadata.get('page_title')}"
        )
        
        assert "page_url" in result_doc.metadata, (
            "Retrieved document should have page_url in metadata"
        )
        
        assert "author" in result_doc.metadata, (
            "Retrieved document should have author in metadata"
        )
        assert result_doc.metadata["author"] == "test_author", (
            f"Retrieved author should be test_author, got {result_doc.metadata.get('author')}"
        )
        
        assert "modified_date" in result_doc.metadata, (
            "Retrieved document should have modified_date in metadata"
        )
        
        assert "chunk_id" in result_doc.metadata, (
            "Retrieved document should have chunk_id in metadata"
        )
        
        assert "chunk_index" in result_doc.metadata, (
            "Retrieved document should have chunk_index in metadata"
        )
        assert result_doc.metadata["chunk_index"] == 0, (
            f"Retrieved chunk_index should be 0, got {result_doc.metadata.get('chunk_index')}"
        )


@given(
    model_name=model_name_strategy,
    page_id=page_id_strategy,
    page_title=page_title_strategy,
)
@settings(max_examples=10, deadline=None)
def test_metadata_preservation_with_multiple_chunks(
    model_name: str,
    page_id: str,
    page_title: str,
):
    """Test that metadata is preserved for multiple chunks from the same page.
    
    This test verifies that when multiple chunks from the same page are added,
    each chunk's metadata is preserved independently.
    
    **Validates: Requirements 11.1, 11.2**
    """
    from src.models.page import DocumentChunk, to_langchain_documents
    
    # Create embeddings and vector store
    embeddings = get_embeddings(model_name)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store = get_vector_store(
            embeddings=embeddings,
            collection_name="test_multiple_chunks",
            persist_directory=temp_dir
        )
        
        # Create multiple chunks for the same page
        chunks = []
        for i in range(3):
            chunk = DocumentChunk(
                chunk_id=f"{page_id}_{i}",
                page_id=page_id,
                content=f"Content for chunk {i} of page {page_title}",
                metadata={
                    "page_title": page_title,
                    "page_url": f"https://example.com/page/{page_id}",
                    "author": "test_author",
                    "modified_date": "2024-01-01T00:00:00Z",
                },
                chunk_index=i
            )
            chunks.append(chunk)
        
        # Convert to LangChain Documents
        docs = to_langchain_documents(chunks)
        
        # Add to vector store
        doc_ids = vector_store.add_documents(docs)
        assert len(doc_ids) == 3, (
            f"Expected 3 document IDs, got {len(doc_ids)}"
        )
        
        # Search for documents
        results = vector_store.similarity_search(page_title, k=3)
        
        # Verify we got results
        assert len(results) > 0, (
            "Should retrieve at least one result"
        )
        
        # Verify all results have the same page_id but different chunk_index
        page_ids = [doc.metadata.get("page_id") for doc in results]
        chunk_indices = [doc.metadata.get("chunk_index") for doc in results]
        
        # All should have the same page_id
        assert all(pid == page_id for pid in page_ids if pid is not None), (
            f"All results should have page_id {page_id}"
        )
        
        # Chunk indices should be different (if we got multiple results)
        if len(results) > 1:
            assert len(set(chunk_indices)) > 1, (
                "Different chunks should have different chunk_index values"
            )
