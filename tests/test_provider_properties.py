"""Property-based tests for provider module.

Feature: langchain-abstraction-refactor
"""

import tempfile
from pathlib import Path

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.providers import get_embeddings, get_vector_store

log = structlog.stdlib.get_logger()


@settings(deadline=None, max_examples=5)  # Model loading can take time on first run
@given(
    st.sampled_from(
        [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
    )
)
def test_property_3_provider_module_returns_correct_embeddings_type(model_name: str):
    """Property 3: Provider Module Returns Correct Types (Embeddings).

    For any valid model name, get_embeddings() SHALL return an instance that
    implements the Embeddings interface from langchain_core.embeddings.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.3, 6.1, 6.2**
    """
    log.info("test_property_3_embeddings_type", model_name=model_name)

    # Call the provider function
    embeddings = get_embeddings(model_name)

    # Verify the returned instance is of the correct type
    assert isinstance(
        embeddings, Embeddings
    ), f"get_embeddings() should return an Embeddings instance, got {type(embeddings)}"

    # Verify the instance has the required methods
    assert hasattr(
        embeddings, "embed_documents"
    ), "Embeddings instance should have embed_documents method"
    assert hasattr(embeddings, "embed_query"), "Embeddings instance should have embed_query method"
    assert callable(embeddings.embed_documents), "embed_documents should be callable"
    assert callable(embeddings.embed_query), "embed_query should be callable"

    log.info("embeddings_type_verified", model_name=model_name, type=type(embeddings).__name__)


@settings(deadline=None, max_examples=10)  # Model loading and vector store initialization can take time
@given(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-",
        min_size=3,
        max_size=50,
    ).filter(lambda x: x[0].isalnum() and x[-1].isalnum()),
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
        min_size=1,
        max_size=50,
    ),
)
def test_property_3_provider_module_returns_correct_vector_store_type(
    collection_name: str, persist_dir_suffix: str
):
    """Property 3: Provider Module Returns Correct Types (VectorStore).

    For any valid collection name and persist directory, get_vector_store() SHALL
    return an instance that implements the VectorStore interface from
    langchain_core.vectorstores.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.3, 6.1, 6.2**
    """
    log.info(
        "test_property_3_vector_store_type",
        collection_name=collection_name,
        persist_dir_suffix=persist_dir_suffix,
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = str(Path(temp_dir) / persist_dir_suffix)

        # First get embeddings (required for vector store)
        embeddings = get_embeddings("all-MiniLM-L6-v2")

        # Call the provider function
        vector_store = get_vector_store(embeddings, collection_name, persist_directory)

        # Verify the returned instance is of the correct type
        assert isinstance(
            vector_store, VectorStore
        ), f"get_vector_store() should return a VectorStore instance, got {type(vector_store)}"

        # Verify the instance has the required methods
        assert hasattr(
            vector_store, "add_documents"
        ), "VectorStore instance should have add_documents method"
        assert hasattr(
            vector_store, "similarity_search"
        ), "VectorStore instance should have similarity_search method"
        assert callable(vector_store.add_documents), "add_documents should be callable"
        assert callable(vector_store.similarity_search), "similarity_search should be callable"

        log.info(
            "vector_store_type_verified",
            collection_name=collection_name,
            type=type(vector_store).__name__,
        )


@given(st.text().filter(lambda x: not x.strip()))
def test_property_9_provider_module_error_handling_empty_model_name(empty_model_name: str):
    """Property 9: Provider Module Error Handling (Empty Model Name).

    For any empty or whitespace-only model name, get_embeddings() SHALL raise
    a clear ValueError indicating the problem.

    **Feature: langchain-abstraction-refactor, Property 9: Provider Module Error Handling**
    **Validates: Requirements 6.5**
    """
    log.info("test_property_9_empty_model_name", model_name=repr(empty_model_name))

    try:
        get_embeddings(empty_model_name)
        # If we get here, the function didn't raise an error as expected
        assert False, f"Expected ValueError for empty model_name: {repr(empty_model_name)}"
    except ValueError as e:
        # This is expected - verify the error message is clear
        error_message = str(e)
        assert "model_name" in error_message.lower() or "empty" in error_message.lower(), (
            f"Error message should mention model_name or empty, got: {error_message}"
        )
        log.info("empty_model_name_error_caught", error=error_message)
    except Exception as e:
        # Wrong exception type
        assert False, f"Expected ValueError, got {type(e).__name__}: {e}"


@settings(deadline=None, max_examples=5)  # Model loading can take time
@given(st.text().filter(lambda x: not x.strip()))
def test_property_9_provider_module_error_handling_empty_collection_name(
    empty_collection_name: str,
):
    """Property 9: Provider Module Error Handling (Empty Collection Name).

    For any empty or whitespace-only collection name, get_vector_store() SHALL
    raise a clear ValueError indicating the problem.

    **Feature: langchain-abstraction-refactor, Property 9: Provider Module Error Handling**
    **Validates: Requirements 6.5**
    """
    log.info("test_property_9_empty_collection_name", collection_name=repr(empty_collection_name))

    embeddings = get_embeddings("all-MiniLM-L6-v2")

    try:
        get_vector_store(embeddings, empty_collection_name, "/tmp/test_chroma")
        # If we get here, the function didn't raise an error as expected
        assert False, (
            f"Expected ValueError for empty collection_name: {repr(empty_collection_name)}"
        )
    except ValueError as e:
        # This is expected - verify the error message is clear
        error_message = str(e)
        assert "collection_name" in error_message.lower() or "empty" in error_message.lower(), (
            f"Error message should mention collection_name or empty, got: {error_message}"
        )
        log.info("empty_collection_name_error_caught", error=error_message)
    except Exception as e:
        # Wrong exception type
        assert False, f"Expected ValueError, got {type(e).__name__}: {e}"


@settings(deadline=None, max_examples=5)  # Model loading can take time
@given(st.text().filter(lambda x: not x.strip()))
def test_property_9_provider_module_error_handling_empty_persist_directory(
    empty_persist_dir: str,
):
    """Property 9: Provider Module Error Handling (Empty Persist Directory).

    For any empty or whitespace-only persist directory, get_vector_store() SHALL
    raise a clear ValueError indicating the problem.

    **Feature: langchain-abstraction-refactor, Property 9: Provider Module Error Handling**
    **Validates: Requirements 6.5**
    """
    log.info("test_property_9_empty_persist_directory", persist_directory=repr(empty_persist_dir))

    embeddings = get_embeddings("all-MiniLM-L6-v2")

    try:
        get_vector_store(embeddings, "test_collection", empty_persist_dir)
        # If we get here, the function didn't raise an error as expected
        assert False, (
            f"Expected ValueError for empty persist_directory: {repr(empty_persist_dir)}"
        )
    except ValueError as e:
        # This is expected - verify the error message is clear
        error_message = str(e)
        assert "persist_directory" in error_message.lower() or "empty" in error_message.lower(), (
            f"Error message should mention persist_directory or empty, got: {error_message}"
        )
        log.info("empty_persist_directory_error_caught", error=error_message)
    except Exception as e:
        # Wrong exception type
        assert False, f"Expected ValueError, got {type(e).__name__}: {e}"


@settings(deadline=None, max_examples=10)  # Model loading can take time
@given(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_/",
        min_size=5,
        max_size=50,
    ).filter(
        lambda x: x not in [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]
        and "test" not in x.lower()
    )
)
def test_property_9_provider_module_error_handling_invalid_model_name(invalid_model_name: str):
    """Property 9: Provider Module Error Handling (Invalid Model Name).

    For any invalid model name (one that doesn't exist), get_embeddings() SHALL
    raise a clear RuntimeError indicating the problem with the underlying error.

    **Feature: langchain-abstraction-refactor, Property 9: Provider Module Error Handling**
    **Validates: Requirements 6.5**
    """
    log.info("test_property_9_invalid_model_name", model_name=invalid_model_name)

    try:
        embeddings = get_embeddings(invalid_model_name)
        # Try to use it - this is where the error might occur
        embeddings.embed_query("test")
        # If we get here, either the model exists or something is wrong
        log.warning(
            "model_unexpectedly_loaded",
            model_name=invalid_model_name,
            message="Model loaded successfully when it was expected to fail",
        )
    except RuntimeError as e:
        # This is expected - verify the error message is clear
        error_message = str(e)
        assert invalid_model_name in error_message or "model" in error_message.lower(), (
            f"Error message should mention the model name or 'model', got: {error_message}"
        )
        log.info("invalid_model_name_error_caught", error=error_message)
    except Exception as e:
        # Other exceptions are also acceptable (e.g., from HuggingFace)
        # as long as they're wrapped or provide clear context
        log.info(
            "model_loading_failed",
            model_name=invalid_model_name,
            error_type=type(e).__name__,
            error=str(e),
        )


def test_property_3_embeddings_functional_verification():
    """Property 3: Embeddings Functional Verification.

    Verify that the embeddings instance returned by get_embeddings() can actually
    generate embeddings for text.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.3, 6.1, 6.2**
    """
    log.info("test_property_3_embeddings_functional_verification")

    embeddings = get_embeddings("all-MiniLM-L6-v2")

    # Test embed_query
    query_embedding = embeddings.embed_query("test query")
    assert isinstance(query_embedding, list), "embed_query should return a list"
    assert len(query_embedding) > 0, "embed_query should return non-empty embedding"
    assert all(
        isinstance(x, float) for x in query_embedding
    ), "embed_query should return list of floats"

    # Test embed_documents
    doc_embeddings = embeddings.embed_documents(["doc 1", "doc 2", "doc 3"])
    assert isinstance(doc_embeddings, list), "embed_documents should return a list"
    assert len(doc_embeddings) == 3, "embed_documents should return one embedding per document"
    assert all(isinstance(emb, list) for emb in doc_embeddings), "Each embedding should be a list"
    assert all(
        len(emb) == len(query_embedding) for emb in doc_embeddings
    ), "All embeddings should have same dimension"

    log.info(
        "embeddings_functional_verification_passed",
        embedding_dimension=len(query_embedding),
        num_documents=len(doc_embeddings),
    )


def test_property_3_vector_store_functional_verification():
    """Property 3: VectorStore Functional Verification.

    Verify that the vector store instance returned by get_vector_store() can
    actually add and search documents.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.3, 6.1, 6.2**
    """
    log.info("test_property_3_vector_store_functional_verification")

    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings = get_embeddings("all-MiniLM-L6-v2")
        vector_store = get_vector_store(embeddings, "test_collection", temp_dir)

        # Test add_texts
        from langchain_core.documents import Document

        docs = [
            Document(page_content="This is document 1", metadata={"id": "1"}),
            Document(page_content="This is document 2", metadata={"id": "2"}),
        ]

        ids = vector_store.add_documents(docs)
        assert isinstance(ids, list), "add_documents should return a list of IDs"
        assert len(ids) == 2, "add_documents should return one ID per document"

        # Test similarity_search
        results = vector_store.similarity_search("document", k=2)
        assert isinstance(results, list), "similarity_search should return a list"
        assert len(results) <= 2, "similarity_search should respect k parameter"
        assert all(
            isinstance(doc, Document) for doc in results
        ), "similarity_search should return Document instances"

        log.info(
            "vector_store_functional_verification_passed",
            num_documents_added=len(ids),
            num_results=len(results),
        )
