"""Property-based tests for configuration and provider module integration.

Feature: langchain-abstraction-refactor
"""

import tempfile
from pathlib import Path

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from src.models.config import ProcessingConfig, VectorStoreConfig
from src.providers import get_embeddings, get_vector_store

log = structlog.stdlib.get_logger()


@settings(deadline=None, max_examples=10)
@given(
    st.integers(min_value=500, max_value=2000),
    st.integers(min_value=0, max_value=500),
    st.sampled_from(
        [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-MiniLM-L6-v2",
        ]
    ),
)
def test_property_3_configuration_supports_embeddings_provider(
    chunk_size: int, chunk_overlap: int, embedding_model: str
):
    """Property 3: Provider Module Returns Correct Types (Configuration Aspect - Embeddings).

    For any valid ProcessingConfig, the configuration SHALL provide the necessary
    parameters to call get_embeddings() and obtain a valid Embeddings instance.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.1, 3.2**
    """
    log.info(
        "test_property_3_config_embeddings",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )

    # Create a ProcessingConfig with the given parameters
    config = ProcessingConfig(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, embedding_model=embedding_model
    )

    # Verify the configuration has the embedding_model field needed by the provider
    assert hasattr(config, "embedding_model"), "ProcessingConfig should have embedding_model field"
    assert config.embedding_model == embedding_model, "embedding_model should match input"

    # Verify we can use the configuration to call the provider module
    embeddings = get_embeddings(config.embedding_model)

    # Verify the provider returns a valid Embeddings instance
    assert isinstance(
        embeddings, Embeddings
    ), f"get_embeddings() should return an Embeddings instance, got {type(embeddings)}"

    # Verify the instance has the required methods
    assert hasattr(embeddings, "embed_documents"), "Embeddings should have embed_documents method"
    assert hasattr(embeddings, "embed_query"), "Embeddings should have embed_query method"

    log.info(
        "config_embeddings_provider_verified",
        embedding_model=embedding_model,
        embeddings_type=type(embeddings).__name__,
    )


@settings(deadline=None, max_examples=10)
@given(
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
        min_size=3,
        max_size=50,
    ).filter(lambda x: x[0].isalnum() and x[-1].isalnum()),
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-/.",
        min_size=3,
        max_size=50,
    ),
)
def test_property_3_configuration_supports_vector_store_provider(
    collection_name: str, persist_dir_suffix: str
):
    """Property 3: Provider Module Returns Correct Types (Configuration Aspect - VectorStore).

    For any valid VectorStoreConfig, the configuration SHALL provide the necessary
    parameters to call get_vector_store() and obtain a valid VectorStore instance.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.1, 3.2**
    """
    log.info(
        "test_property_3_config_vector_store",
        collection_name=collection_name,
        persist_dir_suffix=persist_dir_suffix,
    )

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        persist_directory = str(Path(temp_dir) / persist_dir_suffix)

        # Create a VectorStoreConfig with the given parameters
        config = VectorStoreConfig(
            collection_name=collection_name, persist_directory=persist_directory
        )

        # Verify the configuration has the fields needed by the provider
        assert hasattr(
            config, "collection_name"
        ), "VectorStoreConfig should have collection_name field"
        assert hasattr(
            config, "persist_directory"
        ), "VectorStoreConfig should have persist_directory field"
        assert config.collection_name == collection_name, "collection_name should match input"
        assert config.persist_directory == persist_directory, "persist_directory should match input"

        # Get embeddings (required for vector store)
        embeddings = get_embeddings("all-MiniLM-L6-v2")

        # Verify we can use the configuration to call the provider module
        vector_store = get_vector_store(
            embeddings, config.collection_name, config.persist_directory
        )

        # Verify the provider returns a valid VectorStore instance
        assert isinstance(
            vector_store, VectorStore
        ), f"get_vector_store() should return a VectorStore instance, got {type(vector_store)}"

        # Verify the instance has the required methods
        assert hasattr(
            vector_store, "add_documents"
        ), "VectorStore should have add_documents method"
        assert hasattr(
            vector_store, "similarity_search"
        ), "VectorStore should have similarity_search method"

        log.info(
            "config_vector_store_provider_verified",
            collection_name=collection_name,
            vector_store_type=type(vector_store).__name__,
        )


def test_property_3_configuration_provides_all_required_fields():
    """Property 3: Configuration Provides All Required Fields.

    The ProcessingConfig and VectorStoreConfig SHALL provide exactly the fields
    needed by the provider module functions, no more and no less.

    **Feature: langchain-abstraction-refactor, Property 3: Provider Module Returns Correct Types**
    **Validates: Requirements 3.1, 3.2**
    """
    log.info("test_property_3_config_required_fields")

    # Create default configurations
    processing_config = ProcessingConfig()
    vector_store_config = VectorStoreConfig()

    # Verify ProcessingConfig has embedding_model (required by get_embeddings)
    assert hasattr(
        processing_config, "embedding_model"
    ), "ProcessingConfig must have embedding_model"
    assert isinstance(
        processing_config.embedding_model, str
    ), "embedding_model must be a string"

    # Verify ProcessingConfig still has chunking parameters (not removed)
    assert hasattr(processing_config, "chunk_size"), "ProcessingConfig must have chunk_size"
    assert hasattr(processing_config, "chunk_overlap"), "ProcessingConfig must have chunk_overlap"

    # Verify VectorStoreConfig has collection_name and persist_directory
    # (required by get_vector_store)
    assert hasattr(
        vector_store_config, "collection_name"
    ), "VectorStoreConfig must have collection_name"
    assert hasattr(
        vector_store_config, "persist_directory"
    ), "VectorStoreConfig must have persist_directory"
    assert isinstance(
        vector_store_config.collection_name, str
    ), "collection_name must be a string"
    assert isinstance(
        vector_store_config.persist_directory, str
    ), "persist_directory must be a string"

    # Verify VectorStoreConfig does NOT have old fields (type, config)
    assert not hasattr(
        vector_store_config, "type"
    ), "VectorStoreConfig should not have 'type' field (removed in refactor)"
    assert not hasattr(
        vector_store_config, "config"
    ), "VectorStoreConfig should not have 'config' field (removed in refactor)"

    log.info(
        "config_required_fields_verified",
        processing_fields=["chunk_size", "chunk_overlap", "embedding_model"],
        vector_store_fields=["collection_name", "persist_directory"],
    )
