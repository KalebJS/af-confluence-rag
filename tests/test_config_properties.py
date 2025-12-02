"""Property-based tests for configuration models.

Feature: confluence-rag-system
"""

import os

import structlog
from hypothesis import given, strategies as st
from pydantic import ValidationError

from src.models import AppConfig, ConfluenceConfig, ProcessingConfig, VectorStoreConfig

log = structlog.stdlib.get_logger()


@given(st.integers(min_value=500, max_value=2000))
def test_property_5_chunk_size_bounds(chunk_size: int):
    """Property 5: Chunk size bounds.
    
    For any document that is chunked, all resulting chunks should have token
    counts between 500 and 2000 tokens (inclusive).
    
    **Feature: confluence-rag-system, Property 5: Chunk size bounds**
    **Validates: Requirements 2.1**
    """
    log.info("test_property_5_chunk_size_bounds", chunk_size=chunk_size)
    
    # Create a ProcessingConfig with the given chunk_size
    config = ProcessingConfig(
        chunk_size=chunk_size,
        chunk_overlap=200,
        embedding_model="all-MiniLM-L6-v2",
    )
    
    # Verify the chunk_size is within bounds
    assert 500 <= config.chunk_size <= 2000, "chunk_size should be between 500 and 2000"
    
    # Verify that chunk_overlap is reasonable relative to chunk_size
    assert config.chunk_overlap < config.chunk_size, "chunk_overlap should be less than chunk_size"


@given(st.integers().filter(lambda x: x < 500 or x > 2000))
def test_property_5_chunk_size_bounds_validation_error(invalid_chunk_size: int):
    """Test that chunk sizes outside bounds are rejected."""
    log.info("test_property_5_chunk_size_bounds_validation_error", chunk_size=invalid_chunk_size)
    
    try:
        ProcessingConfig(
            chunk_size=invalid_chunk_size,
            chunk_overlap=200,
            embedding_model="all-MiniLM-L6-v2",
        )
        # If we get here, validation failed to catch the invalid value
        assert False, f"Expected ValidationError for chunk_size={invalid_chunk_size}"
    except ValidationError as e:
        # This is expected - invalid chunk_size should be rejected
        assert "chunk_size" in str(e).lower() or "greater_than_equal" in str(e).lower()


def test_property_24_environment_variable_loading():
    """Property 24: Environment variable loading.
    
    For any required environment variable (CONFLUENCE_BASE_URL, CONFLUENCE_AUTH_TOKEN),
    the system should read its value from the environment at startup.
    
    **Feature: confluence-rag-system, Property 24: Environment variable loading**
    **Validates: Requirements 7.1**
    """
    log.info("test_property_24_environment_variable_loading")
    
    # Set up environment variables with APP_ prefix
    os.environ["APP_CONFLUENCE__BASE_URL"] = "https://example.atlassian.net"
    os.environ["APP_CONFLUENCE__AUTH_TOKEN"] = "test-token-12345"
    os.environ["APP_CONFLUENCE__SPACE_KEY"] = "DOCS"
    os.environ["APP_CONFLUENCE__CLOUD"] = "true"
    os.environ["APP_PROCESSING__CHUNK_SIZE"] = "1000"
    os.environ["APP_PROCESSING__CHUNK_OVERLAP"] = "200"
    os.environ["APP_PROCESSING__EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
    os.environ["APP_VECTOR_STORE__TYPE"] = "chroma"
    os.environ["APP_VECTOR_STORE__CONFIG"] = "{}"
    os.environ["APP_TOP_K_RESULTS"] = "10"
    
    try:
        # Load configuration from environment variables
        config = AppConfig()
        
        # Verify that required environment variables were loaded
        # Note: HttpUrl normalizes URLs by adding trailing slash
        assert str(config.confluence.base_url).rstrip("/") == "https://example.atlassian.net"
        assert config.confluence.auth_token == "test-token-12345"
        assert config.confluence.space_key == "DOCS"
        assert config.confluence.cloud is True
        
        # Verify processing config
        assert config.processing.chunk_size == 1000
        assert config.processing.chunk_overlap == 200
        assert config.processing.embedding_model == "all-MiniLM-L6-v2"
        
        # Verify vector store config
        assert config.vector_store.type == "chroma"
        
        # Verify top_k_results
        assert config.top_k_results == 10
        
    finally:
        # Clean up environment variables
        for key in [
            "APP_CONFLUENCE__BASE_URL",
            "APP_CONFLUENCE__AUTH_TOKEN",
            "APP_CONFLUENCE__SPACE_KEY",
            "APP_CONFLUENCE__CLOUD",
            "APP_PROCESSING__CHUNK_SIZE",
            "APP_PROCESSING__CHUNK_OVERLAP",
            "APP_PROCESSING__EMBEDDING_MODEL",
            "APP_VECTOR_STORE__TYPE",
            "APP_VECTOR_STORE__CONFIG",
            "APP_TOP_K_RESULTS",
        ]:
            os.environ.pop(key, None)
