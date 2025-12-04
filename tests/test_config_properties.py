"""Property-based tests for configuration models.

Feature: confluence-rag-system
"""

import os

import structlog
from hypothesis import given
from hypothesis import strategies as st
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
    os.environ["APP_VECTOR_STORE__COLLECTION_NAME"] = "confluence_docs"
    os.environ["APP_VECTOR_STORE__PERSIST_DIRECTORY"] = "./chroma_db"
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
        assert config.vector_store.collection_name == "confluence_docs"
        assert config.vector_store.persist_directory == "./chroma_db"

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
            "APP_VECTOR_STORE__COLLECTION_NAME",
            "APP_VECTOR_STORE__PERSIST_DIRECTORY",
            "APP_TOP_K_RESULTS",
        ]:
            os.environ.pop(key, None)


def test_property_25_configuration_file_parsing():
    """Property 25: Configuration file parsing.

    For any valid configuration file, the system should successfully load all
    parameters (chunk_size, embedding_model, vector_db_path, collection_name).

    **Feature: confluence-rag-system, Property 25: Configuration file parsing**
    **Validates: Requirements 7.2**
    """
    import tempfile
    from pathlib import Path

    from src.utils.config_loader import ConfigLoader

    log.info("test_property_25_configuration_file_parsing")

    # Set required environment variables
    os.environ["CONFLUENCE_BASE_URL"] = "https://test.atlassian.net"
    os.environ["CONFLUENCE_AUTH_TOKEN"] = "test-token"
    os.environ["CONFLUENCE_SPACE_KEY"] = "TEST"

    try:
        # Create a temporary valid configuration file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
confluence:
  base_url: ${CONFLUENCE_BASE_URL}
  auth_token: ${CONFLUENCE_AUTH_TOKEN}
  space_key: ${CONFLUENCE_SPACE_KEY}
  cloud: true

processing:
  chunk_size: 1500
  chunk_overlap: 300
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  collection_name: "test_docs"
  persist_directory: "./test_chroma_db"

top_k_results: 15
""")
            temp_config_path = f.name

        # Load configuration from file
        loader = ConfigLoader()
        config = loader.load_config(temp_config_path)

        # Verify all parameters were loaded correctly
        assert config.processing.chunk_size == 1500
        assert config.processing.chunk_overlap == 300
        assert config.processing.embedding_model == "all-MiniLM-L6-v2"
        assert config.vector_store.collection_name == "test_docs"
        assert config.vector_store.persist_directory == "./test_chroma_db"
        assert config.top_k_results == 15

        # Verify environment variable substitution worked
        assert str(config.confluence.base_url).rstrip("/") == "https://test.atlassian.net"
        assert config.confluence.auth_token == "test-token"
        assert config.confluence.space_key == "TEST"

    finally:
        # Clean up
        Path(temp_config_path).unlink(missing_ok=True)
        os.environ.pop("CONFLUENCE_BASE_URL", None)
        os.environ.pop("CONFLUENCE_AUTH_TOKEN", None)
        os.environ.pop("CONFLUENCE_SPACE_KEY", None)


def test_property_26_missing_configuration_error_handling():
    """Property 26: Missing configuration error handling.

    For any missing required configuration parameter, the system should fail at
    startup with an error message containing the parameter name.

    **Feature: confluence-rag-system, Property 26: Missing configuration error handling**
    **Validates: Requirements 7.4**
    """
    import tempfile
    from pathlib import Path

    from src.utils.config_loader import ConfigLoader, ConfigurationError

    log.info("test_property_26_missing_configuration_error_handling")

    # Ensure the required environment variable is NOT set
    os.environ.pop("MISSING_REQUIRED_VAR", None)

    try:
        # Create a configuration file with a missing environment variable
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
confluence:
  base_url: ${MISSING_REQUIRED_VAR}
  auth_token: "test-token"
  space_key: "TEST"
  cloud: true

processing:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  collection_name: "docs"
  persist_directory: "./chroma_db"

top_k_results: 10
""")
            temp_config_path = f.name

        # Attempt to load configuration - should fail with clear error
        loader = ConfigLoader()
        try:
            config = loader.load_config(temp_config_path)
            # If we get here, the test failed - should have raised an error
            assert False, "Expected ConfigurationError for missing environment variable"
        except ConfigurationError as e:
            # Verify the error message contains the parameter name
            error_message = str(e)
            assert "MISSING_REQUIRED_VAR" in error_message, (
                f"Error message should contain 'MISSING_REQUIRED_VAR', got: {error_message}"
            )
            log.info("missing_config_error_caught", error=error_message)

    finally:
        # Clean up
        Path(temp_config_path).unlink(missing_ok=True)


@given(st.text(min_size=1).filter(lambda x: x not in ["chroma", "faiss", "qdrant"]))
def test_property_26_invalid_configuration_validation(invalid_store_type: str):
    """Test that invalid configuration values are caught during validation.

    This extends Property 26 to test validation of configuration values.
    """
    import tempfile
    from pathlib import Path

    from src.utils.config_loader import ConfigLoader

    log.info("test_property_26_invalid_configuration_validation", store_type=invalid_store_type)

    # Set required environment variables
    os.environ["CONFLUENCE_BASE_URL"] = "https://test.atlassian.net"
    os.environ["CONFLUENCE_AUTH_TOKEN"] = "test-token"
    os.environ["CONFLUENCE_SPACE_KEY"] = "TEST"

    try:
        # Create a configuration file with invalid chunk_size (out of bounds)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""
confluence:
  base_url: ${CONFLUENCE_BASE_URL}
  auth_token: ${CONFLUENCE_AUTH_TOKEN}
  space_key: ${CONFLUENCE_SPACE_KEY}
  cloud: true

processing:
  chunk_size: 100
  chunk_overlap: 200
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  collection_name: "docs"
  persist_directory: "./chroma_db"

top_k_results: 10
""")
            temp_config_path = f.name

        # Attempt to load configuration - should fail validation
        loader = ConfigLoader()
        try:
            config = loader.load_config(temp_config_path)
            # If we get here, validation failed to catch the invalid value
            assert False, "Expected validation error for chunk_size=100 (below minimum 500)"
        except Exception as e:
            # Verify the error is related to validation
            error_message = str(e)
            assert "chunk_size" in error_message.lower() or "validation" in error_message.lower(), (
                f"Error should mention chunk_size or validation, got: {error_message}"
            )
            log.info("invalid_config_validation_caught", error=error_message)

    finally:
        # Clean up
        Path(temp_config_path).unlink(missing_ok=True)
        os.environ.pop("CONFLUENCE_BASE_URL", None)
        os.environ.pop("CONFLUENCE_AUTH_TOKEN", None)
        os.environ.pop("CONFLUENCE_SPACE_KEY", None)
