"""Centralized provider module for embeddings and vector store implementations.

This module provides factory functions for creating Embeddings and VectorStore instances.
Developers can modify these functions to swap implementations without changing other code.

Default implementations:
- Embeddings: HuggingFaceEmbeddings (local, no API keys required)
- VectorStore: Chroma (local, no external services required)
"""

import structlog
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

log = structlog.stdlib.get_logger()


def get_embeddings(model_name: str) -> Embeddings:
    """Get the configured embeddings implementation.

    Developers: Modify this function to change the embedding provider.
    Default: HuggingFaceEmbeddings (wraps sentence-transformers, runs locally)

    Example - Swap to OpenAI embeddings:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)

    Example - Swap to Cohere embeddings:
        from langchain_cohere import CohereEmbeddings
        return CohereEmbeddings(model=model_name)

    Args:
        model_name: Name of the model to use (e.g., "all-MiniLM-L6-v2")

    Returns:
        Embeddings instance

    Raises:
        ValueError: If model_name is invalid or cannot be loaded
        RuntimeError: If the embeddings provider cannot be initialized
    """
    if not model_name or not model_name.strip():
        error_msg = "model_name cannot be empty"
        log.error("get_embeddings_failed", error=error_msg)
        raise ValueError(error_msg)

    try:
        log.info("initializing_embeddings", model_name=model_name, provider="HuggingFace")

        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        log.info("embeddings_initialized_successfully", model_name=model_name)
        return embeddings

    except Exception as e:
        error_msg = f"Failed to initialize embeddings with model '{model_name}': {e}"
        log.error(
            "get_embeddings_failed",
            model_name=model_name,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(error_msg) from e


def get_vector_store(
    embeddings: Embeddings, collection_name: str, persist_directory: str
) -> VectorStore:
    """Get the configured vector store implementation.

    Developers: Modify this function to change the vector store provider.
    Default: Chroma (local vector database, no external services required)

    Example - Swap to FAISS:
        from langchain_community.vectorstores import FAISS
        return FAISS(
            embedding_function=embeddings,
            index_name=collection_name,
            folder_path=persist_directory
        )

    Example - Swap to Snowflake:
        from langchain_snowflake import SnowflakeVectorStore
        return SnowflakeVectorStore(
            embedding=embeddings,
            connection_params={...}
        )

    Args:
        embeddings: Embeddings instance to use for vectorization
        collection_name: Name of the collection/index
        persist_directory: Directory for persistence (if supported)

    Returns:
        VectorStore instance

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If the vector store cannot be initialized
    """
    if not collection_name or not collection_name.strip():
        error_msg = "collection_name cannot be empty"
        log.error("get_vector_store_failed", error=error_msg)
        raise ValueError(error_msg)

    if not persist_directory or not persist_directory.strip():
        error_msg = "persist_directory cannot be empty"
        log.error("get_vector_store_failed", error=error_msg)
        raise ValueError(error_msg)

    try:
        log.info(
            "initializing_vector_store",
            collection_name=collection_name,
            persist_directory=persist_directory,
            provider="Chroma",
        )

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        log.info(
            "vector_store_initialized_successfully",
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        return vector_store

    except Exception as e:
        error_msg = (
            f"Failed to initialize vector store with collection '{collection_name}' "
            f"at '{persist_directory}': {e}"
        )
        log.error(
            "get_vector_store_failed",
            collection_name=collection_name,
            persist_directory=persist_directory,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise RuntimeError(error_msg) from e
