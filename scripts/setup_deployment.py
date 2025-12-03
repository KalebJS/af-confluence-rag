#!/usr/bin/env python3
"""
Initial setup script for Confluence RAG System deployment.

This script performs initial setup tasks:
- Validates environment configuration
- Creates necessary directories
- Initializes the vector database
- Performs a test ingestion

Usage:
    python scripts/setup_deployment.py [--config CONFIG_PATH]
"""

import argparse
import sys
from pathlib import Path

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.ingestion.ingestion_service import IngestionService
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.processing.metadata_enricher import MetadataEnricher
from src.storage.vector_store import ChromaStore, VectorStoreFactory
from src.utils.config_loader import ConfigLoader

log = structlog.stdlib.get_logger()


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.

    Returns:
        True if environment is valid, False otherwise
    """
    log.info("Validating environment configuration")

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config()

        log.info(
            "Environment validation successful",
            confluence_url=str(config.confluence.base_url),
            space_key=config.confluence.space_key,
            vector_store_type=config.vector_store.type,
        )
        return True

    except Exception as e:
        log.error("Environment validation failed", error=str(e))
        return False


def create_directories(config_path: str | None = None) -> bool:
    """
    Create necessary directories for the application.

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if directories created successfully, False otherwise
    """
    log.info("Creating necessary directories")

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        # Create vector store directory if using Chroma
        if config.vector_store.type == "chroma":
            persist_dir = Path(
                config.vector_store.config.get("persist_directory", "./chroma_db")
            )
            persist_dir.mkdir(parents=True, exist_ok=True)
            log.info("Created vector store directory", path=str(persist_dir))

        # Create logs directory
        logs_dir = Path("./logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log.info("Created logs directory", path=str(logs_dir))

        return True

    except Exception as e:
        log.error("Failed to create directories", error=str(e))
        return False


def initialize_vector_store(config_path: str | None = None) -> bool:
    """
    Initialize the vector database.

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if initialization successful, False otherwise
    """
    log.info("Initializing vector store")

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        # Create vector store instance
        vector_store = VectorStoreFactory.create_vector_store(
            config.vector_store.type, config.vector_store.config
        )

        log.info(
            "Vector store initialized successfully",
            store_type=config.vector_store.type,
        )
        return True

    except Exception as e:
        log.error("Failed to initialize vector store", error=str(e))
        return False


def test_confluence_connection(config_path: str | None = None) -> bool:
    """
    Test connection to Confluence API.

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if connection successful, False otherwise
    """
    log.info("Testing Confluence connection")

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        # Create Confluence client
        client = ConfluenceClient(
            base_url=str(config.confluence.base_url),
            auth_token=config.confluence.auth_token,
            cloud=config.confluence.cloud,
        )

        # Try to get space info
        space_key = config.confluence.space_key
        log.info("Attempting to retrieve space information", space_key=space_key)

        # Get first page to test connection
        pages = list(client.get_space_pages(space_key, limit=1))

        if pages:
            log.info(
                "Confluence connection successful",
                space_key=space_key,
                sample_page=pages[0].title,
            )
            return True
        else:
            log.warning(
                "Confluence connection successful but space is empty",
                space_key=space_key,
            )
            return True

    except Exception as e:
        log.error("Failed to connect to Confluence", error=str(e))
        return False


def perform_test_ingestion(config_path: str | None = None) -> bool:
    """
    Perform a test ingestion of a small number of pages.

    Args:
        config_path: Optional path to configuration file

    Returns:
        True if test ingestion successful, False otherwise
    """
    log.info("Performing test ingestion")

    try:
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        # Create components
        confluence_client = ConfluenceClient(
            base_url=str(config.confluence.base_url),
            auth_token=config.confluence.auth_token,
            cloud=config.confluence.cloud,
        )

        chunker = DocumentChunker(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap,
        )

        embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)

        metadata_enricher = MetadataEnricher()

        vector_store = VectorStoreFactory.create_vector_store(
            config.vector_store.type, config.vector_store.config
        )

        # Create ingestion service
        ingestion_service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            metadata_enricher=metadata_enricher,
            vector_store=vector_store,
        )

        # Ingest first 5 pages as a test
        log.info("Ingesting first 5 pages as test", space_key=config.confluence.space_key)
        result = ingestion_service.ingest_space(
            space_key=config.confluence.space_key, max_pages=5
        )

        log.info(
            "Test ingestion completed",
            pages_processed=result.pages_processed,
            chunks_created=result.chunks_created,
            duration_seconds=result.duration_seconds,
        )

        return result.pages_processed > 0

    except Exception as e:
        log.error("Test ingestion failed", error=str(e))
        return False


def run_health_check(config_path: str | None = None) -> dict[str, bool]:
    """
    Run all health checks and return results.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Dictionary of check names to results
    """
    checks = {
        "environment": validate_environment(),
        "directories": create_directories(config_path),
        "vector_store": initialize_vector_store(config_path),
        "confluence": test_confluence_connection(config_path),
        "test_ingestion": perform_test_ingestion(config_path),
    }

    return checks


def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(
        description="Initial setup for Confluence RAG System deployment"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "--skip-test-ingestion",
        action="store_true",
        help="Skip the test ingestion step",
    )

    args = parser.parse_args()

    log.info("Starting deployment setup")

    # Run health checks
    checks = {
        "environment": validate_environment(),
        "directories": create_directories(args.config),
        "vector_store": initialize_vector_store(args.config),
        "confluence": test_confluence_connection(args.config),
    }

    if not args.skip_test_ingestion:
        checks["test_ingestion"] = perform_test_ingestion(args.config)

    # Print summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT SETUP SUMMARY")
    print("=" * 60)

    all_passed = True
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{check_name.replace('_', ' ').title():.<40} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! Deployment is ready.")
        log.info("Deployment setup completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed. Please review the logs and fix issues.")
        log.error("Deployment setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
