#!/usr/bin/env python3
"""CLI script for running Confluence ingestion.

This script provides a command-line interface for ingesting Confluence
documentation into the vector database.
"""

import argparse
import sys

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.ingestion.ingestion_service import IngestionService
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import VectorStoreFactory
from src.utils.config_loader import ConfigLoader, ConfigurationError
from src.utils.logging_config import configure_logging

log = structlog.stdlib.get_logger()


def setup_logging(verbose: bool, config_path: str | None) -> None:
    """Configure logging based on verbosity level and config.

    Args:
        verbose: If True, set log level to DEBUG, otherwise use config
        config_path: Path to configuration file
    """
    try:
        # Load configuration to get logging settings
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        # Override log level if verbose flag is set
        log_level = "DEBUG" if verbose else config.logging.log_level

        # Configure logging
        configure_logging(
            log_level=log_level,
            json_logs=config.logging.json_logs,
            log_file=config.logging.log_file,
        )

        log.info(
            "logging_configured",
            log_level=log_level,
            json_logs=config.logging.json_logs,
            log_file=config.logging.log_file,
        )
    except Exception as e:
        # Fallback to basic logging if config loading fails
        configure_logging(
            log_level="DEBUG" if verbose else "INFO",
            json_logs=False,
        )
        log.warning("failed_to_load_logging_config", error=str(e))


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Ingest Confluence documentation into vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest using default configuration
  python scripts/ingest.py
  
  # Ingest using custom configuration file
  python scripts/ingest.py --config config/production.yaml
  
  # Ingest specific space with verbose logging
  python scripts/ingest.py --space DOCS --verbose
  
  # Perform full re-ingestion (not incremental)
  python scripts/ingest.py --full
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration YAML file (default: config/default.yaml)",
        default=None,
    )

    parser.add_argument(
        "--space",
        "-s",
        type=str,
        help="Confluence space key to ingest (overrides config file)",
        default=None,
    )

    parser.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Perform full re-ingestion instead of incremental sync",
        default=False,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
        default=False,
    )

    parser.add_argument(
        "--page-id",
        "-p",
        type=str,
        help="Ingest a single page by ID (for testing)",
        default=None,
    )

    return parser.parse_args()


def create_ingestion_service(config_path: str | None) -> tuple[IngestionService, str]:
    """Create and configure the ingestion service.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (IngestionService instance, space_key from config)

    Raises:
        ConfigurationError: If configuration is invalid
        RuntimeError: If service initialization fails
    """
    log.info("initializing_ingestion_service", config_path=config_path)

    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config(config_path)

        log.info(
            "configuration_loaded",
            confluence_url=str(config.confluence.base_url),
            space_key=config.confluence.space_key,
            chunk_size=config.processing.chunk_size,
            embedding_model=config.processing.embedding_model,
            vector_store_type=config.vector_store.type,
        )

        # Initialize components
        confluence_client = ConfluenceClient(
            base_url=str(config.confluence.base_url),
            auth_token=config.confluence.auth_token,
            cloud=config.confluence.cloud,
        )

        chunker = DocumentChunker(
            chunk_size=config.processing.chunk_size,
            chunk_overlap=config.processing.chunk_overlap,
        )

        embedder = EmbeddingGenerator(
            model_name=config.processing.embedding_model,
        )

        vector_store = VectorStoreFactory.create_vector_store(
            store_type=config.vector_store.type,
            config=config.vector_store.config,
        )

        # Create ingestion service
        ingestion_service = IngestionService(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        log.info("ingestion_service_initialized_successfully")

        return ingestion_service, config.confluence.space_key

    except ConfigurationError as e:
        log.error("configuration_error", error=str(e))
        raise
    except Exception as e:
        log.error("service_initialization_failed", error=str(e))
        raise RuntimeError(f"Failed to initialize ingestion service: {e}") from e


def ingest_space(
    service: IngestionService,
    space_key: str,
    incremental: bool,
) -> int:
    """Ingest a Confluence space.

    Args:
        service: IngestionService instance
        space_key: Confluence space key to ingest
        incremental: If True, perform incremental sync

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    log.info(
        "starting_space_ingestion",
        space_key=space_key,
        incremental=incremental,
    )

    try:
        result = service.ingest_space(space_key, incremental=incremental)

        # Log results
        if result["success"]:
            log.info(
                "ingestion_completed_successfully",
                space_key=space_key,
                pages_processed=result["pages_processed"],
                chunks_created=result.get("chunks_created", 0),
                duration_seconds=result["duration_seconds"],
            )
            print(f"\n✓ Ingestion completed successfully!")
            print(f"  Pages processed: {result['pages_processed']}")
            print(f"  Chunks created: {result.get('chunks_created', 0)}")
            print(f"  Duration: {result['duration_seconds']:.2f} seconds")
            return 0
        else:
            log.error(
                "ingestion_completed_with_errors",
                space_key=space_key,
                pages_processed=result["pages_processed"],
                error_count=len(result.get("errors", [])),
            )
            print(f"\n✗ Ingestion completed with errors!")
            print(f"  Pages processed: {result['pages_processed']}")
            print(f"  Errors: {len(result.get('errors', []))}")
            for error in result.get("errors", [])[:5]:  # Show first 5 errors
                print(f"    - {error}")
            if len(result.get("errors", [])) > 5:
                print(f"    ... and {len(result['errors']) - 5} more errors")
            return 1

    except Exception as e:
        log.error("ingestion_failed", space_key=space_key, error=str(e))
        print(f"\n✗ Ingestion failed: {e}")
        return 1


def ingest_page(service: IngestionService, page_id: str) -> int:
    """Ingest a single Confluence page.

    Args:
        service: IngestionService instance
        page_id: Confluence page ID to ingest

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    log.info("starting_page_ingestion", page_id=page_id)

    try:
        result = service.ingest_page(page_id)

        if result["success"]:
            log.info(
                "page_ingestion_completed_successfully",
                page_id=page_id,
                chunks_created=result["chunks_created"],
            )
            print(f"\n✓ Page ingestion completed successfully!")
            print(f"  Chunks created: {result['chunks_created']}")
            return 0
        else:
            log.error(
                "page_ingestion_failed",
                page_id=page_id,
                error=result.get("error"),
            )
            print(f"\n✗ Page ingestion failed: {result.get('error')}")
            return 1

    except Exception as e:
        log.error("page_ingestion_failed", page_id=page_id, error=str(e))
        print(f"\n✗ Page ingestion failed: {e}")
        return 1


def main() -> int:
    """Main entry point for the ingestion CLI.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose, args.config)

    log.info(
        "ingestion_cli_started",
        config=args.config,
        space=args.space,
        full=args.full,
        page_id=args.page_id,
    )

    try:
        # Create ingestion service
        service, default_space_key = create_ingestion_service(args.config)

        # Determine space key (command-line arg overrides config)
        space_key = args.space or default_space_key

        # Ingest page or space
        if args.page_id:
            return ingest_page(service, args.page_id)
        else:
            return ingest_space(service, space_key, incremental=not args.full)

    except ConfigurationError as e:
        print(f"\n✗ Configuration error: {e}")
        print("\nPlease check your configuration file and environment variables.")
        return 1
    except KeyboardInterrupt:
        log.info("ingestion_interrupted_by_user")
        print("\n\nIngestion interrupted by user.")
        return 1
    except Exception as e:
        log.error("unexpected_error", error=str(e))
        print(f"\n✗ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
