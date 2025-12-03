#!/usr/bin/env python3
"""
Scheduled synchronization script for Confluence RAG System.

This script performs incremental synchronization with Confluence:
- Detects new, modified, and deleted pages
- Updates the vector database accordingly
- Logs synchronization statistics

Designed to be run on a schedule (e.g., via cron, Posit Connect scheduler, or Airflow).

Usage:
    python scripts/scheduled_sync.py [--config CONFIG_PATH] [--full-sync]
"""

import argparse
import sys
from datetime import datetime

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.processing.chunker import DocumentChunker
from src.processing.embedder import EmbeddingGenerator
from src.processing.metadata_enricher import MetadataEnricher
from src.storage.vector_store import ChromaStore
from src.sync.change_detector import ChangeDetector
from src.sync.sync_coordinator import SyncCoordinator
from src.sync.timestamp_tracker import TimestampTracker
from src.utils.config_loader import ConfigLoader

log = structlog.stdlib.get_logger()


def perform_sync(config_path: str | None = None, full_sync: bool = False) -> dict:
    """
    Perform synchronization with Confluence.

    Args:
        config_path: Optional path to configuration file
        full_sync: If True, perform full sync instead of incremental

    Returns:
        Dictionary with sync statistics
    """
    start_time = datetime.now()

    log.info(
        "Starting synchronization",
        sync_type="full" if full_sync else "incremental",
        timestamp=start_time.isoformat(),
    )

    try:
        # Load configuration
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

        vector_store = ChromaStore(
            persist_directory=config.vector_store.config["persist_directory"],
            collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
        )

        timestamp_tracker = TimestampTracker(vector_store=vector_store)

        change_detector = ChangeDetector(
            confluence_client=confluence_client, timestamp_tracker=timestamp_tracker
        )

        # Create sync coordinator
        sync_coordinator = SyncCoordinator(
            confluence_client=confluence_client,
            chunker=chunker,
            embedder=embedder,
            metadata_enricher=metadata_enricher,
            vector_store=vector_store,
            change_detector=change_detector,
            timestamp_tracker=timestamp_tracker,
        )

        # Perform sync
        space_key = config.confluence.space_key

        if full_sync:
            log.info("Performing full synchronization", space_key=space_key)
            result = sync_coordinator.sync_space(space_key=space_key, full_sync=True)
        else:
            log.info("Performing incremental synchronization", space_key=space_key)
            result = sync_coordinator.sync_space(space_key=space_key, full_sync=False)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Prepare statistics
        stats = {
            "success": True,
            "sync_type": "full" if full_sync else "incremental",
            "space_key": space_key,
            "pages_added": result.pages_added,
            "pages_updated": result.pages_updated,
            "pages_deleted": result.pages_deleted,
            "total_pages": result.pages_added + result.pages_updated,
            "chunks_created": result.chunks_created,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        }

        log.info(
            "Synchronization completed successfully",
            **stats,
        )

        return stats

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        log.error(
            "Synchronization failed",
            error=str(e),
            duration_seconds=duration,
        )

        return {
            "success": False,
            "error": str(e),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
        }


def send_notification(stats: dict, notification_email: str | None = None):
    """
    Send notification about sync results.

    Args:
        stats: Sync statistics dictionary
        notification_email: Optional email address for notifications
    """
    if not notification_email:
        return

    # This is a placeholder for notification logic
    # In production, integrate with your notification system (email, Slack, etc.)
    log.info(
        "Notification would be sent",
        email=notification_email,
        success=stats.get("success"),
    )


def main():
    """Main entry point for scheduled sync script."""
    parser = argparse.ArgumentParser(
        description="Scheduled synchronization for Confluence RAG System"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "--full-sync",
        action="store_true",
        help="Perform full sync instead of incremental",
    )
    parser.add_argument(
        "--notify",
        type=str,
        help="Email address for notifications",
        default=None,
    )

    args = parser.parse_args()

    # Perform sync
    stats = perform_sync(config_path=args.config, full_sync=args.full_sync)

    # Send notification if configured
    if args.notify:
        send_notification(stats, args.notify)

    # Print summary
    print("\n" + "=" * 60)
    print("SYNCHRONIZATION SUMMARY")
    print("=" * 60)

    if stats.get("success"):
        print(f"Status: ✓ SUCCESS")
        print(f"Sync Type: {stats.get('sync_type', 'unknown')}")
        print(f"Space: {stats.get('space_key', 'unknown')}")
        print(f"Pages Added: {stats.get('pages_added', 0)}")
        print(f"Pages Updated: {stats.get('pages_updated', 0)}")
        print(f"Pages Deleted: {stats.get('pages_deleted', 0)}")
        print(f"Chunks Created: {stats.get('chunks_created', 0)}")
        print(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")
    else:
        print(f"Status: ✗ FAILED")
        print(f"Error: {stats.get('error', 'Unknown error')}")
        print(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")

    print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if stats.get("success") else 1)


if __name__ == "__main__":
    main()
