#!/usr/bin/env python3
"""
Health check script for Confluence RAG System.

This script performs health checks on all system components:
- Configuration validation
- Confluence API connectivity
- Vector database accessibility
- Embedding model availability
- Query interface functionality

Can be used for monitoring, alerting, or pre-deployment validation.

Usage:
    python scripts/health_check.py [--config CONFIG_PATH] [--json]

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

from src.ingestion.confluence_client import ConfluenceClient
from src.processing.embedder import EmbeddingGenerator
from src.query.query_processor import QueryProcessor
from src.storage.vector_store import ChromaStore
from src.utils.config_loader import ConfigLoader

log = structlog.stdlib.get_logger()


class HealthChecker:
    """Performs health checks on system components."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize health checker.

        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path
        self.results: dict[str, dict] = {}

    def check_configuration(self) -> bool:
        """
        Check if configuration is valid.

        Returns:
            True if configuration is valid, False otherwise
        """
        check_name = "configuration"
        log.info("Checking configuration")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            self.results[check_name] = {
                "status": "pass",
                "message": "Configuration loaded successfully",
                "details": {
                    "confluence_url": str(config.confluence.base_url),
                    "space_key": config.confluence.space_key,
                    "vector_store_type": config.vector_store.type,
                    "embedding_model": config.processing.embedding_model,
                },
            }
            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Configuration error: {str(e)}",
                "details": {},
            }
            return False

    def check_confluence_connectivity(self) -> bool:
        """
        Check connectivity to Confluence API.

        Returns:
            True if connection successful, False otherwise
        """
        check_name = "confluence_connectivity"
        log.info("Checking Confluence connectivity")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            client = ConfluenceClient(
                base_url=str(config.confluence.base_url),
                auth_token=config.confluence.auth_token,
                cloud=config.confluence.cloud,
            )

            # Try to get one page
            pages = list(client.get_space_pages(config.confluence.space_key, limit=1))

            self.results[check_name] = {
                "status": "pass",
                "message": "Successfully connected to Confluence",
                "details": {
                    "space_key": config.confluence.space_key,
                    "pages_accessible": len(pages) > 0,
                },
            }
            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Confluence connection failed: {str(e)}",
                "details": {},
            }
            return False

    def check_vector_store(self) -> bool:
        """
        Check vector database accessibility.

        Returns:
            True if vector store is accessible, False otherwise
        """
        check_name = "vector_store"
        log.info("Checking vector store")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            vector_store = ChromaStore(
                persist_directory=config.vector_store.config["persist_directory"],
                collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
            )

            # Try a simple search to verify functionality
            embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)
            test_embedding = embedder.generate_embedding("test query")

            results = vector_store.search(
                query_embedding=test_embedding, top_k=1, space_key=config.confluence.space_key
            )

            self.results[check_name] = {
                "status": "pass",
                "message": "Vector store is accessible",
                "details": {
                    "store_type": config.vector_store.type,
                    "documents_found": len(results),
                },
            }
            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Vector store error: {str(e)}",
                "details": {},
            }
            return False

    def check_embedding_model(self) -> bool:
        """
        Check if embedding model is available and functional.

        Returns:
            True if model works, False otherwise
        """
        check_name = "embedding_model"
        log.info("Checking embedding model")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)

            # Generate a test embedding
            test_text = "This is a test sentence for embedding generation."
            embedding = embedder.generate_embedding(test_text)

            dimension = embedder.get_embedding_dimension()

            self.results[check_name] = {
                "status": "pass",
                "message": "Embedding model is functional",
                "details": {
                    "model_name": config.processing.embedding_model,
                    "embedding_dimension": dimension,
                    "test_embedding_length": len(embedding),
                },
            }
            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Embedding model error: {str(e)}",
                "details": {},
            }
            return False

    def check_query_functionality(self) -> bool:
        """
        Check if query processing works end-to-end.

        Returns:
            True if queries work, False otherwise
        """
        check_name = "query_functionality"
        log.info("Checking query functionality")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)

            vector_store = ChromaStore(
                persist_directory=config.vector_store.config["persist_directory"],
                collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
            )

            query_processor = QueryProcessor(
                embedder=embedder,
                vector_store=vector_store,
                space_key=config.confluence.space_key,
            )

            # Try a test query
            results = query_processor.process_query(
                query="test query", top_k=config.top_k_results
            )

            self.results[check_name] = {
                "status": "pass",
                "message": "Query processing is functional",
                "details": {
                    "test_query": "test query",
                    "results_returned": len(results),
                },
            }
            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Query processing error: {str(e)}",
                "details": {},
            }
            return False

    def check_storage_space(self) -> bool:
        """
        Check available storage space for vector database.

        Returns:
            True if sufficient space available, False otherwise
        """
        check_name = "storage_space"
        log.info("Checking storage space")

        try:
            config_loader = ConfigLoader()
            config = config_loader.load_config(self.config_path)

            if config.vector_store.type == "chroma":
                persist_dir = Path(
                    config.vector_store.config.get("persist_directory", "./chroma_db")
                )

                if persist_dir.exists():
                    # Calculate directory size
                    total_size = sum(
                        f.stat().st_size for f in persist_dir.rglob("*") if f.is_file()
                    )
                    size_mb = total_size / (1024 * 1024)

                    self.results[check_name] = {
                        "status": "pass",
                        "message": "Storage space check completed",
                        "details": {
                            "persist_directory": str(persist_dir),
                            "size_mb": round(size_mb, 2),
                        },
                    }
                else:
                    self.results[check_name] = {
                        "status": "warn",
                        "message": "Vector store directory does not exist yet",
                        "details": {
                            "persist_directory": str(persist_dir),
                        },
                    }
            else:
                self.results[check_name] = {
                    "status": "skip",
                    "message": f"Storage check not applicable for {config.vector_store.type}",
                    "details": {},
                }

            return True

        except Exception as e:
            self.results[check_name] = {
                "status": "fail",
                "message": f"Storage check error: {str(e)}",
                "details": {},
            }
            return False

    def run_all_checks(self) -> bool:
        """
        Run all health checks.

        Returns:
            True if all checks passed, False otherwise
        """
        checks = [
            self.check_configuration,
            self.check_confluence_connectivity,
            self.check_vector_store,
            self.check_embedding_model,
            self.check_query_functionality,
            self.check_storage_space,
        ]

        all_passed = True
        for check in checks:
            try:
                result = check()
                if not result:
                    all_passed = False
            except Exception as e:
                log.error("Check failed with exception", check=check.__name__, error=str(e))
                all_passed = False

        return all_passed

    def get_summary(self) -> dict:
        """
        Get summary of all health check results.

        Returns:
            Dictionary with summary information
        """
        total_checks = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "pass")
        failed = sum(1 for r in self.results.values() if r["status"] == "fail")
        warnings = sum(1 for r in self.results.values() if r["status"] == "warn")
        skipped = sum(1 for r in self.results.values() if r["status"] == "skip")

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy" if failed == 0 else "unhealthy",
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "warnings": warnings,
            "skipped": skipped,
            "checks": self.results,
        }


def main():
    """Main entry point for health check script."""
    parser = argparse.ArgumentParser(
        description="Health check for Confluence RAG System"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )

    args = parser.parse_args()

    # Run health checks
    checker = HealthChecker(config_path=args.config)
    all_passed = checker.run_all_checks()
    summary = checker.get_summary()

    # Output results
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print("\n" + "=" * 60)
        print("HEALTH CHECK SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Total Checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Skipped: {summary['skipped']}")
        print("\n" + "-" * 60)
        print("DETAILED RESULTS")
        print("-" * 60)

        for check_name, result in summary["checks"].items():
            status_symbol = {
                "pass": "✓",
                "fail": "✗",
                "warn": "⚠",
                "skip": "○",
            }.get(result["status"], "?")

            print(f"\n{status_symbol} {check_name.replace('_', ' ').title()}")
            print(f"  Status: {result['status'].upper()}")
            print(f"  Message: {result['message']}")

            if result["details"]:
                print("  Details:")
                for key, value in result["details"].items():
                    print(f"    - {key}: {value}")

        print("\n" + "=" * 60)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
