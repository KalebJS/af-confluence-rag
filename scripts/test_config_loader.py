#!/usr/bin/env python3
"""Test script to demonstrate ConfigLoader functionality."""

import os


import structlog

from src.utils.config_loader import ConfigLoader, ConfigurationError

# Configure structlog for readable output
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)

log = structlog.stdlib.get_logger()


def main():
    """Test the ConfigLoader with different scenarios."""

    # Set required environment variables for testing
    os.environ["CONFLUENCE_BASE_URL"] = "https://example.atlassian.net"
    os.environ["CONFLUENCE_AUTH_TOKEN"] = "test-token-12345"
    os.environ["CONFLUENCE_SPACE_KEY"] = "DOCS"

    loader = ConfigLoader()

    print("\n" + "=" * 80)
    print("Testing ConfigLoader")
    print("=" * 80)

    # Test 1: Load default configuration
    print("\n1. Loading default configuration...")
    try:
        config = loader.load_config("config/default.yaml")
        print(f"✓ Successfully loaded default.yaml")
        print(f"  - Confluence URL: {config.confluence.base_url}")
        print(f"  - Space Key: {config.confluence.space_key}")
        print(f"  - Chunk Size: {config.processing.chunk_size}")
        print(f"  - Vector Store Type: {config.vector_store.type}")
        print(f"  - Top K Results: {config.top_k_results}")

        # Validate configuration
        warnings = loader.validate_config(config)
        if warnings:
            print(f"  ⚠ Warnings: {warnings}")
        else:
            print(f"  ✓ No validation warnings")
    except ConfigurationError as e:
        print(f"✗ Failed to load configuration: {e}")

    # Test 2: Load development configuration
    print("\n2. Loading development configuration...")
    try:
        config = loader.load_config("config/development.yaml")
        print(f"✓ Successfully loaded development.yaml")
        print(f"  - Chunk Size: {config.processing.chunk_size}")
        print(f"  - Vector Store Directory: {config.vector_store.config.get('persist_directory')}")
        print(f"  - Top K Results: {config.top_k_results}")
    except ConfigurationError as e:
        print(f"✗ Failed to load configuration: {e}")

    # Test 3: Load production configuration
    print("\n3. Loading production configuration...")
    try:
        config = loader.load_config("config/production.yaml")
        print(f"✓ Successfully loaded production.yaml")
        print(f"  - Chunk Size: {config.processing.chunk_size}")
        print(f"  - Vector Store Directory: {config.vector_store.config.get('persist_directory')}")
    except ConfigurationError as e:
        print(f"✗ Failed to load configuration: {e}")

    # Test 4: Test missing environment variable
    print("\n4. Testing missing environment variable handling...")
    os.environ.pop("CONFLUENCE_AUTH_TOKEN", None)
    try:
        config = loader.load_config("config/default.yaml")
        print(f"✗ Should have failed with missing environment variable")
    except ConfigurationError as e:
        print(f"✓ Correctly caught missing environment variable")
        print(f"  - Error: {e}")

    print("\n" + "=" * 80)
    print("ConfigLoader tests complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
