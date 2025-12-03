#!/usr/bin/env python3
"""Script to run the Streamlit query interface.

This script provides a convenient way to start the Streamlit application
with proper configuration and error handling.
"""

import os
import subprocess
import sys
from pathlib import Path

import structlog

log = structlog.stdlib.get_logger()


def check_environment() -> bool:
    """Check if required environment variables are set.
    
    Returns:
        True if all required variables are set, False otherwise
    """
    required_vars = [
        "CONFLUENCE_BASE_URL",
        "CONFLUENCE_AUTH_TOKEN",
        "CONFLUENCE_SPACE_KEY",
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        log.error(
            "missing_environment_variables",
            missing_vars=missing_vars,
        )
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your environment or .env file.")
        return False
    
    log.info("environment_variables_verified")
    return True


def check_vector_store() -> bool:
    """Check if vector store exists and is accessible.
    
    Returns:
        True if vector store is accessible, False otherwise
    """
    # Default Chroma directory
    chroma_dir = Path("./chroma_db")
    
    if not chroma_dir.exists():
        log.warning(
            "vector_store_not_found",
            path=str(chroma_dir),
        )
        print("⚠️  Warning: Vector store directory not found")
        print(f"   Expected location: {chroma_dir}")
        print("\nYou may need to run the ingestion process first:")
        print("   uv run python scripts/ingest.py")
        print("\nContinuing anyway...")
        return True  # Don't block startup
    
    log.info("vector_store_found", path=str(chroma_dir))
    return True


def run_streamlit() -> int:
    """Run the Streamlit application.
    
    Returns:
        Exit code from Streamlit process
    """
    app_path = Path("src/query/app.py")
    
    if not app_path.exists():
        log.error("app_file_not_found", path=str(app_path))
        print(f"❌ Error: Application file not found: {app_path}")
        return 1
    
    log.info("starting_streamlit_app", app_path=str(app_path))
    print("🚀 Starting Streamlit application...")
    print(f"   App: {app_path}")
    print(f"   URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run streamlit with uv
        result = subprocess.run(
            ["uv", "run", "streamlit", "run", str(app_path)],
            check=False,
        )
        return result.returncode
        
    except KeyboardInterrupt:
        log.info("streamlit_app_stopped_by_user")
        print("\n\n✅ Streamlit application stopped")
        return 0
        
    except Exception as e:
        log.error("streamlit_app_failed", error=str(e))
        print(f"\n❌ Error running Streamlit: {e}")
        return 1


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    print("=" * 60)
    print("Confluence RAG System - Streamlit Query Interface")
    print("=" * 60)
    print()
    
    # Check environment
    print("🔍 Checking environment...")
    if not check_environment():
        return 1
    print("✅ Environment variables verified")
    print()
    
    # Check vector store
    print("🔍 Checking vector store...")
    check_vector_store()
    print()
    
    # Run Streamlit
    return run_streamlit()


if __name__ == "__main__":
    sys.exit(main())
