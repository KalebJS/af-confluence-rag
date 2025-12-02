#!/usr/bin/env python3
"""Verify that the project setup is complete and all dependencies are available."""

import sys
from pathlib import Path


def check_directories():
    """Check that all required directories exist."""
    required_dirs = [
        "src",
        "src/ingestion",
        "src/query",
        "src/models",
        "src/utils",
        "tests",
        "config",
        "scripts",
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).is_dir():
            missing.append(dir_path)
    
    if missing:
        print(f"❌ Missing directories: {', '.join(missing)}")
        return False
    
    print("✅ All required directories exist")
    return True


def check_files():
    """Check that all required files exist."""
    required_files = [
        "pyproject.toml",
        "README.md",
        ".env.example",
        "config/default.yaml",
        "config/development.yaml",
        "config/production.yaml",
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).is_file():
            missing.append(file_path)
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    
    print("✅ All required files exist")
    return True


def check_dependencies():
    """Check that all core dependencies can be imported."""
    dependencies = [
        "atlassian",
        "langchain",
        "langchain_community",
        "langchain_core",
        "langchain_text_splitters",
        "langchain_chroma",
        "pydantic",
        "pydantic_settings",
        "streamlit",
        "sentence_transformers",
        "chromadb",
        "numpy",
        "yaml",
        "dotenv",
    ]
    
    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            failed.append(dep)
    
    if failed:
        print(f"❌ Failed to import: {', '.join(failed)}")
        return False
    
    print("✅ All core dependencies can be imported")
    return True


def check_dev_dependencies():
    """Check that all dev dependencies can be imported."""
    dev_dependencies = [
        "pytest",
        "hypothesis",
        "black",
        "ruff",
        "mypy",
    ]
    
    failed = []
    for dep in dev_dependencies:
        try:
            __import__(dep)
        except ImportError:
            failed.append(dep)
    
    if failed:
        print(f"❌ Failed to import dev dependencies: {', '.join(failed)}")
        return False
    
    print("✅ All dev dependencies can be imported")
    return True


def main():
    """Run all verification checks."""
    print("🔍 Verifying Confluence RAG System setup...\n")
    
    checks = [
        check_directories(),
        check_files(),
        check_dependencies(),
        check_dev_dependencies(),
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("✅ Setup verification complete! All checks passed.")
        return 0
    else:
        print("❌ Setup verification failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
