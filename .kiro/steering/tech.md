# Technology Stack

## Language & Runtime

- **Python 3.12+** (required minimum version)
- **uv** package manager for dependency management and script execution

## Core Dependencies

- **LangChain**: Document processing and text splitting
- **Chroma**: Vector database for embeddings storage
- **sentence-transformers**: Embedding generation (default model: all-MiniLM-L6-v2)
- **Streamlit**: Web interface framework
- **atlassian-python-api**: Confluence API client
- **Pydantic**: Data validation and settings management
- **structlog**: Structured logging

## Development Tools

- **pytest**: Testing framework with coverage support
- **Hypothesis**: Property-based testing (configured for 100 iterations)
- **Black**: Code formatting (line length: 100)
- **Ruff**: Linting (Python 3.12 target)
- **mypy**: Type checking

## Common Commands

```bash
# Install dependencies
uv sync

# Run ingestion
uv run python scripts/ingest.py

# Run query interface
uv run python scripts/run_app.py
# or directly:
uv run streamlit run src/query/app.py

# Run incremental sync
uv run python scripts/scheduled_sync.py

# Verify setup
uv run python scripts/verify_setup.py

# Testing
uv run pytest                           # Run all tests
uv run pytest --cov=src                 # With coverage
uv run pytest -k "properties"           # Property-based tests only
uv run pytest tests/test_chunker_properties.py  # Specific test file

# Code quality
uv run black src tests scripts          # Format code
uv run ruff check src tests scripts     # Lint
uv run ruff check --fix src tests scripts  # Auto-fix
uv run mypy src                         # Type check
```

## Configuration

- **Environment variables**: Loaded via `.env` file (see `.env.example`)
- **YAML configs**: Located in `config/` directory (default.yaml, development.yaml, production.yaml)
- **Environment variable interpolation**: Use `${VAR_NAME}` syntax in YAML files
- **Config selection**: Set `APP_ENV` environment variable to choose config file
