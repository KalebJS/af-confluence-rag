# Project Structure

## Directory Organization

```
src/                        # Source code (main application package)
├── ingestion/             # Confluence data extraction and ingestion
├── processing/            # Document chunking, embedding, metadata enrichment
├── storage/               # Vector database operations (Chroma)
├── sync/                  # Incremental synchronization logic
├── query/                 # Streamlit app and query processing
├── models/                # Pydantic data models
└── utils/                 # Shared utilities (config, logging, retry)

tests/                     # Test suite
├── test_*_properties.py   # Property-based tests (Hypothesis)
└── test_*.py              # Unit tests

scripts/                   # Executable scripts
├── ingest.py             # Run full ingestion
├── scheduled_sync.py     # Incremental sync
├── run_app.py            # Start Streamlit app
├── verify_setup.py       # Verify installation
├── setup_deployment.py   # Deployment preparation
└── health_check.py       # Health check endpoint

config/                    # YAML configuration files
├── default.yaml          # Base configuration
├── development.yaml      # Dev environment overrides
└── production.yaml       # Production settings

docs/                      # Documentation
```

## Module Responsibilities

### src/ingestion/
- `confluence_client.py`: Wrapper around atlassian-python-api with retry logic
- `ingestion_service.py`: Orchestrates the full ingestion pipeline

### src/processing/
- `chunker.py`: Text chunking using LangChain splitters
- `embedder.py`: Embedding generation with sentence-transformers
- `metadata_enricher.py`: Metadata extraction and enrichment

### src/storage/
- `vector_store.py`: Chroma vector database operations (CRUD, similarity search)

### src/sync/
- `sync_coordinator.py`: Orchestrates incremental sync process
- `change_detector.py`: Detects new/modified/deleted pages
- `timestamp_tracker.py`: Tracks last sync timestamps

### src/query/
- `app.py`: Streamlit web application
- `query_processor.py`: Query embedding and vector search
- `result_formatter.py`: Format search results for display

### src/models/
- `config.py`: Configuration models (AppConfig, ConfluenceConfig, etc.)
- `page.py`: Data models (Page, DocumentChunk, SearchResult, SyncState)

### src/utils/
- `config_loader.py`: YAML config loading with env var substitution
- `logging_config.py`: Structured logging setup with structlog
- `retry.py`: Exponential backoff retry decorator

## Naming Conventions

- **Files**: Snake case (e.g., `confluence_client.py`)
- **Classes**: Pascal case (e.g., `ConfluenceClient`, `AppConfig`)
- **Functions/methods**: Snake case (e.g., `get_space_pages`, `load_config`)
- **Constants**: Upper snake case (e.g., `MAX_RETRIES`)
- **Private attributes**: Leading underscore (e.g., `_client`, `_base_url`)

## Import Organization

Follow this order:
1. Standard library imports
2. Third-party imports (alphabetical)
3. Local application imports (alphabetical)

Use absolute imports from `src/` package root.
