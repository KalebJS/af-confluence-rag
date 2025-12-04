# Project Structure

## Directory Organization

```
src/                        # Source code (main application package)
├── providers.py           # ⭐ Centralized provider configuration (embeddings & vector stores)
├── ingestion/             # Confluence data extraction and ingestion
├── processing/            # Document chunking and metadata enrichment
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
├── API.md                # Component API reference
├── PROVIDER_SWAPPING.md  # Guide for swapping embeddings/vector stores
├── STREAMLIT_APP.md      # Query interface guide
└── POSIT_CONNECT_DEPLOYMENT.md  # Deployment guide
```

## Module Responsibilities

### src/providers.py ⭐
**THE ONLY FILE TO MODIFY FOR SWAPPING PROVIDERS**
- `get_embeddings()`: Factory function returning Embeddings instance (default: HuggingFaceEmbeddings)
- `get_vector_store()`: Factory function returning VectorStore instance (default: Chroma)
- Centralized provider configuration - modify this file to swap implementations system-wide
- Includes examples for common provider swaps (OpenAI, Snowflake, FAISS, etc.)

### src/ingestion/
- `confluence_client.py`: Wrapper around atlassian-python-api with retry logic
- `ingestion_service.py`: Orchestrates the full ingestion pipeline using LangChain interfaces

### src/processing/
- `chunker.py`: Text chunking using LangChain splitters
- `metadata_enricher.py`: Metadata extraction and enrichment

### src/sync/
- `sync_coordinator.py`: Orchestrates incremental sync process
- `change_detector.py`: Detects new/modified/deleted pages
- `timestamp_tracker.py`: Tracks last sync timestamps

### src/query/
- `app.py`: Streamlit web application
- `query_processor.py`: Query processing using LangChain Embeddings and VectorStore interfaces
- `result_formatter.py`: Format search results for display

### src/models/
- `config.py`: Configuration models (AppConfig, ConfluenceConfig, ProcessingConfig, VectorStoreConfig)
- `page.py`: Data models (Page, DocumentChunk, SearchResult, SyncState) with LangChain Document mapping

### src/utils/
- `config_loader.py`: YAML config loading with env var substitution
- `logging_config.py`: Structured logging setup with structlog
- `retry.py`: Exponential backoff retry decorator

## LangChain Integration

The system uses LangChain's standard abstractions:

### Embeddings Interface
- **Interface**: `langchain_core.embeddings.Embeddings`
- **Default**: `HuggingFaceEmbeddings` from `langchain-huggingface`
- **Methods**: `embed_query()`, `embed_documents()`
- **Swappable**: Modify `get_embeddings()` in `src/providers.py`

### VectorStore Interface
- **Interface**: `langchain_core.vectorstores.VectorStore`
- **Default**: `Chroma` from `langchain-chroma`
- **Methods**: `add_documents()`, `similarity_search()`, `similarity_search_with_score()`, `delete()`
- **Swappable**: Modify `get_vector_store()` in `src/providers.py`

### Document Mapping
- **LangChain Document**: `langchain_core.documents.Document`
- **Mapping functions**: `to_langchain_document()`, `from_langchain_document()` in `src/models/page.py`
- **Preserves**: All metadata fields (page_id, page_title, page_url, author, modified_date)

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
