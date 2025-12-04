# Script Entry Points Update

This document describes the updates made to script entry points to use the LangChain abstraction layer via the provider module.

## Updated Scripts

### 1. `scripts/ingest.py`

**Changes:**
- Removed direct imports of `EmbeddingGenerator` and `ChromaStore`
- Added imports for `get_embeddings` and `get_vector_store` from `src.providers`
- Updated `create_ingestion_service()` to use provider module functions
- Now uses `config.vector_store.collection_name` and `config.vector_store.persist_directory` directly (simplified config)

**Before:**
```python
embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)
vector_store = ChromaStore(
    persist_directory=config.vector_store.config["persist_directory"],
    collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
)
```

**After:**
```python
embeddings = get_embeddings(model_name=config.processing.embedding_model)
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name=config.vector_store.collection_name,
    persist_directory=config.vector_store.persist_directory,
)
```

### 2. `scripts/scheduled_sync.py`

**Changes:**
- Removed direct import of `EmbeddingGenerator` and `ChromaStore`
- Added imports for `get_embeddings` and `get_vector_store` from `src.providers`
- Updated `perform_sync()` to use provider module functions
- Now uses simplified config structure

**Before:**
```python
embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)
vector_store = ChromaStore(
    persist_directory=config.vector_store.config["persist_directory"],
    collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
)
```

**After:**
```python
embeddings = get_embeddings(model_name=config.processing.embedding_model)
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name=config.vector_store.collection_name,
    persist_directory=config.vector_store.persist_directory,
)
```

### 3. `src/query/app.py` (Streamlit Application)

**Changes:**
- Removed direct imports of `EmbeddingGenerator` and `ChromaStore`
- Added imports for `get_embeddings` and `get_vector_store` from `src.providers`
- Updated `initialize_app()` to use provider module functions
- Updated sidebar display to show `collection_name` instead of `type`

**Before:**
```python
embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)
vector_store = ChromaStore(
    persist_directory=config.vector_store.config["persist_directory"],
    collection_name=config.vector_store.config.get("collection_name", "confluence_docs"),
)
query_processor = QueryProcessor(embedder=embedder, vector_store=vector_store)
```

**After:**
```python
embeddings = get_embeddings(model_name=config.processing.embedding_model)
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name=config.vector_store.collection_name,
    persist_directory=config.vector_store.persist_directory,
)
query_processor = QueryProcessor(embeddings=embeddings, vector_store=vector_store)
```

## Benefits

1. **Centralized Provider Management**: All scripts now use the provider module, making it easy to swap implementations by modifying only `src/providers.py`

2. **Simplified Configuration**: Scripts use the simplified config structure with direct field access instead of nested dictionaries

3. **LangChain Abstractions**: All scripts now work with LangChain's `Embeddings` and `VectorStore` interfaces, enabling easy swapping of implementations

4. **Consistency**: All entry points follow the same pattern for initialization

## Testing

All scripts have been verified to:
- Import successfully without errors
- Display help messages correctly
- Initialize components using the provider module
- Pass all existing integration tests

## Usage

Scripts continue to work exactly as before from the user's perspective:

```bash
# Ingestion
uv run python scripts/ingest.py --config config/default.yaml

# Scheduled sync
uv run python scripts/scheduled_sync.py --config config/default.yaml

# Query interface
uv run python scripts/run_app.py
```

## Requirements Validated

This update satisfies the following requirements from the specification:
- **Requirement 3.1**: System calls `get_embeddings()` from centralized providers module
- **Requirement 3.2**: System calls `get_vector_store()` from centralized providers module
- **Requirement 3.5**: Providers module modifications apply without requiring changes to other files
