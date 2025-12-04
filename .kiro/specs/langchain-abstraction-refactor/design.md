# Design Document

## Overview

This design document outlines the refactoring of the Confluence RAG System to use LangChain's standard abstractions for embeddings and vector stores. The refactoring replaces direct usage of sentence-transformers and a custom vector store interface with LangChain's `Embeddings` and `VectorStore` base classes. This change enables users to easily swap implementations by either modifying configuration or providing custom implementations through dependency injection.

The refactoring maintains the system's core functionality while improving extensibility and maintainability. The default configuration will use `HuggingFaceEmbeddings` (which wraps sentence-transformers) and LangChain's `Chroma` implementation, ensuring the system remains usable without API keys.

## Architecture

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingestion Service                         │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │ ConfluenceClient │────────▶│  EmbeddingGenerator      │  │
│  │                  │         │  (sentence-transformers) │  │
│  └──────────────────┘         └──────────────────────────┘  │
│           │                              │                   │
│           ▼                              ▼                   │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │     Chunker      │────────▶│  VectorStoreInterface    │  │
│  │                  │         │  (Custom)                │  │
│  └──────────────────┘         └──────────────────────────┘  │
│                                         │                    │
│                                         ▼                    │
│                               ┌──────────────────────────┐   │
│                               │    ChromaStore           │   │
│                               │    (chromadb direct)     │   │
│                               └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Ingestion Service                         │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │ ConfluenceClient │────────▶│  Embeddings              │  │
│  │                  │         │  (LangChain Interface)   │  │
│  └──────────────────┘         └──────────────────────────┘  │
│           │                              ▲                   │
│           ▼                              │                   │
│  ┌──────────────────┐         ┌──────────────────────────┐  │
│  │     Chunker      │         │  HuggingFaceEmbeddings   │  │
│  │                  │         │  (Default)               │  │
│  └──────────────────┘         └──────────────────────────┘  │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              VectorStore                              │   │
│  │              (LangChain Interface)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                              ▲                               │
│                              │                               │
│                   ┌──────────────────────┐                   │
│                   │  Chroma              │                   │
│                   │  (langchain-chroma)  │                   │
│                   └──────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘

                    Factory Pattern
┌─────────────────────────────────────────────────────────────┐
│                  EmbeddingsFactory                           │
│  create_embeddings(config) -> Embeddings                     │
│    - huggingface -> HuggingFaceEmbeddings                    │
│    - openai -> OpenAIEmbeddings                              │
│    - custom -> User-provided implementation                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                  VectorStoreFactory                          │
│  create_vector_store(config, embeddings) -> VectorStore      │
│    - chroma -> Chroma                                        │
│    - faiss -> FAISS                                          │
│    - custom -> User-provided implementation                  │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Embeddings Abstraction

**Interface**: `langchain_core.embeddings.Embeddings`

**Key Methods**:
- `embed_documents(texts: list[str]) -> list[list[float]]`: Generate embeddings for multiple documents
- `embed_query(text: str) -> list[float]`: Generate embedding for a single query

**Default Implementation**: `HuggingFaceEmbeddings` from `langchain-huggingface`

**Configuration** (model name only):
```yaml
processing:
  embedding_model: "all-MiniLM-L6-v2"
```

### 2. VectorStore Abstraction

**Interface**: `langchain_core.vectorstores.VectorStore`

**Key Methods**:
- `add_documents(documents: list[Document], **kwargs) -> list[str]`: Add documents with automatic embedding
- `add_texts(texts: list[str], metadatas: list[dict] | None, **kwargs) -> list[str]`: Add texts with metadata
- `similarity_search(query: str, k: int, **kwargs) -> list[Document]`: Search by query string
- `similarity_search_with_score(query: str, k: int, **kwargs) -> list[tuple[Document, float]]`: Search with scores
- `delete(ids: list[str], **kwargs) -> bool | None`: Delete documents by ID

**Default Implementation**: `Chroma` from `langchain-chroma`

**Configuration** (persistence settings only):
```yaml
vector_store:
  collection_name: "confluence_docs"
  persist_directory: "./chroma_db"
```

### 3. Provider Module

**Location**: `src/providers.py`

**Purpose**: Centralized location where developers specify which Embeddings and VectorStore implementations to use. This is the ONLY file developers need to modify to swap implementations.

**Functions**:

```python
def get_embeddings(model_name: str) -> Embeddings:
    """Get the configured embeddings implementation.
    
    Developers: Modify this function to change the embedding provider.
    Default: HuggingFaceEmbeddings
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Embeddings instance
    """
    # Default implementation - developers modify this
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
    
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    """Get the configured vector store implementation.
    
    Developers: Modify this function to change the vector store provider.
    Default: Chroma
    
    Args:
        embeddings: Embeddings instance to use
        collection_name: Name of the collection
        persist_directory: Directory for persistence
        
    Returns:
        VectorStore instance
    """
    # Default implementation - developers modify this
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
```

**Example: Swapping to Snowflake VectorStore**:

```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    """Get the configured vector store implementation."""
    from langchain_snowflake import SnowflakeVectorStore
    return SnowflakeVectorStore(
        embedding=embeddings,
        # Snowflake-specific configuration
        connection_params={...}
    )
```

### 4. Updated Configuration Models

**Location**: `src/models/config.py`

**Changes**:
- Keep `embedding_model` field in `ProcessingConfig` (no provider field)
- Simplify `VectorStoreConfig` to only include persistence settings (no provider field)

```python
class ProcessingConfig(BaseModel):
    """Configuration for document processing."""
    
    chunk_size: int = Field(default=1000, ge=500, le=2000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    
    collection_name: str = Field(default="confluence_docs")
    persist_directory: str = Field(default="./chroma_db")
```

### 5. Service Updates

**IngestionService** (`src/ingestion/ingestion_service.py`):
- Accept `Embeddings` and `VectorStore` instances via constructor
- Remove direct instantiation of `EmbeddingGenerator` and `ChromaStore`
- Call `get_embeddings()` and `get_vector_store()` from providers module if instances not provided

**QueryProcessor** (`src/query/query_processor.py`):
- Accept `Embeddings` and `VectorStore` instances via constructor
- Use `embed_query()` method for query embedding
- Use `similarity_search_with_score()` for retrieval
- Call `get_embeddings()` and `get_vector_store()` from providers module if instances not provided

### 6. Document Mapping

**Purpose**: Map between internal `DocumentChunk` model and LangChain's `Document` class

**Location**: `src/models/page.py`

**Methods**:
```python
def to_langchain_document(chunk: DocumentChunk) -> Document:
    """Convert DocumentChunk to LangChain Document."""
    
def from_langchain_document(doc: Document, chunk_id: str, page_id: str) -> DocumentChunk:
    """Convert LangChain Document to DocumentChunk."""
```

## Data Models

### DocumentChunk (Existing)

```python
@dataclass
class DocumentChunk:
    chunk_id: str
    page_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any]
```

### LangChain Document (External)

```python
class Document:
    page_content: str
    metadata: dict[str, Any]
```

### Mapping Strategy

- `DocumentChunk.content` ↔ `Document.page_content`
- `DocumentChunk.metadata` ↔ `Document.metadata` (with additional fields)
- Store `chunk_id`, `page_id`, and `chunk_index` in `Document.metadata`

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the acceptance criteria analysis, the following properties must hold:

### Property 1: Embeddings Interface Compliance

*For any* text input, when the system generates embeddings, the embeddings instance SHALL be of type `langchain_core.embeddings.Embeddings` and SHALL support both `embed_documents` and `embed_query` methods.

**Validates: Requirements 1.1, 1.4**

### Property 2: VectorStore Interface Compliance

*For any* vector operation, the vector store instance SHALL be of type `langchain_core.vectorstores.VectorStore` and SHALL support `add_documents`, `similarity_search`, and deletion methods.

**Validates: Requirements 2.1, 2.3, 2.4, 2.5**

### Property 3: Provider Module Returns Correct Types

*For any* call to `get_embeddings()` or `get_vector_store()`, the returned instances SHALL be of types that implement the `Embeddings` and `VectorStore` interfaces respectively.

**Validates: Requirements 3.3, 6.1, 6.2**

### Property 4: Dependency Injection Override

*For any* service that accepts Embeddings or VectorStore instances, when a custom implementation is provided via constructor, the service SHALL use the provided instance instead of creating one from configuration.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

### Property 5: Model Name Compatibility

*For any* sentence-transformers model name that worked in the previous implementation, the new HuggingFaceEmbeddings SHALL successfully initialize with that model name and produce embeddings of the same dimensionality.

**Validates: Requirements 1.3, 9.3**

### Property 6: Metadata Preservation Through Vector Operations

*For any* DocumentChunk with metadata fields (page_id, page_title, page_url, author, modified_date), when added to the vector store and retrieved via search, all metadata fields SHALL be preserved and accessible.

**Validates: Requirements 11.1, 11.2**

### Property 7: Document Mapping Round-Trip

*For any* DocumentChunk, converting it to a LangChain Document and back SHALL produce an equivalent DocumentChunk with the same content, metadata, chunk_id, page_id, and chunk_index.

**Validates: Requirements 11.3**

### Property 8: Metadata-Based Deletion

*For any* page_id, when documents are deleted by page_id, all documents with that page_id SHALL be removed from the vector store, and subsequent searches SHALL not return any documents with that page_id.

**Validates: Requirements 11.4**

### Property 9: Provider Module Error Handling

*For any* invalid parameters passed to `get_embeddings()` or `get_vector_store()`, the functions SHALL raise clear errors indicating the problem (e.g., invalid model name, missing configuration).

**Validates: Requirements 6.5**

## Error Handling

### Embedding Generation Errors

1. **Empty Text Handling**: When empty or whitespace-only text is provided, the system will delegate to LangChain's default behavior (typically returning a zero vector or raising an error depending on the implementation)

2. **Model Loading Errors**: If the specified embedding model cannot be loaded, the factory will raise a clear error with the model name and suggestions for valid models

3. **Dimension Mismatch**: If embeddings from different models are mixed, the vector store will raise an error during insertion

### Vector Store Errors

1. **Connection Failures**: If the vector store cannot connect to its backend (e.g., Chroma persistence directory is not writable), initialization will fail with a clear error message

2. **Deletion Not Supported**: If a vector store implementation doesn't support deletion, the system will log a warning and document the limitation

3. **Metadata Filtering Limitations**: If a vector store doesn't support metadata-based filtering for deletion, the system will fall back to retrieving all documents and filtering client-side

### Configuration Errors

1. **Missing Required Fields**: If required configuration fields are missing, Pydantic validation will raise a clear error

2. **Invalid Provider Types**: If an unknown provider is specified, the factory will raise a ValueError listing supported providers

3. **Incompatible Configuration**: If provider-specific configuration is invalid, the error will include the provider name and expected configuration format

## Testing Strategy

### Unit Tests

Unit tests will verify specific scenarios and integration points:

1. **Factory Tests**: Verify that factories create correct implementation types for each supported provider
2. **Configuration Tests**: Verify that configuration models validate correctly and reject invalid inputs
3. **Mapping Tests**: Verify DocumentChunk ↔ Document conversion preserves all fields
4. **Service Integration Tests**: Verify services work with both factory-created and injected implementations

### Property-Based Tests

Property-based tests will use Hypothesis to verify universal properties across many inputs:

**Testing Framework**: Hypothesis (Python)
**Minimum Iterations**: 100 per property test

Each property-based test will be tagged with a comment referencing the correctness property:

```python
@given(st.text())
def test_embeddings_interface_compliance(text: str):
    """Feature: langchain-abstraction-refactor, Property 1: Embeddings Interface Compliance"""
    # Test implementation
```

**Property Test Coverage**:

1. **Property 1 Test**: Generate random text, create embeddings via provider module, verify instance type and method availability
2. **Property 2 Test**: Perform random vector operations, verify vector store instance type and method availability
3. **Property 3 Test**: Call provider module functions, verify they return correct interface types
4. **Property 4 Test**: Create services with custom implementations, verify they're used instead of provider module defaults
5. **Property 5 Test**: Test with various sentence-transformers model names, verify dimensionality consistency
6. **Property 6 Test**: Generate random DocumentChunks with metadata, add to vector store, search, verify metadata preserved
7. **Property 7 Test**: Generate random DocumentChunks, test round-trip conversion
8. **Property 8 Test**: Generate random page_ids, add documents, delete by page_id, verify removal
9. **Property 9 Test**: Test with invalid parameters, verify clear error messages

**Test Generators**:

- `document_chunk_strategy`: Generates valid DocumentChunk instances with random content and metadata
- `config_strategy`: Generates valid configuration objects with various provider combinations
- `text_strategy`: Generates text of various lengths including edge cases (empty, very long, special characters)
- `metadata_strategy`: Generates metadata dictionaries with required and optional fields

### Integration Tests

Integration tests will verify end-to-end workflows:

1. **Full Ingestion Pipeline**: Test complete flow from Confluence extraction through embedding to vector storage
2. **Query Pipeline**: Test complete flow from query input through embedding to result retrieval
3. **Incremental Sync**: Test update and deletion workflows with the new abstractions

### Backward Compatibility Tests

Tests will verify that existing functionality is preserved:

1. **Existing Test Suite**: All existing unit and property tests should pass with minimal modifications
2. **Embedding Dimension Tests**: Verify same models produce same dimensions as before
3. **Search Quality Tests**: Verify search results are comparable to previous implementation

## Implementation Notes

### Migration Path

1. **Phase 1**: Add new dependencies and providers module
2. **Phase 2**: Update configuration models (simplify)
3. **Phase 3**: Implement document mapping functions
4. **Phase 4**: Update services to accept LangChain interfaces
5. **Phase 5**: Update all calling code to use provider module
6. **Phase 6**: Remove old EmbeddingGenerator and VectorStoreInterface
7. **Phase 7**: Update tests
8. **Phase 8**: Update documentation

### Dependency Versions

- `langchain-core`: ^0.3.0 (for base abstractions)
- `langchain-huggingface`: ^0.1.0 (for HuggingFaceEmbeddings)
- `langchain-chroma`: ^0.1.0 (for Chroma integration)
- `chromadb`: ^0.4.0 (maintained as direct dependency)

### Type Hints

All functions will use Python 3.12+ type hint syntax:

```python
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

def create_embeddings(config: ProcessingConfig) -> Embeddings:
    """Create embeddings instance."""
    
def process_documents(
    documents: list[Document],
    embeddings: Embeddings,
    vector_store: VectorStore
) -> None:
    """Process documents with provided implementations."""
```

### Performance Considerations

1. **Batch Processing**: Use `embed_documents` for batch embedding generation to leverage model optimizations
2. **Lazy Loading**: Delay model loading until first use to improve startup time
3. **Connection Pooling**: Reuse vector store connections across operations
4. **Caching**: Consider caching embeddings for frequently accessed queries

### Extensibility Points

Developers can extend the system by:

1. **Modify Provider Module**: Edit `src/providers.py` to change default implementations
   - Change `get_embeddings()` to return a different Embeddings implementation
   - Change `get_vector_store()` to return a different VectorStore implementation

2. **Custom Embeddings**: Implement `langchain_core.embeddings.Embeddings` interface and return it from `get_embeddings()`

3. **Custom VectorStore**: Implement `langchain_core.vectorstores.VectorStore` interface and return it from `get_vector_store()`

4. **Dependency Injection**: Pass custom implementations directly to services, bypassing the provider module

Example - Swapping to OpenAI embeddings:

```python
# In src/providers.py
def get_embeddings(model_name: str) -> Embeddings:
    """Get the configured embeddings implementation."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)
```

Example - Using custom implementation via dependency injection:

```python
from langchain_core.embeddings import Embeddings

class MyCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Custom implementation
        pass
    
    def embed_query(self, text: str) -> list[float]:
        # Custom implementation
        pass

# Use via dependency injection
embeddings = MyCustomEmbeddings()
service = IngestionService(embeddings=embeddings, vector_store=vector_store)
```
