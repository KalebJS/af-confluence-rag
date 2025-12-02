# Design Document

## Overview

The Confluence RAG System is a Python-based monorepo application that enables semantic search over Confluence documentation. The system consists of two primary components: an ingestion service that extracts and vectorizes Confluence content, and a query interface that provides users with semantic search capabilities. The architecture follows a modular design with clear separation between data extraction, processing, storage, and presentation layers.

The system leverages LangChain for document processing, Chroma as the vector database for its simplicity and Python compatibility, and Streamlit for the user interface due to its excellent Posit Connect support. The embedding strategy uses sentence-transformers (all-MiniLM-L6-v2) for local execution, providing a balance between performance and resource requirements without external API dependencies.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Confluence RAG System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Ingestion       │         │  Query           │          │
│  │  Service         │         │  Interface       │          │
│  │                  │         │  (Streamlit)     │          │
│  │  ┌────────────┐  │         │                  │          │
│  │  │ Confluence │  │         │  ┌────────────┐  │          │
│  │  │ Extractor  │  │         │  │ Search UI  │  │          │
│  │  └─────┬──────┘  │         │  └─────┬──────┘  │          │
│  │        │         │         │        │         │          │
│  │  ┌─────▼──────┐  │         │  ┌─────▼──────┐  │          │
│  │  │ Document   │  │         │  │ Query      │  │          │
│  │  │ Processor  │  │         │  │ Processor  │  │          │
│  │  └─────┬──────┘  │         │  └─────┬──────┘  │          │
│  │        │         │         │        │         │          │
│  │  ┌─────▼──────┐  │         │  ┌─────▼──────┐  │          │
│  │  │ Embedder   │  │         │  │ Result     │  │          │
│  │  └─────┬──────┘  │         │  │ Formatter  │  │          │
│  │        │         │         │  └─────┬──────┘  │          │
│  └────────┼─────────┘         └────────┼─────────┘          │
│           │                            │                    │
│           └────────┬───────────────────┘                    │
│                    │                                        │
│           ┌────────▼─────────┐                              │
│           │  Vector Database │                              │
│           │  (Chroma)        │                              │
│           └──────────────────┘                              │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Shared Components                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Config       │  │ Logging      │  │ Utilities    │      │
│  │ Manager      │  │ System       │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

**Ingestion Service:**
- Connects to Confluence API and authenticates
- Retrieves pages from specified spaces with pagination handling
- Processes and chunks documents using LangChain text splitters
- Generates embeddings using sentence-transformers
- Stores vectors and metadata in Chroma database
- Manages incremental updates and synchronization

**Query Interface (Streamlit):**
- Provides web-based search interface
- Accepts natural language queries from users
- Converts queries to embeddings
- Retrieves relevant documents from vector database
- Displays formatted results with metadata and links

**Vector Database (Chroma):**
- Persists vector embeddings with metadata
- Performs similarity search operations
- Manages document identifiers and deduplication
- Provides CRUD operations for vector management

**Shared Components:**
- Configuration management for environment-specific settings
- Centralized logging with structured output
- Common utilities for error handling and retries

## Components and Interfaces

### 1. Confluence Extractor

**Purpose:** Interfaces with Confluence REST API to retrieve documentation content using the atlassian-python-api library.

**Key Classes:**
- `ConfluenceClient`: Wraps atlassian-python-api Confluence client
- `SpaceExtractor`: Retrieves all pages from a space
- `PageParser`: Extracts content and metadata from pages

**Interfaces:**
```python
from atlassian import Confluence

class ConfluenceClient:
    """Wrapper around atlassian-python-api Confluence client"""
    def __init__(self, base_url: str, auth_token: str, cloud: bool = True):
        """
        Initialize Confluence client
        
        Args:
            base_url: Confluence instance URL
            auth_token: API token for authentication
            cloud: True for Confluence Cloud, False for Server/Data Center
        """
        self._client = Confluence(
            url=base_url,
            token=auth_token,
            cloud=cloud
        )
    
    def get_space_pages(self, space_key: str) -> List[Page]:
        """Get all pages from a space using generator for memory efficiency"""
        pass
    
    def get_page_content(self, page_id: str, expand: str = 'body.storage,version,history') -> Page:
        """Get page content with metadata"""
        pass
    
    def get_page_by_title(self, space_key: str, title: str) -> Optional[Page]:
        """Get a specific page by title"""
        pass
```

Note: Page model is defined in the Data Models section using Pydantic BaseModel

**Dependencies:**
- `atlassian-python-api` for Confluence API operations (provides built-in authentication, pagination, and error handling)
- `requests` library (dependency of atlassian-python-api)

**Design Rationale:**
Using `atlassian-python-api` provides several advantages:
- Built-in authentication handling for both Cloud (API tokens) and Server (username/password)
- Automatic pagination through generator methods (`get_all_pages_from_space_as_generator`)
- Comprehensive API coverage including content, spaces, users, and attachments
- Active maintenance and community support
- Handles Confluence-specific quirks and API differences between Cloud and Server

**Alternative Approach:**
LangChain provides a `ConfluenceLoader` in `langchain-community` that wraps `atlassian-python-api` and returns LangChain Document objects. This can be used as an alternative to the custom wrapper:

```python
from langchain_community.document_loaders import ConfluenceLoader

loader = ConfluenceLoader(
    url="https://your-domain.atlassian.net",
    token="your-api-token",
    cloud=True
)

# Load all pages from a space
documents = loader.load(space_key="DOCS", include_attachments=False)
```

The custom wrapper approach provides more control over metadata extraction and error handling, while the LangChain loader provides tighter integration with the LangChain ecosystem. The implementation can support both approaches through configuration.

### 2. Document Processor

**Purpose:** Transforms raw Confluence content into processable chunks.

**Key Classes:**
- `DocumentChunker`: Splits documents using LangChain text splitters
- `MetadataEnricher`: Adds contextual metadata to chunks
- `ContentCleaner`: Removes Confluence-specific markup

**Interfaces:**
```python
class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int)
    def chunk_document(self, content: str, metadata: dict) -> List[DocumentChunk]
    
```

Note: DocumentChunk model is defined in the Data Models section using Pydantic BaseModel

**Dependencies:**
- `langchain-text-splitters` for RecursiveCharacterTextSplitter
- `langchain-community` for document loaders

### 3. Embedder

**Purpose:** Generates vector embeddings from text chunks.

**Key Classes:**
- `EmbeddingGenerator`: Creates embeddings using sentence-transformers
- `EmbeddingCache`: Caches embeddings to avoid recomputation

**Interfaces:**
```python
class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2")
    def generate_embedding(self, text: str) -> np.ndarray
    def generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]
    def get_embedding_dimension(self) -> int
```

**Dependencies:**
- `sentence-transformers` for embedding generation
- `numpy` for array operations

### 4. Vector Store Manager

**Purpose:** Manages interactions with vector databases through a pluggable interface.

**Key Classes:**
- `VectorStoreInterface`: Abstract base class defining vector store operations
- `ChromaStore`: Chroma implementation of VectorStoreInterface (default)
- `VectorStoreFactory`: Factory for creating vector store instances based on configuration
- `DocumentIndexer`: Handles document insertion and updates
- `SearchEngine`: Performs similarity searches

**Interfaces:**
```python
from abc import ABC, abstractmethod

class VectorStoreInterface(ABC):
    """Abstract interface for vector database operations"""
    
    @abstractmethod
    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[np.ndarray]):
        """Add documents with embeddings to the vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> List[SearchResult]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_by_page_id(self, page_id: str):
        """Delete all chunks associated with a page"""
        pass
    
    @abstractmethod
    def get_document_metadata(self, page_id: str) -> Optional[dict]:
        """Retrieve metadata for a document"""
        pass

class ChromaStore(VectorStoreInterface):
    """Chroma implementation of vector store"""
    def __init__(self, persist_directory: str, collection_name: str)
    # Implements all abstract methods

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    @staticmethod
    def create_vector_store(store_type: str, config: dict) -> VectorStoreInterface:
        """
        Create a vector store instance based on type
        
        Args:
            store_type: One of 'chroma', 'faiss', 'qdrant', 'pinecone', etc.
            config: Configuration dictionary for the specific store
        
        Returns:
            VectorStoreInterface implementation
        """
        pass

```

Note: SearchResult model is defined in the Data Models section using Pydantic BaseModel

**Dependencies:**
- `langchain-core` for base vector store abstractions
- `langchain-chroma` for Chroma implementation (default)
- Optional: `langchain-community` for additional vector store implementations (FAISS, Qdrant, etc.)

**Design Rationale:**
The vector store is designed as a pluggable component using the Strategy pattern. This allows easy swapping between different LangChain-compatible vector databases (Chroma, FAISS, Qdrant, Pinecone, Weaviate, etc.) by:
1. Defining a common interface (`VectorStoreInterface`)
2. Implementing adapters for each vector database
3. Using a factory pattern to instantiate the correct implementation based on configuration
4. Leveraging LangChain's built-in vector store abstractions where possible

### 5. Synchronization Manager

**Purpose:** Manages incremental updates and change detection.

**Key Classes:**
- `SyncCoordinator`: Orchestrates synchronization process
- `ChangeDetector`: Identifies new, modified, and deleted pages
- `TimestampTracker`: Maintains last sync timestamps

**Interfaces:**
```python
class SyncCoordinator:
    def __init__(self, confluence_client: ConfluenceClient, vector_store: ChromaStore)
    def sync_space(self, space_key: str) -> SyncReport
    def detect_changes(self, space_key: str) -> ChangeSet
    def apply_changes(self, changes: ChangeSet)

class ChangeSet:
    new_pages: List[Page]
    modified_pages: List[Page]
    deleted_page_ids: List[str]
    
class SyncReport:
    pages_added: int
    pages_updated: int
    pages_deleted: int
    duration_seconds: float
```

### 6. Query Interface (Streamlit)

**Purpose:** Provides web-based user interface for searching documentation.

**Key Components:**
- `SearchUI`: Main Streamlit application
- `QueryProcessor`: Handles query embedding and search
- `ResultFormatter`: Formats and displays search results

**Interfaces:**
```python
class QueryProcessor:
    def __init__(self, embedder: EmbeddingGenerator, vector_store: ChromaStore)
    def process_query(self, query: str, top_k: int) -> List[SearchResult]
    
class ResultFormatter:
    def format_results(self, results: List[SearchResult]) -> str
    def create_result_card(self, result: SearchResult) -> dict
```

**Dependencies:**
- `streamlit` for UI framework
- `pandas` for data display

### 7. Configuration Manager

**Purpose:** Manages application configuration and environment variables.

**Key Classes:**
- `ConfigLoader`: Loads configuration from files and environment
- `ConfigValidator`: Validates required configuration parameters

**Interfaces:**
Note: Configuration models (AppConfig, ConfluenceConfig, ProcessingConfig, VectorStoreConfig) are defined in the Data Models section using Pydantic BaseModel

```python

class ConfigLoader:
    def load_config(self, config_path: Optional[str] = None) -> AppConfig:
        """Load configuration from file and environment variables"""
        pass
    
    def validate_config(self, config: AppConfig) -> List[str]:
        """Validate configuration (Pydantic handles most validation automatically)"""
        pass
```

**Example Configuration:**
```yaml
# config/default.yaml
confluence:
  base_url: ${CONFLUENCE_BASE_URL}
  auth_token: ${CONFLUENCE_AUTH_TOKEN}
  space_key: ${CONFLUENCE_SPACE_KEY}

processing:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  type: "chroma"  # Can be changed to 'faiss', 'qdrant', etc.
  config:
    persist_directory: "./chroma_db"
    collection_name: "confluence_docs"
  # Alternative example for FAISS:
  # type: "faiss"
  # config:
  #   index_path: "./faiss_index"
  # Alternative example for Qdrant:
  # type: "qdrant"
  # config:
  #   url: "http://localhost:6333"
  #   collection_name: "confluence_docs"

query:
  top_k_results: 10
```

## Data Models

All data models use Pydantic BaseModel for validation, serialization, and type safety.

### Page Model
```python
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional

class Page(BaseModel):
    """Represents a Confluence page with metadata"""
    id: str = Field(..., description="Unique page identifier")
    title: str = Field(..., description="Page title")
    space_key: str = Field(..., description="Confluence space key")
    content: str = Field(..., description="Page content in storage format")
    author: str = Field(..., description="Page author username")
    created_date: datetime = Field(..., description="Page creation timestamp")
    modified_date: datetime = Field(..., description="Last modification timestamp")
    url: HttpUrl = Field(..., description="Full URL to the page")
    version: int = Field(..., ge=1, description="Page version number")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "123456",
                "title": "Getting Started",
                "space_key": "DOCS",
                "content": "<p>Welcome to our documentation</p>",
                "author": "john.doe",
                "created_date": "2024-01-01T10:00:00Z",
                "modified_date": "2024-01-15T14:30:00Z",
                "url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                "version": 3
            }
        }
```

### Document Chunk Model
```python
class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata"""
    chunk_id: str = Field(..., description="Unique chunk identifier (format: {page_id}_{chunk_index})")
    page_id: str = Field(..., description="Parent page identifier")
    content: str = Field(..., min_length=1, description="Chunk text content")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (title, url, author, dates)")
    chunk_index: int = Field(..., ge=0, description="Zero-based chunk position within page")
    
    @property
    def page_title(self) -> Optional[str]:
        """Extract page title from metadata"""
        return self.metadata.get("page_title")
    
    @property
    def page_url(self) -> Optional[str]:
        """Extract page URL from metadata"""
        return self.metadata.get("page_url")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "123456_0",
                "page_id": "123456",
                "content": "This is the first chunk of content...",
                "metadata": {
                    "page_title": "Getting Started",
                    "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                    "author": "john.doe",
                    "modified_date": "2024-01-15T14:30:00Z"
                },
                "chunk_index": 0
            }
        }
```

### Search Result Model
```python
class SearchResult(BaseModel):
    """Represents a search result with relevance score"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    page_id: str = Field(..., description="Parent page identifier")
    page_title: str = Field(..., description="Page title")
    page_url: HttpUrl = Field(..., description="Full URL to the source page")
    content: str = Field(..., description="Chunk text content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "123456_0",
                "page_id": "123456",
                "page_title": "Getting Started",
                "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
                "content": "This is the first chunk of content...",
                "similarity_score": 0.87,
                "metadata": {
                    "author": "john.doe",
                    "modified_date": "2024-01-15T14:30:00Z"
                }
            }
        }
```

### Sync State Model
```python
class SyncState(BaseModel):
    """Tracks synchronization state for a space"""
    space_key: str = Field(..., description="Confluence space key")
    last_sync_timestamp: datetime = Field(..., description="Last successful sync timestamp")
    page_count: int = Field(..., ge=0, description="Total number of pages synced")
    chunk_count: int = Field(..., ge=0, description="Total number of chunks created")
    
    class Config:
        json_schema_extra = {
            "example": {
                "space_key": "DOCS",
                "last_sync_timestamp": "2024-01-15T14:30:00Z",
                "page_count": 150,
                "chunk_count": 1250
            }
        }
```

### Configuration Model
```python
class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    type: str = Field(..., description="Vector store type (chroma, faiss, qdrant, etc.)")
    config: dict = Field(default_factory=dict, description="Store-specific configuration")

class ProcessingConfig(BaseModel):
    """Configuration for document processing"""
    chunk_size: int = Field(1000, ge=500, le=2000, description="Target chunk size in tokens")
    chunk_overlap: int = Field(200, ge=0, le=500, description="Overlap between chunks in tokens")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model name")

class ConfluenceConfig(BaseModel):
    """Configuration for Confluence connection"""
    base_url: HttpUrl = Field(..., description="Confluence instance URL")
    auth_token: str = Field(..., description="API authentication token")
    space_key: str = Field(..., description="Space key to sync")
    cloud: bool = Field(True, description="True for Cloud, False for Server/Data Center")

class AppConfig(BaseModel):
    """Main application configuration"""
    confluence: ConfluenceConfig
    processing: ProcessingConfig
    vector_store: VectorStoreConfig
    top_k_results: int = Field(10, ge=1, le=100, description="Number of search results to return")
    
    class Config:
        env_prefix = "APP_"
```

**Benefits of Pydantic Models:**
- Automatic validation of data types and constraints
- JSON serialization/deserialization out of the box
- Environment variable parsing with `pydantic-settings`
- Clear documentation through Field descriptions
- Type hints for better IDE support
- Immutability options for thread safety

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Data Extraction and Retrieval Properties

Property 1: Complete page retrieval
*For any* Confluence space, when the system retrieves pages, the count of retrieved pages should equal the total page count reported by the Confluence API
**Validates: Requirements 1.2**

Property 2: Pagination completeness
*For any* paginated API response, the system should retrieve all pages by following pagination links until no next page exists
**Validates: Requirements 1.3**

Property 3: Metadata completeness
*For any* retrieved Confluence page, the extracted data should contain non-empty values for title, author, creation date, modified date, and URL
**Validates: Requirements 1.4**

Property 4: Exponential backoff behavior
*For any* sequence of rate limit errors, the retry delays should increase exponentially (each delay should be at least double the previous delay)
**Validates: Requirements 1.5**

### Document Processing Properties

Property 5: Chunk size bounds
*For any* document that is chunked, all resulting chunks should have token counts between 500 and 2000 tokens (inclusive)
**Validates: Requirements 2.1**

Property 6: Boundary-aware splitting
*For any* text chunk boundary, the split should occur at a paragraph break, sentence boundary, or word boundary (never mid-word)
**Validates: Requirements 2.2**

Property 7: Embedding generation completeness
*For any* set of document chunks, the number of generated embeddings should equal the number of chunks
**Validates: Requirements 2.3**

Property 8: Metadata preservation in embeddings
*For any* generated embedding, the associated metadata should contain page_title, page_url, and chunk_index fields
**Validates: Requirements 2.4**

### Storage and Persistence Properties

Property 9: Storage round-trip consistency
*For any* document chunk and embedding stored in the vector database, retrieving by chunk_id should return the same content and metadata
**Validates: Requirements 2.5, 3.5**

Property 10: Metadata storage completeness
*For any* vector stored in the database, all required metadata fields (page_id, page_title, page_url, chunk_index, content) should be present and non-empty
**Validates: Requirements 3.2**

Property 11: Unique identifier generation
*For any* set of stored chunks, all chunk_ids should be unique and follow the format {page_id}_{chunk_index}
**Validates: Requirements 3.3**

Property 12: Deduplication idempotence
*For any* document chunk, storing it multiple times should result in only one entry in the vector database (subsequent stores should be no-ops or updates)
**Validates: Requirements 3.4**

### Synchronization Properties

Property 13: Timestamp comparison correctness
*For any* page with a modification timestamp newer than the stored timestamp, the system should identify it as modified
**Validates: Requirements 4.1**

Property 14: Update replaces old embeddings
*For any* page that is updated, after synchronization completes, only embeddings with the new modification timestamp should exist (no old embeddings should remain)
**Validates: Requirements 4.2**

Property 15: New page processing
*For any* page that exists in Confluence but not in the vector database, synchronization should result in embeddings being created for that page
**Validates: Requirements 4.3**

Property 16: Deletion completeness
*For any* page_id that is deleted, after synchronization, no embeddings with that page_id should exist in the vector database
**Validates: Requirements 4.4**

Property 17: Sync timestamp update
*For any* synchronization operation that completes successfully, the last_sync_timestamp should be updated to a value greater than the previous timestamp
**Validates: Requirements 4.5**

### Query and Search Properties

Property 18: Embedding model consistency
*For any* query embedding and document embedding, they should have the same dimensionality (indicating the same model was used)
**Validates: Requirements 5.1**

Property 19: Result count correctness
*For any* search query with parameter top_k, the number of returned results should be min(top_k, total_documents_in_database)
**Validates: Requirements 5.2**

Property 20: Search result completeness
*For any* search result, it should contain non-empty values for content, page_title, page_url, and similarity_score
**Validates: Requirements 5.3**

Property 21: Result ranking order
*For any* list of search results, each result's similarity_score should be greater than or equal to the next result's similarity_score (descending order)
**Validates: Requirements 5.4**

Property 22: Valid result URLs
*For any* search result, the page_url field should be a valid URL starting with the configured Confluence base URL
**Validates: Requirements 5.5**

Property 23: Result metadata display
*For any* rendered search result in the UI, the output should contain the page_title and similarity_score
**Validates: Requirements 6.4**

### Configuration and Environment Properties

Property 24: Environment variable loading
*For any* required environment variable (CONFLUENCE_BASE_URL, CONFLUENCE_AUTH_TOKEN), the system should read its value from the environment at startup
**Validates: Requirements 7.1**

Property 25: Configuration file parsing
*For any* valid configuration file, the system should successfully load all parameters (chunk_size, embedding_model, vector_db_path, collection_name)
**Validates: Requirements 7.2**

Property 26: Missing configuration error handling
*For any* missing required configuration parameter, the system should fail at startup with an error message containing the parameter name
**Validates: Requirements 7.4**

Property 27: Multi-environment configuration support
*For any* two different configuration files, the system should be able to load either one based on an environment variable or command-line argument
**Validates: Requirements 7.5**

### Deployment and Compatibility Properties

Property 28: Dependency file presence
*For any* packaged system, a requirements.txt or pyproject.toml file should exist at the repository root
**Validates: Requirements 8.1**

Property 29: Python version verification
*For any* running system instance, the Python version should be 3.12.x
**Validates: Requirements 8.3**

Property 30: Environment-based authentication
*For any* external service connection, authentication credentials should be retrieved from environment variables
**Validates: Requirements 8.4**

### Error Handling and Logging Properties

Property 31: Error log format
*For any* logged error, the log entry should contain timestamp, severity level, and error message fields
**Validates: Requirements 9.1**

Property 32: Retry with backoff
*For any* connection failure to Confluence, the system should retry at least once with a delay before failing permanently
**Validates: Requirements 9.2**

Property 33: Graceful error recovery
*For any* invalid document encountered during batch processing, the system should log the error and continue processing remaining documents
**Validates: Requirements 9.3**

Property 34: Database unavailability error handling
*For any* vector database connection failure, the system should log the error and return a user-friendly error message
**Validates: Requirements 9.4**

Property 35: Completion logging
*For any* completed ingestion operation, the log should contain summary statistics including document_count and duration_seconds
**Validates: Requirements 9.5**

### Code Organization Properties

Property 36: Dependency management file presence
*For any* repository checkout, a single dependency management file (requirements.txt or pyproject.toml) should exist at the root level
**Validates: Requirements 10.3**

Property 37: Documentation presence
*For any* repository checkout, a README.md file should exist at the root containing sections for setup, architecture, and usage
**Validates: Requirements 10.5**

### Vector Store Abstraction Properties

Property 38: Vector store interface compliance
*For any* vector store implementation, it should implement all methods defined in VectorStoreInterface (add_documents, search, delete_by_page_id, get_document_metadata)
**Validates: Design requirement for pluggable vector stores**

Property 39: Vector store factory instantiation
*For any* valid vector_store_type in configuration ('chroma', 'faiss', 'qdrant'), the VectorStoreFactory should successfully create an instance implementing VectorStoreInterface
**Validates: Design requirement for pluggable vector stores**

## Error Handling

### Error Categories

**1. External Service Errors**
- Confluence API unavailable or unreachable
- Authentication failures (invalid tokens, expired credentials)
- Rate limiting and throttling
- Network timeouts and connection failures

**Strategy:**
- Implement exponential backoff with jitter for retries
- Maximum retry attempts: 3
- Log all external service errors with full context
- Provide clear user-facing error messages

**2. Data Processing Errors**
- Invalid or corrupted Confluence content
- Malformed HTML or markup
- Encoding issues
- Empty or null content

**Strategy:**
- Skip invalid documents and continue processing
- Log detailed error information for debugging
- Track skipped documents in sync reports
- Provide warnings in UI when content is skipped

**3. Vector Database Errors**
- Database connection failures
- Storage capacity issues
- Query timeouts
- Index corruption

**Strategy:**
- Fail fast on database initialization errors
- Implement connection pooling and retry logic
- Provide clear error messages to users
- Log database errors with query context

**4. Configuration Errors**
- Missing required environment variables
- Invalid configuration values
- Incompatible settings

**Strategy:**
- Validate configuration at startup
- Fail fast with descriptive error messages
- Provide configuration examples in documentation
- Use sensible defaults where appropriate

### Error Recovery Mechanisms

**Transient Errors:**
- Implement retry logic with exponential backoff
- Use circuit breaker pattern for external services
- Cache successful responses when appropriate

**Permanent Errors:**
- Log error details for debugging
- Skip problematic items and continue processing
- Report errors in sync summaries
- Provide manual intervention options

**User-Facing Errors:**
- Display friendly error messages in UI
- Provide actionable suggestions for resolution
- Include support contact information
- Log technical details separately for administrators

## Testing Strategy

### Unit Testing

**Scope:** Individual functions and classes in isolation

**Key Areas:**
- Configuration loading and validation
- Text chunking logic
- Metadata extraction and enrichment
- URL construction and validation
- Error handling functions
- Utility functions

**Approach:**
- Use pytest as the testing framework
- Mock external dependencies (Confluence API, vector database)
- Test edge cases (empty inputs, null values, boundary conditions)
- Aim for 80%+ code coverage on core logic

**Example Tests:**
- Test chunk size boundaries with various input lengths
- Test metadata extraction with missing fields
- Test URL validation with malformed inputs
- Test configuration validation with missing parameters

### Property-Based Testing

**Scope:** Universal properties that should hold across all inputs

**Framework:** Hypothesis for Python property-based testing

**Configuration:** Minimum 100 iterations per property test

**Key Properties to Test:**

1. **Chunk Size Property:** All chunks should be within configured bounds
2. **Metadata Preservation:** Metadata should survive round-trip through storage
3. **Deduplication:** Storing same content twice should not create duplicates
4. **Sorting Invariant:** Search results should always be in descending score order
5. **Embedding Dimension Consistency:** All embeddings should have same dimensions
6. **Unique Identifiers:** All chunk IDs should be unique within a document

**Generators:**
- Generate random text of varying lengths for chunking tests
- Generate random metadata dictionaries for storage tests
- Generate random search queries for retrieval tests
- Generate random page structures for extraction tests

### Integration Testing

**Scope:** Interactions between components

**Key Scenarios:**
- End-to-end ingestion: Confluence → Processing → Storage
- End-to-end query: User input → Embedding → Search → Results
- Synchronization: Detect changes → Update database
- Configuration loading: Environment → Config → Components

**Approach:**
- Use test fixtures for Confluence API responses
- Use in-memory or temporary Chroma instances
- Test with realistic data samples
- Verify data flow between components

### Performance Testing

**Scope:** System performance under load

**Key Metrics:**
- Ingestion throughput (pages per second)
- Query latency (milliseconds per query)
- Memory usage during processing
- Database query performance

**Benchmarks:**
- Ingest 1000 pages in under 10 minutes
- Query response time under 500ms for 95th percentile
- Memory usage under 2GB for typical workloads
- Support concurrent queries (10+ simultaneous users)

### Deployment Testing

**Scope:** Posit Connect deployment validation

**Key Checks:**
- Application starts successfully on Posit Connect
- Environment variables are correctly loaded
- Dependencies are properly installed
- UI is accessible and functional
- Logs are properly captured

**Approach:**
- Test deployment to staging environment
- Verify all features work in deployed environment
- Check resource usage and performance
- Validate authentication and security

## Deployment Architecture

### Monorepo Structure

```
af-confluence-rag/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── confluence_client.py
│   │   ├── extractor.py
│   │   └── sync_manager.py
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── cleaner.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_store.py
│   │   └── models.py
│   ├── query/
│   │   ├── __init__.py
│   │   ├── search.py
│   │   └── app.py (Streamlit)
│   └── common/
│       ├── __init__.py
│       ├── config.py
│       ├── logging.py
│       └── utils.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── property/
├── config/
│   ├── default.yaml
│   ├── development.yaml
│   └── production.yaml
├── scripts/
│   ├── ingest.py
│   └── sync.py
├── pyproject.toml
├── requirements.txt
├── README.md
└── .env.example
```

### Deployment Models

**Model 1: Separate Services**
- Ingestion service runs as scheduled job (cron/Posit Connect scheduled execution)
- Query interface runs as persistent Streamlit app
- Shared vector database accessed by both

**Model 2: Unified Application**
- Single Streamlit app with ingestion controls
- Admin panel for triggering synchronization
- Real-time status updates during ingestion

**Recommended:** Model 1 for production (separation of concerns, better resource management)

### Posit Connect Deployment

**Requirements:**
- Python 3.12 runtime
- requirements.txt with all dependencies
- Environment variables configured in Posit Connect
- Persistent storage for Chroma database

**Deployment Steps:**
1. Configure environment variables in Posit Connect
2. Set up persistent volume for vector database
3. Deploy ingestion script as scheduled content
4. Deploy Streamlit app as interactive content
5. Configure access controls and permissions

**Environment Variables:**
```
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_AUTH_TOKEN=<secret>
CONFLUENCE_SPACE_KEY=DOCS
VECTOR_STORE_TYPE=chroma
VECTOR_STORE_PATH=/persistent/vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=10
```

### Scalability Considerations

**Current Design (Small to Medium Scale):**
- Single Chroma instance
- Synchronous processing
- Suitable for up to 10,000 pages

**Future Enhancements for Scale:**
- Distributed vector database (Milvus, Qdrant)
- Parallel processing of documents
- Caching layer for frequent queries
- Load balancing for query interface

## Security Considerations

**Authentication:**
- Store Confluence API tokens in environment variables
- Use Posit Connect's secret management
- Never commit credentials to repository
- Rotate tokens regularly

**Access Control:**
- Leverage Posit Connect's authentication
- Implement role-based access if needed
- Respect Confluence page permissions (future enhancement)

**Data Privacy:**
- Document data stored in vector database
- Ensure compliance with data retention policies
- Implement data encryption at rest (if required)
- Audit logging for access tracking

## Monitoring and Observability

**Logging:**
- Structured JSON logging
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Separate log files for ingestion and query services
- Log rotation and retention policies

**Metrics:**
- Ingestion: pages processed, errors encountered, duration
- Query: query count, average latency, error rate
- Storage: database size, document count, chunk count

**Alerting:**
- Failed ingestion jobs
- High error rates
- Database connection failures
- Performance degradation

## Future Enhancements

**Phase 2 Features:**
- Support for multiple Confluence spaces
- Attachment processing (PDFs, images)
- Advanced filtering (by date, author, space)
- Query history and analytics
- Feedback mechanism for result relevance

**Phase 3 Features:**
- Integration with LLM for answer generation (true RAG)
- Multi-language support
- Real-time synchronization via webhooks
- Advanced analytics dashboard
- API for programmatic access
