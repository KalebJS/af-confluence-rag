# API Documentation

This document provides detailed API reference for all public classes and methods in the Confluence RAG System.

## Table of Contents

- [Provider Module](#provider-module)
  - [get_embeddings](#get_embeddings)
  - [get_vector_store](#get_vector_store)
- [Ingestion Components](#ingestion-components)
  - [ConfluenceClient](#confluenceclient)
  - [IngestionService](#ingestionservice)
- [Processing Components](#processing-components)
  - [DocumentChunker](#documentchunker)
  - [MetadataEnricher](#metadataenricher)
- [LangChain Interfaces](#langchain-interfaces)
  - [Embeddings Interface](#embeddings-interface)
  - [VectorStore Interface](#vectorstore-interface)
- [Synchronization Components](#synchronization-components)
  - [SyncCoordinator](#synccoordinator)
  - [ChangeDetector](#changedetector)
  - [TimestampTracker](#timestamptracker)
- [Query Components](#query-components)
  - [QueryProcessor](#queryprocessor)
  - [ResultFormatter](#resultformatter)
- [Data Models](#data-models)
  - [Page](#page)
  - [DocumentChunk](#documentchunk)
  - [SearchResult](#searchresult)
  - [Configuration Models](#configuration-models)
- [Utilities](#utilities)
  - [ConfigLoader](#configloader)
  - [Retry Decorator](#retry-decorator)

---

## Provider Module

The provider module (`src/providers.py`) is the centralized location for configuring embeddings and vector store implementations. This is the ONLY file you need to modify to swap providers system-wide.

### get_embeddings

Factory function that returns an Embeddings instance.

**Location:** `src/providers.py`

#### Signature

```python
def get_embeddings(model_name: str) -> Embeddings
```

**Parameters:**
- `model_name` (str): Name of the embedding model (e.g., "all-MiniLM-L6-v2")

**Returns:**
- `Embeddings`: LangChain Embeddings instance

**Raises:**
- `ValueError`: If model_name is empty or invalid
- `RuntimeError`: If the embeddings provider cannot be initialized

**Default Implementation:**
```python
from langchain_huggingface import HuggingFaceEmbeddings
return HuggingFaceEmbeddings(model_name=model_name)
```

**Example Usage:**
```python
from src.providers import get_embeddings

# Get default HuggingFace embeddings
embeddings = get_embeddings("all-MiniLM-L6-v2")

# Generate embedding for a query
query_embedding = embeddings.embed_query("How do I configure authentication?")

# Generate embeddings for multiple documents
doc_embeddings = embeddings.embed_documents(["Doc 1", "Doc 2", "Doc 3"])
```

**Swapping Providers:**

To use OpenAI embeddings instead, modify the function in `src/providers.py`:

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)
```

---

### get_vector_store

Factory function that returns a VectorStore instance.

**Location:** `src/providers.py`

#### Signature

```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore
```

**Parameters:**
- `embeddings` (Embeddings): Embeddings instance to use for vectorization
- `collection_name` (str): Name of the collection/index
- `persist_directory` (str): Directory for persistence (if supported by the provider)

**Returns:**
- `VectorStore`: LangChain VectorStore instance

**Raises:**
- `ValueError`: If collection_name or persist_directory is empty
- `RuntimeError`: If the vector store cannot be initialized

**Default Implementation:**
```python
from langchain_chroma import Chroma
return Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,
    persist_directory=persist_directory
)
```

**Example Usage:**
```python
from src.providers import get_embeddings, get_vector_store

# Get embeddings and vector store
embeddings = get_embeddings("all-MiniLM-L6-v2")
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name="confluence_docs",
    persist_directory="./chroma_db"
)

# Add documents
from langchain_core.documents import Document
docs = [
    Document(page_content="Content 1", metadata={"page_id": "123"}),
    Document(page_content="Content 2", metadata={"page_id": "456"})
]
vector_store.add_documents(docs)

# Search
results = vector_store.similarity_search_with_score("my query", k=5)
for doc, score in results:
    print(f"Score: {score:.3f} - {doc.page_content[:100]}")
```

**Swapping Providers:**

To use FAISS instead, modify the function in `src/providers.py`:

```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_community.vectorstores import FAISS
    import os
    
    index_path = os.path.join(persist_directory, collection_name)
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)
    else:
        return FAISS.from_texts([""], embeddings)
```

---

## Ingestion Components

### ConfluenceClient

Wrapper around the atlassian-python-api Confluence client for extracting documentation.

**Location:** `src/ingestion/confluence_client.py`

#### Constructor

```python
ConfluenceClient(base_url: str, auth_token: str, cloud: bool = True)
```

**Parameters:**
- `base_url` (str): Confluence instance URL (e.g., "https://your-domain.atlassian.net")
- `auth_token` (str): API token for authentication
- `cloud` (bool): True for Confluence Cloud, False for Server/Data Center

**Example:**
```python
from src.ingestion.confluence_client import ConfluenceClient

client = ConfluenceClient(
    base_url="https://example.atlassian.net",
    auth_token="your-api-token",
    cloud=True
)
```

#### Methods

##### `get_space_pages(space_key: str) -> list[Page]`

Retrieves all pages from a Confluence space.

**Parameters:**
- `space_key` (str): The space key (e.g., "DOCS")

**Returns:**
- `list[Page]`: List of Page objects with content and metadata

**Example:**
```python
pages = client.get_space_pages("DOCS")
for page in pages:
    print(f"{page.title}: {len(page.content)} characters")
```

##### `get_page_content(page_id: str) -> Page`

Retrieves a specific page by ID with full content and metadata.

**Parameters:**
- `page_id` (str): The page ID

**Returns:**
- `Page`: Page object with content and metadata

**Example:**
```python
page = client.get_page_content("123456")
print(f"Title: {page.title}")
print(f"Author: {page.author}")
print(f"Modified: {page.modified_date}")
```

##### `get_page_by_title(space_key: str, title: str) -> Page | None`

Retrieves a page by its title within a space.

**Parameters:**
- `space_key` (str): The space key
- `title` (str): The page title

**Returns:**
- `Page | None`: Page object if found, None otherwise

**Example:**
```python
page = client.get_page_by_title("DOCS", "Getting Started")
if page:
    print(f"Found page: {page.url}")
```

---

### IngestionService

Orchestrates the complete ingestion pipeline from Confluence extraction to vector storage.

**Location:** `src/ingestion/ingestion_service.py`

#### Constructor

```python
IngestionService(
    config: AppConfig,
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None
)
```

**Parameters:**
- `config` (AppConfig): Application configuration object
- `embeddings` (Embeddings | None): Optional custom embeddings implementation. If None, uses provider module.
- `vector_store` (VectorStore | None): Optional custom vector store implementation. If None, uses provider module.

**Example (using defaults):**
```python
from src.ingestion.ingestion_service import IngestionService
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
service = IngestionService(config)
```

**Example (with custom implementations):**
```python
from src.ingestion.ingestion_service import IngestionService
from src.providers import get_embeddings, get_vector_store
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()

# Get custom implementations
embeddings = get_embeddings("all-MiniLM-L6-v2")
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name=config.vector_store.collection_name,
    persist_directory=config.vector_store.persist_directory
)

# Inject into service
service = IngestionService(config, embeddings=embeddings, vector_store=vector_store)
```

#### Methods

##### `ingest_space(space_key: str) -> dict`

Ingests all pages from a Confluence space.

**Parameters:**
- `space_key` (str): The space key to ingest

**Returns:**
- `dict`: Summary statistics with keys:
  - `pages_processed` (int): Number of pages processed
  - `chunks_created` (int): Total chunks created
  - `duration_seconds` (float): Processing time

**Example:**
```python
result = service.ingest_space("DOCS")
print(f"Processed {result['pages_processed']} pages")
print(f"Created {result['chunks_created']} chunks")
print(f"Duration: {result['duration_seconds']:.2f}s")
```

##### `ingest_page(page_id: str) -> dict`

Ingests a single page by ID.

**Parameters:**
- `page_id` (str): The page ID to ingest

**Returns:**
- `dict`: Processing statistics

**Example:**
```python
result = service.ingest_page("123456")
print(f"Created {result['chunks_created']} chunks")
```

---

## Processing Components

### DocumentChunker

Splits documents into chunks using LangChain's RecursiveCharacterTextSplitter.

**Location:** `src/processing/chunker.py`

#### Constructor

```python
DocumentChunker(chunk_size: int = 1000, chunk_overlap: int = 200)
```

**Parameters:**
- `chunk_size` (int): Target chunk size in tokens (500-2000)
- `chunk_overlap` (int): Overlap between chunks in tokens

**Example:**
```python
from src.processing.chunker import DocumentChunker

chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
```

#### Methods

##### `chunk_document(content: str, metadata: dict) -> list[DocumentChunk]`

Splits a document into chunks with metadata preservation.

**Parameters:**
- `content` (str): Document text content
- `metadata` (dict): Document metadata (page_id, title, url, etc.)

**Returns:**
- `list[DocumentChunk]`: List of document chunks

**Example:**
```python
chunks = chunker.chunk_document(
    content="Long document text...",
    metadata={
        "page_id": "123456",
        "page_title": "Getting Started",
        "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456"
    }
)

for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {len(chunk.content)} chars")
```

---

### MetadataEnricher

Enriches document chunks with contextual metadata.

**Location:** `src/processing/metadata_enricher.py`

#### Methods

##### `enrich_chunk(chunk: DocumentChunk, page: Page) -> DocumentChunk`

Adds page metadata to a document chunk.

**Parameters:**
- `chunk` (DocumentChunk): Document chunk to enrich
- `page` (Page): Source page with metadata

**Returns:**
- `DocumentChunk`: Enriched chunk with complete metadata

**Example:**
```python
from src.processing.metadata_enricher import MetadataEnricher

enricher = MetadataEnricher()
enriched_chunk = enricher.enrich_chunk(chunk, page)
print(f"Chunk ID: {enriched_chunk.chunk_id}")
print(f"Metadata: {enriched_chunk.metadata}")
```

---

## LangChain Interfaces

The system uses LangChain's standard interfaces for embeddings and vector stores. These are abstract base classes that define the contract for all implementations.

### Embeddings Interface

**Interface:** `langchain_core.embeddings.Embeddings`

The Embeddings interface defines methods for generating vector embeddings from text.

#### Key Methods

##### `embed_query(text: str) -> list[float]`

Generates an embedding for a single query text.

**Parameters:**
- `text` (str): Input query text

**Returns:**
- `list[float]`: Embedding vector

**Example:**
```python
from src.providers import get_embeddings

embeddings = get_embeddings("all-MiniLM-L6-v2")
query_embedding = embeddings.embed_query("How do I configure authentication?")
print(f"Embedding dimension: {len(query_embedding)}")
```

##### `embed_documents(texts: list[str]) -> list[list[float]]`

Generates embeddings for multiple documents efficiently.

**Parameters:**
- `texts` (list[str]): List of input texts

**Returns:**
- `list[list[float]]`: List of embedding vectors

**Example:**
```python
texts = ["Document 1", "Document 2", "Document 3"]
doc_embeddings = embeddings.embed_documents(texts)
print(f"Generated {len(doc_embeddings)} embeddings")
```

#### Available Implementations

- **HuggingFaceEmbeddings** (default): Local, wraps sentence-transformers
- **OpenAIEmbeddings**: OpenAI's embedding models
- **CohereEmbeddings**: Cohere's embedding models
- **AzureOpenAIEmbeddings**: Azure OpenAI embeddings
- And 20+ more via LangChain ecosystem

---

### VectorStore Interface

**Interface:** `langchain_core.vectorstores.VectorStore`

The VectorStore interface defines methods for storing and searching vector embeddings.

#### Key Methods

##### `add_documents(documents: list[Document], **kwargs) -> list[str]`

Adds documents to the vector store with automatic embedding.

**Parameters:**
- `documents` (list[Document]): LangChain Document objects with content and metadata

**Returns:**
- `list[str]`: List of document IDs

**Example:**
```python
from src.providers import get_embeddings, get_vector_store
from langchain_core.documents import Document

embeddings = get_embeddings("all-MiniLM-L6-v2")
vector_store = get_vector_store(embeddings, "confluence_docs", "./chroma_db")

docs = [
    Document(page_content="Content 1", metadata={"page_id": "123"}),
    Document(page_content="Content 2", metadata={"page_id": "456"})
]
ids = vector_store.add_documents(docs)
print(f"Added documents with IDs: {ids}")
```

##### `similarity_search(query: str, k: int = 4, **kwargs) -> list[Document]`

Searches for similar documents by query string.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of results to return

**Returns:**
- `list[Document]`: List of similar documents

**Example:**
```python
results = vector_store.similarity_search("How do I deploy?", k=5)
for doc in results:
    print(f"Content: {doc.page_content[:100]}")
    print(f"Metadata: {doc.metadata}")
```

##### `similarity_search_with_score(query: str, k: int = 4, **kwargs) -> list[tuple[Document, float]]`

Searches for similar documents with relevance scores.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of results to return

**Returns:**
- `list[tuple[Document, float]]`: List of (document, score) tuples

**Example:**
```python
results = vector_store.similarity_search_with_score("authentication setup", k=5)
for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Title: {doc.metadata.get('page_title', 'Unknown')}")
    print(f"Content: {doc.page_content[:100]}")
```

##### `delete(ids: list[str], **kwargs) -> bool | None`

Deletes documents by their IDs.

**Parameters:**
- `ids` (list[str]): List of document IDs to delete

**Returns:**
- `bool | None`: True if successful, None if not supported

**Example:**
```python
# Delete specific documents
vector_store.delete(["doc_id_1", "doc_id_2"])

# Delete by metadata filter (if supported)
# Note: Not all vector stores support metadata-based deletion
```

#### Available Implementations

- **Chroma** (default): Local vector database with persistence
- **FAISS**: Fast similarity search, local or distributed
- **Pinecone**: Cloud-native vector database
- **Qdrant**: Open-source vector search engine
- **Weaviate**: Open-source vector database
- **Snowflake**: Enterprise data warehouse with vector support
- And 50+ more via LangChain ecosystem

---

## Synchronization Components

### SyncCoordinator

Orchestrates the synchronization process for incremental updates.

**Location:** `src/sync/sync_coordinator.py`

#### Constructor

```python
SyncCoordinator(
    confluence_client: ConfluenceClient,
    ingestion_service: IngestionService,
    vector_store: VectorStoreInterface,
    timestamp_tracker: TimestampTracker
)
```

#### Methods

##### `sync_space(space_key: str) -> dict`

Performs incremental synchronization for a space.

**Parameters:**
- `space_key` (str): Space key to sync

**Returns:**
- `dict`: Sync report with statistics

**Example:**
```python
from src.sync.sync_coordinator import SyncCoordinator

coordinator = SyncCoordinator(client, service, store, tracker)
report = coordinator.sync_space("DOCS")
print(f"Added: {report['pages_added']}")
print(f"Updated: {report['pages_updated']}")
print(f"Deleted: {report['pages_deleted']}")
```

---

### ChangeDetector

Detects changes between Confluence and the vector database.

**Location:** `src/sync/change_detector.py`

#### Methods

##### `detect_changes(space_key: str, last_sync: datetime) -> ChangeSet`

Identifies new, modified, and deleted pages.

**Parameters:**
- `space_key` (str): Space key
- `last_sync` (datetime): Last synchronization timestamp

**Returns:**
- `ChangeSet`: Object containing new, modified, and deleted pages

**Example:**
```python
from src.sync.change_detector import ChangeDetector
from datetime import datetime

detector = ChangeDetector(client, store)
changes = detector.detect_changes("DOCS", datetime.now())
print(f"New pages: {len(changes.new_pages)}")
print(f"Modified pages: {len(changes.modified_pages)}")
print(f"Deleted pages: {len(changes.deleted_page_ids)}")
```

---

### TimestampTracker

Tracks synchronization state and timestamps.

**Location:** `src/sync/timestamp_tracker.py`

#### Methods

##### `save_sync_state(space_key: str, state: SyncState) -> None`

Saves synchronization state.

**Parameters:**
- `space_key` (str): Space key
- `state` (SyncState): Sync state to save

##### `load_sync_state(space_key: str) -> SyncState | None`

Loads synchronization state.

**Parameters:**
- `space_key` (str): Space key

**Returns:**
- `SyncState | None`: Sync state if found

**Example:**
```python
from src.sync.timestamp_tracker import TimestampTracker
from src.models.page import SyncState
from datetime import datetime

tracker = TimestampTracker(store)

# Save state
state = SyncState(
    space_key="DOCS",
    last_sync_timestamp=datetime.now(),
    page_count=150,
    chunk_count=1847
)
tracker.save_sync_state("DOCS", state)

# Load state
loaded_state = tracker.load_sync_state("DOCS")
print(f"Last sync: {loaded_state.last_sync_timestamp}")
```

---

## Query Components

### QueryProcessor

Processes search queries and retrieves relevant documents.

**Location:** `src/query/query_processor.py`

#### Constructor

```python
QueryProcessor(
    embeddings: Embeddings | None = None,
    vector_store: VectorStore | None = None,
    config: AppConfig | None = None
)
```

**Parameters:**
- `embeddings` (Embeddings | None): Optional embeddings implementation. If None, uses provider module.
- `vector_store` (VectorStore | None): Optional vector store implementation. If None, uses provider module.
- `config` (AppConfig | None): Optional configuration. Required if embeddings or vector_store are None.

#### Methods

##### `process_query(query: str, top_k: int = 10) -> list[SearchResult]`

Processes a natural language query and returns results.

**Parameters:**
- `query` (str): Natural language query
- `top_k` (int): Number of results to return

**Returns:**
- `list[SearchResult]`: Ranked search results

**Example (using defaults):**
```python
from src.query.query_processor import QueryProcessor
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
processor = QueryProcessor(config=config)
results = processor.process_query("How do I configure authentication?", top_k=5)

for result in results:
    print(f"{result.page_title}: {result.similarity_score:.2%}")
    print(f"  {result.content[:100]}...")
```

**Example (with custom implementations):**
```python
from src.query.query_processor import QueryProcessor
from src.providers import get_embeddings, get_vector_store

embeddings = get_embeddings("all-MiniLM-L6-v2")
vector_store = get_vector_store(embeddings, "confluence_docs", "./chroma_db")

processor = QueryProcessor(embeddings=embeddings, vector_store=vector_store)
results = processor.process_query("deployment guide", top_k=10)
```

---

### ResultFormatter

Formats search results for display.

**Location:** `src/query/result_formatter.py`

#### Methods

##### `format_results(results: list[SearchResult]) -> str`

Formats results as a readable string.

**Parameters:**
- `results` (list[SearchResult]): Search results

**Returns:**
- `str`: Formatted results

##### `create_result_card(result: SearchResult) -> dict`

Creates a structured result card for UI display.

**Parameters:**
- `result` (SearchResult): Single search result

**Returns:**
- `dict`: Structured result data

**Example:**
```python
from src.query.result_formatter import ResultFormatter

formatter = ResultFormatter()
card = formatter.create_result_card(result)
print(f"Title: {card['title']}")
print(f"Score: {card['score']}")
print(f"URL: {card['url']}")
```

---

## Data Models

### Page

Represents a Confluence page with metadata.

**Location:** `src/models/page.py`

**Fields:**
- `id` (str): Unique page identifier
- `title` (str): Page title
- `space_key` (str): Confluence space key
- `content` (str): Page content in storage format
- `author` (str): Page author username
- `created_date` (datetime): Page creation timestamp
- `modified_date` (datetime): Last modification timestamp
- `url` (HttpUrl): Full URL to the page
- `version` (int): Page version number

**Example:**
```python
from src.models.page import Page
from datetime import datetime

page = Page(
    id="123456",
    title="Getting Started",
    space_key="DOCS",
    content="<p>Welcome to our documentation</p>",
    author="john.doe",
    created_date=datetime.now(),
    modified_date=datetime.now(),
    url="https://example.atlassian.net/wiki/spaces/DOCS/pages/123456",
    version=1
)
```

---

### DocumentChunk

Represents a chunk of a document with metadata.

**Location:** `src/models/page.py`

**Fields:**
- `chunk_id` (str): Unique chunk identifier (format: {page_id}_{chunk_index})
- `page_id` (str): Parent page identifier
- `content` (str): Chunk text content
- `metadata` (dict): Additional metadata
- `chunk_index` (int): Zero-based chunk position within page

**Properties:**
- `page_title` (str | None): Extract page title from metadata
- `page_url` (str | None): Extract page URL from metadata

**Example:**
```python
from src.models.page import DocumentChunk

chunk = DocumentChunk(
    chunk_id="123456_0",
    page_id="123456",
    content="This is the first chunk of content...",
    metadata={
        "page_title": "Getting Started",
        "page_url": "https://example.atlassian.net/wiki/spaces/DOCS/pages/123456"
    },
    chunk_index=0
)
```

---

### SearchResult

Represents a search result with relevance score.

**Location:** `src/models/page.py`

**Fields:**
- `chunk_id` (str): Unique chunk identifier
- `page_id` (str): Parent page identifier
- `page_title` (str): Page title
- `page_url` (HttpUrl): Full URL to the source page
- `content` (str): Chunk text content
- `similarity_score` (float): Cosine similarity score (0.0-1.0)
- `metadata` (dict): Additional metadata

---

### Configuration Models

**Location:** `src/models/config.py`

#### AppConfig

Main application configuration container.

**Fields:**
- `confluence` (ConfluenceConfig): Confluence connection settings
- `processing` (ProcessingConfig): Document processing settings
- `vector_store` (VectorStoreConfig): Vector store configuration
- `top_k_results` (int): Number of search results to return

#### ConfluenceConfig

Confluence connection configuration.

**Fields:**
- `base_url` (HttpUrl): Confluence instance URL
- `auth_token` (str): API authentication token
- `space_key` (str): Space key to sync
- `cloud` (bool): True for Cloud, False for Server/Data Center

#### ProcessingConfig

Document processing configuration.

**Fields:**
- `chunk_size` (int): Target chunk size in tokens (500-2000)
- `chunk_overlap` (int): Overlap between chunks in tokens (0-500)
- `embedding_model` (str): Sentence transformer model name

#### VectorStoreConfig

Vector store configuration.

**Fields:**
- `type` (str): Vector store type (chroma, faiss, qdrant, etc.)
- `config` (dict): Store-specific configuration

---

## Utilities

### ConfigLoader

Loads configuration from files and environment variables.

**Location:** `src/utils/config_loader.py`

#### Methods

##### `load_config(config_path: str | None = None) -> AppConfig`

Loads configuration from file and environment variables.

**Parameters:**
- `config_path` (str | None): Path to configuration file (optional)

**Returns:**
- `AppConfig`: Loaded configuration

**Example:**
```python
from src.utils.config_loader import ConfigLoader

# Load default configuration
loader = ConfigLoader()
config = loader.load_config()

# Load specific configuration file
config = loader.load_config("config/production.yaml")
```

---

### Retry Decorator

Decorator for retrying functions with exponential backoff.

**Location:** `src/utils/retry.py`

#### Usage

```python
from src.utils.retry import retry_with_backoff

@retry_with_backoff(max_retries=3, base_delay=1.0)
def fetch_data():
    # Function that might fail
    pass
```

**Parameters:**
- `max_retries` (int): Maximum number of retry attempts
- `base_delay` (float): Base delay in seconds (doubles each retry)
- `exceptions` (tuple): Exception types to catch (default: Exception)

---

## Configuration Options

### Environment Variables

All configuration can be overridden using environment variables:

```bash
# Confluence
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_AUTH_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=DOCS

# Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=confluence_docs

# Query
TOP_K_RESULTS=10
```

### Configuration Files

YAML configuration files support environment variable interpolation:

```yaml
confluence:
  base_url: ${CONFLUENCE_BASE_URL}
  auth_token: ${CONFLUENCE_AUTH_TOKEN}
  space_key: ${CONFLUENCE_SPACE_KEY}
  cloud: true

processing:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  type: "chroma"
  config:
    persist_directory: "./chroma_db"
    collection_name: "confluence_docs"

query:
  top_k_results: 10
```

---

## Error Handling

All components use structured logging with `structlog` and implement comprehensive error handling:

```python
import structlog

log = structlog.stdlib.get_logger()

try:
    result = service.ingest_space("DOCS")
except Exception as e:
    log.error("ingestion_failed", space_key="DOCS", error=str(e))
    raise
```

Common exceptions:
- `ConnectionError`: Confluence API connection failures
- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Missing configuration files
- `RuntimeError`: Vector store or embedding errors

---

## Type Hints

All components use Python 3.12+ type hints:

```python
def process_query(self, query: str, top_k: int = 10) -> list[SearchResult]:
    """Process a natural language query."""
    pass
```

Use `|` for union types:
```python
def get_page(self, page_id: str) -> Page | None:
    """Get a page by ID, or None if not found."""
    pass
```

---

## Testing

All components have comprehensive test coverage including:

- **Unit tests**: Test specific examples and edge cases
- **Property-based tests**: Verify universal properties using Hypothesis

Run tests:
```bash
uv run pytest tests/
```

Run specific component tests:
```bash
uv run pytest tests/test_chunker_properties.py
```

---

## Additional Resources

- [README.md](../README.md) - Project overview and setup
- [STREAMLIT_APP.md](STREAMLIT_APP.md) - Query interface guide
- [POSIT_CONNECT_DEPLOYMENT.md](POSIT_CONNECT_DEPLOYMENT.md) - Deployment guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
