# API Documentation

This document provides detailed API reference for all public classes and methods in the Confluence RAG System.

## Table of Contents

- [Ingestion Components](#ingestion-components)
  - [ConfluenceClient](#confluenceclient)
  - [IngestionService](#ingestionservice)
- [Processing Components](#processing-components)
  - [DocumentChunker](#documentchunker)
  - [EmbeddingGenerator](#embeddinggenerator)
  - [MetadataEnricher](#metadataenricher)
- [Storage Components](#storage-components)
  - [VectorStoreInterface](#vectorstoreinterface)
  - [ChromaStore](#chromastore)
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
IngestionService(config: AppConfig)
```

**Parameters:**
- `config` (AppConfig): Application configuration object

**Example:**
```python
from src.ingestion.ingestion_service import IngestionService
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
service = IngestionService(config)
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

### EmbeddingGenerator

Generates vector embeddings using sentence-transformers.

**Location:** `src/processing/embedder.py`

#### Constructor

```python
EmbeddingGenerator(model_name: str = "all-MiniLM-L6-v2")
```

**Parameters:**
- `model_name` (str): Sentence transformer model name

**Example:**
```python
from src.processing.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
```

#### Methods

##### `generate_embedding(text: str) -> np.ndarray`

Generates an embedding for a single text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `np.ndarray`: Embedding vector

**Example:**
```python
embedding = embedder.generate_embedding("How do I configure authentication?")
print(f"Embedding dimension: {len(embedding)}")
```

##### `generate_batch_embeddings(texts: list[str]) -> list[np.ndarray]`

Generates embeddings for multiple texts efficiently.

**Parameters:**
- `texts` (list[str]): List of input texts

**Returns:**
- `list[np.ndarray]`: List of embedding vectors

**Example:**
```python
texts = ["Query 1", "Query 2", "Query 3"]
embeddings = embedder.generate_batch_embeddings(texts)
print(f"Generated {len(embeddings)} embeddings")
```

##### `get_embedding_dimension() -> int`

Returns the dimensionality of the embedding model.

**Returns:**
- `int`: Embedding dimension

**Example:**
```python
dim = embedder.get_embedding_dimension()
print(f"Model produces {dim}-dimensional embeddings")
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

## Storage Components

### VectorStoreInterface

Abstract base class defining the interface for vector database operations.

**Location:** `src/storage/vector_store.py`

#### Methods

##### `add_documents(chunks: list[DocumentChunk], embeddings: list[np.ndarray]) -> None`

Adds documents with embeddings to the vector store.

**Parameters:**
- `chunks` (list[DocumentChunk]): Document chunks
- `embeddings` (list[np.ndarray]): Corresponding embeddings

##### `search(query_embedding: np.ndarray, top_k: int) -> list[SearchResult]`

Searches for similar documents.

**Parameters:**
- `query_embedding` (np.ndarray): Query embedding vector
- `top_k` (int): Number of results to return

**Returns:**
- `list[SearchResult]`: Search results ranked by similarity

##### `delete_by_page_id(page_id: str) -> None`

Deletes all chunks associated with a page.

**Parameters:**
- `page_id` (str): Page ID to delete

##### `get_document_metadata(page_id: str) -> dict | None`

Retrieves metadata for a document.

**Parameters:**
- `page_id` (str): Page ID

**Returns:**
- `dict | None`: Document metadata if found

---

### ChromaStore

Chroma implementation of VectorStoreInterface.

**Location:** `src/storage/vector_store.py`

#### Constructor

```python
ChromaStore(persist_directory: str, collection_name: str = "confluence_docs")
```

**Parameters:**
- `persist_directory` (str): Directory for persistent storage
- `collection_name` (str): Name of the Chroma collection

**Example:**
```python
from src.storage.vector_store import ChromaStore

store = ChromaStore(
    persist_directory="./chroma_db",
    collection_name="confluence_docs"
)
```

#### Methods

All methods from VectorStoreInterface, plus:

##### `get_collection_stats() -> dict`

Returns statistics about the collection.

**Returns:**
- `dict`: Statistics including document count

**Example:**
```python
stats = store.get_collection_stats()
print(f"Total documents: {stats['count']}")
```

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
    embedder: EmbeddingGenerator,
    vector_store: VectorStoreInterface
)
```

#### Methods

##### `process_query(query: str, top_k: int = 10) -> list[SearchResult]`

Processes a natural language query and returns results.

**Parameters:**
- `query` (str): Natural language query
- `top_k` (int): Number of results to return

**Returns:**
- `list[SearchResult]`: Ranked search results

**Example:**
```python
from src.query.query_processor import QueryProcessor

processor = QueryProcessor(embedder, store)
results = processor.process_query("How do I configure authentication?", top_k=5)

for result in results:
    print(f"{result.page_title}: {result.similarity_score:.2%}")
    print(f"  {result.content[:100]}...")
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
