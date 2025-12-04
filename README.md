# Confluence RAG System

A Python-based Retrieval-Augmented Generation (RAG) system for semantic search over Confluence documentation.

## Overview

This system enables organizations to extract, vectorize, and query their Confluence documentation using semantic search capabilities. It consists of two primary services:

1. **Ingestion Service**: Extracts and vectorizes Confluence content
2. **Query Interface**: Provides semantic search capabilities through a Streamlit web interface

## Architecture

### High-Level Architecture

The system uses **LangChain's standard abstractions** for embeddings and vector stores, making it easy to swap implementations without modifying core code.

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
│  │  ┌─────▼──────────────┐    │  ┌─────▼──────┐  │          │
│  │  │ Embeddings         │    │  │ Result     │  │          │
│  │  │ (LangChain)        │    │  │ Formatter  │  │          │
│  │  └─────┬──────────────┘    │  └─────┬──────┘  │          │
│  └────────┼─────────────────┘ └────────┼─────────┘          │
│           │                            │                    │
│           └────────┬───────────────────┘                    │
│                    │                                        │
│           ┌────────▼─────────────────┐                      │
│           │  VectorStore             │                      │
│           │  (LangChain)             │                      │
│           └────────┬─────────────────┘                      │
│                    │                                        │
│           ┌────────▼─────────┐                              │
│           │  Chroma          │                              │
│           │  (Default)       │                              │
│           └──────────────────┘                              │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Shared Components                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Providers    │  │ Config       │  │ Logging      │      │
│  │ Module       │  │ Manager      │  │ System       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### LangChain Integration

The system leverages LangChain's abstractions for maximum flexibility:

- **Embeddings Interface**: Uses `langchain_core.embeddings.Embeddings` base class
  - Default: `HuggingFaceEmbeddings` (wraps sentence-transformers, runs locally)
  - Easily swap to OpenAI, Cohere, or any LangChain-compatible provider

- **VectorStore Interface**: Uses `langchain_core.vectorstores.VectorStore` base class
  - Default: `Chroma` (local vector database)
  - Easily swap to FAISS, Pinecone, Qdrant, Snowflake, or any LangChain-compatible store

- **Provider Module**: Centralized `src/providers.py` for swapping implementations
  - Modify ONE file to change providers system-wide
  - No changes needed to ingestion, query, or sync components

### Component Overview

**Ingestion Service:**
- Connects to Confluence API and authenticates
- Retrieves pages from specified spaces with pagination handling
- Processes and chunks documents using LangChain text splitters
- Generates embeddings using LangChain's Embeddings interface (default: HuggingFaceEmbeddings)
- Stores vectors and metadata using LangChain's VectorStore interface (default: Chroma)
- Manages incremental updates and synchronization

**Query Interface (Streamlit):**
- Provides web-based search interface
- Accepts natural language queries from users
- Converts queries to embeddings using LangChain's Embeddings interface
- Retrieves relevant documents using LangChain's VectorStore interface
- Displays formatted results with metadata and links

**Provider Module (`src/providers.py`):**
- Centralized location for configuring embeddings and vector store implementations
- Factory functions: `get_embeddings()` and `get_vector_store()`
- Modify this ONE file to swap providers system-wide
- Includes examples for common provider swaps (OpenAI, Snowflake, FAISS, etc.)

**Shared Components:**
- Configuration management for environment-specific settings
- Centralized logging with structured output (structlog)
- Common utilities for error handling and retries

## Requirements

- Python 3.12 or higher
- uv package manager
- Confluence API access (API token or OAuth credentials)

## Installation

### Prerequisites

- Python 3.12 or higher
- Access to a Confluence instance (Cloud or Server/Data Center)
- Confluence API token or OAuth credentials

### Step 1: Install uv Package Manager

The `uv` package manager provides fast, reliable Python package management. Install it using one of these methods:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

For more installation options, see: https://docs.astral.sh/uv/getting-started/installation/

### Step 2: Clone the Repository

```bash
git clone <repository-url>
cd confluence-rag-system
```

### Step 3: Install Dependencies

Use `uv` to install all project dependencies:

```bash
uv sync
```

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies from `pyproject.toml`
- Set up the project for development

### Step 4: Configure Environment Variables

Copy the example environment file and configure your Confluence credentials:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Confluence Configuration
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_AUTH_TOKEN=your-api-token-here
CONFLUENCE_SPACE_KEY=DOCS

# Vector Store Configuration
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=confluence_docs

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Query Configuration
TOP_K_RESULTS=10
```

### Step 5: Verify Setup

Run the verification script to ensure everything is configured correctly:

```bash
uv run python scripts/verify_setup.py
```

This will check:
- Python version compatibility
- Required environment variables
- Confluence API connectivity
- Vector store accessibility

## Configuration

The system uses a combination of environment variables and YAML configuration files for flexibility across different deployment environments.

### Environment Variables

Required environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `CONFLUENCE_BASE_URL` | Your Confluence instance URL | `https://your-domain.atlassian.net` |
| `CONFLUENCE_AUTH_TOKEN` | API token for authentication | `your-api-token` |
| `CONFLUENCE_SPACE_KEY` | Space key to sync | `DOCS` |

Optional environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VECTOR_STORE_TYPE` | Vector database type | `chroma` |
| `CHROMA_PERSIST_DIR` | Directory for Chroma database | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | Collection name in Chroma | `confluence_docs` |
| `CHUNK_SIZE` | Target chunk size in tokens | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | Number of search results | `10` |

### Configuration Files

Configuration files are located in the `config/` directory and use YAML format:

- **`default.yaml`**: Base configuration with sensible defaults
- **`development.yaml`**: Development environment overrides
- **`production.yaml`**: Production environment settings

Configuration files support environment variable interpolation using `${VAR_NAME}` syntax:

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

### Getting a Confluence API Token

**For Confluence Cloud:**

1. Log in to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a label (e.g., "RAG System")
4. Copy the token and save it securely
5. Use your email address as the username if needed

**For Confluence Server/Data Center:**

1. Log in to your Confluence instance
2. Go to Profile → Personal Access Tokens
3. Create a new token with appropriate permissions
4. Copy the token and save it securely

## Swapping Providers

One of the key benefits of using LangChain abstractions is the ability to easily swap embedding and vector store providers. All provider configuration is centralized in `src/providers.py`.

### Changing Embedding Providers

Edit the `get_embeddings()` function in `src/providers.py`:

**Default (HuggingFace - Local, No API Keys):**
```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
```

**Swap to OpenAI:**
```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)
```

**Swap to Cohere:**
```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_cohere import CohereEmbeddings
    return CohereEmbeddings(model=model_name)
```

**Swap to Azure OpenAI:**
```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        model=model_name,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )
```

### Changing Vector Store Providers

Edit the `get_vector_store()` function in `src/providers.py`:

**Default (Chroma - Local, No External Services):**
```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_chroma import Chroma
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
```

**Swap to FAISS (Local, Fast):**
```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_community.vectorstores import FAISS
    import os
    
    index_path = os.path.join(persist_directory, collection_name)
    
    # Load existing index or create new one
    if os.path.exists(index_path):
        return FAISS.load_local(index_path, embeddings)
    else:
        # Create empty index (will be populated during ingestion)
        return FAISS.from_texts([""], embeddings)
```

**Swap to Pinecone (Cloud, Scalable):**
```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_pinecone import PineconeVectorStore
    import os
    
    return PineconeVectorStore(
        index_name=collection_name,
        embedding=embeddings,
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
```

**Swap to Qdrant (Cloud or Self-Hosted):**
```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    import os
    
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
```

**Swap to Snowflake (Enterprise):**
```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_snowflake import SnowflakeVectorStore
    import os
    
    return SnowflakeVectorStore(
        embedding=embeddings,
        collection_name=collection_name,
        connection_params={
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE")
        }
    )
```

### Installing Additional Dependencies

When swapping providers, you may need to install additional packages:

```bash
# For OpenAI
uv add langchain-openai

# For Cohere
uv add langchain-cohere

# For Pinecone
uv add langchain-pinecone

# For Qdrant
uv add langchain-qdrant

# For FAISS
uv add faiss-cpu  # or faiss-gpu for GPU support
```

### Advanced: Custom Implementations

You can also provide custom implementations via dependency injection:

```python
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

class MyCustomEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Your custom implementation
        pass
    
    def embed_query(self, text: str) -> list[float]:
        # Your custom implementation
        pass

# Use via dependency injection
from src.ingestion.ingestion_service import IngestionService

embeddings = MyCustomEmbeddings()
vector_store = get_vector_store(embeddings, "my_collection", "./data")
service = IngestionService(config, embeddings=embeddings, vector_store=vector_store)
```

## Usage

### Initial Data Ingestion

Before you can search, you need to ingest Confluence content into the vector database.

#### Basic Ingestion

Run the ingestion service to extract and vectorize all pages from your configured Confluence space:

```bash
uv run python scripts/ingest.py
```

This will:
1. Connect to your Confluence instance
2. Retrieve all pages from the specified space
3. Split documents into chunks (500-2000 tokens)
4. Generate embeddings using sentence-transformers
5. Store vectors and metadata in the Chroma database

**Example output:**
```
INFO: Starting ingestion for space: DOCS
INFO: Retrieved 150 pages from Confluence
INFO: Processing page 1/150: Getting Started
INFO: Created 12 chunks for page: Getting Started
...
INFO: Ingestion complete. Processed 150 pages, created 1,847 chunks
INFO: Duration: 5m 32s
```

#### Incremental Sync

To update the vector database with new or modified pages:

```bash
uv run python scripts/scheduled_sync.py
```

This will:
- Compare modification timestamps
- Process only new or updated pages
- Remove embeddings for deleted pages
- Update the sync state

**Scheduling automatic syncs:**

Use cron (Linux/macOS) or Task Scheduler (Windows) to run periodic syncs:

```bash
# Run sync every 6 hours
0 */6 * * * cd /path/to/project && uv run python scripts/scheduled_sync.py
```

### Running the Query Interface

#### Quick Start

The easiest way to start the query interface:

```bash
uv run python scripts/run_app.py
```

This script will:
- Verify environment variables are set
- Check vector store accessibility
- Start the Streamlit application
- Open your browser automatically

#### Manual Start

Alternatively, start Streamlit directly:

```bash
uv run streamlit run src/query/app.py
```

The application will be available at `http://localhost:8501`

#### Using the Search Interface

1. **Enter your query**: Type a natural language question in the search box
   - Example: "How do I configure authentication?"
   - Example: "What are the deployment requirements?"

2. **Adjust settings** (optional):
   - Use the slider to change the number of results (1-50)
   - Default is 10 results

3. **View results**:
   - Each result shows:
     - Page title and excerpt
     - Similarity score (0-100%)
     - Author and last modified date
     - Direct link to source page

4. **Search history**:
   - View previous searches in the sidebar
   - Click to re-run a previous query

#### Query Interface Features

- **Natural Language Search**: Enter questions in plain English
- **Semantic Understanding**: Finds relevant content based on meaning, not just keywords
- **Configurable Results**: Adjust the number of results (1-50)
- **Search History**: View and re-run previous searches
- **Rich Metadata**: View page details, authors, and modification dates
- **Direct Links**: Click results to open source Confluence pages
- **Real-time Search**: Results appear as you search

For detailed usage instructions and screenshots, see [docs/STREAMLIT_APP.md](docs/STREAMLIT_APP.md)

### Advanced Usage

#### Custom Configuration

Use a specific configuration file:

```bash
uv run python scripts/ingest.py --config config/production.yaml
```

#### Verbose Logging

Enable detailed logging for debugging:

```bash
uv run python scripts/ingest.py --verbose
```

#### Ingesting Specific Pages

To ingest only specific pages (useful for testing):

```python
from src.ingestion.ingestion_service import IngestionService
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
service = IngestionService(config)

# Ingest a single page by ID
service.ingest_page(page_id="123456")
```

#### Querying Programmatically

Use the query processor directly in your code:

```python
from src.query.query_processor import QueryProcessor
from src.providers import get_embeddings, get_vector_store
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
embeddings = get_embeddings(config.processing.embedding_model)
vector_store = get_vector_store(
    embeddings=embeddings,
    persist_directory=config.vector_store.persist_directory,
    collection_name=config.vector_store.collection_name
)

processor = QueryProcessor(embedder, vector_store)
results = processor.process_query("How do I deploy?", top_k=5)

for result in results:
    print(f"{result.page_title}: {result.similarity_score:.2%}")
    print(f"  {result.content[:100]}...")
```

## Development

### Running Tests

Run the full test suite:

```bash
uv run pytest
```

Run tests with coverage:

```bash
uv run pytest --cov=src --cov-report=html
```

Run specific test files:

```bash
uv run pytest tests/test_chunker_properties.py
```

Run property-based tests only:

```bash
uv run pytest -k "properties"
```

### Test Structure

The project uses both unit tests and property-based tests:

- **Unit tests**: Test specific examples and edge cases
- **Property-based tests**: Verify universal properties across many inputs using Hypothesis

Property-based tests are configured to run 100 iterations by default to ensure thorough coverage.

### Code Quality

#### Formatting

Format code with Black:

```bash
uv run black src tests scripts
```

#### Linting

Check code with Ruff:

```bash
uv run ruff check src tests scripts
```

Auto-fix issues:

```bash
uv run ruff check --fix src tests scripts
```

#### Type Checking

Run type checking with mypy:

```bash
uv run mypy src
```

### Project Structure

```
confluence-rag-system/
├── src/                        # Source code
│   ├── providers.py           # ⭐ Provider configuration (THE file to modify)
│   ├── ingestion/             # Ingestion service components
│   │   ├── confluence_client.py   # Confluence API wrapper
│   │   └── ingestion_service.py   # Orchestrates ingestion pipeline
│   ├── processing/            # Document processing
│   │   ├── chunker.py            # Text chunking with LangChain
│   │   └── metadata_enricher.py  # Metadata management
│   ├── sync/                  # Synchronization
│   │   ├── sync_coordinator.py   # Orchestrates sync process
│   │   ├── change_detector.py    # Detects changes
│   │   └── timestamp_tracker.py  # Tracks sync state
│   ├── query/                 # Query interface
│   │   ├── app.py                # Streamlit application
│   │   ├── query_processor.py    # Query processing
│   │   └── result_formatter.py   # Result formatting
│   ├── models/                # Data models
│   │   ├── page.py               # Page and chunk models with LangChain mapping
│   │   └── config.py             # Configuration models
│   └── utils/                 # Shared utilities
│       ├── config_loader.py      # Configuration management
│       ├── logging_config.py     # Logging setup
│       └── retry.py              # Retry logic
├── tests/                     # Test suite
│   ├── test_*_properties.py   # Property-based tests
│   └── test_*.py              # Unit tests
├── config/                    # Configuration files
│   ├── default.yaml           # Default configuration
│   ├── development.yaml       # Development settings
│   └── production.yaml        # Production settings
├── scripts/                   # Utility scripts
│   ├── ingest.py             # Run ingestion
│   ├── run_app.py            # Start query interface
│   ├── scheduled_sync.py     # Incremental sync
│   ├── setup_deployment.py   # Deployment setup
│   ├── health_check.py       # Health check endpoint
│   └── verify_setup.py       # Verify installation
├── docs/                      # Documentation
│   ├── PROVIDER_SWAPPING.md  # Provider swapping guide
│   ├── API.md                # API reference
│   ├── STREAMLIT_APP.md      # Query interface guide
│   └── POSIT_CONNECT_DEPLOYMENT.md  # Deployment guide
├── pyproject.toml            # Project dependencies
├── .env.example              # Example environment variables
└── README.md                 # This file
```

### Adding New Features

1. **Define requirements**: Add acceptance criteria to requirements document
2. **Design**: Document the design in the design document
3. **Implement**: Write code following the existing patterns
4. **Test**: Add both unit tests and property-based tests
5. **Document**: Update relevant documentation

## Deployment

This system is designed to be compatible with Posit Connect and other Python hosting platforms.

### Posit Connect Deployment

For detailed Posit Connect deployment instructions, see [docs/POSIT_CONNECT_DEPLOYMENT.md](docs/POSIT_CONNECT_DEPLOYMENT.md)

**Quick deployment steps:**

1. Prepare the deployment:
```bash
uv run python scripts/setup_deployment.py
```

2. Configure environment variables in Posit Connect:
   - Set all required Confluence credentials
   - Configure vector store persistence path
   - Set Python version to 3.12

3. Deploy the Streamlit app:
```bash
rsconnect deploy streamlit src/query/app.py \
  --name confluence-rag \
  --title "Confluence Documentation Search"
```

4. Schedule the ingestion service:
   - Create a scheduled job in Posit Connect
   - Run `scripts/scheduled_sync.py` every 6 hours
   - Monitor logs for sync status

### Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t confluence-rag .

# Run ingestion
docker run --env-file .env confluence-rag python scripts/ingest.py

# Run query interface
docker run --env-file .env -p 8501:8501 confluence-rag streamlit run src/query/app.py
```

### Health Checks

The system includes a health check endpoint for monitoring:

```bash
uv run python scripts/health_check.py
```

This checks:
- Confluence API connectivity
- Vector store accessibility
- Embedding model availability
- Configuration validity

## Troubleshooting

### Common Issues

**Issue: "Confluence authentication failed"**
- Verify your API token is correct
- Check that the base URL includes the protocol (https://)
- For Cloud, ensure you're using an API token, not a password
- For Server, verify your personal access token has appropriate permissions

**Issue: "Vector store not found"**
- Run ingestion first: `uv run python scripts/ingest.py`
- Check that `CHROMA_PERSIST_DIR` points to the correct directory
- Verify the directory has write permissions

**Issue: "No search results found"**
- Ensure ingestion completed successfully
- Check that you're searching the correct space
- Try broader search terms
- Verify the vector store contains data

**Issue: "Out of memory during ingestion"**
- Reduce `CHUNK_SIZE` in configuration
- Process pages in smaller batches
- Increase available system memory

For more troubleshooting tips, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

## Performance Tuning

### Ingestion Performance

- **Batch size**: Process pages in batches to balance memory and speed
- **Chunk size**: Larger chunks (1500-2000 tokens) = fewer embeddings = faster ingestion
- **Embedding model**: Use smaller models for faster processing (e.g., `all-MiniLM-L6-v2`)

### Query Performance

- **Top K**: Reduce the number of results for faster searches
- **Vector store**: Consider FAISS for larger datasets (>100k chunks)
- **Caching**: Enable embedding caching for repeated queries

### Resource Requirements

**Minimum:**
- 2 CPU cores
- 4 GB RAM
- 10 GB disk space

**Recommended:**
- 4 CPU cores
- 8 GB RAM
- 50 GB disk space (for large Confluence spaces)

## Migration Guide

### Migrating from Pre-LangChain Version

If you're upgrading from a version that used direct sentence-transformers and custom vector store interfaces, follow these steps:

#### 1. Update Dependencies

The new version uses LangChain packages:

```bash
uv sync  # This will install the new dependencies
```

Key changes:
- `langchain-core`: Base abstractions
- `langchain-huggingface`: HuggingFace embeddings (wraps sentence-transformers)
- `langchain-chroma`: Chroma integration
- `sentence-transformers`: Now a transitive dependency (no direct import needed)

#### 2. Configuration Changes

**Old configuration** (no longer used):
```yaml
processing:
  embedding_provider: "huggingface"  # REMOVED
  embedding_model: "all-MiniLM-L6-v2"

vector_store:
  type: "chroma"
  provider: "chroma"  # REMOVED
  config:
    persist_directory: "./chroma_db"
    collection_name: "confluence_docs"
```

**New configuration** (simplified):
```yaml
processing:
  embedding_model: "all-MiniLM-L6-v2"  # Only model name needed

vector_store:
  collection_name: "confluence_docs"
  persist_directory: "./chroma_db"
```

Update your `config/default.yaml` to remove the `provider` fields.

#### 3. Code Changes (If You Extended the System)

**Old way** (direct imports):
```python
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import ChromaStore

embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
store = ChromaStore(persist_directory="./chroma_db")
```

**New way** (via providers module):
```python
from src.providers import get_embeddings, get_vector_store

embeddings = get_embeddings(model_name="all-MiniLM-L6-v2")
vector_store = get_vector_store(
    embeddings=embeddings,
    collection_name="confluence_docs",
    persist_directory="./chroma_db"
)
```

#### 4. Vector Database Compatibility

The new version is **fully compatible** with existing Chroma databases. No re-ingestion is required.

- Existing embeddings will work with the new system
- Metadata format is preserved
- Search results will be identical

#### 5. API Changes

**Embeddings:**
- Old: `embedder.generate_embedding(text)` → New: `embeddings.embed_query(text)`
- Old: `embedder.generate_batch_embeddings(texts)` → New: `embeddings.embed_documents(texts)`

**Vector Store:**
- Old: `store.add_documents(chunks, embeddings)` → New: `vector_store.add_documents(langchain_docs)`
- Old: `store.search(embedding, k)` → New: `vector_store.similarity_search_with_score(query, k)`
- Old: `store.delete_by_page_id(page_id)` → New: `vector_store.delete(ids)` (with metadata filtering)

#### 6. Testing Your Migration

After updating, verify everything works:

```bash
# Run tests
uv run pytest

# Verify setup
uv run python scripts/verify_setup.py

# Test query interface
uv run python scripts/run_app.py
```

#### 7. Rollback Plan

If you need to rollback:

1. Checkout the previous version: `git checkout <previous-tag>`
2. Reinstall dependencies: `uv sync`
3. Your vector database will still work with the old version

### Breaking Changes

- **Removed modules**: `src/processing/embedder.py` and `src/storage/vector_store.py`
- **Configuration**: Removed `provider` fields from config files
- **Direct imports**: Can no longer import `EmbeddingGenerator` or `ChromaStore` directly

### Benefits of Migration

- **Flexibility**: Easily swap embedding and vector store providers
- **Standardization**: Uses industry-standard LangChain interfaces
- **Ecosystem**: Access to 100+ LangChain-compatible providers
- **Maintainability**: Simpler codebase with fewer custom abstractions
- **Future-proof**: Automatic compatibility with new LangChain features

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality (both unit and property-based tests)
3. **Follow code style**: Use Black for formatting, Ruff for linting
4. **Update documentation** for user-facing changes
5. **Submit a pull request** with a clear description

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/my-new-feature

# Make changes and add tests
# ...

# Run tests
uv run pytest

# Format code
uv run black src tests scripts

# Check linting
uv run ruff check src tests scripts

# Commit and push
git commit -am "Add new feature"
git push origin feature/my-new-feature
```

## Support

For questions, issues, or feature requests:

- **Issues**: Open an issue on GitHub
- **Documentation**: Check the `docs/` directory
- **Discussions**: Use GitHub Discussions for questions

## License

See LICENSE file for details.

## Acknowledgments

This project uses the following open-source libraries:

- **LangChain**: Document processing and text splitting
- **Chroma**: Vector database for embeddings
- **sentence-transformers**: Embedding generation
- **Streamlit**: Web interface framework
- **atlassian-python-api**: Confluence API client
- **Hypothesis**: Property-based testing framework

## Related Documentation

- [Provider Swapping Guide](docs/PROVIDER_SWAPPING.md) - **How to swap embeddings and vector stores**
- [API Documentation](docs/API.md) - Component API reference with LangChain interfaces
- [Migration Guide](#migration-guide) - Upgrading from pre-LangChain versions
- [Streamlit App Guide](docs/STREAMLIT_APP.md) - Detailed query interface documentation
- [Posit Connect Deployment](docs/POSIT_CONNECT_DEPLOYMENT.md) - Deployment instructions
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md) - Common issues and solutions
