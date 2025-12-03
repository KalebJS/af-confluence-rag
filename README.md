# Confluence RAG System

A Python-based Retrieval-Augmented Generation (RAG) system for semantic search over Confluence documentation.

## Overview

This system enables organizations to extract, vectorize, and query their Confluence documentation using semantic search capabilities. It consists of two primary services:

1. **Ingestion Service**: Extracts and vectorizes Confluence content
2. **Query Interface**: Provides semantic search capabilities through a Streamlit web interface

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

### Component Overview

**Ingestion Service:**
- Connects to Confluence API and authenticates
- Retrieves pages from specified spaces with pagination handling
- Processes and chunks documents using LangChain text splitters
- Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
- Stores vectors and metadata in Chroma database
- Manages incremental updates and synchronization

**Query Interface (Streamlit):**
- Provides web-based search interface
- Accepts natural language queries from users
- Converts queries to embeddings using the same model as ingestion
- Retrieves relevant documents from vector database
- Displays formatted results with metadata and links

**Vector Database (Chroma):**
- Persists vector embeddings with metadata
- Performs similarity search operations
- Manages document identifiers and deduplication
- Provides CRUD operations for vector management

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
from src.processing.embedder import EmbeddingGenerator
from src.storage.vector_store import ChromaStore
from src.utils.config_loader import ConfigLoader

config = ConfigLoader().load_config()
embedder = EmbeddingGenerator(config.processing.embedding_model)
vector_store = ChromaStore(
    persist_directory=config.vector_store.config["persist_directory"],
    collection_name=config.vector_store.config["collection_name"]
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
│   ├── ingestion/             # Ingestion service components
│   │   ├── confluence_client.py   # Confluence API wrapper
│   │   └── ingestion_service.py   # Orchestrates ingestion pipeline
│   ├── processing/            # Document processing
│   │   ├── chunker.py            # Text chunking with LangChain
│   │   ├── embedder.py           # Embedding generation
│   │   └── metadata_enricher.py  # Metadata management
│   ├── storage/               # Vector database
│   │   └── vector_store.py       # Chroma vector store implementation
│   ├── sync/                  # Synchronization
│   │   ├── sync_coordinator.py   # Orchestrates sync process
│   │   ├── change_detector.py    # Detects changes
│   │   └── timestamp_tracker.py  # Tracks sync state
│   ├── query/                 # Query interface
│   │   ├── app.py                # Streamlit application
│   │   ├── query_processor.py    # Query processing
│   │   └── result_formatter.py   # Result formatting
│   ├── models/                # Data models
│   │   ├── page.py               # Page and chunk models
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

This system is designed to be compatible with Posit Connect. See the deployment guide in `docs/deployment.md` for detailed instructions.

## Project Structure

```
af-confluence-rag/
├── src/                    # Source code
│   ├── ingestion/         # Ingestion service components
│   ├── query/             # Query interface components
│   ├── models/            # Data models
│   └── utils/             # Shared utilities
├── tests/                 # Test suite
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── docs/                  # Documentation
├── pyproject.toml         # Project dependencies
└── .env.example           # Example environment variables
```

## License

See LICENSE file for details.
