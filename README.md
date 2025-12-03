# Confluence RAG System

A Python-based Retrieval-Augmented Generation (RAG) system for semantic search over Confluence documentation.

## Overview

This system enables organizations to extract, vectorize, and query their Confluence documentation using semantic search capabilities. It consists of two primary services:

1. **Ingestion Service**: Extracts and vectorizes Confluence content
2. **Query Interface**: Provides semantic search capabilities through a Streamlit web interface

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Confluence RAG System                   │
├─────────────────────────────────────────────────────────────┤
│  Ingestion Service  →  Document Processing  →  Vector Store │
│  Query Interface    →  Semantic Search      →  Results      │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.12 or higher
- uv package manager
- Confluence API access (API token or OAuth credentials)

## Installation

### 1. Install uv package manager

Follow the installation guide at: https://docs.astral.sh/uv/getting-started/installation/

### 2. Clone the repository

```bash
git clone <repository-url>
cd af-confluence-rag
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment variables

Copy the example environment file and update with your Confluence credentials:

```bash
cp .env.example .env
# Edit .env with your Confluence details
```

## Configuration

### Environment Variables

Required environment variables (see `.env.example` for full list):

- `CONFLUENCE_BASE_URL`: Your Confluence instance URL
- `CONFLUENCE_AUTH_TOKEN`: API token for authentication
- `CONFLUENCE_SPACE_KEY`: Space key to sync

### Configuration Files

Configuration files are located in the `config/` directory:

- `default.yaml`: Default configuration
- `development.yaml`: Development environment settings
- `production.yaml`: Production environment settings

## Usage

### Running the Ingestion Service

```bash
uv run python scripts/ingest.py
```

### Running the Query Interface

#### Quick Start

```bash
uv run python scripts/run_app.py
```

This script will:
- Verify environment variables are set
- Check vector store accessibility
- Start the Streamlit application

#### Manual Start

```bash
uv run streamlit run src/query/app.py
```

The application will be available at `http://localhost:8501`

#### Features

- **Natural Language Search**: Enter questions in plain English
- **Configurable Results**: Adjust the number of results (1-50)
- **Search History**: View and re-run previous searches
- **Rich Metadata**: View page details, authors, and modification dates
- **Direct Links**: Click results to open source Confluence pages

For detailed usage instructions, see [docs/STREAMLIT_APP.md](docs/STREAMLIT_APP.md)

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src tests
uv run ruff check src tests
```

### Type Checking

```bash
uv run mypy src
```

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
