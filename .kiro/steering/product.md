# Product Overview

Confluence RAG System is a Python-based Retrieval-Augmented Generation (RAG) system that enables semantic search over Confluence documentation.

## Core Functionality

The system extracts content from Confluence spaces, vectorizes it using embeddings (via LangChain's standard abstractions), and provides a natural language search interface through Streamlit. Users can query their documentation using semantic search rather than keyword matching.

The system is built on LangChain's standard interfaces, making it easy to swap embedding providers (e.g., OpenAI, Cohere) or vector stores (e.g., Pinecone, Weaviate) by modifying a single centralized configuration file.

## Key Components

- **Ingestion Service**: Extracts pages from Confluence API, chunks documents, generates embeddings using LangChain's Embeddings interface, and stores vectors in a vector database
- **Query Interface**: Streamlit web app that accepts natural language queries and returns relevant documentation chunks with metadata
- **Sync System**: Incremental synchronization that detects changes and updates only modified content
- **Provider Module**: Centralized configuration for swapping embeddings and vector store implementations

## Default Configuration

- **Embeddings**: HuggingFaceEmbeddings (wraps sentence-transformers, runs locally without API keys)
- **Vector Store**: Chroma (local vector database, no external services required)
- **Model**: all-MiniLM-L6-v2 (384-dimensional embeddings)

This default setup allows users to run the system immediately without obtaining API keys or configuring external services.

## Extensibility

The system uses dependency injection and a centralized provider module (`src/providers.py`), enabling developers to:
- Swap embedding providers by modifying one function
- Replace the vector store by modifying one function
- Inject custom implementations without modifying core application code

See `docs/PROVIDER_SWAPPING.md` for detailed examples.

## Target Users

Organizations that want to make their Confluence documentation more discoverable through semantic search capabilities. Designed for deployment on Posit Connect and similar Python hosting platforms.
