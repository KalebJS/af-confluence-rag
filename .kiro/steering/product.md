# Product Overview

Confluence RAG System is a Python-based Retrieval-Augmented Generation (RAG) system that enables semantic search over Confluence documentation.

## Core Functionality

The system extracts content from Confluence spaces, vectorizes it using sentence transformers, and provides a natural language search interface through Streamlit. Users can query their documentation using semantic search rather than keyword matching.

## Key Components

- **Ingestion Service**: Extracts pages from Confluence API, chunks documents, generates embeddings, and stores vectors in Chroma database
- **Query Interface**: Streamlit web app that accepts natural language queries and returns relevant documentation chunks with metadata
- **Sync System**: Incremental synchronization that detects changes and updates only modified content

## Target Users

Organizations that want to make their Confluence documentation more discoverable through semantic search capabilities. Designed for deployment on Posit Connect and similar Python hosting platforms.
