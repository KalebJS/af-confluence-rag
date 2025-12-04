# Provider Swapping Guide

This guide explains how to swap embedding and vector store providers in the Confluence RAG System. Thanks to LangChain's standard abstractions, you can easily replace implementations without modifying core application code.

## Overview

All provider configuration is centralized in **`src/providers.py`**. This is the ONLY file you need to modify to swap providers system-wide.

The module provides two factory functions:
- `get_embeddings(model_name: str) -> Embeddings`
- `get_vector_store(embeddings, collection_name, persist_directory) -> VectorStore`

## Embedding Providers

### Default: HuggingFace (Local, No API Keys)

The default implementation uses HuggingFace embeddings, which wrap sentence-transformers and run locally.

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)
```

**Pros:**
- Runs locally, no API keys required
- No usage costs
- Works offline
- Fast for small to medium workloads

**Cons:**
- Requires local compute resources
- Limited to open-source models
- May be slower than cloud APIs for large batches

**Installation:**
```bash
# Already included in default dependencies
uv sync
```

---

### OpenAI Embeddings

OpenAI provides high-quality embeddings via their API.

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings
    import os
    
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
```

**Pros:**
- High-quality embeddings
- Fast API response times
- No local compute needed
- Supports latest models (text-embedding-3-small, text-embedding-3-large)

**Cons:**
- Requires API key
- Usage costs apply
- Requires internet connection

**Installation:**
```bash
uv add langchain-openai
```

**Configuration:**
```bash
# Add to .env
OPENAI_API_KEY=sk-...
```

**Recommended models:**
- `text-embedding-3-small`: Cost-effective, good quality
- `text-embedding-3-large`: Highest quality, more expensive
- `text-embedding-ada-002`: Legacy model, still supported

---

### Azure OpenAI Embeddings

Use OpenAI models deployed on Azure.

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_openai import AzureOpenAIEmbeddings
    import os
    
    return AzureOpenAIEmbeddings(
        model=model_name,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01"
    )
```

**Pros:**
- Enterprise-grade security and compliance
- Data residency control
- Integration with Azure ecosystem
- Same quality as OpenAI

**Cons:**
- Requires Azure subscription
- More complex setup
- Usage costs apply

**Installation:**
```bash
uv add langchain-openai
```

**Configuration:**
```bash
# Add to .env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=...
```

---

### Cohere Embeddings

Cohere provides multilingual embeddings optimized for search.

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_cohere import CohereEmbeddings
    import os
    
    return CohereEmbeddings(
        model=model_name,
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
```

**Pros:**
- Excellent multilingual support
- Optimized for semantic search
- Competitive pricing
- Fast API

**Cons:**
- Requires API key
- Usage costs apply
- Smaller ecosystem than OpenAI

**Installation:**
```bash
uv add langchain-cohere
```

**Configuration:**
```bash
# Add to .env
COHERE_API_KEY=...
```

**Recommended models:**
- `embed-english-v3.0`: English-only, high quality
- `embed-multilingual-v3.0`: 100+ languages
- `embed-english-light-v3.0`: Faster, lower cost

---

### Voyage AI Embeddings

Voyage AI specializes in retrieval-optimized embeddings.

```python
def get_embeddings(model_name: str) -> Embeddings:
    from langchain_voyageai import VoyageAIEmbeddings
    import os
    
    return VoyageAIEmbeddings(
        model=model_name,
        voyage_api_key=os.getenv("VOYAGE_API_KEY")
    )
```

**Pros:**
- Optimized specifically for RAG applications
- High retrieval accuracy
- Competitive pricing

**Cons:**
- Requires API key
- Smaller ecosystem
- Less well-known

**Installation:**
```bash
uv add langchain-voyageai
```

---

## Vector Store Providers

### Default: Chroma (Local, No External Services)

The default implementation uses Chroma, a local vector database with persistence.

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

**Pros:**
- Runs locally, no external services
- No usage costs
- Simple setup
- Good for development and small deployments

**Cons:**
- Limited scalability
- Single-node only
- No built-in replication

**Installation:**
```bash
# Already included in default dependencies
uv sync
```

---

### FAISS (Local, Fast)

FAISS is Facebook's library for efficient similarity search.

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
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        # Create empty index (will be populated during ingestion)
        return FAISS.from_texts([""], embeddings)
```

**Pros:**
- Very fast similarity search
- Efficient memory usage
- Supports GPU acceleration
- Good for large datasets (millions of vectors)

**Cons:**
- More complex persistence
- No built-in metadata filtering
- Requires manual index management

**Installation:**
```bash
# CPU version
uv add faiss-cpu

# GPU version (requires CUDA)
uv add faiss-gpu
```

**Note:** When using FAISS, you need to manually save the index after adding documents:

```python
vector_store.save_local("./faiss_index")
```

---

### Pinecone (Cloud, Scalable)

Pinecone is a fully managed vector database service.

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

**Pros:**
- Fully managed, no infrastructure
- Highly scalable (billions of vectors)
- Fast queries with low latency
- Built-in metadata filtering
- High availability

**Cons:**
- Requires API key
- Usage costs (free tier available)
- Data stored in cloud

**Installation:**
```bash
uv add langchain-pinecone pinecone-client
```

**Configuration:**
```bash
# Add to .env
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1-aws  # or your environment
```

**Setup:**
```python
# Create index first (one-time setup)
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pc.create_index(
    name="confluence_docs",
    dimension=384,  # Must match embedding dimension
    metric="cosine"
)
```

---

### Qdrant (Cloud or Self-Hosted)

Qdrant is an open-source vector search engine with cloud and self-hosted options.

```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    import os
    
    # Cloud option
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # Or local option
    # client = QdrantClient(path=persist_directory)
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
```

**Pros:**
- Open-source with commercial support
- Excellent metadata filtering
- Fast and efficient
- Can run locally or in cloud
- Good documentation

**Cons:**
- Requires setup (cloud or self-hosted)
- Usage costs for cloud version

**Installation:**
```bash
uv add langchain-qdrant qdrant-client
```

**Configuration (Cloud):**
```bash
# Add to .env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=...
```

**Configuration (Local):**
```python
# No API key needed for local
client = QdrantClient(path="./qdrant_db")
```

---

### Weaviate (Cloud or Self-Hosted)

Weaviate is an open-source vector database with GraphQL API.

```python
def get_vector_store(
    embeddings: Embeddings,
    collection_name: str,
    persist_directory: str
) -> VectorStore:
    from langchain_weaviate import WeaviateVectorStore
    import weaviate
    import os
    
    client = weaviate.Client(
        url=os.getenv("WEAVIATE_URL"),
        auth_client_secret=weaviate.AuthApiKey(
            api_key=os.getenv("WEAVIATE_API_KEY")
        )
    )
    
    return WeaviateVectorStore(
        client=client,
        index_name=collection_name,
        text_key="content",
        embedding=embeddings
    )
```

**Pros:**
- GraphQL API
- Strong schema support
- Good for complex queries
- Hybrid search (vector + keyword)

**Cons:**
- More complex setup
- Steeper learning curve

**Installation:**
```bash
uv add langchain-weaviate weaviate-client
```

---

### Snowflake (Enterprise)

Snowflake's data warehouse includes vector search capabilities.

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

**Pros:**
- Enterprise-grade security
- Integration with existing Snowflake data
- Unified data and vector storage
- Strong governance features

**Cons:**
- Requires Snowflake subscription
- Higher cost
- More complex setup

**Installation:**
```bash
uv add langchain-snowflake snowflake-connector-python
```

---

## Testing Your Configuration

After swapping providers, verify everything works:

### 1. Test Embeddings

```python
from src.providers import get_embeddings

embeddings = get_embeddings("all-MiniLM-L6-v2")  # or your model

# Test single query
query_emb = embeddings.embed_query("test query")
print(f"Query embedding dimension: {len(query_emb)}")

# Test batch
doc_embs = embeddings.embed_documents(["doc 1", "doc 2"])
print(f"Generated {len(doc_embs)} document embeddings")
```

### 2. Test Vector Store

```python
from src.providers import get_embeddings, get_vector_store
from langchain_core.documents import Document

embeddings = get_embeddings("all-MiniLM-L6-v2")
vector_store = get_vector_store(embeddings, "test_collection", "./test_db")

# Add test documents
docs = [
    Document(page_content="Test content 1", metadata={"id": "1"}),
    Document(page_content="Test content 2", metadata={"id": "2"})
]
vector_store.add_documents(docs)

# Search
results = vector_store.similarity_search("test", k=2)
print(f"Found {len(results)} results")
```

### 3. Run Full Test Suite

```bash
uv run pytest
```

### 4. Verify Setup

```bash
uv run python scripts/verify_setup.py
```

## Common Issues

### Issue: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'langchain_openai'`

**Solution:** Install the required package:
```bash
uv add langchain-openai
```

### Issue: API Key Not Found

**Problem:** `ValueError: OPENAI_API_KEY not found`

**Solution:** Add the API key to your `.env` file:
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Issue: Dimension Mismatch

**Problem:** `ValueError: Embedding dimension mismatch`

**Solution:** Ensure you're using the same embedding model for ingestion and querying. If you change models, you must re-ingest all documents.

### Issue: Vector Store Connection Failed

**Problem:** `ConnectionError: Failed to connect to vector store`

**Solution:** 
- Verify your connection parameters (URL, API key)
- Check network connectivity
- Ensure the vector store service is running

## Best Practices

### 1. Use Environment Variables

Store sensitive information in `.env`:

```bash
# .env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
QDRANT_URL=https://...
```

### 2. Document Your Choice

Add a comment in `src/providers.py` explaining why you chose a particular provider:

```python
def get_embeddings(model_name: str) -> Embeddings:
    # Using OpenAI for higher quality embeddings
    # Cost: ~$0.0001 per 1K tokens
    # Decision date: 2024-01-15
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)
```

### 3. Test Before Production

Always test provider changes in a development environment before deploying to production.

### 4. Monitor Costs

If using cloud providers, set up billing alerts and monitor usage:
- OpenAI: Check usage dashboard
- Pinecone: Monitor index size and queries
- Cohere: Track API calls

### 5. Consider Hybrid Approaches

You can use different providers for different purposes:

```python
def get_embeddings(model_name: str) -> Embeddings:
    import os
    
    # Use OpenAI in production, HuggingFace in development
    if os.getenv("APP_ENV") == "production":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model_name)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)
```

## Provider Comparison

| Provider | Type | Cost | Setup | Scalability | Best For |
|----------|------|------|-------|-------------|----------|
| HuggingFace | Embeddings | Free | Easy | Medium | Development, small deployments |
| OpenAI | Embeddings | Pay-per-use | Easy | High | Production, high quality needed |
| Cohere | Embeddings | Pay-per-use | Easy | High | Multilingual, search-optimized |
| Chroma | Vector Store | Free | Easy | Low | Development, small datasets |
| FAISS | Vector Store | Free | Medium | High | Large local datasets |
| Pinecone | Vector Store | Pay-per-use | Easy | Very High | Production, managed service |
| Qdrant | Vector Store | Free/Paid | Medium | High | Flexible deployment, good filtering |
| Snowflake | Vector Store | Pay-per-use | Hard | Very High | Enterprise, existing Snowflake users |

## Additional Resources

- [LangChain Embeddings Documentation](https://python.langchain.com/docs/integrations/text_embedding/)
- [LangChain Vector Stores Documentation](https://python.langchain.com/docs/integrations/vectorstores/)
- [Provider-specific documentation links in code comments]

## Support

If you encounter issues swapping providers:

1. Check the provider's documentation
2. Verify your API keys and configuration
3. Review the LangChain integration docs
4. Open an issue on GitHub with details about your setup
