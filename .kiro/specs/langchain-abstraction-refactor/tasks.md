# Implementation Plan

- [x] 1. Update dependencies and create provider module
  - Add langchain-core, langchain-huggingface, and langchain-chroma to dependencies
  - Remove direct sentence-transformers dependency (becomes transitive)
  - Create `src/providers.py` with `get_embeddings()` and `get_vector_store()` functions
  - Implement default HuggingFaceEmbeddings and Chroma providers
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 3.1, 3.2, 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 1.1 Write property test for provider module
  - **Property 3: Provider Module Returns Correct Types**
  - **Validates: Requirements 3.3, 6.1, 6.2**

- [x] 2. Update configuration models
  - Simplify `ProcessingConfig` to remove provider fields, keep only `embedding_model`
  - Simplify `VectorStoreConfig` to only include `collection_name` and `persist_directory`
  - Remove unused configuration fields
  - _Requirements: 3.3, 3.4, 3.5_

- [x] 2.1 Write property test for configuration validation
  - **Property 3: Provider Module Returns Correct Types** (configuration aspect)
  - **Validates: Requirements 3.1, 3.2**

- [x] 3. Implement document mapping functions
  - Create `to_langchain_document()` function to convert DocumentChunk to LangChain Document
  - Create `from_langchain_document()` function to convert LangChain Document to DocumentChunk
  - Ensure all metadata fields are preserved in both directions
  - Add helper functions for batch conversion
  - _Requirements: 11.3_

- [x] 3.1 Write property test for document mapping
  - **Property 7: Document Mapping Round-Trip**
  - **Validates: Requirements 11.3**

- [x] 4. Update IngestionService to use LangChain abstractions
  - Modify constructor to accept `Embeddings` and `VectorStore` instances
  - Add default behavior to call provider module if instances not provided
  - Replace `EmbeddingGenerator` usage with `Embeddings.embed_documents()`
  - Replace `ChromaStore` usage with `VectorStore.add_documents()`
  - Update document processing to use LangChain Document format
  - _Requirements: 1.4, 2.3, 4.1_

- [x] 4.1 Write property test for embeddings interface compliance
  - **Property 1: Embeddings Interface Compliance**
  - **Validates: Requirements 1.1, 1.4**

- [x] 4.2 Write property test for vector store interface compliance
  - **Property 2: VectorStore Interface Compliance**
  - **Validates: Requirements 2.1, 2.3, 2.4, 2.5**

- [x] 4.3 Write property test for metadata preservation
  - **Property 6: Metadata Preservation Through Vector Operations**
  - **Validates: Requirements 11.1, 11.2**

- [x] 5. Update QueryProcessor to use LangChain abstractions
  - Modify constructor to accept `Embeddings` and `VectorStore` instances
  - Add default behavior to call provider module if instances not provided
  - Replace embedding generation with `Embeddings.embed_query()`
  - Replace vector search with `VectorStore.similarity_search_with_score()`
  - Update result formatting to work with LangChain Documents
  - _Requirements: 1.4, 2.4, 4.1_

- [x] 5.1 Write property test for dependency injection
  - **Property 4: Dependency Injection Override**
  - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

- [x] 6. Update SyncCoordinator to use LangChain abstractions
  - Update deletion logic to use `VectorStore` delete methods
  - Implement metadata-based filtering for page_id deletion
  - Handle cases where vector store doesn't support direct deletion
  - _Requirements: 2.5, 11.4_

- [x] 6.1 Write property test for metadata-based deletion
  - **Property 8: Metadata-Based Deletion**
  - **Validates: Requirements 11.4**

- [x] 7. Update all script entry points
  - Update `scripts/ingest.py` to use provider module
  - Update `scripts/run_app.py` to use provider module
  - Update `scripts/scheduled_sync.py` to use provider module
  - Ensure all scripts work with new abstractions
  - _Requirements: 3.1, 3.2, 3.5_

- [x] 8. Remove old implementations
  - Delete `src/processing/embedder.py` (EmbeddingGenerator class)
  - Delete `src/storage/vector_store.py` (VectorStoreInterface and ChromaStore)
  - Remove any imports of deleted modules
  - _Requirements: 1.1, 2.1_

- [x] 9. Update existing tests
  - Update unit tests to work with LangChain interfaces
  - Update test fixtures to use provider module
  - Ensure all existing tests pass
  - _Requirements: 9.1, 9.2_

- [x] 9.1 Write property test for model name compatibility
  - **Property 5: Model Name Compatibility**
  - **Validates: Requirements 1.3, 9.3**

- [x] 9.2 Write property test for provider error handling
  - **Property 9: Provider Module Error Handling**
  - **Validates: Requirements 6.5**

- [x] 10. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Update documentation
  - Update README with new architecture overview
  - Document how to swap implementations in `src/providers.py`
  - Add examples for common provider swaps (OpenAI, Snowflake, etc.)
  - Update API documentation
  - Create migration guide for existing users
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 12. Update steering files
  - Update `tech.md` to reflect new LangChain dependencies
  - Update `structure.md` to document providers module
  - Update `product.md` if needed
  - _Requirements: 10.5_
