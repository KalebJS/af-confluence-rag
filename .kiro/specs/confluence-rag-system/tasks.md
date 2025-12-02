# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create monorepo directory structure with src/, tests/, config/, and scripts/ directories
  - Initialize pyproject.toml with Python 3.12 and core dependencies (atlassian-python-api, langchain, pydantic, streamlit)
  - Set up development environment with uv package manager
  - Create .env.example file with required environment variables
  - _Requirements: 10.1, 10.2, 10.3, 10.5, 8.1_

- [x] 2. Implement core data models with Pydantic
  - [x] 2.1 Create Pydantic models for Page, DocumentChunk, SearchResult, and SyncState
    - Define Page model with validation for all fields
    - Define DocumentChunk model with chunk_id format validation
    - Define SearchResult model with similarity score bounds
    - Define SyncState model for tracking synchronization
    - _Requirements: 1.4, 2.4, 3.2, 4.5_

  - [x] 2.2 Write property test for Pydantic model validation
    - **Property 3: Metadata completeness**
    - **Property 10: Metadata storage completeness**
    - **Validates: Requirements 1.4, 3.2**

  - [x] 2.3 Create configuration models (AppConfig, ConfluenceConfig, ProcessingConfig, VectorStoreConfig)
    - Define ConfluenceConfig with URL and authentication fields
    - Define ProcessingConfig with chunk size bounds (500-2000)
    - Define VectorStoreConfig with pluggable type field
    - Define AppConfig as main configuration container
    - _Requirements: 7.1, 7.2, 8.1_

  - [x] 2.4 Write property test for configuration validation
    - **Property 5: Chunk size bounds**
    - **Property 24: Environment variable loading**
    - **Validates: Requirements 2.1, 7.1**

- [ ] 3. Implement configuration management
  - [ ] 3.1 Create ConfigLoader class with YAML and environment variable support
    - Implement load_config method to read from YAML files
    - Implement environment variable override logic
    - Add validation for required configuration parameters
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ] 3.2 Write property test for configuration loading
    - **Property 25: Configuration file parsing**
    - **Property 26: Missing configuration error handling**
    - **Validates: Requirements 7.2, 7.4**

  - [ ] 3.3 Create default configuration files (default.yaml, development.yaml, production.yaml)
    - Create config/default.yaml with sensible defaults
    - Create config/development.yaml for local development
    - Create config/production.yaml for Posit Connect deployment
    - _Requirements: 7.5_

- [ ] 4. Implement Confluence client wrapper
  - [ ] 4.1 Create ConfluenceClient class wrapping atlassian-python-api
    - Initialize Confluence client with authentication
    - Implement get_space_pages method using generator for memory efficiency
    - Implement get_page_content method with metadata extraction
    - Implement get_page_by_title method for lookups
    - _Requirements: 1.1, 1.2, 1.4_

  - [ ] 4.2 Write property test for page retrieval completeness
    - **Property 1: Complete page retrieval**
    - **Property 2: Pagination completeness**
    - **Validates: Requirements 1.2, 1.3**

  - [ ] 4.3 Implement error handling and retry logic with exponential backoff
    - Add retry decorator for API calls
    - Implement exponential backoff for rate limits
    - Add comprehensive error logging
    - _Requirements: 1.5, 9.1, 9.2_

  - [ ] 4.4 Write property test for exponential backoff
    - **Property 4: Exponential backoff behavior**
    - **Validates: Requirements 1.5**

- [ ] 5. Implement document processing and chunking
  - [ ] 5.1 Create DocumentChunker class using LangChain RecursiveCharacterTextSplitter
    - Initialize text splitter with configurable chunk size and overlap
    - Implement chunk_document method with metadata preservation
    - Add HTML cleaning for Confluence storage format
    - _Requirements: 2.1, 2.2, 2.4_

  - [ ] 5.2 Write property test for chunk size bounds
    - **Property 5: Chunk size bounds**
    - **Property 6: Boundary-aware splitting**
    - **Validates: Requirements 2.1, 2.2**

  - [ ] 5.3 Create MetadataEnricher class for adding contextual metadata
    - Implement enrich_chunk method to add page metadata to chunks
    - Generate unique chunk_ids in format {page_id}_{chunk_index}
    - _Requirements: 2.4, 3.3_

  - [ ] 5.4 Write property test for metadata preservation
    - **Property 8: Metadata preservation in embeddings**
    - **Property 11: Unique identifier generation**
    - **Validates: Requirements 2.4, 3.3**

- [ ] 6. Implement embedding generation
  - [ ] 6.1 Create EmbeddingGenerator class using sentence-transformers
    - Initialize sentence-transformers model (all-MiniLM-L6-v2)
    - Implement generate_embedding method for single text
    - Implement generate_batch_embeddings for efficient batch processing
    - Add get_embedding_dimension method
    - _Requirements: 2.3, 5.1_

  - [ ] 6.2 Write property test for embedding generation
    - **Property 7: Embedding generation completeness**
    - **Property 18: Embedding model consistency**
    - **Validates: Requirements 2.3, 5.1**

- [ ] 7. Implement vector store abstraction layer
  - [ ] 7.1 Create VectorStoreInterface abstract base class
    - Define abstract methods: add_documents, search, delete_by_page_id, get_document_metadata
    - Add type hints and docstrings for all methods
    - _Requirements: 3.1, 3.2_

  - [ ] 7.2 Implement ChromaStore as default VectorStoreInterface implementation
    - Initialize Chroma client with persistence directory
    - Implement add_documents with embedding storage
    - Implement search with similarity scoring
    - Implement delete_by_page_id for updates
    - Implement get_document_metadata for lookups
    - _Requirements: 2.5, 3.1, 3.2, 3.4, 3.5, 5.2_

  - [ ] 7.3 Write property test for vector store operations
    - **Property 9: Storage round-trip consistency**
    - **Property 12: Deduplication idempotence**
    - **Validates: Requirements 2.5, 3.4, 3.5**

  - [ ] 7.4 Create VectorStoreFactory for pluggable vector store instantiation
    - Implement create_vector_store factory method
    - Support 'chroma', 'faiss', 'qdrant' store types
    - Add configuration validation for each store type
    - _Requirements: 3.1_

  - [ ] 7.5 Write property test for vector store factory
    - **Property 38: Vector store interface compliance**
    - **Property 39: Vector store factory instantiation**
    - **Validates: Design requirement for pluggable vector stores**

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement synchronization manager
  - [ ] 9.1 Create SyncCoordinator class for orchestrating synchronization
    - Implement sync_space method for full space synchronization
    - Implement detect_changes method using timestamp comparison
    - Implement apply_changes method for incremental updates
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ] 9.2 Write property test for synchronization logic
    - **Property 13: Timestamp comparison correctness**
    - **Property 14: Update replaces old embeddings**
    - **Property 15: New page processing**
    - **Property 16: Deletion completeness**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4**

  - [ ] 9.2 Create ChangeDetector class for identifying new, modified, and deleted pages
    - Implement detect_new_pages method
    - Implement detect_modified_pages method
    - Implement detect_deleted_pages method
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 9.3 Create TimestampTracker for maintaining sync state
    - Implement save_sync_state method
    - Implement load_sync_state method
    - Store sync state in vector database metadata
    - _Requirements: 4.5_

  - [ ] 9.4 Write property test for sync timestamp updates
    - **Property 17: Sync timestamp update**
    - **Validates: Requirements 4.5**

- [ ] 10. Implement ingestion service orchestration
  - [ ] 10.1 Create IngestionService class to orchestrate the full ingestion pipeline
    - Wire together ConfluenceClient, DocumentChunker, EmbeddingGenerator, and VectorStore
    - Implement ingest_space method for full space ingestion
    - Implement ingest_page method for single page processing
    - Add progress tracking and logging
    - _Requirements: 1.1, 1.2, 2.1, 2.3, 2.5_

  - [ ] 10.2 Add comprehensive error handling and logging
    - Implement error recovery for invalid content
    - Add summary statistics logging
    - Handle database unavailability gracefully
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ] 10.3 Write property test for error recovery
    - **Property 33: Graceful error recovery**
    - **Property 35: Completion logging**
    - **Validates: Requirements 9.3, 9.5**

  - [ ] 10.4 Create CLI script for running ingestion (scripts/ingest.py)
    - Add command-line argument parsing
    - Support configuration file path argument
    - Add verbose logging option
    - _Requirements: 7.1, 7.2_

- [ ] 11. Implement query processing
  - [ ] 11.1 Create QueryProcessor class for handling search queries
    - Implement process_query method that embeds query and searches vector store
    - Add result ranking by similarity score
    - Implement result filtering and deduplication
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ] 11.2 Write property test for query processing
    - **Property 19: Result count correctness**
    - **Property 20: Search result completeness**
    - **Property 21: Result ranking order**
    - **Validates: Requirements 5.2, 5.3, 5.4**

  - [ ] 11.3 Create ResultFormatter class for formatting search results
    - Implement format_results method for display
    - Create result cards with metadata
    - Add URL validation and formatting
    - _Requirements: 5.3, 5.5, 6.4_

  - [ ] 11.4 Write property test for result formatting
    - **Property 22: Valid result URLs**
    - **Property 23: Result metadata display**
    - **Validates: Requirements 5.5, 6.4**

- [ ] 12. Implement Streamlit query interface
  - [ ] 12.1 Create main Streamlit app (src/query/app.py)
    - Set up Streamlit page configuration
    - Add search input field and button
    - Implement loading indicator during search
    - Display search results with formatting
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ] 12.2 Add result display with metadata and links
    - Display page title, content excerpt, and similarity score
    - Add clickable links to source Confluence pages
    - Implement empty results handling with helpful message
    - _Requirements: 5.3, 5.4, 5.5, 6.4, 6.5_

  - [ ] 12.3 Add configuration UI for search parameters
    - Add slider for top_k results
    - Add space selector (if multiple spaces supported)
    - Add search history display
    - _Requirements: 5.2_

  - [ ] 12.4 Implement error handling and user feedback
    - Display friendly error messages for database unavailability
    - Add connection status indicator
    - Implement graceful degradation
    - _Requirements: 9.4_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Add logging and monitoring
  - [ ] 14.1 Create centralized logging configuration
    - Set up structured JSON logging
    - Configure log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Add log rotation and retention
    - _Requirements: 9.1_

  - [ ] 14.2 Add logging throughout the application
    - Log all API calls to Confluence
    - Log ingestion progress and statistics
    - Log query operations and performance
    - Log errors with full context
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ] 14.3 Write property test for logging format
    - **Property 31: Error log format**
    - **Validates: Requirements 9.1**

- [ ] 15. Create deployment configuration for Posit Connect
  - [ ] 15.1 Create requirements.txt from pyproject.toml
    - Export all dependencies with versions
    - Ensure Python 3.12 compatibility
    - _Requirements: 8.1, 8.3_

  - [ ] 15.2 Write property test for deployment requirements
    - **Property 28: Dependency file presence**
    - **Property 29: Python version verification**
    - **Validates: Requirements 8.1, 8.3**

  - [ ] 15.3 Create Posit Connect deployment guide
    - Document environment variable configuration
    - Document persistent storage setup for vector database
    - Document scheduled execution setup for ingestion
    - Document Streamlit app deployment
    - _Requirements: 8.4, 8.5_

  - [ ] 15.4 Create example deployment scripts
    - Create script for initial setup
    - Create script for scheduled synchronization
    - Add health check endpoints
    - _Requirements: 8.5_

- [ ] 16. Create comprehensive documentation
  - [ ] 16.1 Write README.md with setup instructions
    - Add project overview and architecture diagram
    - Add installation instructions using uv
    - Add configuration guide
    - Add usage examples for ingestion and query
    - _Requirements: 10.5_

  - [ ] 16.2 Write property test for documentation presence
    - **Property 37: Documentation presence**
    - **Validates: Requirements 10.5**

  - [ ] 16.3 Create API documentation
    - Document all public classes and methods
    - Add usage examples for each component
    - Document configuration options
    - _Requirements: 10.5_

  - [ ] 16.4 Create troubleshooting guide
    - Document common errors and solutions
    - Add debugging tips
    - Add performance tuning guide
    - _Requirements: 9.1_

- [ ] 17. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
