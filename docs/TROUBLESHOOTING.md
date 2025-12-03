# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Confluence RAG System.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Issues](#configuration-issues)
- [Confluence Connection Issues](#confluence-connection-issues)
- [Ingestion Issues](#ingestion-issues)
- [Vector Store Issues](#vector-store-issues)
- [Query Interface Issues](#query-interface-issues)
- [Performance Issues](#performance-issues)
- [Deployment Issues](#deployment-issues)
- [Debugging Tips](#debugging-tips)
- [Getting Help](#getting-help)

---

## Installation Issues

### Issue: `uv: command not found`

**Symptoms:**
```bash
$ uv sync
bash: uv: command not found
```

**Solution:**

Install the `uv` package manager:

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

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc
```

---

### Issue: Python version mismatch

**Symptoms:**
```
Error: Python 3.12 or higher is required
```

**Solution:**

1. Check your Python version:
```bash
python --version
```

2. Install Python 3.12 or higher:

**macOS (using Homebrew):**
```bash
brew install python@3.12
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12
```

**Windows:**
Download from https://www.python.org/downloads/

3. Verify installation:
```bash
python3.12 --version
```

4. Use `uv` with the correct Python version:
```bash
uv sync --python 3.12
```

---

### Issue: Dependency installation fails

**Symptoms:**
```
Error: Failed to resolve dependencies
```

**Solution:**

1. Clear the cache:
```bash
rm -rf .venv
uv cache clean
```

2. Reinstall dependencies:
```bash
uv sync
```

3. If specific packages fail, try installing them individually:
```bash
uv add package-name
```

---

## Configuration Issues

### Issue: Missing environment variables

**Symptoms:**
```
Error: CONFLUENCE_BASE_URL environment variable is required
```

**Solution:**

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
```bash
CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
CONFLUENCE_AUTH_TOKEN=your-api-token
CONFLUENCE_SPACE_KEY=DOCS
```

3. Verify environment variables are loaded:
```bash
uv run python scripts/verify_setup.py
```

---

### Issue: Configuration file not found

**Symptoms:**
```
Error: Configuration file not found: config/production.yaml
```

**Solution:**

1. Check available configuration files:
```bash
ls config/
```

2. Use an existing configuration file:
```bash
uv run python scripts/ingest.py --config config/default.yaml
```

3. Or create a new configuration file based on the template:
```bash
cp config/default.yaml config/production.yaml
# Edit config/production.yaml with your settings
```

---

### Issue: Invalid configuration values

**Symptoms:**
```
ValidationError: chunk_size must be between 500 and 2000
```

**Solution:**

Check your configuration values against the valid ranges:

- `chunk_size`: 500-2000 tokens
- `chunk_overlap`: 0-500 tokens
- `top_k_results`: 1-100

Edit your configuration file or environment variables to use valid values.

---

## Confluence Connection Issues

### Issue: Authentication failed

**Symptoms:**
```
Error: 401 Unauthorized - Authentication failed
```

**Solutions:**

**For Confluence Cloud:**

1. Verify your API token is correct:
   - Go to https://id.atlassian.com/manage-profile/security/api-tokens
   - Create a new token if needed
   - Copy the token exactly (no extra spaces)

2. Ensure you're using the correct base URL:
   ```bash
   CONFLUENCE_BASE_URL=https://your-domain.atlassian.net
   ```
   Note: Include `https://` and use `.atlassian.net` (not `.com`)

3. Verify the `cloud` setting is `true` in your configuration:
   ```yaml
   confluence:
     cloud: true
   ```

**For Confluence Server/Data Center:**

1. Use a Personal Access Token instead of an API token
2. Set `cloud: false` in your configuration:
   ```yaml
   confluence:
     cloud: false
   ```

3. Verify your base URL includes the context path if applicable:
   ```bash
   CONFLUENCE_BASE_URL=https://confluence.company.com/wiki
   ```

---

### Issue: Connection timeout

**Symptoms:**
```
Error: Connection timeout after 30 seconds
```

**Solutions:**

1. Check your network connection:
```bash
ping your-domain.atlassian.net
```

2. Verify the Confluence URL is accessible:
```bash
curl -I https://your-domain.atlassian.net
```

3. Check if you're behind a proxy:
```bash
export HTTP_PROXY=http://proxy.company.com:8080
export HTTPS_PROXY=http://proxy.company.com:8080
```

4. Increase the timeout in your code (if needed):
```python
# In src/ingestion/confluence_client.py
self._client = Confluence(
    url=base_url,
    token=auth_token,
    cloud=cloud,
    timeout=60  # Increase timeout to 60 seconds
)
```

---

### Issue: Rate limit exceeded

**Symptoms:**
```
Error: 429 Too Many Requests - Rate limit exceeded
```

**Solution:**

The system implements automatic retry with exponential backoff. If you still encounter rate limits:

1. Reduce the ingestion batch size
2. Add delays between requests
3. Contact your Confluence administrator to increase rate limits
4. For Confluence Cloud, check your plan's API rate limits

---

### Issue: Space not found

**Symptoms:**
```
Error: Space 'DOCS' not found
```

**Solutions:**

1. Verify the space key is correct:
   - Go to your Confluence space
   - Check the URL: `https://your-domain.atlassian.net/wiki/spaces/DOCS`
   - The space key is the part after `/spaces/` (e.g., `DOCS`)

2. Ensure you have permission to access the space:
   - Log in to Confluence with the same credentials
   - Try to view the space manually

3. Check if the space key is case-sensitive:
   ```bash
   CONFLUENCE_SPACE_KEY=DOCS  # Not 'docs' or 'Docs'
   ```

---

## Ingestion Issues

### Issue: No pages retrieved

**Symptoms:**
```
INFO: Retrieved 0 pages from Confluence
```

**Solutions:**

1. Verify the space contains pages:
   - Log in to Confluence
   - Navigate to the space
   - Check if there are any pages

2. Check permissions:
   - Ensure your API token has read access to the space
   - Verify you can view the pages manually

3. Check for archived pages:
   - The system only retrieves active pages
   - Archived pages are not included

---

### Issue: Out of memory during ingestion

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. Reduce chunk size to create fewer embeddings:
```bash
CHUNK_SIZE=500  # Reduce from default 1000
```

2. Process pages in smaller batches:
```python
# Modify ingestion to process in batches
for i in range(0, len(pages), 10):
    batch = pages[i:i+10]
    service.ingest_pages(batch)
```

3. Increase available system memory:
   - Close other applications
   - Increase Docker memory limits (if using Docker)
   - Use a machine with more RAM

4. Use a more memory-efficient embedding model:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Smaller model
```

---

### Issue: Ingestion is very slow

**Symptoms:**
- Ingestion takes hours for a small space
- Progress appears stuck

**Solutions:**

1. Check network speed:
```bash
# Test download speed from Confluence
time curl -o /dev/null https://your-domain.atlassian.net
```

2. Enable verbose logging to see progress:
```bash
uv run python scripts/ingest.py --verbose
```

3. Optimize chunk size:
   - Larger chunks = fewer embeddings = faster ingestion
   - But may reduce search accuracy
```bash
CHUNK_SIZE=1500  # Increase from default 1000
```

4. Use batch embedding generation:
   - The system already does this by default
   - Verify `generate_batch_embeddings` is being used

5. Check CPU usage:
   - Embedding generation is CPU-intensive
   - Ensure your system isn't throttling

---

### Issue: HTML parsing errors

**Symptoms:**
```
Error: Failed to parse HTML content
```

**Solutions:**

1. The system handles Confluence storage format automatically
2. If errors persist, check for malformed HTML in the source page
3. Enable error recovery to skip problematic pages:
```python
# The system already implements this
# Check logs for which pages are failing
```

4. Report the issue with the page ID for investigation

---

## Vector Store Issues

### Issue: Vector store not found

**Symptoms:**
```
Error: Collection 'confluence_docs' not found
```

**Solutions:**

1. Run ingestion first to create the vector store:
```bash
uv run python scripts/ingest.py
```

2. Verify the persist directory exists:
```bash
ls -la ./chroma_db
```

3. Check permissions on the directory:
```bash
chmod 755 ./chroma_db
```

4. Ensure the collection name matches your configuration:
```bash
CHROMA_COLLECTION_NAME=confluence_docs
```

---

### Issue: Vector store corruption

**Symptoms:**
```
Error: Database is corrupted or incompatible
```

**Solutions:**

1. Backup the current database (if possible):
```bash
cp -r ./chroma_db ./chroma_db.backup
```

2. Delete and recreate the vector store:
```bash
rm -rf ./chroma_db
uv run python scripts/ingest.py
```

3. If using a different vector store (FAISS, Qdrant), check their specific troubleshooting guides

---

### Issue: Duplicate documents

**Symptoms:**
- Same content appears multiple times in search results
- Vector store size is larger than expected

**Solutions:**

1. The system implements deduplication by default
2. If duplicates persist, clear and re-ingest:
```bash
rm -rf ./chroma_db
uv run python scripts/ingest.py
```

3. Use incremental sync instead of full re-ingestion:
```bash
uv run python scripts/scheduled_sync.py
```

---

## Query Interface Issues

### Issue: Streamlit app won't start

**Symptoms:**
```
Error: Address already in use
```

**Solutions:**

1. Check if another process is using port 8501:
```bash
lsof -i :8501
```

2. Kill the existing process:
```bash
kill -9 <PID>
```

3. Or use a different port:
```bash
uv run streamlit run src/query/app.py --server.port 8502
```

---

### Issue: No search results found

**Symptoms:**
- All queries return 0 results
- "No results found" message appears

**Solutions:**

1. Verify the vector store contains data:
```python
from src.storage.vector_store import ChromaStore

store = ChromaStore("./chroma_db", "confluence_docs")
stats = store.get_collection_stats()
print(f"Documents: {stats['count']}")
```

2. Ensure ingestion completed successfully:
```bash
uv run python scripts/verify_setup.py
```

3. Check if you're searching the correct space:
   - Verify `CONFLUENCE_SPACE_KEY` matches the ingested space

4. Try broader search terms:
   - Instead of "API authentication token", try "authentication"

5. Check embedding model consistency:
   - Query and ingestion must use the same model
   - Verify `EMBEDDING_MODEL` is consistent

---

### Issue: Search results are irrelevant

**Symptoms:**
- Results don't match the query
- Low similarity scores (<30%)

**Solutions:**

1. Increase the number of results to see more options:
```python
results = processor.process_query(query, top_k=20)
```

2. Refine your query:
   - Be more specific
   - Use terminology from your documentation
   - Try different phrasings

3. Check if the content exists in Confluence:
   - Search manually in Confluence
   - Verify the pages were ingested

4. Consider re-ingesting with different chunk settings:
```bash
CHUNK_SIZE=800  # Smaller chunks for more granular search
CHUNK_OVERLAP=150
```

---

### Issue: Slow query response

**Symptoms:**
- Queries take >5 seconds to return results
- UI feels sluggish

**Solutions:**

1. Reduce the number of results:
```bash
TOP_K_RESULTS=5  # Reduce from default 10
```

2. Check vector store performance:
   - Chroma is fast for <100k documents
   - Consider FAISS for larger datasets

3. Optimize embedding generation:
   - Embeddings are cached by default
   - Verify cache is working

4. Check system resources:
```bash
top  # Check CPU and memory usage
```

---

## Performance Issues

### Performance Tuning Guide

#### Ingestion Performance

**Optimize for speed:**
```yaml
processing:
  chunk_size: 1500        # Larger chunks = fewer embeddings
  chunk_overlap: 100      # Less overlap = fewer chunks
  embedding_model: "all-MiniLM-L6-v2"  # Fast model
```

**Optimize for accuracy:**
```yaml
processing:
  chunk_size: 800         # Smaller chunks = more granular
  chunk_overlap: 200      # More overlap = better context
  embedding_model: "all-mpnet-base-v2"  # More accurate model
```

#### Query Performance

**Fast queries:**
- Use Chroma for <100k documents
- Reduce `top_k_results` to 5-10
- Use smaller embedding models

**Accurate queries:**
- Use FAISS for >100k documents
- Increase `top_k_results` to 20-50
- Use larger embedding models (e.g., all-mpnet-base-v2)

#### Memory Optimization

**Reduce memory usage:**
1. Use smaller embedding models
2. Reduce chunk size
3. Process pages in smaller batches
4. Clear cache periodically

**Monitor memory:**
```bash
# macOS
top -o MEM

# Linux
htop

# Python memory profiling
uv add memory-profiler
uv run python -m memory_profiler scripts/ingest.py
```

---

## Deployment Issues

### Issue: Posit Connect deployment fails

**Symptoms:**
```
Error: Deployment failed - missing dependencies
```

**Solutions:**

1. Ensure `requirements.txt` is up to date:
```bash
uv run python scripts/setup_deployment.py
```

2. Verify Python version in Posit Connect:
   - Set Python version to 3.12 in deployment settings

3. Check environment variables in Posit Connect:
   - All required variables must be set
   - Use the Posit Connect UI to configure

4. Review deployment logs:
   - Check for specific error messages
   - Look for missing dependencies or configuration

---

### Issue: Scheduled sync not running

**Symptoms:**
- Vector store is not updating
- New pages don't appear in search

**Solutions:**

1. Verify the scheduled job is configured:
   - Check Posit Connect scheduled jobs
   - Ensure the schedule is active

2. Check job logs:
   - Look for errors in the scheduled job logs
   - Verify the script is executing

3. Test the sync script manually:
```bash
uv run python scripts/scheduled_sync.py
```

4. Verify permissions:
   - Ensure the job has access to the vector store
   - Check file permissions on the persist directory

---

### Issue: Docker container crashes

**Symptoms:**
```
Error: Container exited with code 137
```

**Solutions:**

1. Increase Docker memory limits:
```bash
docker run --memory=4g --env-file .env confluence-rag
```

2. Check container logs:
```bash
docker logs <container-id>
```

3. Verify environment variables are passed:
```bash
docker run --env-file .env confluence-rag env
```

---

## Debugging Tips

### Enable Verbose Logging

Add verbose logging to see detailed information:

```bash
# For ingestion
uv run python scripts/ingest.py --verbose

# For query interface
export LOG_LEVEL=DEBUG
uv run streamlit run src/query/app.py
```

### Check Logs

Logs are written to stdout by default. To save logs to a file:

```bash
uv run python scripts/ingest.py 2>&1 | tee ingestion.log
```

### Use Python Debugger

Add breakpoints to investigate issues:

```python
import pdb; pdb.set_trace()
```

Or use the built-in debugger:

```bash
uv run python -m pdb scripts/ingest.py
```

### Verify Setup

Run the verification script to check all components:

```bash
uv run python scripts/verify_setup.py
```

This checks:
- Python version
- Environment variables
- Confluence connectivity
- Vector store accessibility
- Embedding model availability

### Test Individual Components

Test components in isolation:

```python
# Test Confluence connection
from src.ingestion.confluence_client import ConfluenceClient

client = ConfluenceClient(base_url, auth_token)
pages = client.get_space_pages("DOCS")
print(f"Retrieved {len(pages)} pages")

# Test embedding generation
from src.processing.embedder import EmbeddingGenerator

embedder = EmbeddingGenerator()
embedding = embedder.generate_embedding("test query")
print(f"Embedding dimension: {len(embedding)}")

# Test vector store
from src.storage.vector_store import ChromaStore

store = ChromaStore("./chroma_db", "confluence_docs")
stats = store.get_collection_stats()
print(f"Documents: {stats['count']}")
```

### Check System Resources

Monitor system resources during operation:

```bash
# CPU and memory
top

# Disk space
df -h

# Network connections
netstat -an | grep ESTABLISHED
```

### Profile Performance

Profile code to identify bottlenecks:

```bash
# Install profiler
uv add py-spy

# Profile ingestion
uv run py-spy record -o profile.svg -- python scripts/ingest.py

# View profile
open profile.svg
```

---

## Common Error Messages

### `ModuleNotFoundError: No module named 'src'`

**Solution:**
Run commands from the project root directory:
```bash
cd /path/to/confluence-rag-system
uv run python scripts/ingest.py
```

### `FileNotFoundError: [Errno 2] No such file or directory: '.env'`

**Solution:**
Create the `.env` file:
```bash
cp .env.example .env
# Edit .env with your settings
```

### `ValidationError: 1 validation error for AppConfig`

**Solution:**
Check your configuration for invalid values. The error message will indicate which field is invalid.

### `RuntimeError: Embedding model not found`

**Solution:**
The model will be downloaded automatically on first use. Ensure you have internet connectivity and sufficient disk space.

---

## Getting Help

If you're still experiencing issues:

1. **Check the documentation:**
   - [README.md](../README.md) - Project overview
   - [API.md](API.md) - API reference
   - [STREAMLIT_APP.md](STREAMLIT_APP.md) - Query interface guide

2. **Search existing issues:**
   - Check GitHub issues for similar problems
   - Look for closed issues with solutions

3. **Create a new issue:**
   - Include error messages
   - Provide configuration (remove sensitive data)
   - Describe steps to reproduce
   - Include system information (OS, Python version)

4. **Enable debug logging:**
   ```bash
   export LOG_LEVEL=DEBUG
   uv run python scripts/ingest.py 2>&1 | tee debug.log
   ```
   Include the debug log in your issue report.

5. **Community support:**
   - Use GitHub Discussions for questions
   - Check the project wiki for additional tips

---

## Preventive Maintenance

### Regular Tasks

1. **Update dependencies:**
```bash
uv sync --upgrade
```

2. **Run incremental sync:**
```bash
# Schedule this to run daily or weekly
uv run python scripts/scheduled_sync.py
```

3. **Monitor disk space:**
```bash
du -sh ./chroma_db
```

4. **Check logs for errors:**
```bash
grep ERROR ingestion.log
```

5. **Verify search quality:**
   - Periodically test common queries
   - Ensure results are relevant

### Backup Strategy

1. **Backup vector store:**
```bash
tar -czf chroma_db_backup_$(date +%Y%m%d).tar.gz ./chroma_db
```

2. **Backup configuration:**
```bash
cp .env .env.backup
cp -r config config.backup
```

3. **Document custom changes:**
   - Keep notes on configuration changes
   - Document any code modifications

---

## Performance Benchmarks

### Expected Performance

**Ingestion:**
- Small space (50 pages): 2-5 minutes
- Medium space (500 pages): 20-40 minutes
- Large space (5000 pages): 3-6 hours

**Query:**
- Single query: <1 second
- Batch queries (10): <5 seconds

**Memory Usage:**
- Ingestion: 2-4 GB RAM
- Query interface: 1-2 GB RAM
- Vector store: ~1 MB per 1000 chunks

**Disk Space:**
- Vector store: ~500 KB per page (average)
- Embedding model: ~100 MB (one-time download)

If your performance significantly differs from these benchmarks, review the performance tuning section above.

---

## Additional Resources

- [Confluence API Documentation](https://developer.atlassian.com/cloud/confluence/rest/v2/intro/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [sentence-transformers Documentation](https://www.sbert.net/)
