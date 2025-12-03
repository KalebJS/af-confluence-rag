# Posit Connect Deployment Guide

This guide provides step-by-step instructions for deploying the Confluence RAG System to Posit Connect.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Variable Configuration](#environment-variable-configuration)
- [Persistent Storage Setup](#persistent-storage-setup)
- [Deploying the Streamlit Query Interface](#deploying-the-streamlit-query-interface)
- [Scheduled Execution Setup for Ingestion](#scheduled-execution-setup-for-ingestion)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before deploying to Posit Connect, ensure you have:

1. **Posit Connect Account**: Access to a Posit Connect instance with deployment permissions
2. **Python 3.12**: The application requires Python 3.12.x
3. **Confluence API Access**: Valid Confluence API credentials (base URL and API token)
4. **rsconnect-python**: Install the deployment CLI tool:
   ```bash
   pip install rsconnect-python
   ```

## Environment Variable Configuration

The Confluence RAG System requires several environment variables to be configured in Posit Connect.

### Required Environment Variables

Configure these variables in the Posit Connect dashboard under your application's "Vars" tab:

| Variable Name | Description | Example |
|--------------|-------------|---------|
| `CONFLUENCE_BASE_URL` | Your Confluence instance URL | `https://your-domain.atlassian.net` |
| `CONFLUENCE_AUTH_TOKEN` | Confluence API token | `your-api-token-here` |
| `CONFLUENCE_SPACE_KEY` | Space key to sync | `DOCS` |
| `CONFLUENCE_CLOUD` | Set to `true` for Cloud, `false` for Server | `true` |

### Optional Configuration Variables

| Variable Name | Description | Default |
|--------------|-------------|---------|
| `CHUNK_SIZE` | Document chunk size in tokens | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `EMBEDDING_MODEL` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | Number of search results | `10` |
| `VECTOR_STORE_TYPE` | Vector store backend | `chroma` |
| `CHROMA_PERSIST_DIR` | Chroma database directory | `./chroma_db` |
| `CHROMA_COLLECTION_NAME` | Collection name | `confluence_docs` |

### Setting Environment Variables in Posit Connect

1. Navigate to your deployed application in Posit Connect
2. Click on the "Vars" tab
3. Click "Add Variable"
4. Enter the variable name and value
5. Click "Save"
6. Restart the application for changes to take effect

**Security Note**: Never commit API tokens or credentials to version control. Always use environment variables for sensitive information.

## Persistent Storage Setup

The vector database requires persistent storage to maintain embeddings between application restarts.

### Chroma Database Persistence

By default, the system uses Chroma as the vector database with local file persistence.

#### Option 1: Posit Connect Persistent Storage (Recommended)

Posit Connect provides persistent storage for applications:

1. **Configure Storage Path**: Set the `CHROMA_PERSIST_DIR` environment variable to a path within your application's persistent storage:
   ```
   CHROMA_PERSIST_DIR=/var/data/confluence-rag/chroma_db
   ```

2. **Verify Permissions**: Ensure the application has read/write permissions to the storage directory

3. **Storage Lifecycle**: Data persists across deployments and restarts

#### Option 2: External Vector Database

For production deployments, consider using an external vector database:

**Qdrant** (Recommended for production):
```yaml
# config/production.yaml
vector_store:
  type: "qdrant"
  config:
    url: "https://your-qdrant-instance.com"
    api_key: "${QDRANT_API_KEY}"
    collection_name: "confluence_docs"
```

Environment variables:
```
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=https://your-qdrant-instance.com
QDRANT_API_KEY=your-api-key
```

**FAISS with S3 Backup**:
```yaml
# config/production.yaml
vector_store:
  type: "faiss"
  config:
    index_path: "/var/data/confluence-rag/faiss_index"
    backup_to_s3: true
    s3_bucket: "your-bucket-name"
```

### Storage Size Considerations

Estimate storage requirements:
- **Embeddings**: ~1KB per document chunk
- **Metadata**: ~500 bytes per chunk
- **Example**: 10,000 pages × 10 chunks/page = 100,000 chunks ≈ 150MB

Plan for 2-3x the estimated size to account for growth and overhead.

## Deploying the Streamlit Query Interface

### Step 1: Prepare the Application

1. **Ensure requirements.txt exists**:
   ```bash
   # Already created in repository root
   ls requirements.txt
   ```

2. **Verify Python version**:
   ```bash
   python --version  # Should be 3.12.x
   ```

### Step 2: Deploy to Posit Connect

#### Using rsconnect-python CLI

1. **Configure Posit Connect credentials**:
   ```bash
   rsconnect add \
     --server https://your-posit-connect-server.com \
     --name my-connect \
     --api-key your-api-key
   ```

2. **Deploy the Streamlit app**:
   ```bash
   rsconnect deploy streamlit \
     --server my-connect \
     --title "Confluence Documentation Search" \
     --python /path/to/python3.12 \
     src/query/app.py
   ```

3. **Verify deployment**:
   - Navigate to the application URL provided by rsconnect
   - Check the application logs for any errors
   - Test the search functionality

#### Using Posit Connect UI

1. **Create a new deployment**:
   - Log in to Posit Connect
   - Click "Publish" → "New Content"
   - Select "Streamlit Application"

2. **Upload application files**:
   - Upload the entire `src/` directory
   - Upload `requirements.txt`
   - Upload `config/` directory

3. **Configure the application**:
   - Set Python version to 3.12
   - Set entry point to `src/query/app.py`
   - Configure environment variables (see above)

4. **Deploy and test**:
   - Click "Deploy"
   - Monitor deployment logs
   - Test the application once deployed

### Step 3: Post-Deployment Configuration

1. **Set Access Controls**:
   - Configure who can view the application
   - Set up authentication if required

2. **Configure Resource Limits**:
   - Memory: Recommend at least 2GB for embedding model
   - CPU: 1-2 cores sufficient for typical usage

3. **Enable Logging**:
   - Configure log retention in Posit Connect
   - Set log level via `LOG_LEVEL` environment variable

## Scheduled Execution Setup for Ingestion

The ingestion service should run on a schedule to keep the vector database synchronized with Confluence.

### Option 1: Posit Connect Scheduled Execution

1. **Create a separate deployment for ingestion**:
   ```bash
   rsconnect deploy api \
     --server my-connect \
     --title "Confluence Ingestion Service" \
     --python /path/to/python3.12 \
     --entrypoint scripts/ingest.py
   ```

2. **Configure as a scheduled job**:
   - In Posit Connect, navigate to the ingestion deployment
   - Go to "Schedule" tab
   - Set schedule (e.g., "Daily at 2:00 AM")
   - Configure timeout (recommend 1-2 hours for large spaces)

3. **Monitor execution**:
   - Check execution logs after each run
   - Set up email notifications for failures
   - Monitor storage usage

### Option 2: External Scheduler (cron, Airflow, etc.)

If using an external scheduler:

1. **Create a deployment script** (see `scripts/scheduled_sync.py` below)

2. **Configure the scheduler**:
   ```bash
   # Example cron entry (daily at 2 AM)
   0 2 * * * /path/to/python3.12 /path/to/scripts/scheduled_sync.py
   ```

3. **Ensure environment variables are available**:
   - Load from `.env` file or system environment
   - Use secrets management for credentials

### Ingestion Schedule Recommendations

- **Small spaces (<100 pages)**: Every 6-12 hours
- **Medium spaces (100-1000 pages)**: Daily
- **Large spaces (>1000 pages)**: Daily or weekly
- **Critical documentation**: Every 1-2 hours

### Monitoring Ingestion Jobs

Monitor these metrics:

- **Execution time**: Track how long ingestion takes
- **Pages processed**: Number of new/updated/deleted pages
- **Errors**: Failed page retrievals or processing errors
- **Storage growth**: Vector database size over time

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Symptoms**: Deployment fails or application crashes on startup

**Solutions**:
- Verify Python version is 3.12.x
- Check all required environment variables are set
- Review application logs for specific error messages
- Ensure `requirements.txt` is present and valid

#### 2. Confluence Connection Errors

**Symptoms**: "Failed to connect to Confluence" errors

**Solutions**:
- Verify `CONFLUENCE_BASE_URL` is correct (include `https://`)
- Check `CONFLUENCE_AUTH_TOKEN` is valid and not expired
- Ensure Posit Connect can reach Confluence (firewall/network)
- Verify `CONFLUENCE_CLOUD` setting matches your instance type

#### 3. Vector Database Errors

**Symptoms**: "Failed to initialize vector store" or search errors

**Solutions**:
- Verify `CHROMA_PERSIST_DIR` path exists and is writable
- Check storage quota hasn't been exceeded
- Ensure vector database files aren't corrupted
- Try clearing the database and re-ingesting

#### 4. Out of Memory Errors

**Symptoms**: Application crashes with memory errors

**Solutions**:
- Increase memory allocation in Posit Connect (recommend 2-4GB)
- Reduce `CHUNK_SIZE` to process smaller batches
- Use a smaller embedding model
- Process spaces incrementally

#### 5. Slow Search Performance

**Symptoms**: Queries take >5 seconds to return results

**Solutions**:
- Verify vector database is using persistent storage (not rebuilding)
- Consider using an external vector database (Qdrant, Pinecone)
- Reduce `TOP_K_RESULTS` to return fewer results
- Check CPU/memory allocation

### Getting Help

If you encounter issues not covered here:

1. **Check application logs** in Posit Connect
2. **Review Confluence API logs** for authentication/rate limit issues
3. **Consult the main README** for general troubleshooting
4. **Open an issue** on the project repository with:
   - Error messages and logs
   - Environment configuration (redact credentials)
   - Steps to reproduce the issue

## Best Practices

### Security

- ✅ Use environment variables for all credentials
- ✅ Rotate API tokens regularly
- ✅ Limit Confluence API token permissions to read-only
- ✅ Use Posit Connect access controls to restrict application access
- ✅ Enable HTTPS for all connections

### Performance

- ✅ Schedule ingestion during off-peak hours
- ✅ Monitor storage usage and set up alerts
- ✅ Use incremental sync to minimize processing time
- ✅ Consider caching frequently accessed results
- ✅ Allocate sufficient memory for embedding model

### Reliability

- ✅ Set up monitoring and alerting for failed ingestion jobs
- ✅ Implement backup strategy for vector database
- ✅ Test disaster recovery procedures
- ✅ Document your deployment configuration
- ✅ Keep dependencies up to date

## Next Steps

After successful deployment:

1. **Test the application** with real queries
2. **Monitor performance** and adjust resources as needed
3. **Set up alerting** for failures
4. **Document** any custom configuration for your organization
5. **Train users** on how to use the search interface

For more information, see:
- [Main README](../README.md)
- [Streamlit App Documentation](./STREAMLIT_APP.md)
- [Configuration Guide](../config/README.md)
