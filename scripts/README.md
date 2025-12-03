# Deployment Scripts

This directory contains scripts for deploying and managing the Confluence RAG System.

## Scripts Overview

### 1. setup_deployment.py

Initial setup script for new deployments. Validates configuration, creates directories, initializes the vector database, and performs a test ingestion.

**Usage:**
```bash
python scripts/setup_deployment.py [--config CONFIG_PATH] [--skip-test-ingestion]
```

**Options:**
- `--config`: Path to configuration file (optional, defaults to environment-based config)
- `--skip-test-ingestion`: Skip the test ingestion step

**What it does:**
1. Validates environment variables and configuration
2. Creates necessary directories (vector store, logs)
3. Initializes the vector database
4. Tests Confluence API connectivity
5. Performs a test ingestion of 5 pages (unless skipped)

**Example:**
```bash
# Basic setup
python scripts/setup_deployment.py

# Setup with custom config
python scripts/setup_deployment.py --config config/production.yaml

# Setup without test ingestion
python scripts/setup_deployment.py --skip-test-ingestion
```

**Exit codes:**
- `0`: All checks passed
- `1`: One or more checks failed

---

### 2. scheduled_sync.py

Scheduled synchronization script for keeping the vector database up-to-date with Confluence.

**Usage:**
```bash
python scripts/scheduled_sync.py [--config CONFIG_PATH] [--full-sync] [--notify EMAIL]
```

**Options:**
- `--config`: Path to configuration file (optional)
- `--full-sync`: Perform full sync instead of incremental
- `--notify`: Email address for notifications (placeholder for future implementation)

**What it does:**
1. Detects changes in Confluence (new, modified, deleted pages)
2. Updates the vector database accordingly
3. Logs synchronization statistics
4. Exits with appropriate status code

**Examples:**
```bash
# Incremental sync (default)
python scripts/scheduled_sync.py

# Full sync
python scripts/scheduled_sync.py --full-sync

# Sync with custom config
python scripts/scheduled_sync.py --config config/production.yaml

# Sync with notification
python scripts/scheduled_sync.py --notify admin@example.com
```

**Scheduling:**

**Cron (Linux/macOS):**
```bash
# Daily at 2 AM
0 2 * * * cd /path/to/project && /path/to/python3.12 scripts/scheduled_sync.py

# Every 6 hours
0 */6 * * * cd /path/to/project && /path/to/python3.12 scripts/scheduled_sync.py
```

**Posit Connect:**
1. Deploy as a separate application
2. Configure as a scheduled job in the Posit Connect dashboard
3. Set schedule (e.g., "Daily at 2:00 AM")

**Exit codes:**
- `0`: Sync completed successfully
- `1`: Sync failed

---

### 3. health_check.py

Comprehensive health check script for monitoring system status.

**Usage:**
```bash
python scripts/health_check.py [--config CONFIG_PATH] [--json]
```

**Options:**
- `--config`: Path to configuration file (optional)
- `--json`: Output results in JSON format (useful for monitoring systems)

**What it checks:**
1. Configuration validity
2. Confluence API connectivity
3. Vector database accessibility
4. Embedding model availability
5. Query processing functionality
6. Storage space usage

**Examples:**
```bash
# Basic health check
python scripts/health_check.py

# Health check with JSON output
python scripts/health_check.py --json

# Health check with custom config
python scripts/health_check.py --config config/production.yaml

# Use in monitoring (exit code indicates health)
python scripts/health_check.py && echo "System healthy" || echo "System unhealthy"
```

**JSON Output Format:**
```json
{
  "timestamp": "2024-01-15T14:30:00",
  "overall_status": "healthy",
  "total_checks": 6,
  "passed": 6,
  "failed": 0,
  "warnings": 0,
  "skipped": 0,
  "checks": {
    "configuration": {
      "status": "pass",
      "message": "Configuration loaded successfully",
      "details": {...}
    },
    ...
  }
}
```

**Exit codes:**
- `0`: All checks passed
- `1`: One or more checks failed

---

### 4. ingest.py

Manual ingestion script for one-time or ad-hoc ingestion tasks.

**Usage:**
```bash
python scripts/ingest.py [--config CONFIG_PATH] [--space SPACE_KEY] [--max-pages N]
```

See the script for full documentation.

---

### 5. run_app.py

Script to run the Streamlit query interface locally.

**Usage:**
```bash
python scripts/run_app.py [--config CONFIG_PATH] [--port PORT]
```

See the script for full documentation.

---

## Common Workflows

### Initial Deployment

1. **Set up environment variables:**
   ```bash
   export CONFLUENCE_BASE_URL="https://your-domain.atlassian.net"
   export CONFLUENCE_AUTH_TOKEN="your-api-token"
   export CONFLUENCE_SPACE_KEY="DOCS"
   ```

2. **Run setup script:**
   ```bash
   python scripts/setup_deployment.py
   ```

3. **Verify health:**
   ```bash
   python scripts/health_check.py
   ```

4. **Perform full ingestion:**
   ```bash
   python scripts/ingest.py
   ```

### Ongoing Maintenance

1. **Schedule regular syncs:**
   ```bash
   # Add to crontab
   0 2 * * * cd /path/to/project && python scripts/scheduled_sync.py
   ```

2. **Monitor health:**
   ```bash
   # Add to monitoring system
   */15 * * * * python scripts/health_check.py --json > /var/log/health.json
   ```

3. **Manual sync when needed:**
   ```bash
   python scripts/scheduled_sync.py --full-sync
   ```

### Troubleshooting

1. **Check system health:**
   ```bash
   python scripts/health_check.py
   ```

2. **Test configuration:**
   ```bash
   python scripts/setup_deployment.py --skip-test-ingestion
   ```

3. **Force full re-sync:**
   ```bash
   python scripts/scheduled_sync.py --full-sync
   ```

## Environment Variables

All scripts respect the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `CONFLUENCE_BASE_URL` | Yes | Confluence instance URL |
| `CONFLUENCE_AUTH_TOKEN` | Yes | API authentication token |
| `CONFLUENCE_SPACE_KEY` | Yes | Space key to sync |
| `CONFLUENCE_CLOUD` | No | `true` for Cloud, `false` for Server (default: `true`) |
| `CHUNK_SIZE` | No | Document chunk size (default: `1000`) |
| `CHUNK_OVERLAP` | No | Chunk overlap (default: `200`) |
| `EMBEDDING_MODEL` | No | Model name (default: `all-MiniLM-L6-v2`) |
| `VECTOR_STORE_TYPE` | No | Store type (default: `chroma`) |
| `CHROMA_PERSIST_DIR` | No | Chroma directory (default: `./chroma_db`) |

## Logging

All scripts use structured logging via `structlog`. Logs are written to:
- **stdout**: For normal operation
- **stderr**: For errors

Configure log level via environment variable:
```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Exit Codes

All scripts follow standard exit code conventions:
- `0`: Success
- `1`: Failure

This allows for easy integration with monitoring systems and schedulers.

## Integration Examples

### Posit Connect Scheduled Job

1. Deploy `scheduled_sync.py` as a separate application
2. Configure environment variables in Posit Connect
3. Set up schedule in the "Schedule" tab
4. Monitor execution logs

### Docker Container

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Run health check on startup
CMD ["python", "scripts/health_check.py"]
```

### Kubernetes CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: confluence-sync
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: sync
            image: confluence-rag:latest
            command: ["python", "scripts/scheduled_sync.py"]
            envFrom:
            - secretRef:
                name: confluence-credentials
```

## Support

For issues or questions:
1. Check the main [README](../README.md)
2. Review [Posit Connect Deployment Guide](../docs/POSIT_CONNECT_DEPLOYMENT.md)
3. Check application logs
4. Open an issue on the project repository
