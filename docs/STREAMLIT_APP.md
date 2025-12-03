# Streamlit Query Interface

This document describes how to run and use the Streamlit query interface for the Confluence RAG system.

## Prerequisites

1. Ensure the ingestion process has completed and the vector database is populated
2. Configuration file is set up correctly (see `config/default.yaml`)
3. Required environment variables are set:
   - `CONFLUENCE_BASE_URL`
   - `CONFLUENCE_AUTH_TOKEN`
   - `CONFLUENCE_SPACE_KEY`

## Running the Application

### Local Development

Run the Streamlit app using:

```bash
uv run streamlit run src/query/app.py
```

The application will start on `http://localhost:8501` by default.

### Using a Specific Configuration

To use a different configuration file:

```bash
APP_ENV=production uv run streamlit run src/query/app.py
```

This will load `config/production.yaml` instead of the default configuration.

## Features

### Search Interface

- **Natural Language Queries**: Enter questions or keywords in plain English
- **Configurable Results**: Adjust the maximum number of results (1-50)
- **Real-time Search**: Results appear immediately after clicking the search button

### Search Results

Each result displays:
- **Page Title**: Clickable link to the source Confluence page
- **Similarity Score**: Color-coded relevance indicator
  - 🟢 Green (≥0.8): Highly relevant
  - 🟡 Yellow (≥0.6): Moderately relevant
  - 🟠 Orange (<0.6): Less relevant
- **Content Excerpt**: Preview of the matching content
- **Metadata**: Expandable section with page ID, chunk ID, author, and modification date

### Search History

- **Recent Searches**: View your last 10 searches in the sidebar
- **Quick Re-run**: Click any previous search to run it again
- **Clear History**: Remove all search history with one click

### Configuration Display

The sidebar shows:
- Current Confluence space
- Embedding model in use
- Vector store type

## Error Handling

The application provides user-friendly error messages for common issues:

### Database Unavailable
If the vector database cannot be accessed, you'll see:
- Clear error message
- Troubleshooting steps
- Suggestions for resolution

### Invalid Queries
Empty or invalid queries will show a warning message.

### Configuration Errors
Missing or invalid configuration will display:
- Specific error details
- Required fields
- Configuration file location

## Tips for Best Results

1. **Be Specific**: Use detailed questions rather than single keywords
2. **Natural Language**: Write queries as you would ask a colleague
3. **Try Variations**: If you don't find what you need, rephrase your query
4. **Check Scores**: Higher similarity scores indicate better matches
5. **Use History**: Refine previous searches by clicking them in the sidebar

## Deployment to Posit Connect

### Prerequisites

1. Posit Connect account with deployment permissions
2. Environment variables configured in Posit Connect
3. Vector database accessible from Posit Connect environment

### Deployment Steps

1. Ensure all dependencies are in `pyproject.toml`
2. Set environment variables in Posit Connect:
   - `CONFLUENCE_BASE_URL`
   - `CONFLUENCE_AUTH_TOKEN`
   - `CONFLUENCE_SPACE_KEY`
3. Deploy using the Posit Connect UI or CLI:

```bash
rsconnect deploy streamlit \
  --server https://your-posit-connect-server \
  --api-key YOUR_API_KEY \
  src/query/app.py
```

### Persistent Storage

Ensure the vector database directory is:
- Accessible from the Posit Connect environment
- Configured with appropriate permissions
- Backed up regularly

## Troubleshooting

### App Won't Start

**Symptom**: Application fails to initialize

**Solutions**:
1. Check environment variables are set
2. Verify configuration file exists
3. Ensure vector database is accessible
4. Check application logs for details

### No Search Results

**Symptom**: All searches return empty results

**Solutions**:
1. Verify ingestion has completed
2. Check vector database contains data
3. Ensure embedding model matches ingestion model
4. Verify space key is correct

### Slow Performance

**Symptom**: Searches take a long time

**Solutions**:
1. Reduce `top_k` parameter
2. Check vector database performance
3. Verify embedding model is loaded
4. Consider upgrading hardware resources

## Architecture

The Streamlit app integrates several components:

```
┌─────────────────────────────────────┐
│      Streamlit App (app.py)         │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐  │
│  │   QueryProcessor             │  │
│  │   - Embeds queries           │  │
│  │   - Searches vector store    │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   ResultFormatter            │  │
│  │   - Formats results          │  │
│  │   - Validates URLs           │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   EmbeddingGenerator         │  │
│  │   - Generates embeddings     │  │
│  └──────────────────────────────┘  │
│                                     │
│  ┌──────────────────────────────┐  │
│  │   VectorStore                │  │
│  │   - Similarity search        │  │
│  └──────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
```

## Customization

### Changing the Embedding Model

Edit `config/default.yaml`:

```yaml
processing:
  embedding_model: "all-MiniLM-L6-v2"  # Change to your preferred model
```

**Note**: The model must match the one used during ingestion.

### Adjusting Default Results

Edit `config/default.yaml`:

```yaml
top_k_results: 10  # Change default number of results
```

### Customizing the UI

The app's appearance can be customized by modifying:
- `setup_page_config()`: Page title, icon, layout
- `render_header()`: Header text and styling
- `render_results()`: Result card formatting

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review application logs
3. Verify configuration settings
4. Contact your system administrator
