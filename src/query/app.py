"""Streamlit application for querying Confluence documentation.

This module provides a web-based interface for semantic search over
Confluence documentation using the RAG system.
"""

from typing import Any

import streamlit as st
import structlog

from src.models.config import AppConfig
from src.processing.embedder import EmbeddingGenerator
from src.query.query_processor import QueryProcessor
from src.query.result_formatter import ResultFormatter
from src.storage.vector_store import VectorStoreFactory
from src.utils.config_loader import ConfigLoader, ConfigurationError
from src.utils.logging_config import configure_logging

log = structlog.stdlib.get_logger()


def initialize_app() -> tuple[QueryProcessor, ResultFormatter, AppConfig]:
    """Initialize the application components.

    This function loads configuration, initializes the embedding generator,
    vector store, query processor, and result formatter.

    Returns:
        Tuple of (QueryProcessor, ResultFormatter, AppConfig)

    Raises:
        ConfigurationError: If configuration cannot be loaded
        RuntimeError: If components cannot be initialized
    """
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_config()

        # Configure logging
        configure_logging(
            log_level=config.logging.log_level,
            json_logs=config.logging.json_logs,
            log_file=config.logging.log_file,
        )

        log.info(
            "app_configuration_loaded",
            vector_store_type=config.vector_store.type,
            log_level=config.logging.log_level,
        )

        # Initialize embedding generator
        embedder = EmbeddingGenerator(model_name=config.processing.embedding_model)

        # Initialize vector store
        vector_store = VectorStoreFactory.create_vector_store(
            store_type=config.vector_store.type,
            config=config.vector_store.config,
        )

        # Initialize query processor
        query_processor = QueryProcessor(
            embedder=embedder,
            vector_store=vector_store,
        )

        # Initialize result formatter
        result_formatter = ResultFormatter(base_url=str(config.confluence.base_url))

        log.info("app_initialized_successfully")

        return query_processor, result_formatter, config

    except ConfigurationError as e:
        log.error("app_initialization_failed", error=str(e))
        raise
    except Exception as e:
        log.error("app_initialization_failed", error=str(e))
        raise RuntimeError(f"Failed to initialize application: {e}") from e


def setup_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Confluence Documentation Search",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header() -> None:
    """Render the application header."""
    st.title("🔍 Confluence Documentation Search")
    st.markdown(
        "Search your Confluence documentation using natural language queries. "
        "Results are ranked by semantic similarity.",
    )
    st.divider()


def render_search_interface() -> tuple[str, int]:
    """Render the search input interface.

    Returns:
        Tuple of (query string, top_k value)
    """
    # Check if a history query was selected
    default_query = ""
    if "selected_history_query" in st.session_state:
        default_query = st.session_state.selected_history_query
        del st.session_state.selected_history_query
        st.session_state.search_triggered = True

    # Main search input
    query = st.text_input(
        "Enter your search query:",
        value=default_query,
        placeholder="e.g., How do I configure authentication?",
        help="Enter a natural language question or keywords to search the documentation",
    )

    # Search button and configuration in columns
    col1, col2, _ = st.columns([2, 1, 1])

    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    with col2:
        # Top K results slider
        top_k = st.number_input(
            "Max results",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Maximum number of results to return",
        )

    # Store search button state
    if search_button:
        st.session_state.search_triggered = True

    return query, int(top_k)


def render_results(
    results: list[dict[str, Any]],
    formatter: ResultFormatter,
) -> None:
    """Render search results.

    Args:
        results: List of result card dictionaries
        formatter: Result formatter instance
    """
    if not results:
        st.info("ℹ️ No results found. Try rephrasing your query or using different keywords.")
        return

    st.success(f"✅ Found {len(results)} result(s)")
    st.divider()

    # Display each result as a card
    for i, result in enumerate(results, 1):
        with st.container():
            # Result header with title and score
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {i}. [{result['title']}]({result['url']})")

            with col2:
                # Display similarity score with color coding
                score = result["score"]
                if score >= 0.8:
                    score_color = "🟢"
                elif score >= 0.6:
                    score_color = "🟡"
                else:
                    score_color = "🟠"
                st.markdown(f"{score_color} **Score:** {score:.3f}")

            # Content excerpt
            st.markdown(result["content"])

            # Metadata in expander
            with st.expander("📋 View metadata"):
                metadata = result["metadata"]
                st.markdown(f"**Page ID:** `{metadata['page_id']}`")
                st.markdown(f"**Chunk ID:** `{metadata['chunk_id']}`")
                st.markdown(f"**Author:** {metadata['author']}")
                st.markdown(f"**Last Modified:** {metadata['modified_date']}")
                st.markdown(f"**URL:** {result['url']}")

            st.divider()


def render_sidebar(config: AppConfig) -> None:
    """Render the sidebar with configuration and information.

    Args:
        config: Application configuration
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Display current configuration
        st.markdown("**Confluence Space:**")
        st.code(config.confluence.space_key)

        st.markdown("**Embedding Model:**")
        st.code(config.processing.embedding_model)

        st.markdown("**Vector Store:**")
        st.code(config.vector_store.type)

        st.divider()

        # Search history
        st.header("📜 Search History")
        if "search_history" in st.session_state and st.session_state.search_history:
            # Display last 10 searches
            for i, hist_query in enumerate(reversed(st.session_state.search_history[-10:]), 1):
                if st.button(f"{i}. {hist_query}", key=f"history_{i}", use_container_width=True):
                    st.session_state.selected_history_query = hist_query
                    st.rerun()

            # Clear history button
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.search_history = []
                st.rerun()
        else:
            st.markdown("*No search history yet*")

        st.divider()

        # Help section
        st.header("💡 Tips")
        st.markdown("""
        - Use natural language questions for best results
        - Be specific about what you're looking for
        - Try different phrasings if you don't find what you need
        - Higher scores indicate better matches
        - Click on previous searches to run them again
        """)

        st.divider()

        # About section
        st.header("ℹ️ About")
        st.markdown("""
        This application uses semantic search to find relevant documentation
        from your Confluence space. Results are ranked by similarity to your query.
        """)


def render_error(error: Exception) -> None:
    """Render an error message.

    Args:
        error: Exception that occurred
    """
    st.error(f"❌ An error occurred: {str(error)}")

    with st.expander("🔍 Error details"):
        st.code(str(error))
        st.markdown("""
        **Troubleshooting:**
        - Check that the vector database is accessible
        - Verify your configuration settings
        - Ensure the ingestion process has completed
        - Check the application logs for more details
        """)


def main() -> None:
    """Main application entry point."""
    # Set up page configuration
    setup_page_config()

    # Initialize session state
    if "search_triggered" not in st.session_state:
        st.session_state.search_triggered = False
    if "search_history" not in st.session_state:
        st.session_state.search_history = []

    try:
        # Initialize application components
        with st.spinner("🔄 Initializing application..."):
            query_processor, result_formatter, config = initialize_app()

        # Render header
        render_header()

        # Render sidebar
        render_sidebar(config)

        # Render search interface
        query, top_k = render_search_interface()

        # Process search if triggered
        if st.session_state.search_triggered and query:
            st.session_state.search_triggered = False  # Reset trigger

            # Add to search history (avoid duplicates)
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)

            with st.spinner("🔍 Searching documentation..."):
                try:
                    # Process query
                    results = query_processor.process_query(query, top_k=top_k)

                    # Format results
                    result_cards = result_formatter.create_result_cards(results)

                    # Render results
                    render_results(result_cards, result_formatter)

                    log.info(
                        "search_completed",
                        query_length=len(query),
                        results_count=len(results),
                        top_k=top_k,
                    )

                except ValueError as e:
                    st.warning(f"⚠️ Invalid query: {str(e)}")
                except Exception as e:
                    log.error("search_failed", query=query, error=str(e))
                    render_error(e)

        elif st.session_state.search_triggered and not query:
            st.session_state.search_triggered = False
            st.warning("⚠️ Please enter a search query")

    except ConfigurationError as e:
        st.error("❌ Configuration Error")
        st.markdown(f"**Error:** {str(e)}")
        st.markdown("""
        **Please check:**
        - Environment variables are set correctly
        - Configuration file exists and is valid
        - Required fields are populated
        """)
        log.error("app_configuration_error", error=str(e))

    except Exception as e:
        st.error("❌ Application Initialization Failed")
        st.markdown(f"**Error:** {str(e)}")
        st.markdown("""
        **Possible causes:**
        - Vector database is not accessible
        - Embedding model cannot be loaded
        - Configuration is invalid
        
        Please check the application logs for more details.
        """)
        log.error("app_initialization_error", error=str(e))


if __name__ == "__main__":
    main()
