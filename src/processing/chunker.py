"""Document chunking functionality for processing Confluence pages."""

import re

import structlog
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from structlog.stdlib import BoundLogger

from src.models.page import DocumentChunk, Page
from src.processing.metadata_enricher import MetadataEnricher

log: BoundLogger = structlog.stdlib.get_logger()


class DocumentChunker:
    """Chunks documents using LangChain RecursiveCharacterTextSplitter.

    This class handles splitting Confluence page content into smaller chunks
    while preserving semantic coherence and cleaning HTML markup.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the document chunker.

        Args:
            chunk_size: Target chunk size in characters (500-2000)
            chunk_overlap: Overlap between chunks in characters (0-500)
        """
        if not 500 <= chunk_size <= 2000:
            raise ValueError(f"chunk_size must be between 500 and 2000, got {chunk_size}")
        if not 0 <= chunk_overlap <= 500:
            raise ValueError(f"chunk_overlap must be between 0 and 500, got {chunk_overlap}")

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        # Initialize LangChain text splitter with recursive character splitting
        # This preserves semantic coherence by splitting at paragraph/sentence boundaries
        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",  # Paragraph breaks (highest priority)
                "\n",  # Line breaks
                ". ",  # Sentence boundaries
                "! ",  # Sentence boundaries
                "? ",  # Sentence boundaries
                "; ",  # Clause boundaries
                ", ",  # Phrase boundaries
                " ",  # Word boundaries
                "",  # Character-level (last resort)
            ],
            is_separator_regex=False,
        )

        # Initialize metadata enricher
        self.metadata_enricher: MetadataEnricher = MetadataEnricher()

        log.info(
            "DocumentChunker initialized",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _clean_html(self, html_content: str) -> str:
        """Clean HTML from Confluence storage format.

        Confluence stores content in HTML format. This method extracts
        plain text while preserving structure.

        Args:
            html_content: HTML content from Confluence

        Returns:
            Cleaned plain text
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator="\n")

        # Clean up excessive whitespace while preserving paragraph breaks
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # Remove empty lines
        text = "\n\n".join(lines)

        # Remove excessive spaces
        text = re.sub(r" +", " ", text)

        return text.strip()

    def chunk_document(self, page: Page) -> list[DocumentChunk]:
        """Chunk a Confluence page into smaller segments.

        This method:
        1. Cleans HTML from the page content
        2. Splits the content into chunks using RecursiveCharacterTextSplitter
        3. Creates DocumentChunk objects with preserved metadata

        Args:
            page: Confluence page to chunk

        Returns:
            List of DocumentChunk objects with metadata
        """
        log.info(
            "Chunking document",
            page_id=page.id,
            page_title=page.title,
            content_length=len(page.content),
        )

        # Clean HTML content
        cleaned_content = self._clean_html(page.content)

        if not cleaned_content:
            log.warning("Empty content after HTML cleaning", page_id=page.id)
            return []

        # Split into chunks
        text_chunks = self.text_splitter.split_text(cleaned_content)

        # Enrich chunks with metadata using MetadataEnricher
        chunks = self.metadata_enricher.enrich_chunks(page, text_chunks)

        log.info(
            "Document chunked successfully",
            page_id=page.id,
            num_chunks=len(chunks),
            avg_chunk_size=sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0,
        )

        return chunks
