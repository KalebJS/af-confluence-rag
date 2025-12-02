"""Metadata enrichment functionality for document chunks."""

import structlog

from src.models.page import Page, DocumentChunk

log = structlog.stdlib.get_logger()


class MetadataEnricher:
    """Enriches document chunks with contextual metadata from source pages.
    
    This class is responsible for:
    1. Adding page metadata to chunks (title, URL, author, dates, etc.)
    2. Generating unique chunk_ids in format {page_id}_{chunk_index}
    """

    def enrich_chunk(
        self,
        page: Page,
        chunk_text: str,
        chunk_index: int,
    ) -> DocumentChunk:
        """Enrich a text chunk with metadata from its source page.
        
        This method creates a DocumentChunk with:
        - A unique chunk_id in format {page_id}_{chunk_index}
        - The chunk text content
        - Complete metadata from the source page
        
        Args:
            page: Source Confluence page
            chunk_text: Text content of the chunk
            chunk_index: Zero-based index of this chunk within the page
            
        Returns:
            DocumentChunk with enriched metadata
        """
        chunk_id = f"{page.id}_{chunk_index}"
        
        metadata = {
            "page_title": page.title,
            "page_url": str(page.url),
            "author": page.author,
            "created_date": page.created_date.isoformat(),
            "modified_date": page.modified_date.isoformat(),
            "space_key": page.space_key,
            "version": page.version,
        }
        
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            page_id=page.id,
            content=chunk_text,
            metadata=metadata,
            chunk_index=chunk_index,
        )
        
        log.debug(
            "Chunk enriched with metadata",
            chunk_id=chunk_id,
            page_id=page.id,
            chunk_index=chunk_index,
        )
        
        return chunk

    def enrich_chunks(
        self,
        page: Page,
        chunk_texts: list[str],
    ) -> list[DocumentChunk]:
        """Enrich multiple text chunks with metadata from their source page.
        
        This is a convenience method for enriching multiple chunks at once.
        
        Args:
            page: Source Confluence page
            chunk_texts: List of text chunks to enrich
            
        Returns:
            List of DocumentChunk objects with enriched metadata
        """
        chunks = []
        for idx, chunk_text in enumerate(chunk_texts):
            chunk = self.enrich_chunk(page, chunk_text, idx)
            chunks.append(chunk)
        
        log.info(
            "Multiple chunks enriched with metadata",
            page_id=page.id,
            num_chunks=len(chunks),
        )
        
        return chunks
