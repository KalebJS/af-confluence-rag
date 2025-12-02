"""Embedding generation functionality using sentence-transformers."""

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

log = structlog.stdlib.get_logger()


class EmbeddingGenerator:
    """Generates vector embeddings from text using sentence-transformers.
    
    This class wraps the sentence-transformers library to provide a consistent
    interface for generating embeddings from text chunks.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is "all-MiniLM-L6-v2" which provides a good
                       balance between performance and resource requirements.
        """
        self.model_name: str = model_name
        
        log.info("Loading embedding model", model_name=model_name)
        
        # Load the sentence-transformers model
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        
        log.info(
            "Embedding model loaded successfully",
            model_name=model_name,
            embedding_dimension=self.get_embedding_dimension(),
        )

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a vector embedding for a single text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Numpy array containing the embedding vector
        """
        if not text or not text.strip():
            log.warning("Empty text provided for embedding generation")
            # Return zero vector for empty text
            return np.zeros(self.get_embedding_dimension(), dtype=np.float32)
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        log.debug(
            "Generated embedding",
            text_length=len(text),
            embedding_shape=embedding.shape,
        )
        
        return embedding

    def generate_batch_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts efficiently.
        
        This method uses batch processing for better performance when
        generating embeddings for multiple texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of numpy arrays containing the embedding vectors
        """
        if not texts:
            log.warning("Empty text list provided for batch embedding generation")
            return []
        
        log.info("Generating batch embeddings", batch_size=len(texts))
        
        # Filter out empty texts and track their indices
        non_empty_texts = []
        non_empty_indices = []
        for idx, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(idx)
        
        # Generate embeddings for non-empty texts
        if non_empty_texts:
            embeddings_array = self.model.encode(
                non_empty_texts,
                convert_to_numpy=True,
                show_progress_bar=len(non_empty_texts) > 10,
            )
        else:
            embeddings_array = np.array([])
        
        # Create result list with zero vectors for empty texts
        result: list[np.ndarray] = []
        embedding_dim = self.get_embedding_dimension()
        non_empty_idx = 0
        
        for idx in range(len(texts)):
            if idx in non_empty_indices:
                result.append(embeddings_array[non_empty_idx])
                non_empty_idx += 1
            else:
                # Empty text gets zero vector
                result.append(np.zeros(embedding_dim, dtype=np.float32))
        
        log.info(
            "Batch embeddings generated successfully",
            total_texts=len(texts),
            non_empty_texts=len(non_empty_texts),
        )
        
        return result

    def get_embedding_dimension(self) -> int:
        """Get the dimensionality of the embedding vectors.
        
        Returns:
            Integer representing the embedding dimension
        """
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError("Model does not provide embedding dimension")
        return dim
