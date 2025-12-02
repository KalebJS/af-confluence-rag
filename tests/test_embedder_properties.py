"""Property-based tests for EmbeddingGenerator.

**Feature: confluence-rag-system**
"""

from hypothesis import given, strategies as st, settings
import numpy as np
import pytest

from src.processing.embedder import EmbeddingGenerator


# Fixture to reuse the embedding model across tests
@pytest.fixture(scope="module")
def embedder():
    """Create a single EmbeddingGenerator instance for all tests."""
    return EmbeddingGenerator(model_name="all-MiniLM-L6-v2")


# Strategy for generating text content
text_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cc")),
    min_size=1,
    max_size=1000,
)

# Strategy for generating lists of texts
text_list_strategy = st.lists(
    text_strategy,
    min_size=1,
    max_size=50,
)


@given(texts=text_list_strategy)
@settings(deadline=None)  # Disable deadline since model inference can be slow
def test_property_7_embedding_generation_completeness(embedder, texts: list[str]):
    """Property 7: Embedding generation completeness
    
    *For any* set of document chunks, the number of generated embeddings 
    should equal the number of chunks.
    
    **Validates: Requirements 2.3**
    **Feature: confluence-rag-system, Property 7: Embedding generation completeness**
    """
    # Generate embeddings
    embeddings = embedder.generate_batch_embeddings(texts)
    
    # Check that we got the same number of embeddings as texts
    assert len(embeddings) == len(texts), (
        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
    )
    
    # Check that each embedding is a numpy array
    for idx, embedding in enumerate(embeddings):
        assert isinstance(embedding, np.ndarray), (
            f"Embedding {idx} is not a numpy array, got {type(embedding)}"
        )
        
        # Check that embedding has the correct dimension
        expected_dim = embedder.get_embedding_dimension()
        assert embedding.shape == (expected_dim,), (
            f"Embedding {idx} has shape {embedding.shape}, expected ({expected_dim},)"
        )


@given(
    text1=text_strategy,
    text2=text_strategy,
)
@settings(deadline=None)  # Disable deadline since model inference can be slow
def test_property_18_embedding_model_consistency(embedder, text1: str, text2: str):
    """Property 18: Embedding model consistency
    
    *For any* query embedding and document embedding, they should have the 
    same dimensionality (indicating the same model was used).
    
    **Validates: Requirements 5.1**
    **Feature: confluence-rag-system, Property 18: Embedding model consistency**
    """
    # Generate embeddings for both texts
    embedding1 = embedder.generate_embedding(text1)
    embedding2 = embedder.generate_embedding(text2)
    
    # Check that both embeddings have the same dimensionality
    assert embedding1.shape == embedding2.shape, (
        f"Embeddings have different shapes: {embedding1.shape} vs {embedding2.shape}"
    )
    
    # Check that the dimensionality matches the model's expected dimension
    expected_dim = embedder.get_embedding_dimension()
    assert embedding1.shape == (expected_dim,), (
        f"Embedding 1 has shape {embedding1.shape}, expected ({expected_dim},)"
    )
    assert embedding2.shape == (expected_dim,), (
        f"Embedding 2 has shape {embedding2.shape}, expected ({expected_dim},)"
    )
    
    # Check that embeddings are numpy arrays
    assert isinstance(embedding1, np.ndarray), (
        f"Embedding 1 is not a numpy array, got {type(embedding1)}"
    )
    assert isinstance(embedding2, np.ndarray), (
        f"Embedding 2 is not a numpy array, got {type(embedding2)}"
    )


@given(text=text_strategy)
@settings(deadline=None)  # Disable deadline since model inference can be slow
def test_single_embedding_generation(embedder, text: str):
    """Test that single embedding generation works correctly."""
    # Generate embedding
    embedding = embedder.generate_embedding(text)
    
    # Check that embedding is a numpy array
    assert isinstance(embedding, np.ndarray)
    
    # Check that embedding has the correct dimension
    expected_dim = embedder.get_embedding_dimension()
    assert embedding.shape == (expected_dim,)
    
    # Check that embedding is not all zeros (unless text is empty)
    if text.strip():
        assert not np.allclose(embedding, 0), "Non-empty text should not produce zero embedding"


def test_empty_text_handling(embedder):
    """Test that empty text is handled gracefully."""
    
    # Test empty string
    embedding = embedder.generate_embedding("")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedder.get_embedding_dimension(),)
    
    # Test whitespace-only string
    embedding = embedder.generate_embedding("   ")
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (embedder.get_embedding_dimension(),)


def test_batch_with_empty_texts(embedder):
    """Test that batch embedding handles empty texts correctly."""
    
    texts = ["hello world", "", "   ", "another text"]
    embeddings = embedder.generate_batch_embeddings(texts)
    
    # Check that we got the correct number of embeddings
    assert len(embeddings) == len(texts)
    
    # Check that all embeddings have the correct shape
    expected_dim = embedder.get_embedding_dimension()
    for embedding in embeddings:
        assert embedding.shape == (expected_dim,)
