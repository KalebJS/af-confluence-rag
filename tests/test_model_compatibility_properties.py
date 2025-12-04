"""Property-based tests for model name compatibility.

Feature: langchain-abstraction-refactor
Property 5: Model Name Compatibility
"""

import structlog
from hypothesis import given, settings
from hypothesis import strategies as st

from src.providers import get_embeddings

log = structlog.stdlib.get_logger()


# Common sentence-transformers model names that should work
KNOWN_WORKING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
]

# Expected dimensions for known models
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "sentence-transformers/paraphrase-MiniLM-L6-v2": 384,
}


@settings(deadline=None, max_examples=2)  # Model loading is slow, test just 2 models
@given(st.sampled_from(["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2"]))
def test_property_5_model_name_compatibility(model_name: str):
    """Property 5: Model Name Compatibility.

    *For any* sentence-transformers model name that worked in the previous
    implementation, the new HuggingFaceEmbeddings SHALL successfully initialize
    with that model name and produce embeddings of the same dimensionality.

    **Feature: langchain-abstraction-refactor, Property 5: Model Name Compatibility**
    **Validates: Requirements 1.3, 9.3**
    """
    log.info("test_property_5_model_name_compatibility", model_name=model_name)

    # Get embeddings instance
    embeddings = get_embeddings(model_name)

    # Verify it can generate embeddings
    test_text = "This is a test sentence for embedding generation."
    embedding = embeddings.embed_query(test_text)

    # Verify the embedding is a list of floats
    assert isinstance(embedding, list), f"Embedding should be a list, got {type(embedding)}"
    assert len(embedding) > 0, "Embedding should not be empty"
    assert all(
        isinstance(x, float) for x in embedding
    ), "Embedding should contain only floats"

    # Verify the dimensionality matches expected value
    expected_dim = MODEL_DIMENSIONS.get(model_name)
    if expected_dim:
        assert len(embedding) == expected_dim, (
            f"Model {model_name} should produce embeddings of dimension {expected_dim}, "
            f"got {len(embedding)}"
        )

    log.info(
        "model_name_compatibility_verified",
        model_name=model_name,
        embedding_dimension=len(embedding),
        expected_dimension=expected_dim,
    )


@settings(deadline=None, max_examples=2)  # Model loading is slow
@given(st.sampled_from(["all-MiniLM-L6-v2"]))
def test_property_5_batch_embedding_consistency(model_name: str):
    """Property 5: Batch Embedding Consistency.

    *For any* sentence-transformers model, embeddings generated via embed_query
    and embed_documents should have the same dimensionality.

    **Feature: langchain-abstraction-refactor, Property 5: Model Name Compatibility**
    **Validates: Requirements 1.3, 9.3**
    """
    log.info("test_property_5_batch_embedding_consistency", model_name=model_name)

    embeddings = get_embeddings(model_name)

    # Generate single embedding
    single_text = "Test sentence"
    single_embedding = embeddings.embed_query(single_text)

    # Generate batch embeddings
    batch_texts = ["Test sentence 1", "Test sentence 2", "Test sentence 3"]
    batch_embeddings = embeddings.embed_documents(batch_texts)

    # Verify batch embeddings
    assert isinstance(batch_embeddings, list), "Batch embeddings should be a list"
    assert len(batch_embeddings) == len(
        batch_texts
    ), "Should have one embedding per input text"

    # Verify all embeddings have the same dimension
    for i, batch_embedding in enumerate(batch_embeddings):
        assert isinstance(
            batch_embedding, list
        ), f"Batch embedding {i} should be a list"
        assert len(batch_embedding) == len(single_embedding), (
            f"Batch embedding {i} dimension ({len(batch_embedding)}) should match "
            f"single embedding dimension ({len(single_embedding)})"
        )

    log.info(
        "batch_embedding_consistency_verified",
        model_name=model_name,
        single_dimension=len(single_embedding),
        batch_count=len(batch_embeddings),
    )


@settings(deadline=None, max_examples=5)  # Just test a few examples
@given(
    st.sampled_from(["all-MiniLM-L6-v2"]),
    st.text(min_size=1, max_size=100),
)
def test_property_5_embedding_stability(model_name: str, text: str):
    """Property 5: Embedding Stability.

    *For any* sentence-transformers model and text input, generating embeddings
    multiple times should produce consistent results.

    **Feature: langchain-abstraction-refactor, Property 5: Model Name Compatibility**
    **Validates: Requirements 1.3, 9.3**
    """
    log.info("test_property_5_embedding_stability", model_name=model_name, text_length=len(text))

    embeddings = get_embeddings(model_name)

    # Generate embedding twice
    embedding1 = embeddings.embed_query(text)
    embedding2 = embeddings.embed_query(text)

    # Verify they are identical (or very close due to floating point)
    assert len(embedding1) == len(embedding2), "Embeddings should have same dimension"

    # Check that embeddings are very close (allowing for tiny floating point differences)
    for i, (val1, val2) in enumerate(zip(embedding1, embedding2)):
        assert abs(val1 - val2) < 1e-6, (
            f"Embedding values at index {i} should be identical or very close: "
            f"{val1} vs {val2}"
        )

    log.info(
        "embedding_stability_verified",
        model_name=model_name,
        embedding_dimension=len(embedding1),
    )
