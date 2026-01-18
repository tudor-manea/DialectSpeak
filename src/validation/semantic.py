"""
Semantic Validation Module

Validates that dialect transformations preserve the original meaning
using sentence embeddings and cosine similarity.
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

# Lazy loading for heavy imports
_model = None


def _get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def _cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot_product = np.dot(embedding_a, embedding_b)
    norm_product = np.linalg.norm(embedding_a) * np.linalg.norm(embedding_b)
    if norm_product == 0:
        return 0.0
    return float(dot_product / norm_product)


@dataclass
class SemanticValidationResult:
    """Result of semantic validation."""
    original: str
    transformed: str
    similarity: float
    is_valid: bool
    threshold: float


def compute_semantic_similarity(original: str, transformed: str) -> float:
    """
    Compute semantic similarity between original and transformed text.

    Args:
        original: The original Standard English text
        transformed: The dialect-transformed text

    Returns:
        Cosine similarity score between 0 and 1

    Raises:
        ValueError: If either input is empty
    """
    if not original or not transformed:
        raise ValueError("Both original and transformed text must be non-empty strings")

    model = _get_model()
    embeddings = model.encode([original, transformed], convert_to_numpy=True)

    return _cosine_similarity(embeddings[0], embeddings[1])


def validate_semantic_preservation(
    original: str,
    transformed: str,
    threshold: float = 0.85
) -> SemanticValidationResult:
    """
    Validate that the transformed text preserves the original meaning.

    Args:
        original: The original Standard English text
        transformed: The dialect-transformed text
        threshold: Minimum similarity score to consider valid (default 0.85)

    Returns:
        SemanticValidationResult with similarity score and validity

    Raises:
        ValueError: If threshold is not between 0 and 1
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    similarity = compute_semantic_similarity(original, transformed)

    return SemanticValidationResult(
        original=original,
        transformed=transformed,
        similarity=similarity,
        is_valid=similarity >= threshold,
        threshold=threshold
    )


def batch_validate_semantic(
    pairs: List[Tuple[str, str]],
    threshold: float = 0.85
) -> List[SemanticValidationResult]:
    """
    Batch validate multiple original-transformed pairs.

    Args:
        pairs: List of (original, transformed) tuples
        threshold: Minimum similarity score to consider valid

    Returns:
        List of SemanticValidationResult objects

    Raises:
        ValueError: If threshold is not between 0 and 1, or if pairs is empty
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    if not pairs:
        return []

    model = _get_model()

    # Prepare all texts for batch encoding
    originals = [p[0] for p in pairs]
    transformed = [p[1] for p in pairs]

    # Batch encode
    orig_embeddings = model.encode(originals, convert_to_numpy=True, show_progress_bar=len(pairs) > 10)
    trans_embeddings = model.encode(transformed, convert_to_numpy=True, show_progress_bar=len(pairs) > 10)

    results = []
    for i, (orig, trans) in enumerate(pairs):
        similarity = _cosine_similarity(orig_embeddings[i], trans_embeddings[i])

        results.append(SemanticValidationResult(
            original=orig,
            transformed=trans,
            similarity=similarity,
            is_valid=similarity >= threshold,
            threshold=threshold
        ))

    return results


def get_similarity_stats(results: List[SemanticValidationResult]) -> dict:
    """
    Compute statistics over a list of validation results.

    Args:
        results: List of SemanticValidationResult objects

    Returns:
        Dictionary with mean, std, min, max, and pass_rate
    """
    if not results:
        return {
            'count': 0,
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'pass_rate': 0.0,
            'valid_count': 0,
            'invalid_count': 0
        }

    similarities = [r.similarity for r in results]
    valid_count = sum(1 for r in results if r.is_valid)

    return {
        'count': len(results),
        'mean': float(np.mean(similarities)),
        'std': float(np.std(similarities)),
        'min': float(np.min(similarities)),
        'max': float(np.max(similarities)),
        'pass_rate': valid_count / len(results),
        'valid_count': valid_count,
        'invalid_count': len(results) - valid_count
    }
