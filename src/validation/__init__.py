"""Validation modules for dialect transformation quality."""

from .semantic import (
    SemanticValidationResult,
    compute_semantic_similarity,
    validate_semantic_preservation,
    batch_validate_semantic,
    get_similarity_stats,
)
from .features import (
    FeatureMatch,
    FeatureValidationResult,
    DialectFeatureSpec,
    FeatureValidator,
    validate_hiberno_english,
)
from .authenticity import (
    AuthenticityResult,
    AuthenticityValidator,
    validate_authenticity,
    batch_validate_authenticity,
)
from .pipeline import (
    ValidationStatus,
    PipelineConfig,
    TransformationResult,
    BatchResult,
    ValidationPipeline,
    create_pipeline,
)

__all__ = [
    # Semantic
    "SemanticValidationResult",
    "compute_semantic_similarity",
    "validate_semantic_preservation",
    "batch_validate_semantic",
    "get_similarity_stats",
    # Features
    "FeatureMatch",
    "FeatureValidationResult",
    "DialectFeatureSpec",
    "FeatureValidator",
    "validate_hiberno_english",
    # Authenticity
    "AuthenticityResult",
    "AuthenticityValidator",
    "validate_authenticity",
    "batch_validate_authenticity",
    # Pipeline
    "ValidationStatus",
    "PipelineConfig",
    "TransformationResult",
    "BatchResult",
    "ValidationPipeline",
    "create_pipeline",
]
