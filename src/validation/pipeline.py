"""
Validation Pipeline Module

Integrates semantic, feature, and authenticity validation into a unified pipeline
for validating dialect transformations.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .semantic import (
    SemanticValidationResult,
    validate_semantic_preservation,
    get_similarity_stats,
)
from .features import (
    FeatureValidationResult,
    FeatureValidator,
)
from .authenticity import (
    AuthenticityResult,
    AuthenticityValidator,
)


class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class PipelineConfig:
    """Configuration for the validation pipeline."""
    semantic_threshold: float = 0.80
    min_features: int = 1
    require_high_priority: bool = False
    authenticity_threshold: float = 0.5
    fail_fast: bool = False
    require_all: bool = True


@dataclass
class TransformationResult:
    """Complete validation result for a single transformation."""
    original: str
    transformed: str
    dialect: str
    semantic_result: Optional[SemanticValidationResult] = None
    feature_result: Optional[FeatureValidationResult] = None
    authenticity_result: Optional[AuthenticityResult] = None
    status: ValidationStatus = ValidationStatus.FAILED
    passed_validators: List[str] = field(default_factory=list)
    failed_validators: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def semantic_score(self) -> Optional[float]:
        return self.semantic_result.similarity if self.semantic_result else None

    @property
    def authenticity_score(self) -> Optional[float]:
        return self.authenticity_result.authenticity_score if self.authenticity_result else None

    @property
    def feature_count(self) -> int:
        return len(self.feature_result.features_found) if self.feature_result else 0


@dataclass
class BatchResult:
    """Results for batch validation."""
    results: List[TransformationResult]
    total: int
    passed: int
    failed: int
    partial: int
    pass_rate: float
    semantic_stats: dict = field(default_factory=dict)
    authenticity_mean: float = 0.0
    avg_features: float = 0.0


class ValidationPipeline:
    """
    Integrated validation pipeline for dialect transformations.

    Combines semantic preservation, dialect feature detection, and
    authenticity validation into a single interface.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        dialect_specs_dir: str = "data/dialects"
    ):
        self.config = config or PipelineConfig()
        self.feature_validator = FeatureValidator(dialect_specs_dir)
        self.authenticity_validator = AuthenticityValidator()

    def validate(
        self,
        original: str,
        transformed: str,
        dialect: str = "hiberno_english"
    ) -> TransformationResult:
        """
        Validate a single dialect transformation.

        Args:
            original: Original Standard English text
            transformed: Dialect-transformed text
            dialect: Target dialect identifier

        Returns:
            TransformationResult with all validation details
        """
        result = TransformationResult(
            original=original,
            transformed=transformed,
            dialect=dialect,
            passed_validators=[],
            failed_validators=[]
        )

        # Semantic validation
        result.semantic_result = validate_semantic_preservation(
            original,
            transformed,
            threshold=self.config.semantic_threshold
        )
        self._record_result(result, "semantic", result.semantic_result.is_valid)

        if self.config.fail_fast and not result.semantic_result.is_valid:
            result.status = ValidationStatus.FAILED
            return result

        # Feature validation
        result.feature_result = self.feature_validator.validate_features(
            text=transformed,
            dialect=dialect,
            min_features=self.config.min_features,
            require_high_priority=self.config.require_high_priority
        )
        self._record_result(result, "feature", result.feature_result.is_valid)

        if self.config.fail_fast and not result.feature_result.is_valid:
            result.status = ValidationStatus.FAILED
            return result

        # Authenticity validation
        result.authenticity_result = self.authenticity_validator.validate_authenticity(
            text=transformed,
            dialect=dialect,
            threshold=self.config.authenticity_threshold
        )
        self._record_result(result, "authenticity", result.authenticity_result.is_valid)

        result.status = self._compute_status(result)
        return result

    def _record_result(
        self,
        result: TransformationResult,
        validator_name: str,
        passed: bool
    ) -> None:
        if passed:
            result.passed_validators.append(validator_name)
        else:
            result.failed_validators.append(validator_name)

    def _compute_status(self, result: TransformationResult) -> ValidationStatus:
        if not result.failed_validators:
            return ValidationStatus.PASSED
        if not result.passed_validators:
            return ValidationStatus.FAILED
        if self.config.require_all:
            return ValidationStatus.FAILED
        return ValidationStatus.PARTIAL

    def validate_batch(
        self,
        pairs: List[Tuple[str, str]],
        dialect: str = "hiberno_english"
    ) -> BatchResult:
        """
        Validate multiple transformations.

        Args:
            pairs: List of (original, transformed) tuples
            dialect: Target dialect identifier

        Returns:
            BatchResult with all results and aggregate statistics
        """
        results = [self.validate(orig, trans, dialect) for orig, trans in pairs]
        return self._compute_batch_stats(results)

    def _compute_batch_stats(self, results: List[TransformationResult]) -> BatchResult:
        if not results:
            return BatchResult(
                results=[], total=0, passed=0, failed=0, partial=0, pass_rate=0.0
            )

        total = len(results)
        passed = sum(1 for r in results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in results if r.status == ValidationStatus.FAILED)
        partial = total - passed - failed

        semantic_results = [r.semantic_result for r in results if r.semantic_result]
        semantic_stats = get_similarity_stats(semantic_results) if semantic_results else {}

        auth_scores = [r.authenticity_score for r in results if r.authenticity_score is not None]
        feature_counts = [r.feature_count for r in results]

        return BatchResult(
            results=results,
            total=total,
            passed=passed,
            failed=failed,
            partial=partial,
            pass_rate=passed / total,
            semantic_stats=semantic_stats,
            authenticity_mean=sum(auth_scores) / len(auth_scores) if auth_scores else 0.0,
            avg_features=sum(feature_counts) / total,
        )

    def filter_valid(
        self,
        pairs: List[Tuple[str, str]],
        dialect: str = "hiberno_english"
    ) -> List[Tuple[str, str]]:
        """Filter transformations, keeping only valid ones."""
        batch_result = self.validate_batch(pairs, dialect)
        return [
            (r.original, r.transformed)
            for r in batch_result.results
            if r.is_valid
        ]

    def get_failure_reasons(self, result: TransformationResult) -> List[str]:
        """Get human-readable failure reasons for a transformation."""
        reasons = []

        if result.semantic_result and not result.semantic_result.is_valid:
            score = result.semantic_result.similarity
            threshold = result.semantic_result.threshold
            reasons.append(f"Semantic similarity too low: {score:.3f} < {threshold:.2f}")

        if result.feature_result and not result.feature_result.is_valid:
            found = len(result.feature_result.features_found)
            required = self.config.min_features
            reasons.append(f"Insufficient dialect features: {found} < {required} required")
            if self.config.require_high_priority and result.feature_result.high_priority_found == 0:
                reasons.append("No high-priority features detected")

        if result.authenticity_result and not result.authenticity_result.is_valid:
            score = result.authenticity_result.authenticity_score
            threshold = result.authenticity_result.threshold
            reasons.append(f"Low authenticity score: {score:.3f} < {threshold:.2f}")
            if result.authenticity_result.stereotype_matches:
                stereotypes = ", ".join(result.authenticity_result.stereotype_matches[:3])
                reasons.append(f"Stereotypes detected: {stereotypes}")

        return reasons


def create_pipeline(
    semantic_threshold: float = 0.80,
    min_features: int = 1,
    require_high_priority: bool = False,
    authenticity_threshold: float = 0.5,
    fail_fast: bool = False,
    require_all: bool = True,
    dialect_specs_dir: str = "data/dialects",
) -> ValidationPipeline:
    """Factory function to create a configured validation pipeline."""
    config = PipelineConfig(
        semantic_threshold=semantic_threshold,
        min_features=min_features,
        require_high_priority=require_high_priority,
        authenticity_threshold=authenticity_threshold,
        fail_fast=fail_fast,
        require_all=require_all,
    )
    return ValidationPipeline(config=config, dialect_specs_dir=dialect_specs_dir)
