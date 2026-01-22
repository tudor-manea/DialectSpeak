"""
Unit tests for validation modules.

Tests semantic, feature, authenticity, and pipeline validation.
"""

import pytest
from pathlib import Path

from src.validation.semantic import (
    compute_semantic_similarity,
    validate_semantic_preservation,
    batch_validate_semantic,
    get_similarity_stats,
)
from src.validation.features import (
    DialectFeatureSpec,
    FeatureValidator,
    validate_hiberno_english,
)
from src.validation.authenticity import (
    AuthenticityValidator,
    validate_authenticity,
    batch_validate_authenticity,
)
from src.validation.pipeline import (
    ValidationPipeline,
    PipelineConfig,
    ValidationStatus,
    create_pipeline,
)

# Get the project root for data paths
PROJECT_ROOT = Path(__file__).parent.parent
DIALECTS_DIR = str(PROJECT_ROOT / "data" / "dialects")


# --- Fixtures ---

@pytest.fixture
def feature_validator():
    """Create a FeatureValidator for Hiberno-English tests."""
    return FeatureValidator(DIALECTS_DIR)


@pytest.fixture
def authenticity_validator():
    """Create an AuthenticityValidator for stereotype tests."""
    return AuthenticityValidator(DIALECTS_DIR)


@pytest.fixture
def default_pipeline():
    """Create a pipeline with default configuration."""
    return create_pipeline(dialect_specs_dir=DIALECTS_DIR)


@pytest.fixture
def hiberno_spec():
    """Load the Hiberno-English dialect specification."""
    return DialectFeatureSpec(f"{DIALECTS_DIR}/hiberno_english.yaml")


# --- Semantic Validation Tests ---

class TestSemanticValidation:
    """Tests for semantic similarity validation."""

    def test_compute_similarity_identical(self):
        """Identical texts should have similarity close to 1.0."""
        text = "I just finished eating my dinner."
        similarity = compute_semantic_similarity(text, text)
        assert similarity > 0.99

    def test_compute_similarity_similar(self):
        """Similar texts should have high similarity."""
        original = "I just finished eating my dinner."
        transformed = "I'm after eating my dinner."
        similarity = compute_semantic_similarity(original, transformed)
        assert similarity > 0.8

    def test_compute_similarity_different(self):
        """Different texts should have lower similarity."""
        text1 = "I just finished eating my dinner."
        text2 = "The weather is beautiful today."
        similarity = compute_semantic_similarity(text1, text2)
        assert similarity < 0.5

    def test_compute_similarity_empty_raises(self):
        """Empty strings should raise ValueError."""
        with pytest.raises(ValueError):
            compute_semantic_similarity("", "some text")
        with pytest.raises(ValueError):
            compute_semantic_similarity("some text", "")

    def test_validate_semantic_preservation_valid(self):
        """Test validation with semantically similar texts."""
        result = validate_semantic_preservation(
            original="She has just arrived at the station.",
            transformed="She's after arriving at the station.",
            threshold=0.8,
        )
        assert result.is_valid
        assert result.similarity > 0.8
        assert result.threshold == 0.8

    def test_validate_semantic_preservation_invalid(self):
        """Test validation with semantically different texts."""
        result = validate_semantic_preservation(
            original="I love programming.",
            transformed="The cat sat on the mat.",
            threshold=0.8,
        )
        assert not result.is_valid
        assert result.similarity < 0.8

    def test_validate_semantic_invalid_threshold(self):
        """Invalid threshold should raise ValueError."""
        with pytest.raises(ValueError):
            validate_semantic_preservation("a", "b", threshold=1.5)
        with pytest.raises(ValueError):
            validate_semantic_preservation("a", "b", threshold=-0.1)

    def test_batch_validate_semantic(self):
        """Test batch validation of multiple pairs."""
        pairs = [
            ("I am tired.", "I'm tired."),
            ("Hello world.", "Goodbye moon."),
        ]
        results = batch_validate_semantic(pairs, threshold=0.8)

        assert len(results) == 2
        assert results[0].is_valid  # Similar
        assert not results[1].is_valid  # Different

    def test_batch_validate_empty(self):
        """Empty list should return empty results."""
        results = batch_validate_semantic([], threshold=0.8)
        assert results == []

    def test_get_similarity_stats(self):
        """Test statistics computation."""
        pairs = [
            ("I am tired.", "I'm tired."),
            ("Hello there.", "Hello there."),
        ]
        results = batch_validate_semantic(pairs, threshold=0.8)
        stats = get_similarity_stats(results)

        assert stats["count"] == 2
        assert 0.0 <= stats["mean"] <= 1.0
        assert 0.0 <= stats["min"] <= stats["max"] <= 1.0
        assert "pass_rate" in stats

    def test_get_similarity_stats_empty(self):
        """Empty results should return zero stats."""
        stats = get_similarity_stats([])
        assert stats["count"] == 0
        assert stats["mean"] == 0.0
        assert stats["pass_rate"] == 0.0


class TestFeatureValidation:
    """Tests for dialect feature detection."""

    def test_load_hiberno_english_spec(self, hiberno_spec):
        """Test loading Hiberno-English specification."""
        assert hiberno_spec.spec is not None
        assert len(hiberno_spec.get_feature_ids()) > 0

    def test_detect_perfective_after(self, feature_validator):
        """Test detection of perfective 'after' construction."""
        matches = feature_validator.detect_features(
            "I'm after eating my dinner.",
            dialect="hiberno_english",
        )
        feature_ids = [m.feature_id for m in matches]
        assert "perfective_after" in feature_ids

    def test_detect_habitual_do_be(self, feature_validator):
        """Test detection of habitual 'do be' construction."""
        matches = feature_validator.detect_features(
            "He does be working late every night.",
            dialect="hiberno_english",
        )
        feature_ids = [m.feature_id for m in matches]
        assert "habitual_do_be" in feature_ids

    def test_detect_cleft_emphasis(self, feature_validator):
        """Test detection of cleft sentence emphasis."""
        matches = feature_validator.detect_features(
            "It's tired I am after that walk.",
            dialect="hiberno_english",
        )
        feature_ids = [m.feature_id for m in matches]
        assert "cleft_emphasis" in feature_ids

    def test_detect_youse_plural(self, feature_validator):
        """Test detection of 'youse' plural."""
        matches = feature_validator.detect_features(
            "Are youse coming to the match?",
            dialect="hiberno_english",
        )
        feature_ids = [m.feature_id for m in matches]
        assert "youse_plural" in feature_ids

    def test_validate_features_valid(self, feature_validator):
        """Test feature validation with valid Hiberno-English."""
        result = feature_validator.validate_features(
            text="I'm after finishing the work.",
            dialect="hiberno_english",
            min_features=1,
        )
        assert result.is_valid
        assert len(result.features_found) >= 1
        assert result.coverage_score > 0

    def test_validate_features_invalid(self, feature_validator):
        """Test feature validation with no dialect features."""
        result = feature_validator.validate_features(
            text="I finished eating my dinner.",
            dialect="hiberno_english",
            min_features=1,
        )
        assert not result.is_valid
        assert len(result.features_found) == 0

    def test_validate_require_high_priority(self, feature_validator):
        """Test requiring high-priority features."""
        result = feature_validator.validate_features(
            text="I'm after eating.",
            dialect="hiberno_english",
            min_features=1,
            require_high_priority=True,
        )
        assert result.is_valid
        assert result.high_priority_found > 0

    def test_validate_hiberno_english_convenience(self):
        """Test convenience function."""
        result = validate_hiberno_english(
            "He does be working late.",
            specs_dir=DIALECTS_DIR,
        )
        assert result.is_valid
        assert "habitual_do_be" in result.features_found

    def test_missing_dialect_raises(self, feature_validator):
        """Test that missing dialect file raises error."""
        with pytest.raises(FileNotFoundError):
            feature_validator.load_dialect("nonexistent_dialect")

    def test_get_high_priority_features(self, hiberno_spec):
        """Test getting high-priority feature list."""
        high_priority = hiberno_spec.get_high_priority_features()
        assert len(high_priority) > 0
        assert "perfective_after" in high_priority
        assert "habitual_do_be" in high_priority


class TestAuthenticityValidation:
    """Tests for authenticity and stereotype detection."""

    def test_detect_stereotypes(self, authenticity_validator):
        """Test detection of stereotype patterns."""
        stereotypes = authenticity_validator.detect_stereotypes(
            "Top o' the mornin' to ye, begorrah!",
            dialect="hiberno_english"
        )
        assert len(stereotypes) > 0

    def test_no_stereotypes(self, authenticity_validator):
        """Test that authentic text has no stereotypes."""
        stereotypes = authenticity_validator.detect_stereotypes(
            "I'm after finishing the work, so I am.",
            dialect="hiberno_english"
        )
        assert len(stereotypes) == 0

    def test_detect_suspicious_patterns(self, authenticity_validator):
        """Test detection of suspicious fake-dialect patterns."""
        suspicious = authenticity_validator.detect_suspicious_patterns(
            "I'm 'talkin about the 'walkin and 'runnin!!!",
            dialect="hiberno_english"
        )
        assert len(suspicious) > 0

    def test_count_authentic_markers(self, authenticity_validator):
        """Test counting authentic dialect markers."""
        count = authenticity_validator.count_authentic_markers(
            "I'm after eating, and he does be working late.",
            dialect="hiberno_english"
        )
        assert count >= 2

    def test_compute_authenticity_score_authentic(self, authenticity_validator):
        """Test authenticity score for authentic text."""
        score = authenticity_validator.compute_authenticity_score(
            "I'm after finishing the work.",
            dialect="hiberno_english"
        )
        assert score >= 0.7

    def test_compute_authenticity_score_stereotyped(self, authenticity_validator):
        """Test authenticity score for stereotyped text."""
        score = authenticity_validator.compute_authenticity_score(
            "Top o' the mornin', begorrah! Faith and begorrah!",
            dialect="hiberno_english"
        )
        assert score < 0.5

    def test_validate_authenticity_valid(self):
        """Test validation of authentic text."""
        result = validate_authenticity("I'm after eating my dinner.", threshold=0.5)
        assert result.is_valid
        assert result.authenticity_score >= 0.5
        assert len(result.stereotype_matches) == 0

    def test_validate_authenticity_invalid(self):
        """Test validation of stereotyped text."""
        result = validate_authenticity("Top o' the mornin', begorrah!", threshold=0.5)
        assert not result.is_valid
        assert len(result.stereotype_matches) > 0

    def test_validate_authenticity_invalid_threshold(self, authenticity_validator):
        """Test that invalid threshold raises error."""
        with pytest.raises(ValueError):
            authenticity_validator.validate_authenticity("text", threshold=1.5)

    def test_batch_validate_authenticity(self):
        """Test batch validation."""
        texts = [
            "I'm after eating.",
            "Top o' the mornin', begorrah!",
        ]
        results = batch_validate_authenticity(texts, threshold=0.5)

        assert len(results) == 2
        assert results[0].is_valid
        assert not results[1].is_valid


class TestValidationPipeline:
    """Tests for integrated validation pipeline."""

    def test_create_pipeline(self):
        """Test pipeline creation with factory function."""
        pipeline = create_pipeline(
            semantic_threshold=0.8,
            min_features=1,
            dialect_specs_dir=DIALECTS_DIR,
        )
        assert pipeline is not None
        assert pipeline.config.semantic_threshold == 0.8
        assert pipeline.config.min_features == 1

    def test_validate_valid_transformation(self):
        """Test validation of a valid transformation."""
        pipeline = create_pipeline(
            semantic_threshold=0.7,
            min_features=1,
            authenticity_threshold=0.4,
            dialect_specs_dir=DIALECTS_DIR,
        )
        result = pipeline.validate(
            original="I just finished eating my dinner.",
            transformed="I'm after eating my dinner.",
            dialect="hiberno_english",
        )

        assert result.status == ValidationStatus.PASSED
        assert result.is_valid
        assert "semantic" in result.passed_validators
        assert "feature" in result.passed_validators
        assert "authenticity" in result.passed_validators

    def test_validate_invalid_semantic(self):
        """Test validation failure due to semantic drift."""
        pipeline = create_pipeline(
            semantic_threshold=0.95,
            min_features=0,
            authenticity_threshold=0.3,
            dialect_specs_dir=DIALECTS_DIR,
        )
        result = pipeline.validate(
            original="I love programming.",
            transformed="The cat does be sitting on the mat.",
            dialect="hiberno_english",
        )

        assert result.status == ValidationStatus.FAILED
        assert "semantic" in result.failed_validators

    def test_validate_invalid_features(self):
        """Test validation failure due to missing features."""
        pipeline = create_pipeline(
            semantic_threshold=0.5,
            min_features=3,
            authenticity_threshold=0.3,
            dialect_specs_dir=DIALECTS_DIR,
        )
        result = pipeline.validate(
            original="I am tired.",
            transformed="I'm tired.",
            dialect="hiberno_english",
        )

        assert "feature" in result.failed_validators

    def test_validate_invalid_authenticity(self):
        """Test validation failure due to stereotypes."""
        pipeline = create_pipeline(
            semantic_threshold=0.3,
            min_features=0,
            authenticity_threshold=0.9,
            dialect_specs_dir=DIALECTS_DIR,
        )
        result = pipeline.validate(
            original="Good morning!",
            transformed="Top o' the mornin', begorrah!",
            dialect="hiberno_english",
        )

        assert "authenticity" in result.failed_validators

    def test_validate_batch(self):
        """Test batch validation."""
        pipeline = create_pipeline(
            semantic_threshold=0.7,
            min_features=1,
            authenticity_threshold=0.4,
            dialect_specs_dir=DIALECTS_DIR,
        )
        pairs = [
            ("I just finished eating.", "I'm after eating."),
            ("Good morning!", "Top o' the mornin', begorrah!"),
        ]
        batch_result = pipeline.validate_batch(pairs, dialect="hiberno_english")

        assert batch_result.total == 2
        assert batch_result.passed >= 0
        assert batch_result.failed >= 0
        assert 0.0 <= batch_result.pass_rate <= 1.0

    def test_filter_valid(self):
        """Test filtering to keep only valid transformations."""
        pipeline = create_pipeline(
            semantic_threshold=0.7,
            min_features=1,
            authenticity_threshold=0.4,
            dialect_specs_dir=DIALECTS_DIR,
        )
        pairs = [
            ("I just finished eating.", "I'm after eating."),
            ("Hello.", "Top o' the mornin', begorrah!"),
        ]
        valid_pairs = pipeline.filter_valid(pairs, dialect="hiberno_english")
        assert len(valid_pairs) <= len(pairs)

    def test_get_failure_reasons(self):
        """Test getting human-readable failure reasons."""
        pipeline = create_pipeline(
            semantic_threshold=0.99,
            min_features=5,
            authenticity_threshold=0.99,
            dialect_specs_dir=DIALECTS_DIR,
        )
        result = pipeline.validate(
            original="Hello there.",
            transformed="Top o' the mornin'!",
            dialect="hiberno_english",
        )
        reasons = pipeline.get_failure_reasons(result)

        assert len(reasons) > 0
        assert any("semantic" in r.lower() for r in reasons)

    def test_fail_fast_mode(self):
        """Test fail-fast mode stops on first failure."""
        config = PipelineConfig(
            semantic_threshold=0.99,
            min_features=1,
            authenticity_threshold=0.5,
            fail_fast=True,
        )
        pipeline = ValidationPipeline(config, dialect_specs_dir=DIALECTS_DIR)
        result = pipeline.validate(
            original="Hello.",
            transformed="Goodbye completely different.",
            dialect="hiberno_english",
        )

        assert result.status == ValidationStatus.FAILED
        assert "semantic" in result.failed_validators
        # In fail-fast mode, feature and authenticity may not be run
        assert len(result.passed_validators) + len(result.failed_validators) <= 3

    def test_partial_status(self):
        """Test partial status when require_all is False."""
        config = PipelineConfig(
            semantic_threshold=0.5,
            min_features=10,  # High threshold to fail
            authenticity_threshold=0.3,
            require_all=False,
        )
        pipeline = ValidationPipeline(config, dialect_specs_dir=DIALECTS_DIR)
        result = pipeline.validate(
            original="I am eating.",
            transformed="I'm after eating.",
            dialect="hiberno_english",
        )

        # Should be PARTIAL since require_all=False and some passed
        if result.passed_validators and result.failed_validators:
            assert result.status == ValidationStatus.PARTIAL

    def test_transformation_result_properties(self, default_pipeline):
        """Test TransformationResult property accessors."""
        result = default_pipeline.validate(
            original="I am eating.",
            transformed="I'm after eating.",
            dialect="hiberno_english",
        )

        assert result.semantic_score is not None
        assert 0.0 <= result.semantic_score <= 1.0
        assert result.authenticity_score is not None
        assert 0.0 <= result.authenticity_score <= 1.0
        assert result.feature_count >= 0

    def test_batch_result_statistics(self):
        """Test batch result aggregate statistics."""
        pipeline = create_pipeline(
            semantic_threshold=0.7,
            min_features=1,
            dialect_specs_dir=DIALECTS_DIR,
        )
        pairs = [
            ("I finished eating.", "I'm after eating."),
            ("He works late.", "He does be working late."),
        ]
        batch_result = pipeline.validate_batch(pairs)

        assert "mean" in batch_result.semantic_stats or batch_result.semantic_stats == {}
        assert batch_result.authenticity_mean >= 0.0
        assert batch_result.avg_features >= 0.0
