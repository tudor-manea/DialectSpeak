"""
Feature Validation Module

Validates that dialect transformations include authentic linguistic features
specific to the target dialect.
"""

import re
import yaml
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FeatureMatch:
    """A detected feature in text."""
    feature_id: str
    feature_name: str
    matched_text: str
    start_pos: int
    end_pos: int
    priority: str


@dataclass
class FeatureValidationResult:
    """Result of feature validation."""
    text: str
    dialect: str
    matches: List[FeatureMatch]
    features_found: Set[str]
    is_valid: bool
    coverage_score: float
    high_priority_found: int
    total_high_priority: int


class DialectFeatureSpec:
    """Loads and manages dialect feature specifications."""

    def __init__(self, spec_path: Optional[str] = None):
        self.spec_path = spec_path
        self.spec: Optional[dict] = None
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}

        if spec_path:
            self.load_spec(spec_path)

    def _get_features(self) -> Dict:
        """Return the features dict, or empty dict if not loaded."""
        if not self.spec or 'features' not in self.spec:
            return {}
        return self.spec['features']

    def load_spec(self, spec_path: str) -> None:
        """Load dialect specification from YAML file."""
        with open(spec_path, 'r') as f:
            self.spec = yaml.safe_load(f)
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile all regex patterns for efficiency."""
        for feature_id, feature_data in self._get_features().items():
            patterns = feature_data.get('patterns', [])
            self.compiled_patterns[feature_id] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def get_feature_ids(self) -> List[str]:
        """Get all feature IDs in the specification."""
        return list(self._get_features().keys())

    def get_high_priority_features(self) -> List[str]:
        """Get feature IDs marked as high priority."""
        return [
            fid for fid, fdata in self._get_features().items()
            if fdata.get('priority') == 'high'
        ]

    def get_feature_name(self, feature_id: str) -> str:
        """Get the human-readable name for a feature."""
        return self._get_features().get(feature_id, {}).get('name', feature_id)

    def get_feature_priority(self, feature_id: str) -> str:
        """Get the priority level for a feature."""
        return self._get_features().get(feature_id, {}).get('priority', 'medium')

    def get_avoid_patterns(self) -> List[str]:
        """Get patterns to avoid (stereotypes)."""
        if not self.spec:
            return []
        return self.spec.get('avoid_features', [])


class FeatureValidator:
    """Validates dialect features in text."""

    def __init__(self, dialect_specs_dir: str = "data/dialects"):
        self.specs_dir = Path(dialect_specs_dir)
        self.loaded_specs: Dict[str, DialectFeatureSpec] = {}

    def load_dialect(self, dialect: str) -> DialectFeatureSpec:
        """Load dialect specification if not already loaded."""
        if dialect not in self.loaded_specs:
            spec_path = self.specs_dir / f"{dialect}.yaml"
            if not spec_path.exists():
                raise FileNotFoundError(f"No specification found for dialect: {dialect}")
            self.loaded_specs[dialect] = DialectFeatureSpec(str(spec_path))

        return self.loaded_specs[dialect]

    def detect_features(self, text: str, dialect: str) -> List[FeatureMatch]:
        """Detect all dialect features present in text."""
        spec = self.load_dialect(dialect)
        matches = []

        for feature_id, patterns in spec.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matches.append(FeatureMatch(
                        feature_id=feature_id,
                        feature_name=spec.get_feature_name(feature_id),
                        matched_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        priority=spec.get_feature_priority(feature_id)
                    ))

        return matches

    def validate_features(
        self,
        text: str,
        dialect: str,
        min_features: int = 1,
        require_high_priority: bool = False
    ) -> FeatureValidationResult:
        """
        Validate that text contains required dialect features.

        Args:
            text: Text to validate
            dialect: Target dialect identifier
            min_features: Minimum number of distinct features required
            require_high_priority: If True, at least one high-priority feature required

        Returns:
            FeatureValidationResult with validation details
        """
        spec = self.load_dialect(dialect)
        matches = self.detect_features(text, dialect)

        features_found = set(m.feature_id for m in matches)
        all_feature_ids = spec.get_feature_ids()
        high_priority_ids = spec.get_high_priority_features()

        high_priority_found = len(features_found & set(high_priority_ids))
        total_high_priority = len(high_priority_ids)

        total_features = len(all_feature_ids)
        coverage = len(features_found) / total_features if total_features > 0 else 0.0

        is_valid = len(features_found) >= min_features
        if require_high_priority and high_priority_found == 0:
            is_valid = False

        return FeatureValidationResult(
            text=text,
            dialect=dialect,
            matches=matches,
            features_found=features_found,
            is_valid=is_valid,
            coverage_score=coverage,
            high_priority_found=high_priority_found,
            total_high_priority=total_high_priority
        )

    def contains_stereotypes(self, text: str, dialect: str) -> List[str]:
        """Check if text contains stereotyped/avoided patterns."""
        spec = self.load_dialect(dialect)
        avoid_patterns = spec.get_avoid_patterns()

        found_stereotypes = []
        text_lower = text.lower()

        for pattern in avoid_patterns:
            if pattern.lower() in text_lower:
                found_stereotypes.append(pattern)

        return found_stereotypes


def validate_hiberno_english(
    text: str,
    min_features: int = 1,
    require_high_priority: bool = False,
    specs_dir: str = "data/dialects"
) -> FeatureValidationResult:
    """
    Convenience function to validate Hiberno-English features.

    Args:
        text: Text to validate
        min_features: Minimum distinct features required
        require_high_priority: Require at least one high-priority feature
        specs_dir: Directory containing dialect specifications

    Returns:
        FeatureValidationResult
    """
    validator = FeatureValidator(specs_dir)
    return validator.validate_features(
        text=text,
        dialect="hiberno_english",
        min_features=min_features,
        require_high_priority=require_high_priority
    )
