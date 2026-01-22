"""
Authenticity Validation Module

Filters outputs that look like stereotyped dialect rather than authentic speech.
Loads patterns from YAML configuration files.
"""

import re
import yaml
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class AuthenticityResult:
    """Result of authenticity validation."""
    text: str
    dialect: str
    authenticity_score: float
    is_valid: bool
    stereotype_matches: List[str]
    suspicious_patterns: List[str]
    threshold: float


@dataclass
class AuthenticityPatterns:
    """Patterns for authenticity validation."""
    stereotype_patterns: List[re.Pattern]
    suspicious_patterns: List[re.Pattern]
    authentic_markers: List[re.Pattern]


class AuthenticityPatternLoader:
    """Loads and caches authenticity patterns from YAML files."""

    def __init__(self, dialects_dir: str = "data/dialects"):
        self.dialects_dir = Path(dialects_dir)
        self._cache: dict[str, AuthenticityPatterns] = {}

    def _load_dialect(self, dialect: str) -> AuthenticityPatterns:
        """Load authenticity patterns from YAML file."""
        yaml_path = self.dialects_dir / f"{dialect}.yaml"

        if not yaml_path.exists():
            raise ValueError(
                f"No configuration found for dialect: {dialect}. "
                f"Expected file: {yaml_path}"
            )

        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f)

        if "authenticity" not in spec:
            raise ValueError(
                f"Dialect config {yaml_path} missing 'authenticity' section"
            )

        auth = spec["authenticity"]

        def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
            return [re.compile(p, re.IGNORECASE) for p in patterns]

        return AuthenticityPatterns(
            stereotype_patterns=compile_patterns(
                auth.get("stereotype_patterns", [])
            ),
            suspicious_patterns=compile_patterns(
                auth.get("suspicious_patterns", [])
            ),
            authentic_markers=compile_patterns(
                auth.get("authentic_markers", [])
            ),
        )

    def get(self, dialect: str) -> AuthenticityPatterns:
        """Get patterns for a dialect, with caching."""
        if dialect not in self._cache:
            self._cache[dialect] = self._load_dialect(dialect)
        return self._cache[dialect]


# Global loader instance
_loader: Optional[AuthenticityPatternLoader] = None


def _get_loader() -> AuthenticityPatternLoader:
    """Get or create the global loader instance."""
    global _loader
    if _loader is None:
        _loader = AuthenticityPatternLoader()
    return _loader


def set_dialects_dir(dialects_dir: str) -> None:
    """Set a custom dialects directory (useful for testing)."""
    global _loader
    _loader = AuthenticityPatternLoader(dialects_dir)


class AuthenticityValidator:
    """Validates dialect authenticity and filters stereotyped outputs."""

    def __init__(self, dialects_dir: str = "data/dialects"):
        self.loader = AuthenticityPatternLoader(dialects_dir)

    def _get_patterns(self, dialect: str) -> AuthenticityPatterns:
        """Get patterns for a dialect."""
        return self.loader.get(dialect)

    def detect_stereotypes(self, text: str, dialect: str) -> List[str]:
        """Find stereotype patterns in text."""
        patterns = self._get_patterns(dialect)
        found = []
        for pattern in patterns.stereotype_patterns:
            matches = pattern.findall(text)
            found.extend(matches)
        return found

    def detect_suspicious_patterns(self, text: str, dialect: str) -> List[str]:
        """Find suspicious fake-dialect patterns."""
        patterns = self._get_patterns(dialect)
        found = []
        for pattern in patterns.suspicious_patterns:
            matches = pattern.findall(text)
            found.extend(matches)
        return found

    def count_authentic_markers(self, text: str, dialect: str) -> int:
        """Count authentic dialect markers present."""
        patterns = self._get_patterns(dialect)
        count = 0
        for pattern in patterns.authentic_markers:
            if pattern.search(text):
                count += 1
        return count

    def compute_authenticity_score(self, text: str, dialect: str) -> float:
        """
        Compute authenticity score for dialect text.

        Score is based on:
        - Presence of authentic markers (positive)
        - Absence of stereotypes (positive)
        - Absence of suspicious patterns (positive)

        Returns:
            Score between 0.0 (likely fake) and 1.0 (likely authentic)
        """
        stereotype_count = len(self.detect_stereotypes(text, dialect))
        suspicious_count = len(self.detect_suspicious_patterns(text, dialect))
        authentic_count = self.count_authentic_markers(text, dialect)

        # Base score starts at 0.7
        score = 0.7

        # Penalize stereotypes heavily
        score -= stereotype_count * 0.2

        # Penalize suspicious patterns
        score -= suspicious_count * 0.1

        # Reward authentic markers
        score += authentic_count * 0.1

        return max(0.0, min(1.0, score))

    def validate_authenticity(
        self,
        text: str,
        dialect: str = "hiberno_english",
        threshold: float = 0.5
    ) -> AuthenticityResult:
        """
        Validate that text appears authentic rather than stereotyped.

        Args:
            text: Text to validate
            dialect: Target dialect
            threshold: Minimum authenticity score to pass (default 0.5)

        Returns:
            AuthenticityResult with score and validation status
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        stereotypes = self.detect_stereotypes(text, dialect)
        suspicious = self.detect_suspicious_patterns(text, dialect)
        score = self.compute_authenticity_score(text, dialect)

        return AuthenticityResult(
            text=text,
            dialect=dialect,
            authenticity_score=score,
            is_valid=score >= threshold,
            stereotype_matches=stereotypes,
            suspicious_patterns=suspicious,
            threshold=threshold
        )


def validate_authenticity(
    text: str,
    dialect: str = "hiberno_english",
    threshold: float = 0.5
) -> AuthenticityResult:
    """
    Convenience function to validate dialect authenticity.

    Args:
        text: Text to validate
        dialect: Target dialect
        threshold: Minimum score to consider authentic

    Returns:
        AuthenticityResult
    """
    validator = AuthenticityValidator()
    return validator.validate_authenticity(text, dialect, threshold)


def batch_validate_authenticity(
    texts: List[str],
    dialect: str = "hiberno_english",
    threshold: float = 0.5
) -> List[AuthenticityResult]:
    """
    Validate authenticity for multiple texts.

    Args:
        texts: List of texts to validate
        dialect: Target dialect
        threshold: Minimum score to consider authentic

    Returns:
        List of AuthenticityResult objects
    """
    validator = AuthenticityValidator()
    return [
        validator.validate_authenticity(text, dialect, threshold)
        for text in texts
    ]
