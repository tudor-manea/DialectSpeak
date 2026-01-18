"""
Authenticity Validation Module

Filters outputs that look like stereotyped dialect rather than authentic speech.
Uses n-gram analysis and stereotype detection.
"""

import re
from typing import List
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


# Hiberno-English stereotype patterns to flag
HIBERNO_STEREOTYPES = [
    r"\btop o[f']? the mornin",
    r"\bbegorrah\b",
    r"\bfaith and begorrah\b",
    r"\bto be sure,? to be sure\b",
    r"\bbegorra\b",
    r"\boch\s+aye\b",
    r"\blassie\b",
    r"\bladdie\b",
    r"\bme darlin[g']?\b",
    r"\bwee bit o[f']?\b",
    r"\bpotato famine\b",
    r"\bleprechaun\b",
    r"\bpot o[f']? gold\b",
    r"\bshamrock\b",
    r"\bpaddy\b",
    r"\bmick\b",
]

# Suspicious patterns that suggest fake dialect
SUSPICIOUS_PATTERNS = [
    r"'[a-z]+in\b",  # Excessive apostrophe usage like 'talkin
    r"\b[a-z]+'[a-z]+\b.*\b[a-z]+'[a-z]+\b.*\b[a-z]+'[a-z]+\b",  # Multiple apostrophe words in sequence
    r"\baye\b.*\baye\b.*\baye\b",  # Repetitive "aye"
    r"!{3,}",  # Excessive exclamation marks
]

# Authentic Hiberno-English markers (positive indicators)
AUTHENTIC_MARKERS = [
    r"\b(?:I'm|he's|she's|we're|they're)\s+after\s+\w+ing\b",  # Perfective after
    r"\b(?:do|does)\s+be\s+\w+ing\b",  # Habitual do be
    r"\bamn't\b",  # Authentic contraction
    r"\byouse\b",  # Plural you
    r"^[Ss]ure,?\s+",  # Discourse marker
    r"\b[Ii]t's\s+\w+\s+(?:I|he|she|we|they)\s+(?:am|is|are)\b",  # Cleft
    r",\s+so\s+(?:it|I|he|she)\s+(?:is|am|did)\s*[.!?]?$",  # Tag
]


class AuthenticityValidator:
    """Validates dialect authenticity and filters stereotyped outputs."""

    def __init__(self):
        self.stereotype_patterns = [re.compile(p, re.IGNORECASE) for p in HIBERNO_STEREOTYPES]
        self.suspicious_patterns = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]
        self.authentic_patterns = [re.compile(p, re.IGNORECASE) for p in AUTHENTIC_MARKERS]

    def detect_stereotypes(self, text: str) -> List[str]:
        """Find stereotype patterns in text."""
        found = []
        for pattern in self.stereotype_patterns:
            matches = pattern.findall(text)
            found.extend(matches)
        return found

    def detect_suspicious_patterns(self, text: str) -> List[str]:
        """Find suspicious fake-dialect patterns."""
        found = []
        for pattern in self.suspicious_patterns:
            matches = pattern.findall(text)
            found.extend(matches)
        return found

    def count_authentic_markers(self, text: str) -> int:
        """Count authentic dialect markers present."""
        count = 0
        for pattern in self.authentic_patterns:
            if pattern.search(text):
                count += 1
        return count

    def compute_authenticity_score(self, text: str) -> float:
        """
        Compute authenticity score for dialect text.

        Score is based on:
        - Presence of authentic markers (positive)
        - Absence of stereotypes (positive)
        - Absence of suspicious patterns (positive)

        Returns:
            Score between 0.0 (likely fake) and 1.0 (likely authentic)
        """
        stereotype_count = len(self.detect_stereotypes(text))
        suspicious_count = len(self.detect_suspicious_patterns(text))
        authentic_count = self.count_authentic_markers(text)

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
            dialect: Target dialect (currently only hiberno_english supported)
            threshold: Minimum authenticity score to pass (default 0.5)

        Returns:
            AuthenticityResult with score and validation status
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        stereotypes = self.detect_stereotypes(text)
        suspicious = self.detect_suspicious_patterns(text)
        score = self.compute_authenticity_score(text)

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
