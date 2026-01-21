"""
Post-processing module for cleaning LLM transformation outputs.

Removes common padding patterns and unwanted commentary.
"""

import re
from typing import Optional


# Patterns that indicate padding/commentary at the start
START_PADDING_PATTERNS = [
    r"^['\"]?(?:'Tis|Sure|Ah|Well|Now|So|Right|Ok|Okay),?\s+(?:a\s+)?(?:grand|good|fine|great|lovely)\s+(?:question|one).*?[!.]\s*",
    r"^(?:Let's see|Let me see|Right then|Now then|Here we go|Here goes).*?[!.]\s*",
    r"^(?:This is|That's).*?(?:question|text|sentence).*?[!.]\s*",
    r"^(?:Ah|Oh|Aye|Sure),?\s+",
    r"^(?:Here's|Here is)\s+(?:the|my)\s+(?:transformation|answer|response).*?[:.]\s*",
    r"^Transformed(?:\s+text)?:\s*",
]

# Patterns that indicate the LLM solved the problem instead of transforming
SOLUTION_INDICATORS = [
    r"=\s*\d+(?:\.\d+)?",  # Contains "= 123" (mathematical answer)
    r"(?:the answer is|answer:|result:|total:|sum:)\s*\d+",
    r"(?:that's|which is|equals?)\s+\d+\s+(?:sheep|dollars|miles|hours|days|years|pounds|euros|items|things)",
    r"(?:so|therefore|thus),?\s+(?:altogether|in total|combined)",
]

# Patterns for trailing commentary
END_PADDING_PATTERNS = [
    r"\s*(?:So there you go|There you are|Hope that helps).*?[!.]?\s*$",
    r"\s*(?:Is that|Does that|Would that).*?\?\s*$",
    r"\s*(?:Let me know if).*$",
]


def clean_transformation(text: str, original: str) -> str:
    """
    Clean LLM transformation output by removing padding and commentary.

    Args:
        text: Raw LLM output
        original: Original text (for reference)

    Returns:
        Cleaned transformation
    """
    if not text:
        return text

    cleaned = text.strip()

    # Remove start padding
    for pattern in START_PADDING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Remove end padding
    for pattern in END_PADDING_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # Remove quotes if the LLM wrapped the output in quotes
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1]

    return cleaned.strip()


def detect_solution_instead_of_transform(
    transformed: str,
    original: str,
) -> bool:
    """
    Detect if the LLM solved the problem instead of transforming it.

    This happens when the LLM sees a math problem and provides the answer
    rather than just transforming the dialect.

    Args:
        transformed: The transformed text
        original: The original text

    Returns:
        True if the output appears to be a solution rather than transformation
    """
    # Check if original is a question
    original_is_question = original.strip().endswith("?")

    # Check if transformed contains solution indicators
    for pattern in SOLUTION_INDICATORS:
        if re.search(pattern, transformed, re.IGNORECASE):
            # Also check if the transformed is significantly longer
            # (solutions tend to be longer than simple transformations)
            if len(transformed) > len(original) * 1.5:
                return True

    # If original is a question but transformed is not, likely a solution
    if original_is_question and not transformed.strip().endswith("?"):
        # Only flag if significantly longer
        if len(transformed) > len(original) * 1.3:
            return True

    return False


def is_valid_transformation(
    transformed: str,
    original: str,
    max_length_ratio: float = 2.0,
) -> tuple[bool, Optional[str]]:
    """
    Check if a transformation is structurally valid.

    Args:
        transformed: The transformed text
        original: The original text
        max_length_ratio: Maximum allowed length ratio (transformed/original)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not transformed:
        return False, "Empty transformation"

    if detect_solution_instead_of_transform(transformed, original):
        return False, "LLM solved the problem instead of transforming"

    length_ratio = len(transformed) / len(original) if original else float("inf")
    if length_ratio > max_length_ratio:
        return False, f"Output too long (ratio: {length_ratio:.1f})"

    return True, None
