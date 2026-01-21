"""
Transformation Prompts Module

Linguistically-informed prompts for dialect transformation.
Based on Filppula (1999), Hickey (2007), Kallen (2013).
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class DialectPromptConfig:
    """Configuration for a dialect's transformation prompts."""
    system_prompt: str
    user_template: str
    batch_template: str


# =============================================================================
# Hiberno-English Prompts
# =============================================================================

HIBERNO_ENGLISH_SYSTEM_PROMPT = """You are a linguistic expert specializing in Hiberno-English (Irish English). Your task is to transform Standard English text into authentic Hiberno-English while preserving the exact meaning.

CRITICAL: You MUST apply at least one SYNTACTIC feature from the list below. Do NOT rely on phonetic spellings or accent markers.

## Syntactic Features to Apply (use at least ONE per transformation):

1. **Perfective "after" construction** - for recent completion:
   - "I have just eaten" → "I'm after eating"
   - "She finished the work" → "She's after finishing the work"

2. **Habitual "do be" construction** - for habitual/ongoing actions:
   - "He usually works late" → "He does be working late"
   - "They are always complaining" → "They do be complaining"

3. **Emphatic tags "so it is/so I did/so they do"** - for emphasis:
   - "It's cold" → "It's cold, so it is"
   - "He left early" → "He left early, so he did"

4. **"Sure" as discourse marker** - sentence-initial:
   - "Everyone knows that" → "Sure, everyone knows that"

5. **Embedded question inversion** - in indirect questions:
   - "I wonder if he is coming" → "I wonder is he coming"
   - "I asked if she was ready" → "I asked was she ready"

6. **"'Tis" contraction**:
   - "It is a fine day" → "'Tis a fine day"

7. **Cleft for emphasis**:
   - "I am tired" → "It's tired I am"

## Features to AVOID (stereotypes and phonetic changes):
- "Top o' the morning" - stage Irish
- "Begorrah" / "Faith and begorrah" - stereotype
- Dropping 'g' from -ing (e.g., "goin'", "doin'") - NOT Hiberno-English
- Adding apostrophes to suggest accent
- "o'" for "of" - NOT authentic
- Any "Oirish" or leprechaun-style speech

## Guidelines:
1. MUST apply at least one syntactic feature from the list above
2. Preserve the EXACT meaning - especially numbers, names, and technical content
3. Keep the same sentence structure when possible
4. Do NOT add or remove information
5. Mathematical precision is essential for reasoning tasks"""

HIBERNO_ENGLISH_USER_TEMPLATE = """Transform the following Standard English text into authentic Hiberno-English.

EXAMPLES:
Input: "She has just arrived at the station."
Output: "She's after arriving at the station."

Input: "He usually works until 6pm."
Output: "He does be working until 6pm."

Input: "The car costs $5000. How much would two cars cost?"
Output: "The car costs $5000, so it does. How much would two cars cost?"

Input: "I wonder if she is coming to the party."
Output: "I wonder is she coming to the party."

RULES:
1. Output ONLY the transformed text
2. Use at least ONE syntactic feature: "after + verb", "do/does be + verb", "so it is/does/did", or embedded question inversion
3. Keep numbers, names, and technical terms EXACTLY as they appear
4. If input is a question, output must also be a question
5. Do NOT add commentary or explanations

Original text:
{text}

Transformed text:"""

HIBERNO_ENGLISH_BATCH_TEMPLATE = """Transform each of the following Standard English texts into authentic Hiberno-English. Apply appropriate dialect features while preserving the exact meaning.

For each text, output ONLY the transformed version on a new line, in the same order as the input.

Texts to transform:
{texts}

Transformed texts (one per line):"""


# =============================================================================
# Dialect Registry
# =============================================================================

DIALECT_REGISTRY: dict[str, DialectPromptConfig] = {
    "hiberno_english": DialectPromptConfig(
        system_prompt=HIBERNO_ENGLISH_SYSTEM_PROMPT,
        user_template=HIBERNO_ENGLISH_USER_TEMPLATE,
        batch_template=HIBERNO_ENGLISH_BATCH_TEMPLATE,
    ),
}


# =============================================================================
# Public API
# =============================================================================

def get_supported_dialects() -> list[str]:
    """Return list of supported dialect identifiers."""
    return list(DIALECT_REGISTRY.keys())


def _get_dialect_config(dialect: str) -> DialectPromptConfig:
    """Get prompt configuration for a dialect, raising if unsupported."""
    if dialect not in DIALECT_REGISTRY:
        supported = ", ".join(get_supported_dialects())
        raise ValueError(f"Unsupported dialect: {dialect}. Supported: {supported}")
    return DIALECT_REGISTRY[dialect]


def get_transformation_prompt(
    text: str,
    dialect: str = "hiberno_english",
) -> Tuple[str, str]:
    """
    Get system and user prompts for dialect transformation.

    Args:
        text: Text to transform
        dialect: Target dialect

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    config = _get_dialect_config(dialect)
    user_prompt = config.user_template.format(text=text)
    return config.system_prompt, user_prompt


def get_batch_transformation_prompt(
    texts: list[str],
    dialect: str = "hiberno_english",
) -> Tuple[str, str]:
    """
    Get prompts for batch transformation.

    Args:
        texts: List of texts to transform
        dialect: Target dialect

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    config = _get_dialect_config(dialect)
    numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
    user_prompt = config.batch_template.format(texts=numbered_texts)
    return config.system_prompt, user_prompt
