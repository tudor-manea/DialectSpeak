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

## Key Hiberno-English Features to Apply

### High-Priority Syntactic Features (use when appropriate):

1. **Perfective "after" construction**: Express recent past/perfect aspect
   - "I have just eaten" → "I'm after eating"
   - "She has just left" → "She's after leaving"

2. **Habitual "do be" construction**: Express habitual actions
   - "He usually works late" → "He does be working late"
   - "They are always complaining" → "They do be complaining"

3. **Cleft sentences for emphasis**: Front predicates for emphasis
   - "I am very tired" → "It's tired I am"
   - "He is really clever" → "It's clever he is"

### Medium-Priority Features:

4. **"Amn't" contraction**: Use instead of "aren't I"
   - "I'm not sure" → "I amn't sure"
   - "Am I not right?" → "Amn't I right?"

5. **Plural "youse"**: Second person plural
   - "Are you all coming?" → "Are youse coming?"

6. **"Sure" as discourse marker**: Sentence-initial pragmatic marker
   - "Everyone knows that" → "Sure, everyone knows that"

7. **Emphatic tags "so it is/so I did"**:
   - "It's cold today" → "It's cold today, so it is"

8. **Embedded question inversion**:
   - "I wonder if he is coming" → "I wonder is he coming"

### Features to AVOID (stereotypes):
- "Top o' the morning"
- "Begorrah" / "Faith and begorrah"
- "To be sure, to be sure"
- Excessive apostrophes to suggest accent
- "Oirish" spellings
- Leprechaun-style speech

## Guidelines:
1. Apply 1-3 features naturally per sentence - don't overload
2. Preserve the EXACT meaning and information content
3. Maintain mathematical/logical precision in reasoning tasks
4. Keep proper nouns, numbers, and technical terms unchanged
5. The transformation should sound natural to an Irish English speaker
6. When in doubt, prefer subtle transformation over heavy-handed changes"""

HIBERNO_ENGLISH_USER_TEMPLATE = """Transform the following Standard English text into authentic Hiberno-English. Apply appropriate dialect features while preserving the exact meaning.

Original text:
{text}

Transformed text (Hiberno-English):"""

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
