"""Dialect transformation modules."""

from .transformer import (
    LLMBackend,
    TransformationConfig,
    TransformationResult,
    DialectTransformer,
    create_transformer,
)
from .prompts import (
    HIBERNO_ENGLISH_SYSTEM_PROMPT,
    HIBERNO_ENGLISH_USER_TEMPLATE,
    get_transformation_prompt,
    get_batch_transformation_prompt,
    get_supported_dialects,
)
from .postprocess import (
    clean_transformation,
    detect_solution_instead_of_transform,
    is_valid_transformation,
)

__all__ = [
    "LLMBackend",
    "TransformationConfig",
    "TransformationResult",
    "DialectTransformer",
    "create_transformer",
    "HIBERNO_ENGLISH_SYSTEM_PROMPT",
    "HIBERNO_ENGLISH_USER_TEMPLATE",
    "get_transformation_prompt",
    "get_batch_transformation_prompt",
    "get_supported_dialects",
    "clean_transformation",
    "detect_solution_instead_of_transform",
    "is_valid_transformation",
]
