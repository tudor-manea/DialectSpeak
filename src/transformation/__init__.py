"""Dialect transformation modules."""

from .transformer import (
    LLMBackend,
    TransformationConfig,
    TransformationResult,
    DialectTransformer,
    create_transformer,
)
from .prompts import (
    DialectPromptConfig,
    get_transformation_prompt,
    get_batch_transformation_prompt,
    get_supported_dialects,
    set_dialects_dir,
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
    "DialectPromptConfig",
    "get_transformation_prompt",
    "get_batch_transformation_prompt",
    "get_supported_dialects",
    "set_dialects_dir",
    "clean_transformation",
    "detect_solution_instead_of_transform",
    "is_valid_transformation",
]
