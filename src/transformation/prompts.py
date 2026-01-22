"""
Transformation Prompts Module

Loads dialect transformation prompts from YAML configuration files.
Supports multiple dialects with configurable prompts.
"""

import yaml
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DialectPromptConfig:
    """Configuration for a dialect's transformation prompts."""
    system_prompt: str
    user_template: str
    batch_template: str


class DialectPromptLoader:
    """Loads and caches dialect prompts from YAML files."""

    def __init__(self, dialects_dir: str = "data/dialects"):
        self.dialects_dir = Path(dialects_dir)
        self._cache: dict[str, DialectPromptConfig] = {}

    def _load_dialect(self, dialect: str) -> DialectPromptConfig:
        """Load dialect prompts from YAML file."""
        yaml_path = self.dialects_dir / f"{dialect}.yaml"

        if not yaml_path.exists():
            raise ValueError(
                f"No configuration found for dialect: {dialect}. "
                f"Expected file: {yaml_path}"
            )

        with open(yaml_path, "r") as f:
            spec = yaml.safe_load(f)

        if "prompts" not in spec:
            raise ValueError(
                f"Dialect config {yaml_path} missing 'prompts' section"
            )

        prompts = spec["prompts"]
        required_keys = ["system", "user_template", "batch_template"]
        for key in required_keys:
            if key not in prompts:
                raise ValueError(
                    f"Dialect config {yaml_path} missing 'prompts.{key}'"
                )

        return DialectPromptConfig(
            system_prompt=prompts["system"].strip(),
            user_template=prompts["user_template"].strip(),
            batch_template=prompts["batch_template"].strip(),
        )

    def get(self, dialect: str) -> DialectPromptConfig:
        """Get prompt configuration for a dialect, with caching."""
        if dialect not in self._cache:
            self._cache[dialect] = self._load_dialect(dialect)
        return self._cache[dialect]

    def get_supported_dialects(self) -> list[str]:
        """Return list of available dialect identifiers."""
        if not self.dialects_dir.exists():
            return []
        return [
            p.stem for p in self.dialects_dir.glob("*.yaml")
            if self._has_prompts(p)
        ]

    def _has_prompts(self, yaml_path: Path) -> bool:
        """Check if a YAML file contains prompts section."""
        try:
            with open(yaml_path, "r") as f:
                spec = yaml.safe_load(f)
            return "prompts" in spec
        except Exception:
            return False


# Global loader instance
_loader: Optional[DialectPromptLoader] = None


def _get_loader() -> DialectPromptLoader:
    """Get or create the global loader instance."""
    global _loader
    if _loader is None:
        _loader = DialectPromptLoader()
    return _loader


def set_dialects_dir(dialects_dir: str) -> None:
    """Set a custom dialects directory (useful for testing)."""
    global _loader
    _loader = DialectPromptLoader(dialects_dir)


# =============================================================================
# Public API
# =============================================================================

def get_supported_dialects() -> list[str]:
    """Return list of supported dialect identifiers."""
    return _get_loader().get_supported_dialects()


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
    config = _get_loader().get(dialect)
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
    config = _get_loader().get(dialect)
    numbered_texts = "\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
    user_prompt = config.batch_template.format(texts=numbered_texts)
    return config.system_prompt, user_prompt
