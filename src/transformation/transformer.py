"""
Dialect Transformer Module

Transforms Standard English to target dialects using LLM APIs.
Supports Ollama (local) and OpenAI backends.
"""

import os
from typing import List, Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum

from .prompts import get_transformation_prompt
from .postprocess import clean_transformation, is_valid_transformation


class LLMBackend(Enum):
    """Supported LLM backends."""
    OLLAMA = "ollama"
    OPENAI = "openai"


@dataclass
class TransformationConfig:
    """Configuration for dialect transformation."""
    dialect: str = "hiberno_english"
    backend: LLMBackend = LLMBackend.OLLAMA
    model: str = "llama3.1:8b"
    temperature: float = 0.3
    max_tokens: int = 1024
    ollama_host: str = "http://localhost:11434"


@dataclass
class TransformationResult:
    """Result of a single transformation."""
    original: str
    transformed: str
    dialect: str
    model: str
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class DialectTransformer:
    """
    Transforms text to target dialects using LLM APIs.

    Supports Ollama (local) and OpenAI backends.
    """

    def __init__(self, config: Optional[TransformationConfig] = None):
        """
        Initialize the transformer.

        Args:
            config: Transformation configuration
        """
        self.config = config or TransformationConfig()
        self._http_client: Optional[Any] = None  # httpx.Client when using Ollama
        self._openai_client: Optional[Any] = None  # OpenAI client when using OpenAI
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client based on backend."""
        if self.config.backend == LLMBackend.OLLAMA:
            self._init_ollama()
        elif self.config.backend == LLMBackend.OPENAI:
            self._init_openai()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _init_ollama(self) -> None:
        """Initialize Ollama HTTP client."""
        try:
            import httpx
            self._http_client = httpx.Client(timeout=120.0)
        except ImportError:
            raise ImportError("httpx package not installed. Run: pip install httpx")

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._openai_client = OpenAI(api_key=api_key)

    def transform(self, text: str, max_retries: int = 2) -> TransformationResult:
        """
        Transform a single text to the target dialect.

        Args:
            text: Text to transform
            max_retries: Number of retries if transformation is invalid

        Returns:
            TransformationResult
        """
        system_prompt, user_prompt = get_transformation_prompt(
            text, self.config.dialect
        )

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if self.config.backend == LLMBackend.OLLAMA:
                    raw_output = self._call_ollama(system_prompt, user_prompt)
                else:
                    raw_output = self._call_openai(system_prompt, user_prompt)

                # Clean the output
                transformed = clean_transformation(raw_output, text)

                # Validate the transformation
                is_valid, error_msg = is_valid_transformation(transformed, text)

                if is_valid:
                    return TransformationResult(
                        original=text,
                        transformed=transformed,
                        dialect=self.config.dialect,
                        model=self.config.model,
                        success=True,
                        metadata={"attempts": attempt + 1},
                    )
                else:
                    last_error = error_msg
                    # Continue to retry

            except Exception as e:
                last_error = str(e)

        # All retries failed
        return TransformationResult(
            original=text,
            transformed="",
            dialect=self.config.dialect,
            model=self.config.model,
            success=False,
            error=last_error or "Unknown error",
        )

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Make Ollama API call."""
        url = f"{self.config.ollama_host}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        response = self._http_client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Make OpenAI API call."""
        response = self._openai_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def transform_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[TransformationResult]:
        """
        Transform multiple texts to the target dialect.

        Args:
            texts: List of texts to transform
            show_progress: Whether to show progress bar

        Returns:
            List of TransformationResult objects
        """
        results = []
        iterator = texts

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Transforming")
            except ImportError:
                pass

        for text in iterator:
            result = self.transform(text)
            results.append(result)

        return results

    def check_connection(self) -> bool:
        """Check if the backend is accessible."""
        try:
            if self.config.backend == LLMBackend.OLLAMA:
                url = f"{self.config.ollama_host}/api/tags"
                response = self._http_client.get(url)
                return response.status_code == 200
            else:
                # OpenAI - try a minimal request
                self._openai_client.models.list()
                return True
        except Exception:
            return False

    def list_ollama_models(self) -> List[str]:
        """List available Ollama models."""
        if self.config.backend != LLMBackend.OLLAMA:
            raise ValueError("This method is only for Ollama backend")
        url = f"{self.config.ollama_host}/api/tags"
        response = self._http_client.get(url)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [m["name"] for m in models]


def create_transformer(
    dialect: str = "hiberno_english",
    backend: Literal["ollama", "openai"] = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.3,
    ollama_host: str = "http://localhost:11434",
) -> DialectTransformer:
    """
    Factory function to create a configured transformer.

    Args:
        dialect: Target dialect
        backend: LLM backend ('ollama' or 'openai')
        model: Model name (defaults based on backend)
        temperature: Sampling temperature
        ollama_host: Ollama server URL (for ollama backend)

    Returns:
        Configured DialectTransformer
    """
    backend_enum = LLMBackend(backend)

    # Default models per backend
    if model is None:
        model = {
            LLMBackend.OLLAMA: "llama3.1:8b",
            LLMBackend.OPENAI: "gpt-4o-mini",
        }[backend_enum]

    config = TransformationConfig(
        dialect=dialect,
        backend=backend_enum,
        model=model,
        temperature=temperature,
        ollama_host=ollama_host,
    )

    return DialectTransformer(config)
