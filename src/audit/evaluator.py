"""
Fairness Audit Evaluator

Compares LLM responses on original vs dialect-transformed prompts to detect bias.
"""

import json
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

import httpx


@dataclass
class AuditConfig:
    """Configuration for fairness audit."""
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0
    temperature: float = 0.0  # Deterministic for fair comparison


@dataclass
class PairResult:
    """Result for a single original/transformed pair."""
    id: str
    original_prompt: str
    transformed_prompt: str
    original_response: str
    transformed_response: str
    original_answer: Optional[str] = None
    transformed_answer: Optional[str] = None
    expected_answer: Optional[str] = None
    original_correct: Optional[bool] = None
    transformed_correct: Optional[bool] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    """Aggregated audit results."""
    benchmark: str
    dialect: str
    model: str
    total_pairs: int
    original_correct: int
    transformed_correct: int
    both_correct: int
    both_wrong: int
    original_only_correct: int
    transformed_only_correct: int
    original_accuracy: float
    transformed_accuracy: float
    accuracy_gap: float
    pairs: List[PairResult] = field(default_factory=list)
    audit_time: str = ""
    errors: int = 0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["pairs"] = [p.to_dict() for p in self.pairs]
        return result

    def summary(self) -> str:
        """Return a formatted summary of the audit results."""
        lines = [
            f"=== Fairness Audit Results ===",
            f"Model: {self.model}",
            f"Benchmark: {self.benchmark}",
            f"Dialect: {self.dialect}",
            f"Total pairs: {self.total_pairs}",
            f"",
            f"Original accuracy:    {self.original_accuracy:.1%} ({self.original_correct}/{self.total_pairs})",
            f"Transformed accuracy: {self.transformed_accuracy:.1%} ({self.transformed_correct}/{self.total_pairs})",
            f"Accuracy gap:         {self.accuracy_gap:+.1%}",
            f"",
            f"Both correct:              {self.both_correct}",
            f"Both wrong:                {self.both_wrong}",
            f"Original only correct:     {self.original_only_correct}",
            f"Transformed only correct:  {self.transformed_only_correct}",
        ]
        if self.errors > 0:
            lines.append(f"Errors: {self.errors}")
        return "\n".join(lines)


class FairnessAuditor:
    """
    Audits LLM fairness by comparing responses to original vs dialect prompts.
    """

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.client = httpx.Client(timeout=self.config.timeout)

    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama API for a response."""
        response = self.client.post(
            f"{self.config.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.config.temperature},
            },
        )
        response.raise_for_status()
        return response.json()["response"]

    def _query_llm(self, prompt: str) -> str:
        """Query the configured LLM backend."""
        if self.config.backend == "ollama":
            return self._query_ollama(prompt)
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

    def _extract_numerical_answer(self, response: str) -> Optional[str]:
        """
        Extract numerical answer from LLM response.
        Looks for patterns like "#### 42" or "the answer is 42".
        """
        # GSM8K format: #### number
        match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", response)
        if match:
            return match.group(1).replace(",", "")

        # Common answer patterns
        patterns = [
            r"(?:the\s+)?answer\s+is\s*[:\s]*([\d,]+(?:\.\d+)?)",
            r"(?:=|equals?)\s*([\d,]+(?:\.\d+)?)\s*$",
            r"([\d,]+(?:\.\d+)?)\s*(?:dollars?|items?|people|hours?|minutes?|days?)?\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).replace(",", "")

        # Last number in response as fallback
        numbers = re.findall(r"\d+(?:\.\d+)?", response)
        if numbers:
            return numbers[-1]

        return None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison (remove trailing zeros, etc.)."""
        try:
            num = float(answer)
            if num == int(num):
                return str(int(num))
            return str(num)
        except (ValueError, TypeError):
            return str(answer).strip().lower()

    def _check_answer(self, extracted: Optional[str], expected: str) -> bool:
        """Check if extracted answer matches expected."""
        if extracted is None:
            return False
        return self._normalize_answer(extracted) == self._normalize_answer(expected)

    def audit_pair(
        self,
        pair_id: str,
        original: str,
        transformed: str,
        expected_answer: Optional[str] = None,
    ) -> PairResult:
        """
        Audit a single original/transformed pair.

        Args:
            pair_id: Unique identifier for the pair
            original: Original prompt
            transformed: Dialect-transformed prompt
            expected_answer: Expected answer for correctness checking

        Returns:
            PairResult with responses and correctness info
        """
        try:
            original_response = self._query_llm(original)
            transformed_response = self._query_llm(transformed)

            original_answer = self._extract_numerical_answer(original_response)
            transformed_answer = self._extract_numerical_answer(transformed_response)

            original_correct = None
            transformed_correct = None
            if expected_answer:
                original_correct = self._check_answer(original_answer, expected_answer)
                transformed_correct = self._check_answer(transformed_answer, expected_answer)

            return PairResult(
                id=pair_id,
                original_prompt=original,
                transformed_prompt=transformed,
                original_response=original_response,
                transformed_response=transformed_response,
                original_answer=original_answer,
                transformed_answer=transformed_answer,
                expected_answer=expected_answer,
                original_correct=original_correct,
                transformed_correct=transformed_correct,
            )

        except Exception as e:
            return PairResult(
                id=pair_id,
                original_prompt=original,
                transformed_prompt=transformed,
                original_response="",
                transformed_response="",
                error=str(e),
            )

    def audit(
        self,
        pairs: List[dict],
        benchmark: str,
        dialect: str,
        show_progress: bool = True,
    ) -> AuditResult:
        """
        Run fairness audit on a list of pairs.

        Args:
            pairs: List of dicts with 'id', 'original', 'transformed', and optionally 'metadata.answer'
            benchmark: Benchmark name (e.g., 'gsm8k')
            dialect: Dialect name (e.g., 'hiberno_english')
            show_progress: Whether to show progress bar

        Returns:
            AuditResult with all metrics and pair results
        """
        results = []
        original_correct = 0
        transformed_correct = 0
        both_correct = 0
        both_wrong = 0
        original_only = 0
        transformed_only = 0
        errors = 0

        iterator = pairs
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(pairs, desc=f"Auditing {benchmark}")
            except ImportError:
                pass

        for pair in iterator:
            expected = pair.get("metadata", {}).get("answer")

            result = self.audit_pair(
                pair_id=pair["id"],
                original=pair["original"],
                transformed=pair["transformed"],
                expected_answer=expected,
            )
            results.append(result)

            if result.error:
                errors += 1
                continue

            if result.original_correct:
                original_correct += 1
            if result.transformed_correct:
                transformed_correct += 1

            if result.original_correct and result.transformed_correct:
                both_correct += 1
            elif not result.original_correct and not result.transformed_correct:
                both_wrong += 1
            elif result.original_correct:
                original_only += 1
            else:
                transformed_only += 1

        total = len(pairs) - errors
        orig_acc = original_correct / total if total else 0.0
        trans_acc = transformed_correct / total if total else 0.0

        return AuditResult(
            benchmark=benchmark,
            dialect=dialect,
            model=self.config.model,
            total_pairs=total,
            original_correct=original_correct,
            transformed_correct=transformed_correct,
            both_correct=both_correct,
            both_wrong=both_wrong,
            original_only_correct=original_only,
            transformed_only_correct=transformed_only,
            original_accuracy=orig_acc,
            transformed_accuracy=trans_acc,
            accuracy_gap=trans_acc - orig_acc,
            pairs=results,
            audit_time=datetime.now().isoformat(),
            errors=errors,
        )

    def save_result(self, result: AuditResult, output_path: Path) -> None:
        """Save audit result to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)


def load_generated_pairs(path: Path) -> tuple[list[dict], str, str]:
    """
    Load generated pairs from a JSON file.

    Returns:
        Tuple of (pairs, benchmark_name, dialect)
    """
    with open(path) as f:
        data = json.load(f)
    return data["pairs"], data["benchmark"], data["dialect"]


def run_audit(
    pairs_path: Path,
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    output_dir: Optional[Path] = None,
    show_progress: bool = True,
    save: bool = True,
) -> AuditResult:
    """
    Convenience function to run a fairness audit.

    Args:
        pairs_path: Path to generated pairs JSON file
        backend: LLM backend
        model: Model name
        output_dir: Directory to save results
        show_progress: Whether to show progress
        save: Whether to save results

    Returns:
        AuditResult with all metrics
    """
    pairs, benchmark, dialect = load_generated_pairs(pairs_path)

    config = AuditConfig(backend=backend, model=model)
    auditor = FairnessAuditor(config)

    result = auditor.audit(pairs, benchmark, dialect, show_progress)

    if save:
        output_dir = output_dir or Path("data/audits")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audit_{benchmark}_{dialect}_{model.replace(':', '_')}_{timestamp}.json"
        output_path = output_dir / filename
        auditor.save_result(result, output_path)
        print(f"Saved to: {output_path}")

    return result
