"""
Dataset Generation Pipeline

End-to-end pipeline for generating validated dialect transformation pairs
from benchmark datasets.
"""

import json
from pathlib import Path
from typing import List, Optional, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime

from ..benchmarks import BenchmarkDataset, BenchmarkSample
from ..transformation import create_transformer, DialectTransformer, TransformationResult
from ..validation import create_pipeline, ValidationPipeline


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    dialect: str = "hiberno_english"
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    semantic_threshold: float = 0.80
    min_features: int = 1
    authenticity_threshold: float = 0.5
    max_retries: int = 2
    output_dir: str = "data/benchmarks"


@dataclass
class GeneratedPair:
    """A validated original-transformed pair."""
    id: str
    original: str
    transformed: str
    dialect: str
    benchmark: str
    subject: Optional[str] = None
    semantic_score: float = 0.0
    feature_count: int = 0
    authenticity_score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FailedPair:
    """Summary of a failed validation for diagnostics."""
    id: str
    failed_validators: List[str]
    semantic_score: Optional[float] = None
    feature_count: int = 0
    authenticity_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationResult:
    """Result of dataset generation."""
    benchmark: str
    dialect: str
    total_samples: int
    successful_transforms: int
    valid_pairs: int
    failed_validation: int
    transform_errors: int
    pass_rate: float
    pairs: List[GeneratedPair]
    generation_time: str
    failed_pairs: List[FailedPair] = field(default_factory=list)
    failure_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["pairs"] = [p.to_dict() for p in self.pairs]
        result["failed_pairs"] = [fp.to_dict() for fp in self.failed_pairs]
        return result


class DatasetGenerator:
    """
    Generates validated dialect transformation datasets from benchmarks.

    Pipeline:
    1. Load benchmark samples
    2. Transform each prompt to target dialect
    3. Validate transformation (semantic, features, authenticity)
    4. Store valid pairs
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        transformer: Optional[DialectTransformer] = None,
        validator: Optional[ValidationPipeline] = None,
    ):
        self.config = config or GenerationConfig()

        # Initialize transformer
        self.transformer = transformer or create_transformer(
            dialect=self.config.dialect,
            backend=self.config.backend,
            model=self.config.model,
        )

        # Initialize validator
        self.validator = validator or create_pipeline(
            semantic_threshold=self.config.semantic_threshold,
            min_features=self.config.min_features,
            authenticity_threshold=self.config.authenticity_threshold,
        )

    def generate(
        self,
        benchmark: BenchmarkDataset,
        show_progress: bool = True,
    ) -> GenerationResult:
        """
        Generate validated dialect pairs from a benchmark dataset.

        Args:
            benchmark: Loaded benchmark dataset
            show_progress: Whether to show progress bar

        Returns:
            GenerationResult with all valid pairs and statistics
        """
        pairs = []
        failed_pairs_list: List[FailedPair] = []
        successful_transforms = 0
        valid_pairs = 0
        failed_validation = 0
        transform_errors = 0
        failure_counts: dict = {}

        samples = list(benchmark.samples)
        iterator: Iterator[BenchmarkSample] = samples

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(samples, desc=f"Generating {benchmark.name}")
            except ImportError:
                pass

        for sample in iterator:
            # Transform the question/prompt
            transform_result = self._transform_with_retry(sample.question)

            if not transform_result.success:
                transform_errors += 1
                continue

            successful_transforms += 1

            # Validate the transformation
            validation = self.validator.validate(
                original=sample.question,
                transformed=transform_result.transformed,
                dialect=self.config.dialect,
            )

            if not validation.is_valid:
                failed_validation += 1
                failed_pairs_list.append(FailedPair(
                    id=sample.id,
                    failed_validators=list(validation.failed_validators),
                    semantic_score=validation.semantic_score,
                    feature_count=validation.feature_count,
                    authenticity_score=validation.authenticity_score,
                ))
                for v in validation.failed_validators:
                    failure_counts[v] = failure_counts.get(v, 0) + 1
                continue

            valid_pairs += 1
            pair = GeneratedPair(
                id=sample.id,
                original=sample.question,
                transformed=transform_result.transformed,
                dialect=self.config.dialect,
                benchmark=benchmark.name,
                subject=sample.subject,
                semantic_score=validation.semantic_score or 0.0,
                feature_count=validation.feature_count,
                authenticity_score=validation.authenticity_score or 0.0,
                metadata=self._build_sample_metadata(sample),
            )
            pairs.append(pair)

        total = len(samples)
        pass_rate = valid_pairs / total if total else 0.0

        if show_progress and failed_validation > 0:
            breakdown = ", ".join(f"{count} {name}" for name, count in sorted(failure_counts.items(), key=lambda x: -x[1]))
            print(f"\nFailure breakdown ({failed_validation} total): {breakdown}")

        return GenerationResult(
            benchmark=benchmark.name,
            dialect=self.config.dialect,
            total_samples=total,
            successful_transforms=successful_transforms,
            valid_pairs=valid_pairs,
            failed_validation=failed_validation,
            transform_errors=transform_errors,
            pass_rate=pass_rate,
            pairs=pairs,
            generation_time=datetime.now().isoformat(),
            failed_pairs=failed_pairs_list,
            failure_breakdown=failure_counts,
        )

    def _build_sample_metadata(self, sample: BenchmarkSample) -> dict:
        """Build metadata dict from benchmark sample, excluding None values."""
        metadata = {"answer": sample.answer}
        if sample.choices is not None:
            metadata["choices"] = sample.choices
        if sample.correct_choice is not None:
            metadata["correct_choice"] = sample.correct_choice
        return metadata

    def _transform_with_retry(self, text: str) -> TransformationResult:
        """Transform text with retries on failure."""
        attempts = self.config.max_retries + 1
        for _ in range(attempts):
            result = self.transformer.transform(text)
            if result.success:
                return result

        return TransformationResult(
            original=text,
            transformed="",
            dialect=self.config.dialect,
            model=self.transformer.config.model,
            success=False,
            error=f"Failed after {attempts} attempts",
        )

    def save_result(self, result: GenerationResult, filename: Optional[str] = None) -> Path:
        """
        Save generation result to JSON file.

        Args:
            result: GenerationResult to save
            filename: Optional filename (auto-generated if not provided)

        Returns:
            Path to saved file
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.benchmark}_{result.dialect}_{timestamp}.json"

        output_path = output_dir / filename
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return output_path


def generate_dataset(
    benchmark: BenchmarkDataset,
    dialect: str = "hiberno_english",
    backend: str = "ollama",
    model: str = "llama3.1:8b",
    output_dir: str = "data/benchmarks",
    show_progress: bool = True,
    save: bool = True,
) -> GenerationResult:
    """
    Convenience function to generate a validated dialect dataset.

    Args:
        benchmark: Loaded benchmark dataset
        dialect: Target dialect
        backend: LLM backend ('ollama' or 'openai')
        model: Model name
        output_dir: Directory to save results
        show_progress: Whether to show progress bar
        save: Whether to save results to file

    Returns:
        GenerationResult with all valid pairs
    """
    config = GenerationConfig(
        dialect=dialect,
        backend=backend,
        model=model,
        output_dir=output_dir,
    )

    generator = DatasetGenerator(config=config)
    result = generator.generate(benchmark, show_progress=show_progress)

    if save:
        path = generator.save_result(result)
        print(f"Saved to: {path}")

    return result
