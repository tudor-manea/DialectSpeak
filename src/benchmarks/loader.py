"""
Benchmark Loader Module

Loads reasoning and safety benchmarks for dialect fairness auditing.
Supports GSM8K, MMLU, and custom benchmark formats.
"""

from typing import List, Optional, Iterator
from dataclasses import dataclass, field
from enum import Enum
from datasets import load_dataset


class BenchmarkType(Enum):
    """Type of benchmark dataset."""
    REASONING = "reasoning"
    SAFETY = "safety"


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""
    id: str
    question: str
    answer: Optional[str] = None
    choices: Optional[List[str]] = None
    correct_choice: Optional[int] = None
    subject: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.choices)


@dataclass
class BenchmarkDataset:
    """A loaded benchmark dataset."""
    name: str
    benchmark_type: BenchmarkType
    samples: List[BenchmarkSample]
    description: str = ""

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.samples)

    def __getitem__(self, idx: int) -> BenchmarkSample:
        return self.samples[idx]

    def subset(self, n: int, seed: int = 42) -> "BenchmarkDataset":
        """Return a random subset of n samples."""
        import random
        rng = random.Random(seed)
        selected = rng.sample(self.samples, min(n, len(self.samples)))
        return BenchmarkDataset(
            name=f"{self.name}_subset_{n}",
            benchmark_type=self.benchmark_type,
            samples=selected,
            description=f"Subset of {n} samples from {self.name}",
        )

    def filter_by_subject(self, subject: str) -> "BenchmarkDataset":
        """Filter samples by subject (for MMLU)."""
        filtered = [s for s in self.samples if s.subject == subject]
        return BenchmarkDataset(
            name=f"{self.name}_{subject}",
            benchmark_type=self.benchmark_type,
            samples=filtered,
            description=f"{self.name} filtered by subject: {subject}",
        )


def _extract_gsm8k_answer(answer_text: str) -> Optional[str]:
    """Extract the final numeric answer from GSM8K answer format (#### <answer>)."""
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    return None


def load_gsm8k(
    split: str = "test",
    max_samples: Optional[int] = None,
) -> BenchmarkDataset:
    """
    Load GSM8K (Grade School Math 8K) benchmark.

    Args:
        split: Dataset split ('train' or 'test')
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        BenchmarkDataset with GSM8K samples
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)

    samples = []
    for idx, item in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        answer_text = item["answer"]
        samples.append(BenchmarkSample(
            id=f"gsm8k_{split}_{idx}",
            question=item["question"],
            answer=_extract_gsm8k_answer(answer_text),
            metadata={
                "full_answer": answer_text,
                "split": split,
            },
        ))

    return BenchmarkDataset(
        name="gsm8k",
        benchmark_type=BenchmarkType.REASONING,
        samples=samples,
        description="Grade School Math 8K - math word problems",
    )


def load_mmlu(
    split: str = "test",
    subjects: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
) -> BenchmarkDataset:
    """
    Load MMLU (Massive Multitask Language Understanding) benchmark.

    Args:
        split: Dataset split ('test', 'validation', 'dev', or 'auxiliary_train')
        subjects: List of subjects to load (None for all)
        max_samples: Maximum samples per subject (None for all)

    Returns:
        BenchmarkDataset with MMLU samples
    """
    dataset = load_dataset("cais/mmlu", "all", split=split)
    subject_set = set(subjects) if subjects else None

    samples = []
    subject_counts: dict[str, int] = {}

    for idx, item in enumerate(dataset):
        subject = item["subject"]

        # Filter by subjects if specified
        if subject_set is not None and subject not in subject_set:
            continue

        # Limit samples per subject
        if max_samples is not None:
            current_count = subject_counts.get(subject, 0)
            if current_count >= max_samples:
                continue
            subject_counts[subject] = current_count + 1

        samples.append(BenchmarkSample(
            id=f"mmlu_{split}_{idx}",
            question=item["question"],
            choices=item["choices"],
            correct_choice=item["answer"],
            subject=subject,
            metadata={"split": split},
        ))

    return BenchmarkDataset(
        name="mmlu",
        benchmark_type=BenchmarkType.REASONING,
        samples=samples,
        description="Massive Multitask Language Understanding",
    )


# Registry mapping benchmark names to their loader functions
BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
}


def get_available_benchmarks() -> List[str]:
    """Return list of available benchmark names."""
    return list(BENCHMARK_LOADERS.keys())


def load_benchmark(
    name: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    **kwargs,
) -> BenchmarkDataset:
    """
    Load a benchmark by name.

    Args:
        name: Benchmark name ('gsm8k', 'mmlu')
        split: Dataset split
        max_samples: Maximum number of samples
        **kwargs: Additional arguments passed to specific loaders

    Returns:
        BenchmarkDataset
    """
    name_lower = name.lower()
    if name_lower not in BENCHMARK_LOADERS:
        available = ", ".join(get_available_benchmarks())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")

    return BENCHMARK_LOADERS[name_lower](split=split, max_samples=max_samples, **kwargs)


def get_mmlu_subjects() -> List[str]:
    """Get list of all MMLU subjects."""
    return [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]
