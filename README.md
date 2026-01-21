# Dialectal Fairness Auditing in LLMs

A pipeline for auditing Large Language Models for dialectal bias in reasoning and safety mechanisms.

## Motivation

LLMs may exhibit performance disparities when processing non-standard English dialects. A model that solves a math problem correctly in Standard American English might fail when the same problem is phrased in Hiberno-English, AAVE, or Singlish. Similarly, safety mechanisms might trigger inconsistently across dialects.

This project provides tools to systematically measure these disparities across multiple dialects, models, and benchmarks.

## How It Works

The pipeline operates in three stages:

### 1. Transformation

Benchmark prompts (e.g., math problems, safety scenarios) are transformed from Standard American English into target dialects using LLMs. Each transformation preserves the original meaning while introducing authentic dialectal features.

### 2. Validation

A three-part validation pipeline ensures transformation quality:
- **Semantic validation**: Confirms meaning is preserved between original and transformed text
- **Feature detection**: Verifies presence of authentic dialectal markers
- **Authenticity checking**: Flags stereotypical or inauthentic representations

### 3. Auditing

The same model is evaluated on both original and transformed prompts. Performance differences reveal dialectal bias—if a model answers correctly in Standard English but incorrectly in a dialect variant, that indicates unfair treatment.

## Scope

The framework is designed to be extensible across:

- **Dialects**: Hiberno-English, AAVE, Indian English, Scottish English, Singlish, Nigerian English, and more
- **Benchmarks**: Reasoning (GSM8K, MMLU), safety (SORRY-Bench), toxicity detection, and others
- **Models**: Any LLM accessible via API or local inference

## Project Structure

```
├── src/
│   ├── validation/        # Semantic, feature, authenticity validators
│   ├── benchmarks/        # Benchmark loaders
│   ├── transformation/    # Dialect transformation
│   ├── generation/        # Dataset generation pipeline
│   └── audit/             # Fairness auditing
├── data/
│   ├── dialects/          # Dialect specifications (YAML)
│   ├── benchmarks/        # Generated dialect pairs
│   └── audits/            # Audit results
└── tests/
```

## License

MIT License
