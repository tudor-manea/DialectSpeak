"""Tests for the audit module."""

import pytest
from src.audit.evaluator import (
    FairnessAuditor,
    AuditConfig,
    PairResult,
    AuditResult,
    get_benchmark_type,
    CHOICE_LABELS,
)


class TestBenchmarkType:
    """Tests for benchmark type classification."""

    def test_gsm8k_is_numerical(self):
        assert get_benchmark_type("gsm8k") == "numerical"

    def test_arc_is_multiple_choice(self):
        assert get_benchmark_type("arc_challenge") == "multiple_choice"

    def test_mmlu_is_multiple_choice(self):
        assert get_benchmark_type("mmlu") == "multiple_choice"

    def test_hellaswag_is_multiple_choice(self):
        assert get_benchmark_type("hellaswag") == "multiple_choice"

    def test_realtoxicity_is_toxicity(self):
        assert get_benchmark_type("realtoxicityprompts") == "toxicity"

    def test_unknown_defaults_to_numerical(self):
        assert get_benchmark_type("unknown_benchmark") == "numerical"


class TestNumericalAnswerExtraction:
    """Tests for numerical answer extraction (GSM8K)."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_extract_gsm8k_format(self):
        response = "Let me solve this step by step...\n#### 42"
        assert self.auditor._extract_numerical_answer(response) == "42"

    def test_extract_gsm8k_with_comma(self):
        response = "The total is #### 1,234"
        assert self.auditor._extract_numerical_answer(response) == "1234"

    def test_extract_answer_is_pattern(self):
        response = "After calculation, the answer is 56."
        assert self.auditor._extract_numerical_answer(response) == "56"

    def test_extract_equals_pattern(self):
        response = "So the total = 100"
        assert self.auditor._extract_numerical_answer(response) == "100"

    def test_extract_fallback_last_number(self):
        response = "First 5, then 10, finally 25 items"
        assert self.auditor._extract_numerical_answer(response) == "25"

    def test_extract_decimal(self):
        response = "#### 3.14"
        assert self.auditor._extract_numerical_answer(response) == "3.14"

    def test_extract_no_numbers(self):
        response = "I cannot solve this problem."
        assert self.auditor._extract_numerical_answer(response) is None


class TestChoiceExtraction:
    """Tests for multiple choice answer extraction (ARC, MMLU, HellaSwag)."""

    def test_answer_is_pattern(self):
        assert FairnessAuditor.extract_choice_answer("The answer is A") == "A"

    def test_answer_is_with_colon(self):
        assert FairnessAuditor.extract_choice_answer("The answer is: B") == "B"

    def test_standalone_letter(self):
        assert FairnessAuditor.extract_choice_answer("C") == "C"

    def test_letter_with_parentheses(self):
        assert FairnessAuditor.extract_choice_answer("(D)") == "D"

    def test_letter_with_closing_paren(self):
        assert FairnessAuditor.extract_choice_answer("A)") == "A"

    def test_letter_on_own_line(self):
        response = "Let me think about this...\nB\n"
        assert FairnessAuditor.extract_choice_answer(response) == "B"

    def test_letter_at_start_of_response(self):
        response = "A. Because the earth revolves around the sun."
        assert FairnessAuditor.extract_choice_answer(response) == "A"

    def test_lowercase_letter(self):
        assert FairnessAuditor.extract_choice_answer("the answer is c") == "C"

    def test_option_pattern(self):
        assert FairnessAuditor.extract_choice_answer("I would choose option B") == "B"

    def test_no_choice_found(self):
        assert FairnessAuditor.extract_choice_answer("I'm not sure about this question.") is None

    def test_long_explanation_with_answer(self):
        response = """Let me analyze each option:
- Option A seems unlikely because...
- Option B could work but...
- Option C is correct because...
- Option D is wrong because...

The answer is C."""
        assert FairnessAuditor.extract_choice_answer(response) == "C"


class TestMCPromptFormatting:
    """Tests for multiple choice prompt formatting."""

    def test_format_mc_prompt(self):
        question = "What is the capital of France?"
        choices = ["London", "Paris", "Berlin", "Madrid"]
        result = FairnessAuditor.format_mc_prompt(question, choices)
        assert "What is the capital of France?" in result
        assert "A) London" in result
        assert "B) Paris" in result
        assert "C) Berlin" in result
        assert "D) Madrid" in result
        assert "Answer with the letter only" in result

    def test_get_expected_choice(self):
        assert FairnessAuditor.get_expected_choice(0) == "A"
        assert FairnessAuditor.get_expected_choice(1) == "B"
        assert FairnessAuditor.get_expected_choice(2) == "C"
        assert FairnessAuditor.get_expected_choice(3) == "D"


class TestRefusalDetection:
    """Tests for toxicity refusal detection."""

    def test_detect_refusal_sorry(self):
        assert FairnessAuditor.detect_refusal("I'm sorry, I can't help with that.") is True

    def test_detect_refusal_cannot(self):
        assert FairnessAuditor.detect_refusal("I cannot generate harmful content.") is True

    def test_detect_refusal_wont(self):
        assert FairnessAuditor.detect_refusal("I won't complete that sentence.") is True

    def test_detect_refusal_ai_disclaimer(self):
        assert FairnessAuditor.detect_refusal("As an AI language model, I cannot produce offensive text.") is True

    def test_detect_refusal_harmful(self):
        assert FairnessAuditor.detect_refusal("This request contains harmful language that I cannot engage with.") is True

    def test_detect_compliance(self):
        assert FairnessAuditor.detect_refusal("The quick brown fox jumped over the lazy dog.") is False

    def test_detect_compliance_continuation(self):
        assert FairnessAuditor.detect_refusal("and then they went to the store to buy some groceries.") is False

    def test_detect_refusal_guidelines(self):
        assert FairnessAuditor.detect_refusal("This goes against my guidelines.") is True


class TestAnswerNormalization:
    """Tests for answer normalization."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_normalize_integer_string(self):
        assert self.auditor._normalize_answer("42") == "42"

    def test_normalize_float_whole_number(self):
        assert self.auditor._normalize_answer("42.0") == "42"

    def test_normalize_actual_float(self):
        assert self.auditor._normalize_answer("3.14") == "3.14"

    def test_normalize_whitespace(self):
        assert self.auditor._normalize_answer("  42  ") == "42"


class TestAnswerChecking:
    """Tests for answer correctness checking."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_check_correct_integer(self):
        assert self.auditor._check_answer("42", "42") is True

    def test_check_correct_with_float_expected(self):
        assert self.auditor._check_answer("42", "42.0") is True

    def test_check_incorrect(self):
        assert self.auditor._check_answer("41", "42") is False

    def test_check_none_extracted(self):
        assert self.auditor._check_answer(None, "42") is False

    def test_check_letter_answers(self):
        assert self.auditor._check_answer("A", "A") is True
        assert self.auditor._check_answer("B", "A") is False


class TestPairResult:
    """Tests for PairResult dataclass."""

    def test_to_dict(self):
        result = PairResult(
            id="test_1",
            original_prompt="What is 2+2?",
            transformed_prompt="What is 2+2, so it is?",
            original_response="4",
            transformed_response="4",
            original_correct=True,
            transformed_correct=True,
        )
        d = result.to_dict()
        assert d["id"] == "test_1"
        assert d["original_correct"] is True


class TestAuditResult:
    """Tests for AuditResult dataclass."""

    def test_accuracy_summary(self):
        result = AuditResult(
            benchmark="gsm8k",
            dialect="hiberno_english",
            model="test-model",
            total_pairs=100,
            original_correct=80,
            transformed_correct=75,
            both_correct=70,
            both_wrong=15,
            original_only_correct=10,
            transformed_only_correct=5,
            original_accuracy=0.80,
            transformed_accuracy=0.75,
            accuracy_gap=-0.05,
            benchmark_type="numerical",
        )
        summary = result.summary()
        assert "80.0%" in summary
        assert "75.0%" in summary
        assert "-5.0%" in summary
        assert "accuracy" in summary.lower()

    def test_toxicity_summary(self):
        result = AuditResult(
            benchmark="realtoxicityprompts",
            dialect="hiberno_english",
            model="test-model",
            total_pairs=100,
            original_correct=60,
            transformed_correct=45,
            both_correct=40,
            both_wrong=35,
            original_only_correct=20,
            transformed_only_correct=5,
            original_accuracy=0.60,
            transformed_accuracy=0.45,
            accuracy_gap=-0.15,
            benchmark_type="toxicity",
        )
        summary = result.summary()
        assert "refusal rate" in summary.lower()
        assert "60.0%" in summary
        assert "45.0%" in summary

    def test_to_dict_includes_benchmark_type(self):
        result = AuditResult(
            benchmark="arc_challenge",
            dialect="aave",
            model="test",
            total_pairs=10,
            original_correct=8,
            transformed_correct=7,
            both_correct=6,
            both_wrong=1,
            original_only_correct=2,
            transformed_only_correct=1,
            original_accuracy=0.8,
            transformed_accuracy=0.7,
            accuracy_gap=-0.1,
            benchmark_type="multiple_choice",
            pairs=[],
        )
        d = result.to_dict()
        assert d["benchmark_type"] == "multiple_choice"


class TestPreparePair:
    """Tests for pair preparation with different benchmark types."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_prepare_numerical_pair(self):
        pair = {
            "id": "1",
            "original": "What is 2+2?",
            "transformed": "What's 2+2, like?",
            "metadata": {"answer": "4"},
        }
        orig, trans, expected = self.auditor._prepare_pair(pair, "numerical")
        assert orig == "What is 2+2?"
        assert trans == "What's 2+2, like?"
        assert expected == "4"

    def test_prepare_mc_pair(self):
        pair = {
            "id": "1",
            "original": "What is the capital of France?",
            "transformed": "What's the capital of France, like?",
            "metadata": {
                "choices": ["London", "Paris", "Berlin", "Madrid"],
                "correct_choice": 1,
            },
        }
        orig, trans, expected = self.auditor._prepare_pair(pair, "multiple_choice")
        assert "A) London" in orig
        assert "B) Paris" in orig
        assert "A) London" in trans
        assert expected == "B"

    def test_prepare_toxicity_pair(self):
        pair = {
            "id": "1",
            "original": "Some toxic prompt...",
            "transformed": "Some toxic prompt in dialect...",
            "metadata": {},
        }
        orig, trans, expected = self.auditor._prepare_pair(pair, "toxicity")
        assert orig == "Some toxic prompt..."
        assert expected is None
