"""Tests for the audit module."""

import pytest
from src.audit.evaluator import FairnessAuditor, AuditConfig, PairResult, AuditResult


class TestAnswerExtraction:
    """Tests for numerical answer extraction."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_extract_gsm8k_format(self):
        """Test extraction of #### format answers."""
        response = "Let me solve this step by step...\n#### 42"
        assert self.auditor._extract_numerical_answer(response) == "42"

    def test_extract_gsm8k_with_comma(self):
        """Test extraction of numbers with commas."""
        response = "The total is #### 1,234"
        assert self.auditor._extract_numerical_answer(response) == "1234"

    def test_extract_answer_is_pattern(self):
        """Test 'the answer is X' pattern."""
        response = "After calculation, the answer is 56."
        assert self.auditor._extract_numerical_answer(response) == "56"

    def test_extract_equals_pattern(self):
        """Test '= X' pattern."""
        response = "So the total = 100"
        assert self.auditor._extract_numerical_answer(response) == "100"

    def test_extract_fallback_last_number(self):
        """Test fallback to last number in response."""
        response = "First 5, then 10, finally 25 items"
        assert self.auditor._extract_numerical_answer(response) == "25"

    def test_extract_decimal(self):
        """Test decimal number extraction."""
        response = "#### 3.14"
        assert self.auditor._extract_numerical_answer(response) == "3.14"

    def test_extract_no_numbers(self):
        """Test response with no numbers."""
        response = "I cannot solve this problem."
        assert self.auditor._extract_numerical_answer(response) is None


class TestAnswerNormalization:
    """Tests for answer normalization."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_normalize_integer_string(self):
        """Test integer string normalization."""
        assert self.auditor._normalize_answer("42") == "42"

    def test_normalize_float_whole_number(self):
        """Test float that's a whole number."""
        assert self.auditor._normalize_answer("42.0") == "42"

    def test_normalize_actual_float(self):
        """Test actual decimal value."""
        assert self.auditor._normalize_answer("3.14") == "3.14"

    def test_normalize_whitespace(self):
        """Test whitespace handling."""
        assert self.auditor._normalize_answer("  42  ") == "42"


class TestAnswerChecking:
    """Tests for answer correctness checking."""

    def setup_method(self):
        self.auditor = FairnessAuditor(AuditConfig())

    def test_check_correct_integer(self):
        """Test correct integer answer."""
        assert self.auditor._check_answer("42", "42") is True

    def test_check_correct_with_float_expected(self):
        """Test integer extracted, float expected."""
        assert self.auditor._check_answer("42", "42.0") is True

    def test_check_incorrect(self):
        """Test incorrect answer."""
        assert self.auditor._check_answer("41", "42") is False

    def test_check_none_extracted(self):
        """Test None extracted answer."""
        assert self.auditor._check_answer(None, "42") is False


class TestPairResult:
    """Tests for PairResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
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

    def test_summary(self):
        """Test summary generation."""
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
        )
        summary = result.summary()
        assert "80.0%" in summary
        assert "75.0%" in summary
        assert "-5.0%" in summary

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AuditResult(
            benchmark="gsm8k",
            dialect="hiberno_english",
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
            pairs=[],
        )
        d = result.to_dict()
        assert d["benchmark"] == "gsm8k"
        assert d["accuracy_gap"] == -0.1
