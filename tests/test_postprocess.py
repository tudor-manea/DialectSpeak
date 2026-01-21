"""Tests for transformation post-processing module."""

import pytest
from src.transformation.postprocess import (
    clean_transformation,
    detect_solution_instead_of_transform,
    is_valid_transformation,
)


class TestCleanTransformation:
    """Tests for the clean_transformation function."""

    def test_clean_no_padding(self):
        """Text without padding should be unchanged."""
        original = "How many sheep are there?"
        transformed = "How many sheep are there, so it is?"
        result = clean_transformation(transformed, original)
        assert result == "How many sheep are there, so it is?"

    def test_clean_start_padding_grand_question(self):
        """Remove 'grand question' style padding."""
        original = "How much does it cost?"
        transformed = "'Tis a grand question! How much does it cost, then?"
        result = clean_transformation(transformed, original)
        assert "'Tis a grand question" not in result
        assert "cost" in result

    def test_clean_start_padding_lets_see(self):
        """Remove 'Let's see' style padding."""
        original = "What is 2 + 2?"
        transformed = "Let's see now. What is 2 + 2?"
        result = clean_transformation(transformed, original)
        assert "Let's see" not in result
        assert "2 + 2" in result

    def test_clean_quoted_output(self):
        """Remove quotes wrapping the output."""
        original = "Hello there"
        transformed = '"Hello there, so I am"'
        result = clean_transformation(transformed, original)
        assert result == "Hello there, so I am"

    def test_clean_single_quoted_output(self):
        """Remove single quotes wrapping the output."""
        original = "Hello"
        transformed = "'Sure, hello'"
        result = clean_transformation(transformed, original)
        assert result == "Sure, hello"

    def test_clean_whitespace(self):
        """Strip leading/trailing whitespace."""
        original = "Test"
        transformed = "  Test, so it is  "
        result = clean_transformation(transformed, original)
        assert result == "Test, so it is"

    def test_clean_empty_input(self):
        """Handle empty input gracefully."""
        assert clean_transformation("", "test") == ""

    def test_clean_transformed_text_prefix(self):
        """Remove 'Transformed text:' prefix."""
        original = "Hello"
        transformed = "Transformed text: Hello, so I am"
        result = clean_transformation(transformed, original)
        assert result == "Hello, so I am"


class TestDetectSolutionInsteadOfTransform:
    """Tests for detecting when LLM solves instead of transforms."""

    def test_detect_math_solution(self):
        """Detect when LLM provides mathematical answer."""
        original = "How many sheep are there?"
        transformed = "Sure, let me work that out. The total = 260 sheep altogether."
        assert detect_solution_instead_of_transform(transformed, original) is True

    def test_detect_answer_pattern(self):
        """Detect 'the answer is' pattern."""
        original = "What is the total cost?"
        transformed = "So the answer is 500 dollars, which is the total you'd be paying."
        assert detect_solution_instead_of_transform(transformed, original) is True

    def test_valid_transformation_not_flagged(self):
        """Valid transformation should not be flagged."""
        original = "How many sheep are there?"
        transformed = "How many sheep are there, so it is?"
        assert detect_solution_instead_of_transform(transformed, original) is False

    def test_question_transformed_to_question(self):
        """Question transformed to question should be valid."""
        original = "What time is it?"
        transformed = "What time is it, then?"
        assert detect_solution_instead_of_transform(transformed, original) is False

    def test_statement_with_numbers_ok(self):
        """Statements with numbers that aren't solutions should pass."""
        original = "She bought 3 dozen donuts."
        transformed = "She's after buying 3 dozen donuts."
        assert detect_solution_instead_of_transform(transformed, original) is False


class TestIsValidTransformation:
    """Tests for the is_valid_transformation function."""

    def test_valid_transformation(self):
        """Valid transformation should pass."""
        original = "I just finished eating."
        transformed = "I'm after eating."
        is_valid, error = is_valid_transformation(transformed, original)
        assert is_valid is True
        assert error is None

    def test_empty_transformation_invalid(self):
        """Empty transformation should fail."""
        is_valid, error = is_valid_transformation("", "test")
        assert is_valid is False
        assert "Empty" in error

    def test_too_long_transformation(self):
        """Overly long transformation should fail."""
        original = "Hello."
        transformed = "Hello " * 100  # Much longer than original
        is_valid, error = is_valid_transformation(transformed, original)
        assert is_valid is False
        assert "too long" in error

    def test_solution_instead_of_transform(self):
        """LLM solving problem should fail."""
        original = "How many sheep are there?"
        transformed = "Let me calculate. Seattle has 20 sheep, Charleston has 4 times that which is 80 sheep, and Toulouse has twice 80 which equals 160 sheep. So altogether they have 260 sheep."
        is_valid, error = is_valid_transformation(transformed, original)
        assert is_valid is False
        assert "solved" in error.lower()

    def test_reasonable_length_increase_ok(self):
        """Reasonable length increase should pass."""
        original = "I finished eating."
        transformed = "I'm after finishing eating, so I am."  # 1.8x length
        is_valid, error = is_valid_transformation(transformed, original)
        assert is_valid is True
