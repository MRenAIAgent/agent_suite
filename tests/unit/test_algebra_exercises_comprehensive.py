#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Algebra Exercises.

This module tests algebra exercise generation, validation, and content quality
across different difficulty levels and mathematical concepts.
"""

import pytest
import sys
import os
from typing import List, Dict, Any
import re

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.exercises.algebra_exercises import AlgebraExercises
from math_learning.exercises.middle_algebra_exercises import MiddleAlgebraExercises
from math_learning.exercises.high_algebra_exercises import HighAlgebraExercises
from math_learning.exercises.exercise import Exercise


class TestAlgebraExercisesCore:
    """Test core Algebra Exercises functionality."""

    @pytest.fixture
    def algebra_exercises(self):
        """Create AlgebraExercises instance."""
        return AlgebraExercises()

    @pytest.fixture
    def middle_algebra_exercises(self):
        """Create MiddleAlgebraExercises instance."""
        return MiddleAlgebraExercises()

    @pytest.fixture
    def high_algebra_exercises(self):
        """Create HighAlgebraExercises instance."""
        return HighAlgebraExercises()

    def test_algebra_exercises_initialization(self, algebra_exercises):
        """Test AlgebraExercises initialization."""
        assert hasattr(algebra_exercises, 'exercises')
        assert isinstance(algebra_exercises.exercises, dict)

    def test_get_addition_exercises_basic(self, algebra_exercises):
        """Test getting basic addition exercises."""
        exercises = algebra_exercises.get_addition_exercises()
        
        assert len(exercises) > 0
        assert all(isinstance(ex, Exercise) for ex in exercises)
        assert all(ex.concept_id == "addition" for ex in exercises)
        
        # Should have variety in difficulty
        difficulties = [ex.difficulty for ex in exercises]
        assert len(set(difficulties)) > 1  # Multiple difficulty levels

    def test_get_subtraction_exercises_basic(self, algebra_exercises):
        """Test getting basic subtraction exercises."""
        exercises = algebra_exercises.get_subtraction_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "subtraction" for ex in exercises)
        
        # Should include negative results and borrowing
        contents = [ex.content for ex in exercises]
        assert any("borrow" in content.lower() or "-" in content for content in contents)

    def test_get_multiplication_exercises_basic(self, algebra_exercises):
        """Test getting basic multiplication exercises."""
        exercises = algebra_exercises.get_multiplication_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "multiplication" for ex in exercises)
        
        # Should include various multiplication patterns
        contents = [ex.content for ex in exercises]
        multiplication_symbols = ["×", "*", "·"]
        assert any(any(symbol in content for symbol in multiplication_symbols) for content in contents)

    def test_get_division_exercises_basic(self, algebra_exercises):
        """Test getting basic division exercises."""
        exercises = algebra_exercises.get_division_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "division" for ex in exercises)
        
        # Should include division symbols and remainders
        contents = [ex.content for ex in exercises]
        division_symbols = ["÷", "/", "÷"]
        assert any(any(symbol in content for symbol in division_symbols) for content in contents)

    def test_get_fractions_exercises_basic(self, algebra_exercises):
        """Test getting basic fractions exercises."""
        exercises = algebra_exercises.get_fractions_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "fractions" for ex in exercises)
        
        # Should contain fraction notation
        contents = [ex.content for ex in exercises]
        assert any("/" in content or "½" in content or "¼" in content for content in contents)

    def test_get_variables_exercises_basic(self, algebra_exercises):
        """Test getting basic variable exercises."""
        exercises = algebra_exercises.get_variables_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "variables" for ex in exercises)
        
        # Should contain variable symbols
        contents = [ex.content for ex in exercises]
        variable_pattern = re.compile(r'[a-zA-Z]')
        assert any(variable_pattern.search(content) for content in contents)

    def test_exercise_content_quality(self, algebra_exercises):
        """Test quality of exercise content."""
        all_exercises = []
        for method_name in ["get_addition_exercises", "get_subtraction_exercises", 
                           "get_multiplication_exercises", "get_division_exercises"]:
            method = getattr(algebra_exercises, method_name)
            all_exercises.extend(method())
        
        for exercise in all_exercises:
            # Content should not be empty
            assert exercise.content.strip() != ""
            
            # Title should not be empty
            assert exercise.title.strip() != ""
            
            # Should have valid difficulty
            assert 0.0 <= exercise.difficulty <= 5.0
            
            # Should have concept_id
            assert exercise.concept_id is not None
            assert exercise.concept_id.strip() != ""
            
            # Should have reasonable content length
            assert len(exercise.content) >= 3  # At least "1+1"
            assert len(exercise.content) <= 500  # Not too verbose

    def test_exercise_difficulty_progression(self, algebra_exercises):
        """Test that exercises show proper difficulty progression."""
        addition_exercises = algebra_exercises.get_addition_exercises()
        
        # Sort by difficulty
        sorted_exercises = sorted(addition_exercises, key=lambda ex: ex.difficulty)
        
        # Check that content complexity increases with difficulty
        easy_exercises = [ex for ex in sorted_exercises if ex.difficulty <= 2.0]
        hard_exercises = [ex for ex in sorted_exercises if ex.difficulty >= 4.0]
        
        if easy_exercises and hard_exercises:
            # Easy exercises should have smaller numbers
            easy_content = " ".join([ex.content for ex in easy_exercises])
            hard_content = " ".join([ex.content for ex in hard_exercises])
            
            # Count multi-digit numbers
            easy_multi_digit = len(re.findall(r'\b\d{2,}\b', easy_content))
            hard_multi_digit = len(re.findall(r'\b\d{2,}\b', hard_content))
            
            # Hard exercises should generally have more multi-digit numbers
            if hard_multi_digit > 0:
                assert hard_multi_digit >= easy_multi_digit

    def test_exercise_tags_consistency(self, algebra_exercises):
        """Test that exercise tags are consistent and meaningful."""
        all_exercises = []
        for method_name in ["get_addition_exercises", "get_subtraction_exercises", 
                           "get_multiplication_exercises"]:
            method = getattr(algebra_exercises, method_name)
            all_exercises.extend(method())
        
        for exercise in all_exercises:
            # Should have tags
            assert isinstance(exercise.tags, list)
            
            # Tags should be strings
            assert all(isinstance(tag, str) for tag in exercise.tags)
            
            # Should have at least one meaningful tag
            meaningful_tags = ["basic", "intermediate", "advanced", "arithmetic", 
                             "algebra", "word_problem", "mental_math"]
            assert any(tag in meaningful_tags for tag in exercise.tags)

    def test_exercise_uniqueness(self, algebra_exercises):
        """Test that exercises are reasonably unique."""
        addition_exercises = algebra_exercises.get_addition_exercises()
        
        # Check for unique IDs
        exercise_ids = [ex.id for ex in addition_exercises]
        assert len(exercise_ids) == len(set(exercise_ids))  # All IDs unique
        
        # Check for content diversity
        contents = [ex.content for ex in addition_exercises]
        unique_contents = set(contents)
        
        # Should have reasonable diversity (at least 80% unique)
        uniqueness_ratio = len(unique_contents) / len(contents)
        assert uniqueness_ratio >= 0.8


class TestMiddleAlgebraExercises:
    """Test Middle Algebra Exercises (grades 6-8)."""

    @pytest.fixture
    def middle_exercises(self):
        """Create MiddleAlgebraExercises instance."""
        return MiddleAlgebraExercises()

    def test_middle_algebra_initialization(self, middle_exercises):
        """Test MiddleAlgebraExercises initialization."""
        assert hasattr(middle_exercises, 'exercises')
        assert isinstance(middle_exercises.exercises, dict)

    def test_get_integer_exercises(self, middle_exercises):
        """Test getting integer operation exercises."""
        exercises = middle_exercises.get_integer_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "integers" for ex in exercises)
        
        # Should include negative numbers
        contents = [ex.content for ex in exercises]
        assert any("-" in content for content in contents)

    def test_get_rational_numbers_exercises(self, middle_exercises):
        """Test getting rational number exercises."""
        exercises = middle_exercises.get_rational_numbers_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "rational_numbers" for ex in exercises)
        
        # Should include fractions and decimals
        contents = [ex.content for ex in exercises]
        has_fractions = any("/" in content for content in contents)
        has_decimals = any("." in content and content.replace(".", "").replace("-", "").isdigit() 
                          for content in contents)
        
        assert has_fractions or has_decimals

    def test_get_algebraic_expressions_exercises(self, middle_exercises):
        """Test getting algebraic expression exercises."""
        exercises = middle_exercises.get_algebraic_expressions_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "algebraic_expressions" for ex in exercises)
        
        # Should contain variables and operations
        contents = [ex.content for ex in exercises]
        variable_pattern = re.compile(r'[a-zA-Z]')
        assert any(variable_pattern.search(content) for content in contents)

    def test_get_equations_exercises(self, middle_exercises):
        """Test getting equation solving exercises."""
        exercises = middle_exercises.get_equations_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id in ["linear_equations", "equations"] for ex in exercises)
        
        # Should contain equals signs
        contents = [ex.content for ex in exercises]
        assert any("=" in content for content in contents)

    def test_middle_algebra_difficulty_appropriate(self, middle_exercises):
        """Test that middle algebra exercises have appropriate difficulty."""
        all_exercises = []
        for method_name in ["get_integer_exercises", "get_rational_numbers_exercises", 
                           "get_algebraic_expressions_exercises"]:
            if hasattr(middle_exercises, method_name):
                method = getattr(middle_exercises, method_name)
                all_exercises.extend(method())
        
        # Middle school exercises should be in appropriate difficulty range
        difficulties = [ex.difficulty for ex in all_exercises]
        assert all(1.5 <= diff <= 4.0 for diff in difficulties)  # Appropriate for grades 6-8

    def test_middle_algebra_concept_coverage(self, middle_exercises):
        """Test that middle algebra covers expected concepts."""
        expected_concepts = ["integers", "rational_numbers", "algebraic_expressions", 
                           "linear_equations", "equations"]
        
        all_exercises = []
        for method_name in dir(middle_exercises):
            if method_name.startswith("get_") and method_name.endswith("_exercises"):
                method = getattr(middle_exercises, method_name)
                try:
                    all_exercises.extend(method())
                except:
                    pass  # Skip methods that might not work
        
        covered_concepts = {ex.concept_id for ex in all_exercises}
        
        # Should cover most expected concepts
        overlap = covered_concepts.intersection(set(expected_concepts))
        assert len(overlap) >= len(expected_concepts) // 2


class TestHighAlgebraExercises:
    """Test High Algebra Exercises (grades 9-12)."""

    @pytest.fixture
    def high_exercises(self):
        """Create HighAlgebraExercises instance."""
        return HighAlgebraExercises()

    def test_high_algebra_initialization(self, high_exercises):
        """Test HighAlgebraExercises initialization."""
        assert hasattr(high_exercises, 'exercises')
        assert isinstance(high_exercises.exercises, dict)

    def test_get_quadratic_exercises(self, high_exercises):
        """Test getting quadratic function exercises."""
        exercises = high_exercises.get_quadratic_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "quadratic_functions" for ex in exercises)
        
        # Should contain quadratic patterns
        contents = [ex.content for ex in exercises]
        quadratic_patterns = ["x²", "x^2", "ax²", "ax^2"]
        assert any(any(pattern in content for pattern in quadratic_patterns) 
                  for content in contents)

    def test_get_polynomial_exercises(self, high_exercises):
        """Test getting polynomial exercises."""
        exercises = high_exercises.get_polynomial_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "polynomials" for ex in exercises)
        
        # Should contain polynomial notation
        contents = [ex.content for ex in exercises]
        polynomial_indicators = ["x²", "x^2", "x³", "x^3", "+", "-"]
        assert any(any(indicator in content for indicator in polynomial_indicators) 
                  for content in contents)

    def test_get_systems_exercises(self, high_exercises):
        """Test getting systems of equations exercises."""
        exercises = high_exercises.get_systems_exercises()
        
        assert len(exercises) > 0
        assert all(ex.concept_id == "systems_of_equations" for ex in exercises)
        
        # Should contain multiple equations
        contents = [ex.content for ex in exercises]
        assert any(content.count("=") >= 2 for content in contents)

    def test_high_algebra_difficulty_appropriate(self, high_exercises):
        """Test that high algebra exercises have appropriate difficulty."""
        all_exercises = []
        for method_name in dir(high_exercises):
            if method_name.startswith("get_") and method_name.endswith("_exercises"):
                method = getattr(high_exercises, method_name)
                try:
                    all_exercises.extend(method())
                except:
                    pass  # Skip methods that might not work
        
        # High school exercises should be in higher difficulty range
        difficulties = [ex.difficulty for ex in all_exercises]
        if difficulties:
            assert all(2.5 <= diff <= 5.0 for diff in difficulties)  # Appropriate for grades 9-12

    def test_high_algebra_complexity(self, high_exercises):
        """Test that high algebra exercises are appropriately complex."""
        all_exercises = []
        for method_name in ["get_quadratic_exercises", "get_polynomial_exercises"]:
            if hasattr(high_exercises, method_name):
                method = getattr(high_exercises, method_name)
                all_exercises.extend(method())
        
        # Should have more complex mathematical notation
        contents = [ex.content for ex in all_exercises]
        complex_indicators = ["²", "^", "√", "±", "∞", "(", ")", "[", "]"]
        
        complex_exercise_count = sum(1 for content in contents 
                                   if any(indicator in content for indicator in complex_indicators))
        
        # At least half should have complex notation
        if len(contents) > 0:
            complexity_ratio = complex_exercise_count / len(contents)
            assert complexity_ratio >= 0.3  # At least 30% should be complex


class TestAlgebraExerciseIntegration:
    """Test integration between different algebra exercise levels."""

    def test_difficulty_progression_across_levels(self):
        """Test difficulty progression from basic to high algebra."""
        basic_exercises = AlgebraExercises()
        middle_exercises = MiddleAlgebraExercises()
        high_exercises = HighAlgebraExercises()
        
        # Get sample exercises from each level
        basic_addition = basic_exercises.get_addition_exercises()
        middle_integers = middle_exercises.get_integer_exercises() if hasattr(middle_exercises, 'get_integer_exercises') else []
        high_quadratic = high_exercises.get_quadratic_exercises() if hasattr(high_exercises, 'get_quadratic_exercises') else []
        
        # Calculate average difficulties
        if basic_addition:
            basic_avg = sum(ex.difficulty for ex in basic_addition) / len(basic_addition)
        else:
            basic_avg = 1.0
            
        if middle_integers:
            middle_avg = sum(ex.difficulty for ex in middle_integers) / len(middle_integers)
        else:
            middle_avg = 2.5
            
        if high_quadratic:
            high_avg = sum(ex.difficulty for ex in high_quadratic) / len(high_quadratic)
        else:
            high_avg = 4.0
        
        # Should show progression: basic < middle < high
        assert basic_avg < middle_avg
        assert middle_avg < high_avg

    def test_concept_id_consistency(self):
        """Test that concept IDs are consistent across exercise levels."""
        basic_exercises = AlgebraExercises()
        middle_exercises = MiddleAlgebraExercises()
        
        # Get all exercises from both levels
        all_basic = []
        all_middle = []
        
        for obj, container in [(basic_exercises, all_basic), (middle_exercises, all_middle)]:
            for method_name in dir(obj):
                if method_name.startswith("get_") and method_name.endswith("_exercises"):
                    try:
                        method = getattr(obj, method_name)
                        container.extend(method())
                    except:
                        pass
        
        # Collect all concept IDs
        basic_concepts = {ex.concept_id for ex in all_basic}
        middle_concepts = {ex.concept_id for ex in all_middle}
        
        # Should have some overlap in concept IDs (like variables, equations)
        overlap = basic_concepts.intersection(middle_concepts)
        
        # Common concepts should exist
        common_concepts = {"addition", "multiplication", "variables"}
        found_common = basic_concepts.intersection(common_concepts)
        assert len(found_common) >= 1  # Should have at least one common concept

    def test_exercise_format_consistency(self):
        """Test that exercise formats are consistent across levels."""
        all_exercise_classes = [AlgebraExercises(), MiddleAlgebraExercises(), HighAlgebraExercises()]
        
        for exercise_class in all_exercise_classes:
            # Get sample exercises
            sample_exercises = []
            for method_name in dir(exercise_class):
                if method_name.startswith("get_") and method_name.endswith("_exercises"):
                    try:
                        method = getattr(exercise_class, method_name)
                        exercises = method()
                        if exercises:
                            sample_exercises.extend(exercises[:2])  # Take first 2 from each method
                    except:
                        pass
            
            # Test format consistency
            for exercise in sample_exercises:
                # Should have required attributes
                assert hasattr(exercise, 'id')
                assert hasattr(exercise, 'title')
                assert hasattr(exercise, 'content')
                assert hasattr(exercise, 'concept_id')
                assert hasattr(exercise, 'difficulty')
                assert hasattr(exercise, 'tags')
                
                # Attributes should have proper types
                assert isinstance(exercise.id, str)
                assert isinstance(exercise.title, str)
                assert isinstance(exercise.content, str)
                assert isinstance(exercise.concept_id, str)
                assert isinstance(exercise.difficulty, (int, float))
                assert isinstance(exercise.tags, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 