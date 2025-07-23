#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Exercise Bank.

This module tests all Exercise Bank functionality including exercise storage,
retrieval, filtering, and management operations.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.algebra_exercises import AlgebraExercises


class TestExerciseBankCore:
    """Test core Exercise Bank functionality."""

    @pytest.fixture
    def sample_exercises(self):
        """Create sample exercises for testing."""
        return [
            Exercise(
                id="ex1", title="Basic Addition", content="2 + 3 = ?",
                concept_id="addition", difficulty=1.0, tags=["basic", "arithmetic"]
            ),
            Exercise(
                id="ex2", title="Advanced Addition", content="15 + 27 = ?",
                concept_id="addition", difficulty=1.5, tags=["arithmetic"]
            ),
            Exercise(
                id="ex3", title="Multiplication", content="4 × 5 = ?",
                concept_id="multiplication", difficulty=2.0, tags=["basic", "arithmetic"]
            ),
            Exercise(
                id="ex4", title="Variables", content="Solve: x + 5 = 10",
                concept_id="variables", difficulty=3.0, tags=["algebra", "variables"]
            ),
            Exercise(
                id="ex5", title="Linear Equations", content="Solve: 2x + 3 = 11",
                concept_id="linear_equations", difficulty=4.0, tags=["algebra", "equations"]
            )
        ]

    @pytest.fixture
    def exercise_bank(self, sample_exercises):
        """Create an exercise bank with sample exercises."""
        bank = ExerciseBank()
        for exercise in sample_exercises:
            bank.add_exercise(exercise)
        return bank

    def test_exercise_bank_initialization(self):
        """Test ExerciseBank initialization."""
        bank = ExerciseBank()
        
        assert hasattr(bank, 'exercises')
        assert isinstance(bank.exercises, dict)
        assert len(bank.exercises) == 0

    def test_add_exercise_basic(self):
        """Test adding a single exercise."""
        bank = ExerciseBank()
        exercise = Exercise("ex1", "Test", "content", "addition", 1.0, [])
        
        bank.add_exercise(exercise)
        
        assert len(bank.exercises) == 1
        assert "ex1" in bank.exercises
        assert bank.exercises["ex1"] == exercise

    def test_add_exercise_duplicate_id(self, exercise_bank, sample_exercises):
        """Test adding exercise with duplicate ID."""
        original_count = len(exercise_bank.exercises)
        
        # Try to add exercise with existing ID
        duplicate = Exercise("ex1", "Duplicate", "content", "addition", 2.0, [])
        exercise_bank.add_exercise(duplicate)
        
        # Should replace the original exercise
        assert len(exercise_bank.exercises) == original_count
        assert exercise_bank.exercises["ex1"].title == "Duplicate"

    def test_get_exercise_by_id_existing(self, exercise_bank):
        """Test retrieving exercise by existing ID."""
        exercise = exercise_bank.get_exercise("ex1")
        
        assert exercise is not None
        assert exercise.id == "ex1"
        assert exercise.title == "Basic Addition"

    def test_get_exercise_by_id_nonexistent(self, exercise_bank):
        """Test retrieving exercise by non-existent ID."""
        exercise = exercise_bank.get_exercise("nonexistent")
        
        assert exercise is None

    def test_get_all_exercises(self, exercise_bank, sample_exercises):
        """Test retrieving all exercises."""
        all_exercises = exercise_bank.get_all_exercises()
        
        assert len(all_exercises) == len(sample_exercises)
        assert all(isinstance(ex, Exercise) for ex in all_exercises)
        
        # Should contain all sample exercises
        exercise_ids = {ex.id for ex in all_exercises}
        expected_ids = {ex.id for ex in sample_exercises}
        assert exercise_ids == expected_ids

    def test_get_exercises_by_concept_existing(self, exercise_bank):
        """Test retrieving exercises by existing concept."""
        addition_exercises = exercise_bank.get_exercises_by_concept("addition")
        
        assert len(addition_exercises) == 2  # ex1 and ex2
        assert all(ex.concept_id == "addition" for ex in addition_exercises)
        
        # Should be sorted by difficulty (ascending)
        difficulties = [ex.difficulty for ex in addition_exercises]
        assert difficulties == sorted(difficulties)

    def test_get_exercises_by_concept_nonexistent(self, exercise_bank):
        """Test retrieving exercises by non-existent concept."""
        exercises = exercise_bank.get_exercises_by_concept("nonexistent")
        
        assert exercises == []

    def test_get_exercises_by_difficulty_range_inclusive(self, exercise_bank):
        """Test retrieving exercises by difficulty range (inclusive)."""
        exercises = exercise_bank.get_exercises_by_difficulty_range(1.0, 3.0)
        
        # Should include ex1 (1.0), ex2 (1.5), ex3 (2.0), ex4 (3.0)
        assert len(exercises) == 4
        assert all(1.0 <= ex.difficulty <= 3.0 for ex in exercises)
        
        # Should be sorted by difficulty
        difficulties = [ex.difficulty for ex in exercises]
        assert difficulties == sorted(difficulties)

    def test_get_exercises_by_difficulty_range_empty(self, exercise_bank):
        """Test retrieving exercises by difficulty range with no matches."""
        exercises = exercise_bank.get_exercises_by_difficulty_range(5.0, 6.0)
        
        assert exercises == []

    def test_get_exercises_by_tags_single_tag(self, exercise_bank):
        """Test retrieving exercises by single tag."""
        basic_exercises = exercise_bank.get_exercises_by_tags(["basic"])
        
        # Should include ex1 and ex3 (both have "basic" tag)
        assert len(basic_exercises) == 2
        assert all("basic" in ex.tags for ex in basic_exercises)

    def test_get_exercises_by_tags_multiple_tags(self, exercise_bank):
        """Test retrieving exercises by multiple tags (AND logic)."""
        exercises = exercise_bank.get_exercises_by_tags(["basic", "arithmetic"])
        
        # Should include ex1 and ex3 (both have both tags)
        assert len(exercises) == 2
        assert all("basic" in ex.tags and "arithmetic" in ex.tags for ex in exercises)

    def test_get_exercises_by_tags_no_matches(self, exercise_bank):
        """Test retrieving exercises by tags with no matches."""
        exercises = exercise_bank.get_exercises_by_tags(["nonexistent"])
        
        assert exercises == []

    def test_remove_exercise_existing(self, exercise_bank):
        """Test removing existing exercise."""
        original_count = len(exercise_bank.exercises)
        
        success = exercise_bank.remove_exercise("ex1")
        
        assert success is True
        assert len(exercise_bank.exercises) == original_count - 1
        assert "ex1" not in exercise_bank.exercises

    def test_remove_exercise_nonexistent(self, exercise_bank):
        """Test removing non-existent exercise."""
        original_count = len(exercise_bank.exercises)
        
        success = exercise_bank.remove_exercise("nonexistent")
        
        assert success is False
        assert len(exercise_bank.exercises) == original_count

    def test_get_concepts_list(self, exercise_bank):
        """Test getting list of all concepts."""
        concepts = exercise_bank.get_concepts()
        
        expected_concepts = {"addition", "multiplication", "variables", "linear_equations"}
        assert set(concepts) == expected_concepts

    def test_get_difficulty_range(self, exercise_bank):
        """Test getting difficulty range."""
        min_diff, max_diff = exercise_bank.get_difficulty_range()
        
        assert min_diff == 1.0  # ex1
        assert max_diff == 4.0  # ex5

    def test_get_exercise_count(self, exercise_bank, sample_exercises):
        """Test getting total exercise count."""
        count = exercise_bank.get_exercise_count()
        
        assert count == len(sample_exercises)

    def test_get_exercises_by_concept_and_difficulty(self, exercise_bank):
        """Test combined concept and difficulty filtering."""
        exercises = exercise_bank.get_exercises_by_concept_and_difficulty(
            "addition", min_difficulty=1.0, max_difficulty=1.5
        )
        
        # Should include ex1 (1.0) and ex2 (1.5)
        assert len(exercises) == 2
        assert all(ex.concept_id == "addition" for ex in exercises)
        assert all(1.0 <= ex.difficulty <= 1.5 for ex in exercises)


class TestExerciseBankIntegration:
    """Test Exercise Bank integration with algebra exercises."""

    def test_load_algebra_exercises_basic(self):
        """Test loading basic algebra exercises."""
        bank = ExerciseBank()
        algebra_exercises = AlgebraExercises()
        
        # Load some exercises
        exercises = algebra_exercises.get_addition_exercises()
        for exercise in exercises:
            bank.add_exercise(exercise)
        
        # Should have loaded exercises
        assert bank.get_exercise_count() > 0
        
        # Should have addition exercises
        addition_exercises = bank.get_exercises_by_concept("addition")
        assert len(addition_exercises) > 0

    def test_exercise_bank_statistics(self, exercise_bank):
        """Test getting exercise bank statistics."""
        stats = exercise_bank.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_exercises' in stats
        assert 'concepts_covered' in stats
        assert 'difficulty_range' in stats
        assert 'exercises_by_concept' in stats
        
        # Verify statistics accuracy
        assert stats['total_exercises'] == 5
        assert stats['concepts_covered'] == 4
        assert stats['difficulty_range'] == (1.0, 4.0)

    def test_exercise_bank_search_functionality(self, exercise_bank):
        """Test search functionality."""
        # Search by title
        results = exercise_bank.search_exercises(query="addition", search_in=["title"])
        addition_exercises = [ex for ex in results if "addition" in ex.title.lower()]
        assert len(addition_exercises) >= 2

        # Search by content
        results = exercise_bank.search_exercises(query="solve", search_in=["content"])
        solve_exercises = [ex for ex in results if "solve" in ex.content.lower()]
        assert len(solve_exercises) >= 2

    def test_exercise_bank_filtering_combinations(self, exercise_bank):
        """Test complex filtering combinations."""
        # Filter by concept, difficulty, and tags
        filtered = exercise_bank.filter_exercises(
            concept_ids=["addition", "multiplication"],
            min_difficulty=1.0,
            max_difficulty=2.5,
            required_tags=["arithmetic"]
        )
        
        # Should include exercises that match all criteria
        assert len(filtered) >= 2
        for ex in filtered:
            assert ex.concept_id in ["addition", "multiplication"]
            assert 1.0 <= ex.difficulty <= 2.5
            assert "arithmetic" in ex.tags

    def test_exercise_bank_export_import(self, exercise_bank, sample_exercises):
        """Test exporting and importing exercise data."""
        # Export exercises
        exported_data = exercise_bank.export_exercises()
        
        assert isinstance(exported_data, list)
        assert len(exported_data) == len(sample_exercises)
        
        # Create new bank and import
        new_bank = ExerciseBank()
        new_bank.import_exercises(exported_data)
        
        # Should have same exercises
        assert new_bank.get_exercise_count() == exercise_bank.get_exercise_count()
        
        # Verify exercise integrity
        for original_ex in sample_exercises:
            imported_ex = new_bank.get_exercise(original_ex.id)
            assert imported_ex is not None
            assert imported_ex.title == original_ex.title
            assert imported_ex.concept_id == original_ex.concept_id

    def test_exercise_bank_validation(self, exercise_bank):
        """Test exercise validation functionality."""
        # Test with valid exercise
        valid_exercise = Exercise("valid", "Valid Exercise", "Content", "concept", 2.0, [])
        validation_result = exercise_bank.validate_exercise(valid_exercise)
        assert validation_result['is_valid'] is True
        assert len(validation_result['errors']) == 0

        # Test with invalid exercise (missing required fields)
        invalid_exercise = Exercise("", "", "", "", -1.0, [])
        validation_result = exercise_bank.validate_exercise(invalid_exercise)
        assert validation_result['is_valid'] is False
        assert len(validation_result['errors']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 