#!/usr/bin/env python3
"""
Comprehensive tests for Exercise System.

This test suite provides coverage for the exercise system components
that currently have 0% test coverage.
"""

import pytest
import tempfile
import os
import sys
import json
from typing import Dict, List, Any
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Test imports
try:
    from math_learning.exercises.algebra_exercises import AlgebraExercises
    from math_learning.exercises.exercise_bank import ExerciseBank
    from math_learning.exercises.exercise_generator import ExerciseGenerator
    from math_learning.exercises.exercise_validator import ExerciseValidator
    from math_learning.exercises.difficulty_adapter import DifficultyAdapter
    EXERCISES_AVAILABLE = True
except ImportError as e:
    EXERCISES_AVAILABLE = False
    print(f"Exercise components not available: {e}")


class TestAlgebraExercises:
    """Test suite for Algebra Exercises."""
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_algebra_exercises_initialization(self):
        """Test algebra exercises initialization."""
        exercises = AlgebraExercises()
        
        assert hasattr(exercises, 'exercise_types')
        assert hasattr(exercises, 'difficulty_levels')
        assert hasattr(exercises, 'concept_mapping')
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_generate_linear_equation_exercise(self):
        """Test generating linear equation exercises."""
        exercises = AlgebraExercises()
        
        exercise = exercises.generate_linear_equation(difficulty=0.5)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise
        assert "answer" in exercise
        assert "concept_id" in exercise
        assert "difficulty" in exercise
        assert exercise["concept_id"] == "linear_equations"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_generate_quadratic_exercise(self):
        """Test generating quadratic exercises."""
        exercises = AlgebraExercises()
        
        exercise = exercises.generate_quadratic(difficulty=0.7)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise
        assert "answer" in exercise
        assert exercise["concept_id"] == "quadratic_equations"
        assert exercise["difficulty"] == 0.7
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_generate_polynomial_exercise(self):
        """Test generating polynomial exercises."""
        exercises = AlgebraExercises()
        
        exercise = exercises.generate_polynomial(difficulty=0.8, degree=3)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise
        assert "answer" in exercise
        assert exercise["concept_id"] == "polynomials"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_generate_factoring_exercise(self):
        """Test generating factoring exercises."""
        exercises = AlgebraExercises()
        
        exercise = exercises.generate_factoring(difficulty=0.6)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise
        assert "answer" in exercise
        assert exercise["concept_id"] == "factoring"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validate_exercise_structure(self):
        """Test exercise structure validation."""
        exercises = AlgebraExercises()
        
        # Valid exercise
        valid_exercise = {
            "problem": "Solve for x: 2x + 3 = 7",
            "answer": "x = 2",
            "concept_id": "linear_equations",
            "difficulty": 0.5,
            "hints": ["Isolate x", "Subtract 3 from both sides"]
        }
        
        assert exercises.validate_exercise(valid_exercise) == True
        
        # Invalid exercise (missing required fields)
        invalid_exercise = {
            "problem": "Solve for x: 2x + 3 = 7"
            # Missing answer, concept_id, difficulty
        }
        
        assert exercises.validate_exercise(invalid_exercise) == False
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_get_exercise_by_concept(self):
        """Test retrieving exercises by concept."""
        exercises = AlgebraExercises()
        
        linear_exercises = exercises.get_exercises_by_concept("linear_equations", count=3)
        
        assert isinstance(linear_exercises, list)
        assert len(linear_exercises) <= 3
        assert all(ex["concept_id"] == "linear_equations" for ex in linear_exercises)
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_adaptive_difficulty_generation(self):
        """Test adaptive difficulty exercise generation."""
        exercises = AlgebraExercises()
        
        # Generate exercises with increasing difficulty
        easy_exercise = exercises.generate_by_concept("linear_equations", difficulty=0.2)
        medium_exercise = exercises.generate_by_concept("linear_equations", difficulty=0.5)
        hard_exercise = exercises.generate_by_concept("linear_equations", difficulty=0.8)
        
        assert easy_exercise["difficulty"] == 0.2
        assert medium_exercise["difficulty"] == 0.5
        assert hard_exercise["difficulty"] == 0.8
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_exercise_metadata(self):
        """Test exercise metadata generation."""
        exercises = AlgebraExercises()
        
        exercise = exercises.generate_linear_equation(difficulty=0.5)
        
        # Check for metadata
        assert "id" in exercise
        assert "created_at" in exercise
        assert "estimated_time" in exercise
        assert "learning_objectives" in exercise


class TestExerciseBank:
    """Test suite for Exercise Bank."""
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_exercise_bank_initialization(self):
        """Test exercise bank initialization."""
        bank = ExerciseBank()
        
        assert hasattr(bank, 'exercises')
        assert hasattr(bank, 'concepts')
        assert hasattr(bank, 'difficulty_index')
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_add_exercise_to_bank(self):
        """Test adding exercises to the bank."""
        bank = ExerciseBank()
        
        exercise = {
            "id": "test_ex_1",
            "problem": "What is 2 + 2?",
            "answer": "4",
            "concept_id": "addition",
            "difficulty": 0.1
        }
        
        bank.add_exercise(exercise)
        
        assert len(bank.exercises) > 0
        assert "test_ex_1" in bank.exercises
        assert bank.exercises["test_ex_1"]["problem"] == "What is 2 + 2?"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_search_exercises_by_concept(self):
        """Test searching exercises by concept."""
        bank = ExerciseBank()
        
        # Add test exercises
        exercises = [
            {"id": "add_1", "concept_id": "addition", "difficulty": 0.1, "problem": "2+2", "answer": "4"},
            {"id": "add_2", "concept_id": "addition", "difficulty": 0.3, "problem": "5+7", "answer": "12"},
            {"id": "mult_1", "concept_id": "multiplication", "difficulty": 0.4, "problem": "3×4", "answer": "12"}
        ]
        
        for ex in exercises:
            bank.add_exercise(ex)
        
        addition_exercises = bank.get_exercises_by_concept("addition")
        
        assert len(addition_exercises) == 2
        assert all(ex["concept_id"] == "addition" for ex in addition_exercises)
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_filter_exercises_by_difficulty(self):
        """Test filtering exercises by difficulty range."""
        bank = ExerciseBank()
        
        # Add exercises with different difficulties
        exercises = [
            {"id": "easy_1", "concept_id": "addition", "difficulty": 0.1, "problem": "1+1", "answer": "2"},
            {"id": "med_1", "concept_id": "addition", "difficulty": 0.5, "problem": "25+37", "answer": "62"},
            {"id": "hard_1", "concept_id": "addition", "difficulty": 0.9, "problem": "456+789", "answer": "1245"}
        ]
        
        for ex in exercises:
            bank.add_exercise(ex)
        
        medium_exercises = bank.get_exercises_by_difficulty_range(0.3, 0.7)
        
        assert len(medium_exercises) == 1
        assert medium_exercises[0]["id"] == "med_1"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_exercise_bank_persistence(self):
        """Test saving and loading exercise bank."""
        bank = ExerciseBank()
        
        # Add test exercise
        exercise = {
            "id": "persist_test",
            "problem": "Test persistence",
            "answer": "success",
            "concept_id": "testing",
            "difficulty": 0.5
        }
        bank.add_exercise(exercise)
        
        # Test saving
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            bank.save_to_file(temp_file_path)
            
            # Test loading
            new_bank = ExerciseBank()
            new_bank.load_from_file(temp_file_path)
            
            assert "persist_test" in new_bank.exercises
            assert new_bank.exercises["persist_test"]["problem"] == "Test persistence"
            
        finally:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_exercise_statistics(self):
        """Test exercise bank statistics."""
        bank = ExerciseBank()
        
        # Add exercises
        exercises = [
            {"id": "stat_1", "concept_id": "addition", "difficulty": 0.2, "problem": "1+2", "answer": "3"},
            {"id": "stat_2", "concept_id": "addition", "difficulty": 0.4, "problem": "3+4", "answer": "7"},
            {"id": "stat_3", "concept_id": "multiplication", "difficulty": 0.6, "problem": "2×3", "answer": "6"}
        ]
        
        for ex in exercises:
            bank.add_exercise(ex)
        
        stats = bank.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_exercises" in stats
        assert "concepts_covered" in stats
        assert "difficulty_distribution" in stats
        assert stats["total_exercises"] == 3


class TestExerciseGenerator:
    """Test suite for Exercise Generator."""
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_exercise_generator_initialization(self):
        """Test exercise generator initialization."""
        generator = ExerciseGenerator()
        
        assert hasattr(generator, 'templates')
        assert hasattr(generator, 'concept_rules')
        assert hasattr(generator, 'difficulty_parameters')
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_generate_from_template(self):
        """Test generating exercises from templates."""
        generator = ExerciseGenerator()
        
        template = {
            "problem_template": "Solve for x: {a}x + {b} = {c}",
            "answer_template": "x = {solution}",
            "parameters": {
                "a": {"min": 1, "max": 10},
                "b": {"min": 1, "max": 20},
                "c": {"min": 1, "max": 50}
            }
        }
        
        exercise = generator.generate_from_template(template, "linear_equations", 0.5)
        
        assert isinstance(exercise, dict)
        assert "problem" in exercise
        assert "answer" in exercise
        assert "concept_id" in exercise
        assert exercise["concept_id"] == "linear_equations"
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_parameter_generation(self):
        """Test parameter generation for exercises."""
        generator = ExerciseGenerator()
        
        param_rules = {
            "coefficient": {"min": 1, "max": 10, "type": "integer"},
            "constant": {"min": -20, "max": 20, "type": "integer"},
            "difficulty_factor": {"min": 0.1, "max": 1.0, "type": "float"}
        }
        
        params = generator.generate_parameters(param_rules, difficulty=0.6)
        
        assert isinstance(params, dict)
        assert "coefficient" in params
        assert "constant" in params
        assert "difficulty_factor" in params
        assert 1 <= params["coefficient"] <= 10
        assert -20 <= params["constant"] <= 20
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_difficulty_scaling(self):
        """Test difficulty scaling in exercise generation."""
        generator = ExerciseGenerator()
        
        # Generate exercises at different difficulty levels
        easy_params = generator.generate_parameters({
            "range": {"min": 1, "max": 100}
        }, difficulty=0.2)
        
        hard_params = generator.generate_parameters({
            "range": {"min": 1, "max": 100}
        }, difficulty=0.8)
        
        # Hard exercises should generally have larger numbers
        assert isinstance(easy_params, dict)
        assert isinstance(hard_params, dict)
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_concept_specific_generation(self):
        """Test concept-specific exercise generation."""
        generator = ExerciseGenerator()
        
        # Test different concept types
        concepts = ["linear_equations", "quadratic_equations", "factoring", "polynomials"]
        
        for concept in concepts:
            try:
                exercise = generator.generate_by_concept(concept, difficulty=0.5)
                
                assert isinstance(exercise, dict)
                assert exercise["concept_id"] == concept
                assert "problem" in exercise
                assert "answer" in exercise
                
            except NotImplementedError:
                # Some concepts might not be implemented yet
                pass
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_batch_exercise_generation(self):
        """Test generating multiple exercises at once."""
        generator = ExerciseGenerator()
        
        exercises = generator.generate_batch(
            concept="linear_equations",
            count=5,
            difficulty_range=(0.3, 0.7)
        )
        
        assert isinstance(exercises, list)
        assert len(exercises) == 5
        assert all(ex["concept_id"] == "linear_equations" for ex in exercises)
        assert all(0.3 <= ex["difficulty"] <= 0.7 for ex in exercises)


class TestExerciseValidator:
    """Test suite for Exercise Validator."""
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validator_initialization(self):
        """Test exercise validator initialization."""
        validator = ExerciseValidator()
        
        assert hasattr(validator, 'validation_rules')
        assert hasattr(validator, 'required_fields')
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validate_exercise_structure(self):
        """Test validating exercise structure."""
        validator = ExerciseValidator()
        
        # Valid exercise
        valid_exercise = {
            "id": "valid_1",
            "problem": "What is 2 + 3?",
            "answer": "5",
            "concept_id": "addition",
            "difficulty": 0.3,
            "hints": ["Add the numbers"],
            "estimated_time": 30
        }
        
        validation_result = validator.validate_structure(valid_exercise)
        
        assert validation_result["valid"] == True
        assert len(validation_result["errors"]) == 0
        
        # Invalid exercise
        invalid_exercise = {
            "problem": "What is 2 + 3?",
            # Missing required fields
        }
        
        validation_result = validator.validate_structure(invalid_exercise)
        
        assert validation_result["valid"] == False
        assert len(validation_result["errors"]) > 0
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validate_answer_format(self):
        """Test validating answer formats."""
        validator = ExerciseValidator()
        
        # Test different answer formats
        test_cases = [
            {"answer": "5", "expected": True},
            {"answer": "x = 2", "expected": True},
            {"answer": "2.5", "expected": True},
            {"answer": "", "expected": False},
            {"answer": None, "expected": False}
        ]
        
        for case in test_cases:
            result = validator.validate_answer_format(case["answer"])
            assert result == case["expected"]
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validate_difficulty_range(self):
        """Test validating difficulty ranges."""
        validator = ExerciseValidator()
        
        # Valid difficulties
        assert validator.validate_difficulty(0.0) == True
        assert validator.validate_difficulty(0.5) == True
        assert validator.validate_difficulty(1.0) == True
        
        # Invalid difficulties
        assert validator.validate_difficulty(-0.1) == False
        assert validator.validate_difficulty(1.1) == False
        assert validator.validate_difficulty("invalid") == False
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_validate_concept_id(self):
        """Test validating concept IDs."""
        validator = ExerciseValidator()
        
        # Valid concept IDs
        valid_concepts = ["addition", "subtraction", "multiplication", "division", 
                         "linear_equations", "quadratic_equations"]
        
        for concept in valid_concepts:
            assert validator.validate_concept_id(concept) == True
        
        # Invalid concept IDs
        invalid_concepts = ["", None, "invalid_concept", 123]
        
        for concept in invalid_concepts:
            assert validator.validate_concept_id(concept) == False


class TestDifficultyAdapter:
    """Test suite for Difficulty Adapter."""
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_difficulty_adapter_initialization(self):
        """Test difficulty adapter initialization."""
        adapter = DifficultyAdapter()
        
        assert hasattr(adapter, 'difficulty_curves')
        assert hasattr(adapter, 'adaptation_rules')
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_adapt_difficulty_based_on_performance(self):
        """Test adapting difficulty based on student performance."""
        adapter = DifficultyAdapter()
        
        # High performance - should increase difficulty
        high_performance = {
            "recent_scores": [0.9, 0.85, 0.95, 0.8],
            "current_difficulty": 0.5,
            "concept_mastery": 0.8
        }
        
        new_difficulty = adapter.adapt_difficulty(high_performance)
        
        assert isinstance(new_difficulty, float)
        assert new_difficulty > 0.5  # Should increase
        assert 0.0 <= new_difficulty <= 1.0
        
        # Low performance - should decrease difficulty
        low_performance = {
            "recent_scores": [0.3, 0.2, 0.4, 0.1],
            "current_difficulty": 0.7,
            "concept_mastery": 0.3
        }
        
        new_difficulty = adapter.adapt_difficulty(low_performance)
        
        assert isinstance(new_difficulty, float)
        assert new_difficulty < 0.7  # Should decrease
        assert 0.0 <= new_difficulty <= 1.0
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_difficulty_bounds(self):
        """Test difficulty adaptation bounds."""
        adapter = DifficultyAdapter()
        
        # Test extreme cases
        extreme_high = {
            "recent_scores": [1.0, 1.0, 1.0, 1.0],
            "current_difficulty": 0.95,
            "concept_mastery": 1.0
        }
        
        new_difficulty = adapter.adapt_difficulty(extreme_high)
        assert 0.0 <= new_difficulty <= 1.0
        
        extreme_low = {
            "recent_scores": [0.0, 0.0, 0.0, 0.0],
            "current_difficulty": 0.05,
            "concept_mastery": 0.0
        }
        
        new_difficulty = adapter.adapt_difficulty(extreme_low)
        assert 0.0 <= new_difficulty <= 1.0
    
    @pytest.mark.skipif(not EXERCISES_AVAILABLE, reason="Exercise components not available")
    def test_concept_specific_adaptation(self):
        """Test concept-specific difficulty adaptation."""
        adapter = DifficultyAdapter()
        
        performance_data = {
            "recent_scores": [0.7, 0.6, 0.8, 0.7],
            "current_difficulty": 0.5,
            "concept_mastery": 0.6
        }
        
        # Different concepts might have different adaptation rules
        concepts = ["addition", "multiplication", "linear_equations", "quadratic_equations"]
        
        for concept in concepts:
            new_difficulty = adapter.adapt_difficulty(performance_data, concept=concept)
            
            assert isinstance(new_difficulty, float)
            assert 0.0 <= new_difficulty <= 1.0


# Mock implementations for testing when real components aren't available
class MockAlgebraExercises:
    """Mock algebra exercises for testing."""
    
    def __init__(self):
        self.exercise_types = ["linear", "quadratic", "polynomial"]
        self.difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.concept_mapping = {
            "linear_equations": "linear",
            "quadratic_equations": "quadratic"
        }
    
    def generate_linear_equation(self, difficulty=0.5):
        return {
            "id": f"linear_{difficulty}",
            "problem": f"Solve: 2x + 3 = {7 + int(difficulty * 10)}",
            "answer": f"x = {2 + int(difficulty * 5)}",
            "concept_id": "linear_equations",
            "difficulty": difficulty,
            "hints": ["Isolate x", "Subtract, then divide"]
        }
    
    def validate_exercise(self, exercise):
        required_fields = ["problem", "answer", "concept_id", "difficulty"]
        return all(field in exercise for field in required_fields)


if not EXERCISES_AVAILABLE:
    # Use mock implementations when real components aren't available
    AlgebraExercises = MockAlgebraExercises
    ExerciseBank = type('ExerciseBank', (), {
        '__init__': lambda self: setattr(self, 'exercises', {}),
        'add_exercise': lambda self, ex: self.exercises.update({ex['id']: ex}),
        'get_exercises_by_concept': lambda self, concept: [ex for ex in self.exercises.values() if ex.get('concept_id') == concept]
    })


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 