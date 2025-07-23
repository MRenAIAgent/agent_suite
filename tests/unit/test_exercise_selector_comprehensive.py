#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Exercise Selector.

This module tests all Exercise Selector functionality including selection algorithms,
difficulty adaptation, and intelligent exercise recommendation logic.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.exercises.exercise_selector import ExerciseSelector
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.algebra_graph import AlgebraGraph


class TestExerciseSelectorCore:
    """Test core Exercise Selector functionality."""

    @pytest.fixture
    def sample_exercises(self):
        """Create sample exercises with varied difficulties and concepts."""
        return [
            Exercise("ex1", "Easy Addition", "2 + 3", "addition", 1.0, ["basic"]),
            Exercise("ex2", "Medium Addition", "15 + 27", "addition", 2.0, ["intermediate"]),
            Exercise("ex3", "Hard Addition", "125 + 376", "addition", 3.0, ["advanced"]),
            Exercise("ex4", "Easy Multiplication", "3 × 4", "multiplication", 1.5, ["basic"]),
            Exercise("ex5", "Medium Multiplication", "12 × 15", "multiplication", 2.5, ["intermediate"]),
            Exercise("ex6", "Variables Intro", "x + 5 = 10", "variables", 2.0, ["algebra"]),
            Exercise("ex7", "Variables Advanced", "2x + 3y = 12", "variables", 4.0, ["algebra", "advanced"]),
            Exercise("ex8", "Fractions Basic", "1/2 + 1/4", "fractions", 2.5, ["fractions"]),
        ]

    @pytest.fixture
    def exercise_bank(self, sample_exercises):
        """Create exercise bank with sample exercises."""
        bank = ExerciseBank()
        for exercise in sample_exercises:
            bank.add_exercise(exercise)
        return bank

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph."""
        kg = Mock(spec=AlgebraGraph)
        kg.get_concept.return_value = Mock(id="addition", prerequisites=[])
        kg.get_prerequisites.return_value = []
        kg.get_related_concepts.return_value = ["subtraction", "multiplication"]
        kg.find_learning_path.return_value = ["addition", "multiplication", "variables"]
        return kg

    @pytest.fixture
    def sample_student(self):
        """Create sample student with learning progress."""
        student = LearningGraph("test_student", "Test Student")
        student.set_mastery("addition", 0.8, 0.9)      # High mastery
        student.set_mastery("multiplication", 0.6, 0.7)  # Medium mastery
        student.set_mastery("variables", 0.3, 0.4)      # Low mastery
        student.set_mastery("fractions", 0.1, 0.2)      # Very low mastery
        return student

    @pytest.fixture
    def exercise_selector(self, exercise_bank, mock_knowledge_graph):
        """Create exercise selector with dependencies."""
        return ExerciseSelector(exercise_bank, mock_knowledge_graph)

    def test_exercise_selector_initialization(self, exercise_bank, mock_knowledge_graph):
        """Test ExerciseSelector initialization."""
        selector = ExerciseSelector(exercise_bank, mock_knowledge_graph)
        
        assert selector.exercise_bank == exercise_bank
        assert selector.knowledge_graph == mock_knowledge_graph
        assert hasattr(selector, 'selection_history')

    def test_select_by_difficulty_exact_match(self, exercise_selector):
        """Test selecting exercises by exact difficulty."""
        exercises = exercise_selector.select_by_difficulty(
            difficulty=2.0, count=2, tolerance=0.1
        )
        
        # Should find exercises close to difficulty 2.0
        assert len(exercises) <= 2
        for ex in exercises:
            assert abs(ex.difficulty - 2.0) <= 0.1

    def test_select_by_difficulty_with_tolerance(self, exercise_selector):
        """Test selecting exercises with difficulty tolerance."""
        exercises = exercise_selector.select_by_difficulty(
            difficulty=2.0, count=5, tolerance=1.0
        )
        
        # Should find exercises within tolerance range
        assert len(exercises) > 0
        for ex in exercises:
            assert abs(ex.difficulty - 2.0) <= 1.0

    def test_select_by_difficulty_no_matches(self, exercise_selector):
        """Test selecting exercises with no difficulty matches."""
        exercises = exercise_selector.select_by_difficulty(
            difficulty=10.0, count=5, tolerance=0.1
        )
        
        # Should return empty list
        assert exercises == []

    def test_select_adaptive_for_high_mastery_student(self, exercise_selector, sample_student):
        """Test adaptive selection for student with high mastery."""
        # Student has high mastery in addition (0.8)
        exercises = exercise_selector.select_adaptive(
            student=sample_student,
            concept_id="addition",
            count=3
        )
        
        # Should select more challenging addition exercises
        assert len(exercises) > 0
        addition_exercises = [ex for ex in exercises if ex.concept_id == "addition"]
        if addition_exercises:
            # Should tend toward higher difficulty for high mastery
            avg_difficulty = sum(ex.difficulty for ex in addition_exercises) / len(addition_exercises)
            assert avg_difficulty >= 1.5  # Should be above beginner level

    def test_select_adaptive_for_low_mastery_student(self, exercise_selector, sample_student):
        """Test adaptive selection for student with low mastery."""
        # Student has low mastery in variables (0.3)
        exercises = exercise_selector.select_adaptive(
            student=sample_student,
            concept_id="variables",
            count=3
        )
        
        # Should select easier variable exercises
        assert len(exercises) > 0
        variable_exercises = [ex for ex in exercises if ex.concept_id == "variables"]
        if variable_exercises:
            # Should include easier exercises for low mastery
            min_difficulty = min(ex.difficulty for ex in variable_exercises)
            assert min_difficulty <= 3.0  # Should include easier options

    def test_select_progressive_difficulty(self, exercise_selector, sample_student):
        """Test selecting exercises with progressive difficulty."""
        exercises = exercise_selector.select_progressive(
            student=sample_student,
            concept_id="addition",
            count=3,
            start_difficulty=1.0,
            end_difficulty=3.0
        )
        
        # Should return exercises with increasing difficulty
        assert len(exercises) > 0
        if len(exercises) > 1:
            difficulties = [ex.difficulty for ex in exercises]
            # Should be in increasing order (or at least not decreasing)
            for i in range(1, len(difficulties)):
                assert difficulties[i] >= difficulties[i-1]

    def test_select_by_concept_priority(self, exercise_selector, sample_student):
        """Test selecting exercises based on concept priority."""
        concept_priorities = {
            "variables": 0.9,      # High priority (low mastery)
            "fractions": 0.8,      # High priority (very low mastery)
            "multiplication": 0.5,  # Medium priority
            "addition": 0.2        # Low priority (high mastery)
        }
        
        exercises = exercise_selector.select_by_priority(
            student=sample_student,
            concept_priorities=concept_priorities,
            count=4
        )
        
        # Should prioritize concepts with high priority scores
        assert len(exercises) > 0
        
        # Count exercises by concept
        concept_counts = {}
        for ex in exercises:
            concept_counts[ex.concept_id] = concept_counts.get(ex.concept_id, 0) + 1
        
        # Variables and fractions should have more exercises than addition
        variables_count = concept_counts.get("variables", 0)
        fractions_count = concept_counts.get("fractions", 0)
        addition_count = concept_counts.get("addition", 0)
        
        assert variables_count + fractions_count >= addition_count

    def test_select_spaced_repetition(self, exercise_selector, sample_student):
        """Test spaced repetition exercise selection."""
        # Simulate previous attempts
        exercise_history = {
            "ex1": {"last_attempt": "2024-01-01", "success_rate": 0.8, "attempts": 3},
            "ex2": {"last_attempt": "2024-01-02", "success_rate": 0.6, "attempts": 2},
            "ex3": {"last_attempt": "2024-01-05", "success_rate": 0.4, "attempts": 1},
        }
        
        exercises = exercise_selector.select_spaced_repetition(
            student=sample_student,
            exercise_history=exercise_history,
            count=2
        )
        
        # Should prioritize exercises that need reinforcement
        assert isinstance(exercises, list)
        # Exercises with lower success rates should be prioritized

    def test_select_diverse_concepts(self, exercise_selector, sample_student):
        """Test selecting diverse exercises across multiple concepts."""
        exercises = exercise_selector.select_diverse(
            student=sample_student,
            count=6,
            max_per_concept=2
        )
        
        # Should include exercises from multiple concepts
        assert len(exercises) > 0
        
        concept_counts = {}
        for ex in exercises:
            concept_counts[ex.concept_id] = concept_counts.get(ex.concept_id, 0) + 1
        
        # Should respect max_per_concept limit
        for count in concept_counts.values():
            assert count <= 2
        
        # Should include multiple concepts
        assert len(concept_counts) >= 2

    def test_select_prerequisite_aware(self, exercise_selector, sample_student, mock_knowledge_graph):
        """Test prerequisite-aware exercise selection."""
        # Mock prerequisite relationships
        mock_knowledge_graph.get_prerequisites.side_effect = lambda concept: {
            "variables": ["addition", "multiplication"],
            "fractions": ["addition"],
            "addition": [],
            "multiplication": ["addition"]
        }.get(concept, [])
        
        exercises = exercise_selector.select_prerequisite_aware(
            student=sample_student,
            target_concept="variables",
            count=4
        )
        
        # Should include prerequisite concepts if student needs them
        assert len(exercises) > 0
        
        # Should include exercises for prerequisites if mastery is low
        concept_ids = {ex.concept_id for ex in exercises}
        
        # Since student has low variables mastery (0.3), should include prerequisites
        assert len(concept_ids) >= 1

    def test_calculate_exercise_score_high_mastery(self, exercise_selector, sample_student):
        """Test exercise scoring for high mastery student."""
        easy_exercise = Exercise("ex1", "Easy", "content", "addition", 1.0, [])
        hard_exercise = Exercise("ex2", "Hard", "content", "addition", 3.0, [])
        
        # For high mastery student (addition: 0.8), harder exercise should score higher
        easy_score = exercise_selector.calculate_exercise_score(easy_exercise, sample_student)
        hard_score = exercise_selector.calculate_exercise_score(hard_exercise, sample_student)
        
        # Hard exercise should be more suitable for high mastery student
        assert hard_score > easy_score

    def test_calculate_exercise_score_low_mastery(self, exercise_selector, sample_student):
        """Test exercise scoring for low mastery student."""
        easy_exercise = Exercise("ex1", "Easy", "content", "variables", 2.0, [])
        hard_exercise = Exercise("ex2", "Hard", "content", "variables", 4.0, [])
        
        # For low mastery student (variables: 0.3), easier exercise should score higher
        easy_score = exercise_selector.calculate_exercise_score(easy_exercise, sample_student)
        hard_score = exercise_selector.calculate_exercise_score(hard_exercise, sample_student)
        
        # Easy exercise should be more suitable for low mastery student
        assert easy_score > hard_score

    def test_filter_by_success_rate_threshold(self, exercise_selector, sample_student):
        """Test filtering exercises by estimated success rate."""
        exercises = exercise_selector.filter_by_success_rate(
            student=sample_student,
            exercises=exercise_selector.exercise_bank.get_all_exercises(),
            min_success_rate=0.4,
            max_success_rate=0.8
        )
        
        # Should return exercises within the success rate range
        assert isinstance(exercises, list)
        
        # Verify success rates are within range
        for ex in exercises:
            success_rate = exercise_selector.estimate_success_rate(ex, sample_student)
            assert 0.4 <= success_rate <= 0.8

    def test_select_session_exercises(self, exercise_selector, sample_student):
        """Test selecting exercises for a complete session."""
        session_exercises = exercise_selector.select_session_exercises(
            student=sample_student,
            session_type="practice",
            duration_minutes=15,
            target_concepts=["addition", "multiplication"]
        )
        
        # Should return appropriate exercises for session
        assert len(session_exercises) > 0
        assert len(session_exercises) <= 10  # Reasonable for 15-minute session
        
        # Should focus on target concepts
        concept_ids = {ex.concept_id for ex in session_exercises}
        assert concept_ids.intersection({"addition", "multiplication"})


class TestExerciseSelectorAdvanced:
    """Test advanced Exercise Selector functionality."""

    @pytest.fixture
    def exercise_selector_with_history(self, exercise_selector):
        """Create exercise selector with selection history."""
        # Add some selection history
        exercise_selector.selection_history = {
            "test_student": [
                {"exercise_id": "ex1", "timestamp": "2024-01-01", "success": True},
                {"exercise_id": "ex2", "timestamp": "2024-01-02", "success": False},
                {"exercise_id": "ex3", "timestamp": "2024-01-03", "success": True},
            ]
        }
        return exercise_selector

    def test_avoid_recent_exercises(self, exercise_selector_with_history, sample_student):
        """Test avoiding recently attempted exercises."""
        exercises = exercise_selector_with_history.select_avoiding_recent(
            student=sample_student,
            concept_id="addition",
            count=3,
            avoid_recent_hours=24
        )
        
        # Should avoid exercises attempted recently
        recent_exercise_ids = {"ex1", "ex2", "ex3"}  # From history
        selected_ids = {ex.id for ex in exercises}
        
        # Should minimize overlap with recent exercises
        overlap = selected_ids.intersection(recent_exercise_ids)
        assert len(overlap) <= len(selected_ids) // 2  # At most half should be recent

    def test_personalized_difficulty_adjustment(self, exercise_selector, sample_student):
        """Test personalized difficulty adjustment based on performance."""
        # Simulate performance data
        performance_data = {
            "addition": {"avg_score": 0.9, "attempts": 10},
            "multiplication": {"avg_score": 0.7, "attempts": 8},
            "variables": {"avg_score": 0.4, "attempts": 5}
        }
        
        # Test difficulty adjustment for each concept
        addition_difficulty = exercise_selector.adjust_difficulty_for_student(
            base_difficulty=2.0,
            concept_id="addition",
            student=sample_student,
            performance_data=performance_data
        )
        
        variables_difficulty = exercise_selector.adjust_difficulty_for_student(
            base_difficulty=2.0,
            concept_id="variables",
            student=sample_student,
            performance_data=performance_data
        )
        
        # High-performing concept should get harder exercises
        # Low-performing concept should get easier exercises
        assert addition_difficulty >= variables_difficulty

    def test_multi_objective_selection(self, exercise_selector, sample_student):
        """Test multi-objective exercise selection optimization."""
        objectives = {
            "skill_development": 0.4,  # Focus on improving weak skills
            "engagement": 0.3,         # Keep student engaged
            "difficulty_progression": 0.2,  # Gradual difficulty increase
            "concept_diversity": 0.1   # Cover multiple concepts
        }
        
        exercises = exercise_selector.select_multi_objective(
            student=sample_student,
            objectives=objectives,
            count=5
        )
        
        # Should return exercises optimized for multiple objectives
        assert len(exercises) > 0
        assert len(exercises) <= 5
        
        # Should balance different objectives
        difficulties = [ex.difficulty for ex in exercises]
        concepts = {ex.concept_id for ex in exercises}
        
        # Should have reasonable difficulty spread
        if len(difficulties) > 1:
            difficulty_range = max(difficulties) - min(difficulties)
            assert difficulty_range > 0  # Some variety in difficulty
        
        # Should cover multiple concepts for diversity
        assert len(concepts) >= min(2, len(exercises))


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 