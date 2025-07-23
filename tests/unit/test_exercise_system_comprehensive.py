#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Exercise System.

Tests all exercise system functionality with proper mocking and isolation.
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.exercises.exercise_system import ExerciseSystem
from math_learning.exercises.exercise import Exercise
from math_learning.learning_graph.user_model import LearningGraph


class TestExerciseSystemCore:
    """Test core Exercise System functionality."""

    @pytest.fixture
    def mock_exercise_bank(self):
        """Create a mock exercise bank."""
        bank = Mock()
        exercises = [
            Exercise("ex1", "Addition", "2+3", "addition", 1.0, ["basic"]),
            Exercise("ex2", "Multiplication", "4×5", "multiplication", 2.0, ["basic"]),
        ]
        bank.get_exercises_by_concept.return_value = exercises
        bank.get_all_exercises.return_value = exercises
        return bank

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create a mock knowledge graph."""
        kg = Mock()
        kg.get_concept.return_value = Mock(id="addition", name="Addition")
        kg.get_prerequisites.return_value = []
        return kg

    @pytest.fixture
    def sample_student(self):
        """Create a sample student."""
        student = LearningGraph("test_student", "Test Student")
        student.set_mastery("addition", 0.8, 0.9)
        student.set_mastery("multiplication", 0.6, 0.7)
        return student

    @pytest.fixture
    def exercise_system(self, mock_exercise_bank, mock_knowledge_graph):
        """Create exercise system with mocks."""
        return ExerciseSystem(mock_exercise_bank, mock_knowledge_graph)

    def test_exercise_system_initialization(self, mock_exercise_bank, mock_knowledge_graph):
        """Test ExerciseSystem initialization."""
        system = ExerciseSystem(mock_exercise_bank, mock_knowledge_graph)
        assert system.exercise_bank == mock_exercise_bank
        assert system.knowledge_graph == mock_knowledge_graph

    def test_get_exercises_by_concept_difficulty(self, exercise_system, mock_exercise_bank):
        """Test getting exercises by concept and difficulty."""
        exercises = exercise_system.get_exercises_by_concept_difficulty("addition", 1.5, count=2)
        
        # Should call the exercise bank
        mock_exercise_bank.get_exercises_by_concept.assert_called_with("addition")
        
        # Should return exercises
        assert isinstance(exercises, list)
        assert len(exercises) <= 2

    def test_get_adaptive_exercises(self, exercise_system, sample_student):
        """Test adaptive exercise selection."""
        recommendations = exercise_system.get_adaptive_exercises(sample_student, count=2)
        
        # Should return recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 2

    def test_get_ai_optimized_exercises(self, exercise_system, sample_student):
        """Test AI-optimized exercise selection."""
        session = exercise_system.get_ai_optimized_exercises(
            sample_student, "practice", 15
        )
        
        # Should return a session
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'exercises')
        assert hasattr(session, 'session_type')
        assert session.session_type == "practice"

    def test_create_learning_path_session(self, exercise_system, sample_student):
        """Test learning path session creation."""
        session = exercise_system.create_learning_path_session(
            sample_student, ["addition", "multiplication"], 20
        )
        
        # Should return a valid session
        assert hasattr(session, 'session_id')
        assert hasattr(session, 'exercises')

    def test_get_session_analytics_valid_session(self, exercise_system, sample_student):
        """Test getting analytics for valid session."""
        # First create a session
        session = exercise_system.get_ai_optimized_exercises(sample_student, "practice", 10)
        
        # Then get analytics
        analytics = exercise_system.get_session_analytics(session.session_id)
        
        assert isinstance(analytics, dict)
        assert 'session_id' in analytics

    def test_get_session_analytics_invalid_session(self, exercise_system):
        """Test getting analytics for invalid session."""
        analytics = exercise_system.get_session_analytics("nonexistent")
        
        assert isinstance(analytics, dict)
        assert 'error' in analytics

    def test_estimate_success_rate(self, exercise_system):
        """Test success rate estimation."""
        exercise = Exercise("ex1", "Test", "content", "addition", 2.0, [])
        
        # Test with different mastery levels
        success_rate = exercise_system._estimate_success_rate(exercise, 0.8, 0.9)
        assert 0.0 <= success_rate <= 1.0
        
        # Higher mastery should generally give higher success rate for same difficulty
        high_mastery_rate = exercise_system._estimate_success_rate(exercise, 0.9, 0.9)
        low_mastery_rate = exercise_system._estimate_success_rate(exercise, 0.3, 0.4)
        
        assert high_mastery_rate >= low_mastery_rate


class TestExerciseSystemIntegration:
    """Test Exercise System integration scenarios."""

    @pytest.fixture
    def real_exercise_system(self):
        """Create exercise system with real components for integration testing."""
        try:
            from math_learning.algebra_learning import AlgebraLearningSystem
            algebra_system = AlgebraLearningSystem()
            return ExerciseSystem(
                algebra_system.exercise_bank,
                algebra_system.knowledge_graph
            )
        except ImportError:
            pytest.skip("Real components not available")

    def test_integration_with_real_components(self, real_exercise_system):
        """Test integration with real algebra components."""
        student = LearningGraph("integration_test", "Integration Student")
        student.set_mastery("addition", 0.7, 0.8)
        
        # Test basic functionality
        exercises = real_exercise_system.get_exercises_by_concept_difficulty(
            "addition", 2.0, count=3
        )
        
        assert isinstance(exercises, list)
        assert len(exercises) <= 3

    def test_session_workflow(self, real_exercise_system):
        """Test complete session workflow."""
        student = LearningGraph("workflow_test", "Workflow Student")
        student.set_mastery("addition", 0.5, 0.6)
        
        # Create session
        session = real_exercise_system.get_ai_optimized_exercises(
            student, "practice", 10
        )
        
        # Get analytics
        analytics = real_exercise_system.get_session_analytics(session.session_id)
        
        assert 'session_id' in analytics
        assert analytics['session_id'] == session.session_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 