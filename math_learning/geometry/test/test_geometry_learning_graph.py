"""
Unit tests for GeometryLearningGraph class.

Tests inheritance from base LearningGraph and geometry-specific functionality.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Add the necessary paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from geometry.src.geometry_learning_graph import GeometryLearningGraph
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometry_learning_graph", 
        os.path.join(os.path.dirname(__file__), '..', 'src', 'geometry_learning_graph.py')
    )
    geometry_lg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_lg_module)
    GeometryLearningGraph = geometry_lg_module.GeometryLearningGraph


class TestGeometryLearningGraph:
    """Test cases for GeometryLearningGraph."""

    @pytest.mark.unit
    def test_inheritance(self):
        """Test that GeometryLearningGraph properly inherits from LearningGraph."""
        lg = GeometryLearningGraph("test_student")
        
        # Check that it has inherited methods
        assert hasattr(lg, 'get_mastery_level')
        assert hasattr(lg, 'update_mastery')
        assert hasattr(lg, 'record_exercise_attempt')
        assert hasattr(lg, 'get_learning_patterns')

    @pytest.mark.unit
    def test_initialization(self):
        """Test GeometryLearningGraph initialization."""
        student_id = "test_student_001"
        lg = GeometryLearningGraph(student_id)
        
        # Check basic initialization
        assert lg.student_id == student_id
        
        # Check geometry-specific attributes
        assert hasattr(lg, 'visual_learning_preference')
        assert hasattr(lg, 'spatial_reasoning_score')
        assert hasattr(lg, 'construction_familiarity')
        
        # Check default values
        assert 0.0 <= lg.visual_learning_preference <= 1.0
        assert 0.0 <= lg.spatial_reasoning_score <= 1.0
        assert 0.0 <= lg.construction_familiarity <= 1.0

    @pytest.mark.unit
    def test_visual_learning_preference_tracking(self, sample_student_data):
        """Test visual learning preference tracking."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Test setting visual learning preference
        lg.set_visual_learning_preference(0.8)
        assert lg.visual_learning_preference == 0.8
        
        # Test getting visual learning preference
        preference = lg.get_visual_learning_preference()
        assert preference == 0.8
        
        # Test boundary values
        lg.set_visual_learning_preference(0.0)
        assert lg.visual_learning_preference == 0.0
        
        lg.set_visual_learning_preference(1.0)
        assert lg.visual_learning_preference == 1.0

    @pytest.mark.unit
    def test_spatial_reasoning_assessment(self, sample_student_data):
        """Test spatial reasoning score assessment."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Test setting spatial reasoning score
        lg.set_spatial_reasoning_score(0.7)
        assert lg.spatial_reasoning_score == 0.7
        
        # Test getting spatial reasoning score
        score = lg.get_spatial_reasoning_score()
        assert score == 0.7
        
        # Test assessment update based on exercise performance
        lg.update_spatial_reasoning_from_exercise("geo_004", True, 3)
        # Score should be updated (exact value depends on implementation)
        assert isinstance(lg.spatial_reasoning_score, float)

    @pytest.mark.unit
    def test_construction_familiarity_tracking(self, sample_student_data):
        """Test construction tool familiarity tracking."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Test setting construction familiarity
        lg.set_construction_familiarity(0.6)
        assert lg.construction_familiarity == 0.6
        
        # Test getting construction familiarity
        familiarity = lg.get_construction_familiarity()
        assert familiarity == 0.6
        
        # Test updating based on construction exercise performance
        lg.update_construction_familiarity("geo_004", True, construction_used=True)
        # Familiarity should be updated
        assert isinstance(lg.construction_familiarity, float)

    @pytest.mark.unit
    def test_geometry_exercise_recording(self, sample_student_data):
        """Test recording of geometry exercises with metadata."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Record a geometry exercise
        exercise_data = {
            "exercise_id": "geo_001",
            "concept": "basic_shapes",
            "correct": True,
            "time_taken": 30,
            "attempts": 1,
            "visual_aids_used": True,
            "construction_used": False,
            "spatial_reasoning_level": 1
        }
        
        lg.record_geometry_exercise(exercise_data)
        
        # Check that exercise was recorded
        history = lg.get_exercise_history()
        assert len(history) >= 1
        
        # Check geometry-specific metadata was stored
        recorded_exercise = history[-1]  # Get last recorded exercise
        assert recorded_exercise.get("visual_aids_used") == True
        assert recorded_exercise.get("construction_used") == False
        assert recorded_exercise.get("spatial_reasoning_level") == 1

    @pytest.mark.unit
    def test_visual_learning_analysis(self, sample_student_data):
        """Test analysis of visual learning effectiveness."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add some exercise history with visual aids
        exercises = [
            {"exercise_id": "geo_001", "correct": True, "visual_aids_used": True},
            {"exercise_id": "geo_002", "correct": False, "visual_aids_used": True},
            {"exercise_id": "geo_003", "correct": True, "visual_aids_used": False},
            {"exercise_id": "geo_004", "correct": False, "visual_aids_used": False}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Analyze visual learning effectiveness
        analysis = lg.analyze_visual_learning_effectiveness()
        
        assert isinstance(analysis, dict)
        assert "visual_aids_success_rate" in analysis
        assert "non_visual_success_rate" in analysis
        assert "visual_preference_recommendation" in analysis

    @pytest.mark.unit
    def test_construction_performance_analysis(self, sample_student_data):
        """Test analysis of construction exercise performance."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add construction exercises
        construction_exercises = [
            {"exercise_id": "geo_004", "correct": True, "construction_used": True},
            {"exercise_id": "geo_005", "correct": False, "construction_used": True},
            {"exercise_id": "geo_006", "correct": True, "construction_used": False}
        ]
        
        for exercise in construction_exercises:
            lg.record_geometry_exercise(exercise)
        
        # Analyze construction performance
        analysis = lg.analyze_construction_performance()
        
        assert isinstance(analysis, dict)
        assert "construction_success_rate" in analysis
        assert "construction_familiarity_level" in analysis

    @pytest.mark.unit
    def test_learning_style_identification(self, sample_student_data):
        """Test identification of learning style preferences."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Set some preferences
        lg.set_visual_learning_preference(0.8)
        lg.set_spatial_reasoning_score(0.7)
        lg.set_construction_familiarity(0.6)
        
        # Identify learning style
        learning_style = lg.identify_learning_style()
        
        assert learning_style in ["visual", "kinesthetic", "analytical", "balanced"]

    @pytest.mark.unit
    def test_performance_by_exercise_type(self, sample_student_data):
        """Test performance analysis by exercise type."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add exercises of different types
        exercises = [
            {"exercise_id": "geo_001", "correct": True, "format_type": "multiple_choice"},
            {"exercise_id": "geo_002", "correct": False, "format_type": "multiple_choice"},
            {"exercise_id": "geo_003", "correct": True, "format_type": "calculation"},
            {"exercise_id": "geo_004", "correct": True, "format_type": "calculation"},
            {"exercise_id": "geo_005", "correct": False, "format_type": "construction"}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Analyze performance by type
        performance = lg.get_performance_by_exercise_type()
        
        assert isinstance(performance, dict)
        assert "multiple_choice" in performance
        assert "calculation" in performance

    @pytest.mark.unit
    def test_spatial_reasoning_progression(self, sample_student_data):
        """Test tracking of spatial reasoning progression."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add exercises with different spatial reasoning levels
        exercises = [
            {"exercise_id": "geo_001", "correct": True, "spatial_reasoning_level": 1},
            {"exercise_id": "geo_002", "correct": True, "spatial_reasoning_level": 2},
            {"exercise_id": "geo_003", "correct": False, "spatial_reasoning_level": 3},
            {"exercise_id": "geo_004", "correct": True, "spatial_reasoning_level": 2}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Get spatial reasoning progression
        progression = lg.get_spatial_reasoning_progression()
        
        assert isinstance(progression, dict)
        assert "current_level" in progression
        assert "progression_trend" in progression

    @pytest.mark.unit
    def test_geometry_insights_generation(self, sample_student_data):
        """Test generation of geometry-specific learning insights."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Set up some learning data
        lg.set_visual_learning_preference(0.8)
        lg.set_spatial_reasoning_score(0.6)
        lg.set_construction_familiarity(0.4)
        
        # Add some exercise history
        exercises = [
            {"exercise_id": "geo_001", "correct": True, "concept": "basic_shapes"},
            {"exercise_id": "geo_002", "correct": False, "concept": "angles"},
            {"exercise_id": "geo_003", "correct": True, "concept": "area_calculation"}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Generate insights
        insights = lg.generate_geometry_insights()
        
        assert isinstance(insights, dict)
        assert "learning_strengths" in insights
        assert "improvement_areas" in insights
        assert "recommended_strategies" in insights

    @pytest.mark.unit
    def test_error_handling_invalid_values(self, sample_student_data):
        """Test error handling for invalid input values."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Test invalid preference values
        with pytest.raises(ValueError):
            lg.set_visual_learning_preference(-0.1)
        
        with pytest.raises(ValueError):
            lg.set_visual_learning_preference(1.1)
        
        # Test invalid spatial reasoning score
        with pytest.raises(ValueError):
            lg.set_spatial_reasoning_score(-0.1)
        
        with pytest.raises(ValueError):
            lg.set_spatial_reasoning_score(1.1)

    @pytest.mark.unit
    def test_learning_velocity_calculation(self, sample_student_data):
        """Test calculation of learning velocity for geometry concepts."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add exercises over time to calculate velocity
        import time
        exercises = [
            {"exercise_id": "geo_001", "correct": True, "concept": "basic_shapes", "timestamp": "2024-01-01T10:00:00"},
            {"exercise_id": "geo_002", "correct": True, "concept": "basic_shapes", "timestamp": "2024-01-01T10:05:00"},
            {"exercise_id": "geo_003", "correct": True, "concept": "angles", "timestamp": "2024-01-01T10:10:00"}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Calculate learning velocity
        velocity = lg.calculate_geometry_learning_velocity()
        
        assert isinstance(velocity, (int, float))
        assert velocity >= 0

    @pytest.mark.integration
    def test_complete_learning_session(self, sample_student_data):
        """Test a complete learning session with multiple geometry exercises."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Initialize student preferences
        lg.set_visual_learning_preference(0.7)
        lg.set_spatial_reasoning_score(0.6)
        lg.set_construction_familiarity(0.5)
        
        # Simulate a learning session
        session_exercises = [
            {
                "exercise_id": "geo_001",
                "concept": "basic_shapes",
                "correct": True,
                "time_taken": 30,
                "visual_aids_used": True,
                "construction_used": False,
                "spatial_reasoning_level": 1
            },
            {
                "exercise_id": "geo_002",
                "concept": "angles",
                "correct": False,
                "time_taken": 120,
                "visual_aids_used": True,
                "construction_used": False,
                "spatial_reasoning_level": 2
            },
            {
                "exercise_id": "geo_003",
                "concept": "area_calculation",
                "correct": True,
                "time_taken": 45,
                "visual_aids_used": False,
                "construction_used": False,
                "spatial_reasoning_level": 2
            }
        ]
        
        # Record all exercises
        for exercise in session_exercises:
            lg.record_geometry_exercise(exercise)
        
        # Update learning metrics
        lg.update_spatial_reasoning_from_exercise("geo_002", False, 2)
        lg.update_construction_familiarity("geo_003", True, False)
        
        # Generate session summary
        summary = lg.generate_session_summary()
        
        assert isinstance(summary, dict)
        assert "exercises_completed" in summary
        assert "concepts_practiced" in summary
        assert "performance_summary" in summary

    @pytest.mark.slow
    def test_large_exercise_history_performance(self, sample_student_data):
        """Test performance with large exercise history."""
        lg = GeometryLearningGraph(sample_student_data["student_id"])
        
        # Add a large number of exercises
        for i in range(1000):
            exercise = {
                "exercise_id": f"geo_{i:04d}",
                "concept": f"concept_{i % 10}",
                "correct": i % 3 != 0,  # 2/3 success rate
                "time_taken": 30 + (i % 60),
                "visual_aids_used": i % 2 == 0,
                "construction_used": i % 5 == 0,
                "spatial_reasoning_level": (i % 3) + 1
            }
            lg.record_geometry_exercise(exercise)
        
        # Test that analysis still works efficiently
        insights = lg.generate_geometry_insights()
        performance = lg.get_performance_by_exercise_type()
        progression = lg.get_spatial_reasoning_progression()
        
        assert isinstance(insights, dict)
        assert isinstance(performance, dict)
        assert isinstance(progression, dict) 