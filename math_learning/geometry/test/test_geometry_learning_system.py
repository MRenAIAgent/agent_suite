"""
Unit and integration tests for GeometryLearningSystem class.

Tests the complete geometry learning system integration and coordination.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the necessary paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from geometry.src.geometry_learning_system import GeometryLearningSystem
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometry_learning_system", 
        os.path.join(os.path.dirname(__file__), '..', 'src', 'geometry_learning_system.py')
    )
    geometry_ls_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_ls_module)
    GeometryLearningSystem = geometry_ls_module.GeometryLearningSystem


class TestGeometryLearningSystem:
    """Test cases for GeometryLearningSystem."""

    @pytest.mark.unit
    def test_initialization(self, temp_data_dir, sample_geometry_exercises):
        """Test GeometryLearningSystem initialization."""
        # Create test exercise file
        exercise_file = os.path.join(temp_data_dir, "test_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        
        # Check that all components are initialized
        assert system.knowledge_graph is not None
        assert system.learning_graph is not None
        assert system.gap_analyzer is not None
        assert system.recommender is not None
        
        # Check system-specific attributes
        assert hasattr(system, 'exercise_file')
        assert hasattr(system, 'student_sessions')
        assert hasattr(system, 'system_analytics')

    @pytest.mark.unit
    def test_student_session_creation(self, geometry_learning_system):
        """Test creation of student learning sessions."""
        system = geometry_learning_system
        
        # Create a new student session
        student_id = "test_student_001"
        session = system.create_student_session(student_id)
        
        assert session is not None
        assert student_id in system.student_sessions
        
        # Check session structure
        assert 'learning_graph' in session
        assert 'gap_analyzer' in session
        assert 'recommender' in session
        assert 'session_start_time' in session

    @pytest.mark.unit
    def test_student_learning_graph_creation(self, geometry_learning_system):
        """Test creation of student-specific learning graph."""
        system = geometry_learning_system
        
        student_id = "test_student_002"
        learning_graph = system.create_student_learning_graph(student_id)
        
        assert learning_graph is not None
        assert learning_graph.student_id == student_id
        
        # Check geometry-specific attributes
        assert hasattr(learning_graph, 'visual_learning_preference')
        assert hasattr(learning_graph, 'spatial_reasoning_score')
        assert hasattr(learning_graph, 'construction_familiarity')

    @pytest.mark.unit
    def test_gap_analysis_integration(self, geometry_learning_system):
        """Test integration of gap analysis functionality."""
        system = geometry_learning_system
        
        student_id = "test_student_003"
        system.create_student_session(student_id)
        
        # Perform gap analysis
        gaps = system.analyze_student_gaps(student_id)
        
        assert isinstance(gaps, dict)
        assert 'knowledge_gaps' in gaps
        assert 'visual_gaps' in gaps
        assert 'spatial_reasoning_gaps' in gaps
        assert 'construction_gaps' in gaps

    @pytest.mark.unit
    def test_exercise_recommendation_integration(self, geometry_learning_system):
        """Test integration of exercise recommendation functionality."""
        system = geometry_learning_system
        
        student_id = "test_student_004"
        system.create_student_session(student_id)
        
        # Get exercise recommendations
        recommendations = system.recommend_exercises(student_id, count=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        
        # Check recommendation structure
        for rec in recommendations:
            assert isinstance(rec, dict)
            assert 'exercise_id' in rec
            assert 'confidence_score' in rec

    @pytest.mark.unit
    def test_learning_path_generation(self, geometry_learning_system):
        """Test generation of personalized learning paths."""
        system = geometry_learning_system
        
        student_id = "test_student_005"
        system.create_student_session(student_id)
        
        # Generate learning path
        learning_path = system.generate_learning_path(student_id, target_concept='pythagorean_theorem')
        
        assert isinstance(learning_path, list)
        
        # Check learning path structure
        for step in learning_path:
            assert isinstance(step, dict)
            assert 'concept' in step
            assert 'exercises' in step

    @pytest.mark.unit
    def test_exercise_recording(self, geometry_learning_system):
        """Test recording of student exercise attempts."""
        system = geometry_learning_system
        
        student_id = "test_student_006"
        system.create_student_session(student_id)
        
        # Record an exercise attempt
        exercise_data = {
            'exercise_id': 'geo_001',
            'concept': 'basic_shapes',
            'correct': True,
            'time_taken': 30,
            'visual_aids_used': True,
            'construction_used': False
        }
        
        result = system.record_exercise_attempt(student_id, exercise_data)
        
        assert result is True
        
        # Check that exercise was recorded in learning graph
        session = system.student_sessions[student_id]
        history = session['learning_graph'].get_exercise_history()
        assert len(history) >= 1

    @pytest.mark.unit
    def test_student_progress_tracking(self, geometry_learning_system):
        """Test tracking of student progress over time."""
        system = geometry_learning_system
        
        student_id = "test_student_007"
        system.create_student_session(student_id)
        
        # Record multiple exercises
        exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True},
            {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': False},
            {'exercise_id': 'geo_003', 'concept': 'area_calculation', 'correct': True}
        ]
        
        for exercise in exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Get progress summary
        progress = system.get_student_progress(student_id)
        
        assert isinstance(progress, dict)
        assert 'exercises_completed' in progress
        assert 'concepts_practiced' in progress
        assert 'overall_accuracy' in progress

    @pytest.mark.unit
    def test_learning_insights_generation(self, geometry_learning_system):
        """Test generation of learning insights."""
        system = geometry_learning_system
        
        student_id = "test_student_008"
        system.create_student_session(student_id)
        
        # Generate insights
        insights = system.generate_learning_insights(student_id)
        
        assert isinstance(insights, dict)
        assert 'learning_strengths' in insights
        assert 'improvement_areas' in insights
        assert 'recommended_strategies' in insights
        assert 'geometry_specific_insights' in insights

    @pytest.mark.unit
    def test_adaptive_difficulty_adjustment(self, geometry_learning_system):
        """Test adaptive difficulty adjustment based on performance."""
        system = geometry_learning_system
        
        student_id = "test_student_009"
        system.create_student_session(student_id)
        
        # Simulate consistent success - should increase difficulty
        for i in range(5):
            exercise = {
                'exercise_id': f'geo_{i:03d}',
                'concept': 'basic_shapes',
                'correct': True,
                'difficulty': 1
            }
            system.record_exercise_attempt(student_id, exercise)
        
        # Get next recommendations - should be harder
        recommendations = system.recommend_exercises(student_id, count=3)
        
        assert isinstance(recommendations, list)
        # Check if difficulty has been adjusted (implementation dependent)

    @pytest.mark.unit
    def test_visual_learning_support(self, geometry_learning_system):
        """Test visual learning support features."""
        system = geometry_learning_system
        
        student_id = "test_student_010"
        system.create_student_session(student_id)
        
        # Set high visual learning preference
        session = system.student_sessions[student_id]
        session['learning_graph'].set_visual_learning_preference(0.9)
        
        # Get visual learning recommendations
        visual_recs = system.get_visual_learning_recommendations(student_id)
        
        assert isinstance(visual_recs, list)
        
        # Check that recommendations include visual components
        for rec in visual_recs:
            assert isinstance(rec, dict)
            assert 'visual_aids' in rec

    @pytest.mark.unit
    def test_construction_learning_support(self, geometry_learning_system):
        """Test construction learning support features."""
        system = geometry_learning_system
        
        student_id = "test_student_011"
        system.create_student_session(student_id)
        
        # Get construction learning recommendations
        construction_recs = system.get_construction_recommendations(student_id)
        
        assert isinstance(construction_recs, list)
        
        # Check construction recommendation structure
        for rec in construction_recs:
            assert isinstance(rec, dict)
            assert 'concept' in rec
            assert 'construction_tools' in rec

    @pytest.mark.unit
    def test_spatial_reasoning_assessment(self, geometry_learning_system):
        """Test spatial reasoning assessment and support."""
        system = geometry_learning_system
        
        student_id = "test_student_012"
        system.create_student_session(student_id)
        
        # Assess spatial reasoning
        assessment = system.assess_spatial_reasoning(student_id)
        
        assert isinstance(assessment, dict)
        assert 'current_level' in assessment
        assert 'recommended_exercises' in assessment
        assert 'progression_plan' in assessment

    @pytest.mark.unit
    def test_system_analytics(self, geometry_learning_system):
        """Test system-wide analytics and reporting."""
        system = geometry_learning_system
        
        # Create multiple student sessions
        student_ids = ["student_001", "student_002", "student_003"]
        for student_id in student_ids:
            system.create_student_session(student_id)
            
            # Add some exercise data
            exercise = {
                'exercise_id': 'geo_001',
                'concept': 'basic_shapes',
                'correct': True
            }
            system.record_exercise_attempt(student_id, exercise)
        
        # Get system analytics
        analytics = system.get_system_analytics()
        
        assert isinstance(analytics, dict)
        assert 'total_students' in analytics
        assert 'total_exercises_completed' in analytics
        assert 'concept_popularity' in analytics

    @pytest.mark.unit
    def test_error_handling_invalid_student(self, geometry_learning_system):
        """Test error handling for invalid student operations."""
        system = geometry_learning_system
        
        # Try to get recommendations for non-existent student
        with pytest.raises((KeyError, ValueError)):
            system.recommend_exercises("non_existent_student")

    @pytest.mark.unit
    def test_session_persistence(self, geometry_learning_system, temp_data_dir):
        """Test saving and loading of student sessions."""
        system = geometry_learning_system
        
        student_id = "test_student_013"
        system.create_student_session(student_id)
        
        # Add some data
        exercise = {
            'exercise_id': 'geo_001',
            'concept': 'basic_shapes',
            'correct': True
        }
        system.record_exercise_attempt(student_id, exercise)
        
        # Save session
        session_file = os.path.join(temp_data_dir, "session.json")
        system.save_student_session(student_id, session_file)
        
        assert os.path.exists(session_file)
        
        # Load session
        loaded_system = GeometryLearningSystem()
        loaded_system.load_student_session(student_id, session_file)
        
        assert student_id in loaded_system.student_sessions

    @pytest.mark.integration
    def test_complete_learning_workflow(self, geometry_learning_system):
        """Test complete learning workflow from start to finish."""
        system = geometry_learning_system
        
        student_id = "integration_test_student"
        
        # 1. Create student session
        system.create_student_session(student_id)
        
        # 2. Set student preferences
        session = system.student_sessions[student_id]
        session['learning_graph'].set_visual_learning_preference(0.8)
        session['learning_graph'].set_spatial_reasoning_score(0.6)
        session['learning_graph'].set_construction_familiarity(0.4)
        
        # 3. Get initial gap analysis
        initial_gaps = system.analyze_student_gaps(student_id)
        assert isinstance(initial_gaps, dict)
        
        # 4. Get initial recommendations
        initial_recs = system.recommend_exercises(student_id, count=3)
        assert len(initial_recs) <= 3
        
        # 5. Record exercise attempts
        exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True, 'time_taken': 30},
            {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': False, 'time_taken': 120},
            {'exercise_id': 'geo_003', 'concept': 'area_calculation', 'correct': True, 'time_taken': 45}
        ]
        
        for exercise in exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # 6. Get updated gap analysis
        updated_gaps = system.analyze_student_gaps(student_id)
        assert isinstance(updated_gaps, dict)
        
        # 7. Get adaptive recommendations
        adaptive_recs = system.recommend_exercises(student_id, count=3)
        assert len(adaptive_recs) <= 3
        
        # 8. Generate learning insights
        insights = system.generate_learning_insights(student_id)
        assert isinstance(insights, dict)
        
        # 9. Get progress summary
        progress = system.get_student_progress(student_id)
        assert progress['exercises_completed'] == 3

    @pytest.mark.integration
    def test_multi_student_system_performance(self, geometry_learning_system):
        """Test system performance with multiple concurrent students."""
        system = geometry_learning_system
        
        # Create multiple student sessions
        num_students = 10
        student_ids = [f"perf_test_student_{i:03d}" for i in range(num_students)]
        
        # Create sessions for all students
        for student_id in student_ids:
            system.create_student_session(student_id)
        
        # Record exercises for all students
        for student_id in student_ids:
            exercises = [
                {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True},
                {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': False},
                {'exercise_id': 'geo_003', 'concept': 'area_calculation', 'correct': True}
            ]
            
            for exercise in exercises:
                system.record_exercise_attempt(student_id, exercise)
        
        # Get recommendations for all students
        all_recommendations = {}
        for student_id in student_ids:
            recommendations = system.recommend_exercises(student_id, count=2)
            all_recommendations[student_id] = recommendations
            assert len(recommendations) <= 2
        
        # Check system analytics
        analytics = system.get_system_analytics()
        assert analytics['total_students'] == num_students
        assert analytics['total_exercises_completed'] == num_students * 3

    @pytest.mark.slow
    def test_large_exercise_dataset_performance(self, temp_data_dir):
        """Test system performance with large exercise dataset."""
        # Create large exercise dataset
        large_dataset = []
        for i in range(500):
            exercise = {
                "id": f"geo_{i:04d}",
                "concept": f"concept_{i % 20}",
                "difficulty": (i % 5) + 1,
                "grade_level": ["K-5", "6-8", "9-12"][i % 3],
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "category": f"category_{i % 8}",
                "format_type": ["multiple_choice", "calculation", "construction"][i % 3],
                "visual_aids": i % 2 == 0,
                "construction_required": i % 5 == 0,
                "spatial_reasoning_level": (i % 3) + 1,
                "learning_objectives": [f"objective_{i % 10}"]
            }
            large_dataset.append(exercise)
        
        # Create exercise file
        exercise_file = os.path.join(temp_data_dir, "large_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(large_dataset, f)
        
        # Test system initialization and performance
        system = GeometryLearningSystem(exercise_file=exercise_file)
        
        student_id = "large_dataset_test_student"
        system.create_student_session(student_id)
        
        # Test that operations still work efficiently
        recommendations = system.recommend_exercises(student_id, count=10)
        gaps = system.analyze_student_gaps(student_id)
        
        assert len(recommendations) <= 10
        assert isinstance(gaps, dict)

    @pytest.mark.unit
    def test_system_configuration(self, geometry_learning_system):
        """Test system configuration and customization."""
        system = geometry_learning_system
        
        # Test configuration updates
        config = {
            'visual_weight': 0.8,
            'spatial_reasoning_weight': 0.7,
            'construction_weight': 0.6,
            'recommendation_count': 5
        }
        
        system.update_configuration(config)
        
        # Check that configuration was applied
        assert system.config['visual_weight'] == 0.8
        assert system.config['spatial_reasoning_weight'] == 0.7

    @pytest.mark.unit
    def test_system_state_management(self, geometry_learning_system):
        """Test system state management and cleanup."""
        system = geometry_learning_system
        
        # Create several student sessions
        student_ids = ["state_test_001", "state_test_002", "state_test_003"]
        for student_id in student_ids:
            system.create_student_session(student_id)
        
        # Check initial state
        assert len(system.student_sessions) == 3
        
        # Remove a session
        system.remove_student_session("state_test_002")
        assert len(system.student_sessions) == 2
        assert "state_test_002" not in system.student_sessions
        
        # Clear all sessions
        system.clear_all_sessions()
        assert len(system.student_sessions) == 0 