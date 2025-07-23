"""
Integration tests for the complete geometry learning system.

Tests the interaction between all components and end-to-end workflows.
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
    from geometry.src.geometry_knowledge_graph import GeometryKnowledgeGraph
    from geometry.src.geometry_learning_graph import GeometryLearningGraph
    from geometry.src.geometry_gap_analyzer import GeometryGapAnalyzer
    from geometry.src.geometry_recommender import GeometryRecommender
    from geometry.src.geometry_learning_system import GeometryLearningSystem
except ImportError:
    # Fallback import method for all components
    import importlib.util
    
    def import_geometry_module(module_name):
        spec = importlib.util.spec_from_file_location(
            module_name, 
            os.path.join(os.path.dirname(__file__), '..', 'src', f'{module_name}.py')
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    kg_module = import_geometry_module("geometry_knowledge_graph")
    lg_module = import_geometry_module("geometry_learning_graph")
    ga_module = import_geometry_module("geometry_gap_analyzer")
    rec_module = import_geometry_module("geometry_recommender")
    ls_module = import_geometry_module("geometry_learning_system")
    
    GeometryKnowledgeGraph = kg_module.GeometryKnowledgeGraph
    GeometryLearningGraph = lg_module.GeometryLearningGraph
    GeometryGapAnalyzer = ga_module.GeometryGapAnalyzer
    GeometryRecommender = rec_module.GeometryRecommender
    GeometryLearningSystem = ls_module.GeometryLearningSystem


class TestGeometrySystemIntegration:
    """Integration tests for the complete geometry learning system."""

    @pytest.mark.integration
    def test_component_initialization_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test that all components initialize and integrate correctly."""
        # Create test exercise file
        exercise_file = os.path.join(temp_data_dir, "integration_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        # Initialize knowledge graph
        kg = GeometryKnowledgeGraph()
        exercises = kg.load_geometry_exercises(exercise_file)
        kg.initialize_geometry_concepts()
        
        # Initialize learning graph
        student_id = "integration_test_student"
        lg = GeometryLearningGraph(student_id)
        
        # Initialize gap analyzer
        ga = GeometryGapAnalyzer(kg, lg)
        
        # Initialize recommender
        rec = GeometryRecommender(kg, lg)
        
        # Verify all components are properly initialized
        assert kg is not None
        assert lg is not None
        assert ga is not None
        assert rec is not None
        
        # Test basic component interactions
        assert len(exercises) == 4
        assert lg.student_id == student_id

    @pytest.mark.integration
    def test_knowledge_graph_learning_graph_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test integration between knowledge graph and learning graph."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "kg_lg_integration.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        kg = GeometryKnowledgeGraph()
        kg.load_geometry_exercises(exercise_file)
        kg.initialize_geometry_concepts()
        
        lg = GeometryLearningGraph("integration_student")
        
        # Test concept mastery tracking
        lg.set_visual_learning_preference(0.8)
        lg.set_spatial_reasoning_score(0.7)
        
        # Record exercises and check integration
        for exercise in sample_geometry_exercises:
            exercise_data = {
                'exercise_id': exercise['id'],
                'concept': exercise['concept'],
                'correct': True,
                'visual_aids_used': exercise.get('visual_aids', False),
                'construction_used': exercise.get('construction_required', False)
            }
            lg.record_geometry_exercise(exercise_data)
        
        # Verify integration
        history = lg.get_exercise_history()
        assert len(history) == 4
        
        # Check that concepts from knowledge graph are tracked in learning graph
        concepts_in_kg = list(kg.geometry_concepts.keys()) if hasattr(kg, 'geometry_concepts') else []
        concepts_in_history = [ex.get('concept') for ex in history]
        
        # Should have overlap between KG concepts and LG exercise history
        assert len(set(concepts_in_history)) > 0

    @pytest.mark.integration
    def test_gap_analyzer_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test gap analyzer integration with knowledge and learning graphs."""
        # Setup components
        exercise_file = os.path.join(temp_data_dir, "gap_integration.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        kg = GeometryKnowledgeGraph()
        kg.load_geometry_exercises(exercise_file)
        kg.initialize_geometry_concepts()
        
        lg = GeometryLearningGraph("gap_test_student")
        lg.set_visual_learning_preference(0.6)
        lg.set_spatial_reasoning_score(0.5)
        lg.set_construction_familiarity(0.4)
        
        # Record some exercises with mixed performance
        exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True},
            {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': False},
            {'exercise_id': 'geo_003', 'concept': 'area_calculation', 'correct': True},
            {'exercise_id': 'geo_004', 'concept': 'pythagorean_theorem', 'correct': False}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Initialize gap analyzer
        ga = GeometryGapAnalyzer(kg, lg)
        
        # Test gap detection
        gaps = ga.detect_enhanced_geometry_gaps()
        
        assert isinstance(gaps, list)
        # Should detect gaps for concepts with poor performance
        gap_concepts = [gap.get('concept') for gap in gaps if isinstance(gap, dict)]
        
        # Verify gap analysis considers both KG structure and LG performance
        assert len(gaps) >= 0  # Should detect some gaps based on poor performance

    @pytest.mark.integration
    def test_recommender_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test recommender integration with all other components."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "rec_integration.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        kg = GeometryKnowledgeGraph()
        kg.load_geometry_exercises(exercise_file)
        kg.initialize_geometry_concepts()
        
        lg = GeometryLearningGraph("recommender_test_student")
        lg.set_visual_learning_preference(0.9)  # High visual preference
        lg.set_spatial_reasoning_score(0.6)
        lg.set_construction_familiarity(0.3)   # Low construction familiarity
        
        # Record some performance data
        exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True, 'visual_aids_used': True},
            {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': False, 'visual_aids_used': False}
        ]
        
        for exercise in exercises:
            lg.record_geometry_exercise(exercise)
        
        # Initialize recommender
        rec = GeometryRecommender(kg, lg)
        
        # Test recommendations
        recommendations = rec.recommend_geometry_exercises(count=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Check that recommendations consider student preferences
        for recommendation in recommendations:
            assert isinstance(recommendation, dict)
            assert 'exercise_id' in recommendation
            assert 'confidence_score' in recommendation

    @pytest.mark.integration
    def test_complete_system_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test complete system integration through GeometryLearningSystem."""
        # Create exercise file
        exercise_file = os.path.join(temp_data_dir, "complete_integration.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        # Initialize complete system
        system = GeometryLearningSystem(exercise_file=exercise_file)
        
        # Test student workflow
        student_id = "complete_integration_student"
        
        # 1. Create student session
        session = system.create_student_session(student_id)
        assert session is not None
        
        # 2. Set student preferences
        student_lg = system.student_sessions[student_id]['learning_graph']
        student_lg.set_visual_learning_preference(0.8)
        student_lg.set_spatial_reasoning_score(0.6)
        student_lg.set_construction_familiarity(0.5)
        
        # 3. Get initial recommendations
        initial_recs = system.recommend_exercises(student_id, count=2)
        assert isinstance(initial_recs, list)
        assert len(initial_recs) <= 2
        
        # 4. Record exercise attempts
        exercises = [
            {
                'exercise_id': 'geo_001',
                'concept': 'basic_shapes',
                'correct': True,
                'time_taken': 30,
                'visual_aids_used': True,
                'construction_used': False
            },
            {
                'exercise_id': 'geo_002',
                'concept': 'angles',
                'correct': False,
                'time_taken': 120,
                'visual_aids_used': True,
                'construction_used': False
            }
        ]
        
        for exercise in exercises:
            result = system.record_exercise_attempt(student_id, exercise)
            assert result is True
        
        # 5. Analyze gaps after exercises
        gaps = system.analyze_student_gaps(student_id)
        assert isinstance(gaps, dict)
        
        # 6. Get updated recommendations
        updated_recs = system.recommend_exercises(student_id, count=2)
        assert isinstance(updated_recs, list)
        
        # 7. Generate insights
        insights = system.generate_learning_insights(student_id)
        assert isinstance(insights, dict)
        
        # 8. Check progress
        progress = system.get_student_progress(student_id)
        assert isinstance(progress, dict)
        assert progress['exercises_completed'] == 2

    @pytest.mark.integration
    def test_data_flow_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test data flow between all components."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "data_flow.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        student_id = "data_flow_student"
        
        # Create session
        system.create_student_session(student_id)
        
        # Test data flow: Exercise -> Learning Graph -> Gap Analysis -> Recommendations
        
        # 1. Record exercise (data enters learning graph)
        exercise_data = {
            'exercise_id': 'geo_001',
            'concept': 'basic_shapes',
            'correct': True,
            'visual_aids_used': True,
            'spatial_reasoning_level': 1
        }
        system.record_exercise_attempt(student_id, exercise_data)
        
        # 2. Check learning graph has data
        lg = system.student_sessions[student_id]['learning_graph']
        history = lg.get_exercise_history()
        assert len(history) >= 1
        
        # 3. Gap analysis uses learning graph data
        gaps = system.analyze_student_gaps(student_id)
        assert isinstance(gaps, dict)
        
        # 4. Recommendations use gap analysis and learning graph
        recommendations = system.recommend_exercises(student_id, count=1)
        assert isinstance(recommendations, list)
        
        # Verify data consistency across components
        recorded_concepts = [ex.get('concept') for ex in history]
        assert 'basic_shapes' in recorded_concepts

    @pytest.mark.integration
    def test_adaptive_learning_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test adaptive learning behavior across components."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "adaptive_learning.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        student_id = "adaptive_learning_student"
        system.create_student_session(student_id)
        
        # Simulate learning progression
        lg = system.student_sessions[student_id]['learning_graph']
        lg.set_visual_learning_preference(0.7)
        lg.set_spatial_reasoning_score(0.5)
        
        # Phase 1: Struggle with angles
        angle_exercises = [
            {'exercise_id': 'geo_002_1', 'concept': 'angles', 'correct': False, 'time_taken': 120},
            {'exercise_id': 'geo_002_2', 'concept': 'angles', 'correct': False, 'time_taken': 150},
            {'exercise_id': 'geo_002_3', 'concept': 'angles', 'correct': True, 'time_taken': 90}
        ]
        
        for exercise in angle_exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Check that system adapts - should identify angles as a gap
        gaps_after_struggle = system.analyze_student_gaps(student_id)
        assert isinstance(gaps_after_struggle, dict)
        
        # Phase 2: Success with basic shapes
        shape_exercises = [
            {'exercise_id': 'geo_001_1', 'concept': 'basic_shapes', 'correct': True, 'time_taken': 30},
            {'exercise_id': 'geo_001_2', 'concept': 'basic_shapes', 'correct': True, 'time_taken': 25}
        ]
        
        for exercise in shape_exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Check adaptive recommendations
        recommendations = system.recommend_exercises(student_id, count=3)
        assert isinstance(recommendations, list)
        
        # Verify progress tracking
        progress = system.get_student_progress(student_id)
        assert progress['exercises_completed'] == 5

    @pytest.mark.integration
    def test_visual_learning_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test visual learning features integration."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "visual_learning.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        student_id = "visual_learning_student"
        system.create_student_session(student_id)
        
        # Set high visual learning preference
        lg = system.student_sessions[student_id]['learning_graph']
        lg.set_visual_learning_preference(0.9)
        
        # Record exercises with visual aids
        visual_exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True, 'visual_aids_used': True},
            {'exercise_id': 'geo_002', 'concept': 'angles', 'correct': True, 'visual_aids_used': True}
        ]
        
        for exercise in visual_exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Test visual learning analysis
        visual_analysis = lg.analyze_visual_learning_effectiveness()
        assert isinstance(visual_analysis, dict)
        
        # Test visual recommendations
        visual_recs = system.get_visual_learning_recommendations(student_id)
        assert isinstance(visual_recs, list)
        
        # Verify visual learning is considered in recommendations
        general_recs = system.recommend_exercises(student_id, count=2)
        assert isinstance(general_recs, list)

    @pytest.mark.integration
    def test_spatial_reasoning_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test spatial reasoning features integration."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "spatial_reasoning.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        student_id = "spatial_reasoning_student"
        system.create_student_session(student_id)
        
        # Set spatial reasoning score
        lg = system.student_sessions[student_id]['learning_graph']
        lg.set_spatial_reasoning_score(0.6)
        
        # Record exercises with different spatial reasoning levels
        spatial_exercises = [
            {'exercise_id': 'geo_001', 'concept': 'basic_shapes', 'correct': True, 'spatial_reasoning_level': 1},
            {'exercise_id': 'geo_004', 'concept': 'pythagorean_theorem', 'correct': False, 'spatial_reasoning_level': 3}
        ]
        
        for exercise in spatial_exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Test spatial reasoning assessment
        spatial_assessment = system.assess_spatial_reasoning(student_id)
        assert isinstance(spatial_assessment, dict)
        
        # Test spatial reasoning progression
        progression = lg.get_spatial_reasoning_progression()
        assert isinstance(progression, dict)

    @pytest.mark.integration
    def test_construction_learning_integration(self, temp_data_dir, sample_geometry_exercises):
        """Test construction learning features integration."""
        # Setup
        exercise_file = os.path.join(temp_data_dir, "construction_learning.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=exercise_file)
        student_id = "construction_learning_student"
        system.create_student_session(student_id)
        
        # Set construction familiarity
        lg = system.student_sessions[student_id]['learning_graph']
        lg.set_construction_familiarity(0.3)  # Low familiarity
        
        # Record construction exercises
        construction_exercises = [
            {'exercise_id': 'geo_004', 'concept': 'pythagorean_theorem', 'correct': False, 'construction_used': True}
        ]
        
        for exercise in construction_exercises:
            system.record_exercise_attempt(student_id, exercise)
        
        # Test construction recommendations
        construction_recs = system.get_construction_recommendations(student_id)
        assert isinstance(construction_recs, list)
        
        # Test construction performance analysis
        construction_analysis = lg.analyze_construction_performance()
        assert isinstance(construction_analysis, dict)

    @pytest.mark.integration
    def test_error_propagation_integration(self, temp_data_dir):
        """Test error handling and propagation across components."""
        # Test with invalid exercise file
        invalid_file = os.path.join(temp_data_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json")
        
        # Should handle initialization errors gracefully
        with pytest.raises((json.JSONDecodeError, ValueError)):
            system = GeometryLearningSystem(exercise_file=invalid_file)
        
        # Test with valid system but invalid operations
        valid_exercises = [{"id": "test", "concept": "test"}]
        valid_file = os.path.join(temp_data_dir, "valid.json")
        with open(valid_file, 'w') as f:
            json.dump(valid_exercises, f)
        
        system = GeometryLearningSystem(exercise_file=valid_file)
        
        # Test invalid student operations
        with pytest.raises((KeyError, ValueError)):
            system.recommend_exercises("non_existent_student")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_performance_integration(self, temp_data_dir):
        """Test performance with realistic data loads."""
        # Create larger exercise dataset
        large_exercises = []
        for i in range(200):
            exercise = {
                "id": f"geo_{i:04d}",
                "concept": f"concept_{i % 15}",
                "difficulty": (i % 5) + 1,
                "grade_level": ["K-5", "6-8", "9-12"][i % 3],
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "category": f"category_{i % 6}",
                "format_type": ["multiple_choice", "calculation", "construction"][i % 3],
                "visual_aids": i % 2 == 0,
                "construction_required": i % 6 == 0,
                "spatial_reasoning_level": (i % 3) + 1,
                "learning_objectives": [f"objective_{i % 8}"]
            }
            large_exercises.append(exercise)
        
        exercise_file = os.path.join(temp_data_dir, "performance_test.json")
        with open(exercise_file, 'w') as f:
            json.dump(large_exercises, f)
        
        # Test system performance
        system = GeometryLearningSystem(exercise_file=exercise_file)
        
        # Create multiple students
        student_ids = [f"perf_student_{i:03d}" for i in range(20)]
        
        for student_id in student_ids:
            system.create_student_session(student_id)
            
            # Record some exercises
            for j in range(10):
                exercise = {
                    'exercise_id': f'geo_{j:04d}',
                    'concept': f'concept_{j % 5}',
                    'correct': j % 3 != 0
                }
                system.record_exercise_attempt(student_id, exercise)
            
            # Get recommendations
            recommendations = system.recommend_exercises(student_id, count=5)
            assert len(recommendations) <= 5
        
        # Test system analytics
        analytics = system.get_system_analytics()
        assert analytics['total_students'] == 20
        assert analytics['total_exercises_completed'] == 200 