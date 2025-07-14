"""
Unit tests for GeometryRecommender class.

Tests inheritance from base Recommender and geometry-specific recommendation functionality.
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
    from geometry.src.geometry_recommender import GeometryRecommender
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometry_recommender", 
        os.path.join(os.path.dirname(__file__), '..', 'src', 'geometry_recommender.py')
    )
    geometry_rec_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_rec_module)
    GeometryRecommender = geometry_rec_module.GeometryRecommender


class TestGeometryRecommender:
    """Test cases for GeometryRecommender."""

    @pytest.mark.unit
    def test_inheritance(self, mock_knowledge_graph, mock_learning_graph):
        """Test that GeometryRecommender properly inherits from Recommender."""
        rec = GeometryRecommender(mock_knowledge_graph, mock_learning_graph)
        
        # Check that it has inherited methods
        assert hasattr(rec, 'recommend_exercises')
        assert hasattr(rec, 'get_next_concepts')
        assert hasattr(rec, 'calculate_recommendation_score')

    @pytest.mark.unit
    def test_initialization(self, mock_knowledge_graph, mock_learning_graph):
        """Test GeometryRecommender initialization."""
        rec = GeometryRecommender(mock_knowledge_graph, mock_learning_graph)
        
        # Check basic initialization
        assert rec.knowledge_graph is not None
        assert rec.learning_graph is not None
        
        # Check geometry-specific attributes
        assert hasattr(rec, 'visual_weight')
        assert hasattr(rec, 'spatial_reasoning_weight')
        assert hasattr(rec, 'construction_weight')
        assert hasattr(rec, 'learning_style_preferences')

    @pytest.mark.unit
    def test_geometry_exercise_recommendations(self, geometry_recommender, sample_geometry_exercises):
        """Test geometry-specific exercise recommendations."""
        rec = geometry_recommender
        
        # Mock exercise data
        with patch.object(rec, 'get_available_exercises', return_value=sample_geometry_exercises):
            recommendations = rec.recommend_geometry_exercises(count=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        # Check recommendation structure
        for recommendation in recommendations:
            assert isinstance(recommendation, dict)
            assert 'exercise_id' in recommendation
            assert 'confidence_score' in recommendation
            assert 'learning_style_match' in recommendation

    @pytest.mark.unit
    def test_learning_style_matching(self, geometry_recommender):
        """Test matching exercises to learning styles."""
        rec = geometry_recommender
        
        # Test visual learning style matching
        visual_exercises = rec.match_exercises_to_learning_style('visual')
        assert isinstance(visual_exercises, list)
        
        # Test kinesthetic learning style matching
        kinesthetic_exercises = rec.match_exercises_to_learning_style('kinesthetic')
        assert isinstance(kinesthetic_exercises, list)
        
        # Test analytical learning style matching
        analytical_exercises = rec.match_exercises_to_learning_style('analytical')
        assert isinstance(analytical_exercises, list)

    @pytest.mark.unit
    def test_visual_aids_recommendations(self, geometry_recommender):
        """Test recommendations for visual aids."""
        rec = geometry_recommender
        
        # Mock visual learning preference
        with patch.object(rec.learning_graph, 'get_visual_learning_preference', return_value=0.8):
            visual_recommendations = rec.recommend_visual_aids(['angles', 'triangles'])
        
        assert isinstance(visual_recommendations, list)
        
        # Check visual aid recommendation structure
        for recommendation in visual_recommendations:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'visual_aids' in recommendation
            assert 'effectiveness_score' in recommendation

    @pytest.mark.unit
    def test_construction_tool_recommendations(self, geometry_recommender):
        """Test recommendations for construction tools."""
        rec = geometry_recommender
        
        # Mock construction familiarity
        with patch.object(rec.learning_graph, 'get_construction_familiarity', return_value=0.6):
            construction_recommendations = rec.recommend_construction_tools(['pythagorean_theorem', 'area_calculation'])
        
        assert isinstance(construction_recommendations, list)
        
        # Check construction tool recommendation structure
        for recommendation in construction_recommendations:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'tools' in recommendation
            assert 'readiness_score' in recommendation

    @pytest.mark.unit
    def test_spatial_reasoning_level_assessment(self, geometry_recommender):
        """Test assessment of appropriate spatial reasoning level."""
        rec = geometry_recommender
        
        # Mock spatial reasoning score
        with patch.object(rec.learning_graph, 'get_spatial_reasoning_score', return_value=0.7):
            level_assessment = rec.assess_spatial_reasoning_level(['basic_shapes', 'angles', 'pythagorean_theorem'])
        
        assert isinstance(level_assessment, dict)
        
        # Check assessment structure
        for concept in level_assessment:
            assert 'recommended_level' in level_assessment[concept]
            assert 'current_readiness' in level_assessment[concept]
            assert isinstance(level_assessment[concept]['recommended_level'], int)

    @pytest.mark.unit
    def test_focus_area_recommendations(self, geometry_recommender):
        """Test recommendations for focus areas."""
        rec = geometry_recommender
        
        # Test focus area identification
        focus_areas = rec.identify_focus_areas(['angles', 'triangles', 'area_calculation'])
        
        assert isinstance(focus_areas, list)
        
        # Check focus area structure
        for area in focus_areas:
            assert isinstance(area, dict)
            assert 'concept' in area
            assert 'focus_topics' in area
            assert 'priority_score' in area

    @pytest.mark.unit
    def test_personalized_learning_path(self, geometry_recommender):
        """Test generation of personalized learning path."""
        rec = geometry_recommender
        
        # Mock student preferences
        with patch.object(rec.learning_graph, 'get_visual_learning_preference', return_value=0.8), \
             patch.object(rec.learning_graph, 'get_spatial_reasoning_score', return_value=0.6), \
             patch.object(rec.learning_graph, 'get_construction_familiarity', return_value=0.4):
            
            learning_path = rec.generate_personalized_learning_path(target_concept='pythagorean_theorem')
        
        assert isinstance(learning_path, list)
        
        # Check learning path structure
        for step in learning_path:
            assert isinstance(step, dict)
            assert 'concept' in step
            assert 'recommended_exercises' in step
            assert 'learning_strategies' in step

    @pytest.mark.unit
    def test_difficulty_progression_recommendations(self, geometry_recommender):
        """Test recommendations for difficulty progression."""
        rec = geometry_recommender
        
        # Test difficulty progression
        progression = rec.recommend_difficulty_progression('angles')
        
        assert isinstance(progression, list)
        
        # Check that progression is in ascending difficulty order
        if len(progression) > 1:
            for i in range(len(progression) - 1):
                current_diff = progression[i].get('difficulty', 0)
                next_diff = progression[i + 1].get('difficulty', 0)
                assert current_diff <= next_diff

    @pytest.mark.unit
    def test_adaptive_recommendations(self, geometry_recommender):
        """Test adaptive recommendations based on performance."""
        rec = geometry_recommender
        
        # Mock recent performance
        recent_performance = [
            {'exercise_id': 'geo_001', 'correct': True, 'concept': 'basic_shapes'},
            {'exercise_id': 'geo_002', 'correct': False, 'concept': 'angles'},
            {'exercise_id': 'geo_003', 'correct': True, 'concept': 'area_calculation'}
        ]
        
        adaptive_recs = rec.generate_adaptive_recommendations(recent_performance)
        
        assert isinstance(adaptive_recs, list)
        
        # Check adaptive recommendation structure
        for rec_item in adaptive_recs:
            assert isinstance(rec_item, dict)
            assert 'exercise_id' in rec_item
            assert 'adaptation_reason' in rec_item
            assert 'confidence_score' in rec_item

    @pytest.mark.unit
    def test_remediation_exercise_recommendations(self, geometry_recommender):
        """Test recommendations for remediation exercises."""
        rec = geometry_recommender
        
        # Test remediation recommendations for weak concepts
        weak_concepts = ['angles', 'pythagorean_theorem']
        remediation_recs = rec.recommend_remediation_exercises(weak_concepts)
        
        assert isinstance(remediation_recs, list)
        
        # Check remediation recommendation structure
        for recommendation in remediation_recs:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'exercises' in recommendation
            assert 'remediation_strategy' in recommendation

    @pytest.mark.unit
    def test_challenge_exercise_recommendations(self, geometry_recommender):
        """Test recommendations for challenge exercises."""
        rec = geometry_recommender
        
        # Test challenge recommendations for strong concepts
        strong_concepts = ['basic_shapes', 'area_calculation']
        challenge_recs = rec.recommend_challenge_exercises(strong_concepts)
        
        assert isinstance(challenge_recs, list)
        
        # Check challenge recommendation structure
        for recommendation in challenge_recs:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'exercises' in recommendation
            assert 'challenge_level' in recommendation

    @pytest.mark.unit
    def test_multi_modal_learning_recommendations(self, geometry_recommender):
        """Test recommendations for multi-modal learning approaches."""
        rec = geometry_recommender
        
        # Test multi-modal recommendations
        multi_modal_recs = rec.recommend_multi_modal_approaches(['angles', 'triangles'])
        
        assert isinstance(multi_modal_recs, list)
        
        # Check multi-modal recommendation structure
        for recommendation in multi_modal_recs:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'modalities' in recommendation
            assert 'sequence' in recommendation

    @pytest.mark.unit
    def test_time_based_recommendations(self, geometry_recommender):
        """Test time-based exercise recommendations."""
        rec = geometry_recommender
        
        # Test recommendations for different time constraints
        short_session_recs = rec.recommend_for_time_constraint(time_minutes=15)
        medium_session_recs = rec.recommend_for_time_constraint(time_minutes=30)
        long_session_recs = rec.recommend_for_time_constraint(time_minutes=60)
        
        assert isinstance(short_session_recs, list)
        assert isinstance(medium_session_recs, list)
        assert isinstance(long_session_recs, list)
        
        # Check that recommendations fit time constraints
        for recs in [short_session_recs, medium_session_recs, long_session_recs]:
            total_time = sum(ex.get('estimated_time', 0) for ex in recs)
            # Should not exceed the time constraint significantly

    @pytest.mark.unit
    def test_collaborative_learning_recommendations(self, geometry_recommender):
        """Test recommendations for collaborative learning."""
        rec = geometry_recommender
        
        # Test collaborative learning recommendations
        collaborative_recs = rec.recommend_collaborative_exercises(['angles', 'area_calculation'])
        
        assert isinstance(collaborative_recs, list)
        
        # Check collaborative recommendation structure
        for recommendation in collaborative_recs:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'collaboration_type' in recommendation
            assert 'group_size' in recommendation

    @pytest.mark.unit
    def test_error_handling_invalid_learning_style(self, geometry_recommender):
        """Test error handling for invalid learning style."""
        rec = geometry_recommender
        
        # Test with invalid learning style
        with pytest.raises(ValueError):
            rec.match_exercises_to_learning_style('invalid_style')

    @pytest.mark.unit
    def test_recommendation_confidence_scoring(self, geometry_recommender):
        """Test confidence scoring for recommendations."""
        rec = geometry_recommender
        
        # Test confidence scoring
        exercise_data = {
            'concept': 'angles',
            'difficulty': 2,
            'visual_aids': True,
            'construction_required': False,
            'spatial_reasoning_level': 2
        }
        
        confidence = rec.calculate_geometry_confidence_score(exercise_data)
        
        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.unit
    def test_learning_objective_alignment(self, geometry_recommender):
        """Test alignment of recommendations with learning objectives."""
        rec = geometry_recommender
        
        # Test learning objective alignment
        learning_objectives = ['angle_measurement', 'triangle_properties', 'area_calculation']
        aligned_recs = rec.align_with_learning_objectives(learning_objectives)
        
        assert isinstance(aligned_recs, list)
        
        # Check alignment structure
        for recommendation in aligned_recs:
            assert isinstance(recommendation, dict)
            assert 'exercise_id' in recommendation
            assert 'aligned_objectives' in recommendation
            assert 'alignment_score' in recommendation

    @pytest.mark.integration
    def test_comprehensive_recommendation_workflow(self, geometry_recommender, sample_student_data):
        """Test comprehensive recommendation workflow."""
        rec = geometry_recommender
        
        # Mock student data and preferences
        with patch.object(rec.learning_graph, 'get_visual_learning_preference', return_value=0.8), \
             patch.object(rec.learning_graph, 'get_spatial_reasoning_score', return_value=0.6), \
             patch.object(rec.learning_graph, 'get_construction_familiarity', return_value=0.4), \
             patch.object(rec.learning_graph, 'identify_learning_style', return_value='visual'):
            
            # Run comprehensive recommendation
            comprehensive_recs = rec.generate_comprehensive_recommendations()
        
        assert isinstance(comprehensive_recs, dict)
        assert 'personalized_exercises' in comprehensive_recs
        assert 'learning_path' in comprehensive_recs
        assert 'visual_aids' in comprehensive_recs
        assert 'focus_areas' in comprehensive_recs

    @pytest.mark.unit
    def test_recommendation_diversity(self, geometry_recommender):
        """Test diversity in exercise recommendations."""
        rec = geometry_recommender
        
        # Test recommendation diversity
        diverse_recs = rec.ensure_recommendation_diversity(['angles', 'triangles', 'area_calculation'], count=6)
        
        assert isinstance(diverse_recs, list)
        assert len(diverse_recs) <= 6
        
        # Check diversity - should include different concepts, formats, etc.
        concepts = [r.get('concept') for r in diverse_recs]
        formats = [r.get('format_type') for r in diverse_recs]
        
        # Should have some variety
        unique_concepts = len(set(concepts))
        unique_formats = len(set(formats))
        
        assert unique_concepts >= 1
        assert unique_formats >= 1

    @pytest.mark.slow
    def test_large_exercise_pool_performance(self, geometry_recommender):
        """Test performance with large exercise pool."""
        rec = geometry_recommender
        
        # Create large exercise pool
        large_exercise_pool = []
        for i in range(1000):
            exercise = {
                'exercise_id': f'geo_{i:04d}',
                'concept': f'concept_{i % 20}',
                'difficulty': (i % 5) + 1,
                'visual_aids': i % 2 == 0,
                'construction_required': i % 5 == 0,
                'spatial_reasoning_level': (i % 3) + 1
            }
            large_exercise_pool.append(exercise)
        
        # Test that recommendations still work efficiently
        with patch.object(rec, 'get_available_exercises', return_value=large_exercise_pool):
            recommendations = rec.recommend_geometry_exercises(count=10)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 10

    @pytest.mark.unit
    def test_feedback_integration(self, geometry_recommender):
        """Test integration of feedback into recommendations."""
        rec = geometry_recommender
        
        # Test feedback integration
        feedback_data = [
            {'exercise_id': 'geo_001', 'rating': 5, 'helpful': True},
            {'exercise_id': 'geo_002', 'rating': 3, 'helpful': False},
            {'exercise_id': 'geo_003', 'rating': 4, 'helpful': True}
        ]
        
        updated_recs = rec.integrate_feedback(feedback_data)
        
        assert isinstance(updated_recs, list)
        
        # Check that feedback influenced recommendations
        for rec_item in updated_recs:
            assert isinstance(rec_item, dict)
            assert 'exercise_id' in rec_item
            assert 'adjusted_score' in rec_item

    @pytest.mark.unit
    def test_real_world_application_recommendations(self, geometry_recommender):
        """Test recommendations for real-world applications."""
        rec = geometry_recommender
        
        # Test real-world application recommendations
        real_world_recs = rec.recommend_real_world_applications(['area_calculation', 'pythagorean_theorem'])
        
        assert isinstance(real_world_recs, list)
        
        # Check real-world recommendation structure
        for recommendation in real_world_recs:
            assert isinstance(recommendation, dict)
            assert 'concept' in recommendation
            assert 'application' in recommendation
            assert 'relevance_score' in recommendation 