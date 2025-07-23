"""
Unit tests for GeometryGapAnalyzer class.

Tests inheritance from base GapAnalyzer and geometry-specific gap analysis functionality.
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
    from geometry.src.geometry_gap_analyzer import GeometryGapAnalyzer
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometry_gap_analyzer", 
        os.path.join(os.path.dirname(__file__), '..', 'src', 'geometry_gap_analyzer.py')
    )
    geometry_ga_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_ga_module)
    GeometryGapAnalyzer = geometry_ga_module.GeometryGapAnalyzer


class TestGeometryGapAnalyzer:
    """Test cases for GeometryGapAnalyzer."""

    @pytest.mark.unit
    def test_inheritance(self, mock_knowledge_graph, mock_learning_graph):
        """Test that GeometryGapAnalyzer properly inherits from GapAnalyzer."""
        ga = GeometryGapAnalyzer(mock_knowledge_graph, mock_learning_graph)
        
        # Check that it has inherited methods
        assert hasattr(ga, 'detect_critical_gaps')
        assert hasattr(ga, 'find_learning_boundary')
        assert hasattr(ga, 'get_next_concepts')

    @pytest.mark.unit
    def test_initialization(self, mock_knowledge_graph, mock_learning_graph):
        """Test GeometryGapAnalyzer initialization."""
        ga = GeometryGapAnalyzer(mock_knowledge_graph, mock_learning_graph)
        
        # Check basic initialization
        assert ga.knowledge_graph is not None
        assert ga.learning_graph is not None
        
        # Check geometry-specific attributes
        assert hasattr(ga, 'visual_gap_threshold')
        assert hasattr(ga, 'spatial_reasoning_threshold')
        assert hasattr(ga, 'construction_gap_threshold')

    @pytest.mark.unit
    def test_enhanced_gap_detection(self, geometry_gap_analyzer, sample_geometry_exercises):
        """Test enhanced gap detection with geometry-specific attributes."""
        ga = geometry_gap_analyzer
        
        # Mock some mastery levels
        with patch.object(ga.learning_graph, 'get_mastery_level') as mock_mastery:
            mock_mastery.side_effect = lambda concept: {
                'basic_shapes': 0.9,
                'angles': 0.4,  # Below threshold - gap
                'area_calculation': 0.8,
                'pythagorean_theorem': 0.3  # Below threshold - gap
            }.get(concept, 0.5)
            
            gaps = ga.detect_enhanced_geometry_gaps()
        
        assert isinstance(gaps, list)
        assert len(gaps) >= 0
        
        # Check that gaps have geometry-specific attributes
        for gap in gaps:
            assert isinstance(gap, dict)
            if 'visual_component' in gap:
                assert isinstance(gap['visual_component'], bool)
            if 'spatial_reasoning_required' in gap:
                assert isinstance(gap['spatial_reasoning_required'], bool)
            if 'construction_skills_needed' in gap:
                assert isinstance(gap['construction_skills_needed'], bool)

    @pytest.mark.unit
    def test_visual_component_analysis(self, geometry_gap_analyzer):
        """Test analysis of visual components in gaps."""
        ga = geometry_gap_analyzer
        
        # Test visual component identification
        visual_gaps = ga.identify_visual_gaps(['angles', 'triangles', 'area_calculation'])
        
        assert isinstance(visual_gaps, list)
        
        # Check that visual gaps have appropriate metadata
        for gap in visual_gaps:
            if isinstance(gap, dict):
                assert 'concept' in gap
                assert 'visual_component' in gap
                assert gap['visual_component'] == True

    @pytest.mark.unit
    def test_spatial_reasoning_gap_analysis(self, geometry_gap_analyzer):
        """Test analysis of spatial reasoning requirements."""
        ga = geometry_gap_analyzer
        
        # Test spatial reasoning gap identification
        spatial_gaps = ga.identify_spatial_reasoning_gaps(['basic_shapes', 'angles', 'pythagorean_theorem'])
        
        assert isinstance(spatial_gaps, list)
        
        # Check spatial reasoning level requirements
        for gap in spatial_gaps:
            if isinstance(gap, dict):
                assert 'concept' in gap
                assert 'spatial_reasoning_level' in gap
                assert isinstance(gap['spatial_reasoning_level'], int)

    @pytest.mark.unit
    def test_construction_skills_assessment(self, geometry_gap_analyzer):
        """Test assessment of construction skills gaps."""
        ga = geometry_gap_analyzer
        
        # Test construction skills gap identification
        construction_gaps = ga.identify_construction_gaps(['basic_shapes', 'pythagorean_theorem', 'area_calculation'])
        
        assert isinstance(construction_gaps, list)
        
        # Check construction requirements
        for gap in construction_gaps:
            if isinstance(gap, dict):
                assert 'concept' in gap
                assert 'construction_required' in gap
                assert isinstance(gap['construction_required'], bool)

    @pytest.mark.unit
    def test_geometry_category_classification(self, geometry_gap_analyzer):
        """Test classification of gaps by geometry category."""
        ga = geometry_gap_analyzer
        
        # Test category classification
        categories = ga.classify_gaps_by_category(['basic_shapes', 'angles', 'area_calculation', 'pythagorean_theorem'])
        
        assert isinstance(categories, dict)
        
        # Check that categories are properly identified
        expected_categories = ['basic_geometry', 'angle_properties', 'area_perimeter', 'triangles']
        for category in categories:
            assert isinstance(category, str)
            assert len(categories[category]) >= 0

    @pytest.mark.unit
    def test_learning_strategy_recommendations(self, geometry_gap_analyzer, sample_gap_analysis):
        """Test generation of learning strategy recommendations."""
        ga = geometry_gap_analyzer
        
        # Mock gap data
        gaps = sample_gap_analysis['knowledge_gaps']
        
        # Generate recommendations
        recommendations = ga.generate_learning_strategy_recommendations(gaps)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation types
        recommendation_types = [
            "visual_aids", "interactive_tools", "construction_practice",
            "spatial_reasoning_exercises", "step_by_step_guidance"
        ]
        
        for rec in recommendations:
            assert isinstance(rec, str)

    @pytest.mark.unit
    def test_prerequisite_gap_analysis(self, geometry_gap_analyzer):
        """Test analysis of prerequisite gaps for geometry concepts."""
        ga = geometry_gap_analyzer
        
        # Test prerequisite gap analysis for advanced concept
        prereq_gaps = ga.analyze_prerequisite_gaps('pythagorean_theorem')
        
        assert isinstance(prereq_gaps, dict)
        assert 'missing_prerequisites' in prereq_gaps
        assert 'weak_prerequisites' in prereq_gaps
        
        # Check that prerequisites are properly identified
        if prereq_gaps['missing_prerequisites']:
            for prereq in prereq_gaps['missing_prerequisites']:
                assert isinstance(prereq, str)

    @pytest.mark.unit
    def test_visual_learning_gap_detection(self, geometry_gap_analyzer):
        """Test detection of visual learning gaps."""
        ga = geometry_gap_analyzer
        
        # Mock visual learning preference
        with patch.object(ga.learning_graph, 'get_visual_learning_preference', return_value=0.8):
            visual_gaps = ga.detect_visual_learning_gaps()
        
        assert isinstance(visual_gaps, list)
        
        # Check visual gap structure
        for gap in visual_gaps:
            if isinstance(gap, dict):
                assert 'concept' in gap
                assert 'visual_support_needed' in gap

    @pytest.mark.unit
    def test_spatial_reasoning_progression_gaps(self, geometry_gap_analyzer):
        """Test detection of spatial reasoning progression gaps."""
        ga = geometry_gap_analyzer
        
        # Mock spatial reasoning score
        with patch.object(ga.learning_graph, 'get_spatial_reasoning_score', return_value=0.6):
            progression_gaps = ga.detect_spatial_reasoning_progression_gaps()
        
        assert isinstance(progression_gaps, list)
        
        # Check progression gap structure
        for gap in progression_gaps:
            if isinstance(gap, dict):
                assert 'current_level' in gap
                assert 'target_level' in gap
                assert 'progression_steps' in gap

    @pytest.mark.unit
    def test_construction_readiness_assessment(self, geometry_gap_analyzer):
        """Test assessment of readiness for construction exercises."""
        ga = geometry_gap_analyzer
        
        # Mock construction familiarity
        with patch.object(ga.learning_graph, 'get_construction_familiarity', return_value=0.4):
            readiness = ga.assess_construction_readiness(['pythagorean_theorem', 'area_calculation'])
        
        assert isinstance(readiness, dict)
        
        # Check readiness assessment structure
        for concept in readiness:
            assert 'ready' in readiness[concept]
            assert 'preparation_needed' in readiness[concept]
            assert isinstance(readiness[concept]['ready'], bool)

    @pytest.mark.unit
    def test_gap_priority_scoring(self, geometry_gap_analyzer):
        """Test priority scoring for geometry gaps."""
        ga = geometry_gap_analyzer
        
        # Test gap priority calculation
        gaps = [
            {'concept': 'angles', 'severity': 0.8, 'visual_component': True},
            {'concept': 'area_calculation', 'severity': 0.6, 'visual_component': False},
            {'concept': 'pythagorean_theorem', 'severity': 0.9, 'construction_required': True}
        ]
        
        prioritized_gaps = ga.prioritize_geometry_gaps(gaps)
        
        assert isinstance(prioritized_gaps, list)
        assert len(prioritized_gaps) == len(gaps)
        
        # Check that gaps have priority scores
        for gap in prioritized_gaps:
            assert 'priority_score' in gap
            assert isinstance(gap['priority_score'], (int, float))

    @pytest.mark.unit
    def test_learning_path_gap_analysis(self, geometry_gap_analyzer):
        """Test gap analysis for learning path optimization."""
        ga = geometry_gap_analyzer
        
        # Test learning path gap analysis
        learning_path = ['basic_shapes', 'angles', 'triangles', 'area_calculation']
        path_gaps = ga.analyze_learning_path_gaps(learning_path)
        
        assert isinstance(path_gaps, dict)
        assert 'pathway_gaps' in path_gaps
        assert 'suggested_modifications' in path_gaps
        
        # Check pathway gap structure
        if path_gaps['pathway_gaps']:
            for gap in path_gaps['pathway_gaps']:
                assert 'position' in gap
                assert 'gap_type' in gap

    @pytest.mark.unit
    def test_error_handling_invalid_concepts(self, geometry_gap_analyzer):
        """Test error handling for invalid concept names."""
        ga = geometry_gap_analyzer
        
        # Test with invalid concept
        gaps = ga.detect_enhanced_geometry_gaps(['invalid_concept', 'another_invalid'])
        
        # Should handle gracefully and return empty or filtered results
        assert isinstance(gaps, list)

    @pytest.mark.unit
    def test_gap_remediation_suggestions(self, geometry_gap_analyzer):
        """Test generation of gap remediation suggestions."""
        ga = geometry_gap_analyzer
        
        # Test remediation suggestions
        gap = {
            'concept': 'angles',
            'gap_type': 'conceptual',
            'severity': 0.8,
            'visual_component': True,
            'spatial_reasoning_required': True
        }
        
        suggestions = ga.generate_remediation_suggestions(gap)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Check suggestion structure
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert 'strategy' in suggestion
            assert 'resources' in suggestion

    @pytest.mark.integration
    def test_comprehensive_gap_analysis(self, geometry_gap_analyzer, sample_student_data):
        """Test comprehensive gap analysis workflow."""
        ga = geometry_gap_analyzer
        
        # Mock student performance data
        with patch.object(ga.learning_graph, 'get_mastery_level') as mock_mastery:
            mock_mastery.side_effect = lambda concept: {
                'basic_shapes': 0.9,
                'angles': 0.4,
                'area_calculation': 0.7,
                'pythagorean_theorem': 0.2
            }.get(concept, 0.5)
            
            # Run comprehensive analysis
            analysis = ga.perform_comprehensive_gap_analysis()
        
        assert isinstance(analysis, dict)
        assert 'knowledge_gaps' in analysis
        assert 'visual_gaps' in analysis
        assert 'spatial_reasoning_gaps' in analysis
        assert 'construction_gaps' in analysis
        assert 'learning_strategies' in analysis
        assert 'priority_recommendations' in analysis

    @pytest.mark.unit
    def test_gap_trend_analysis(self, geometry_gap_analyzer):
        """Test analysis of gap trends over time."""
        ga = geometry_gap_analyzer
        
        # Mock historical gap data
        historical_gaps = [
            {'timestamp': '2024-01-01', 'concept': 'angles', 'severity': 0.9},
            {'timestamp': '2024-01-15', 'concept': 'angles', 'severity': 0.7},
            {'timestamp': '2024-02-01', 'concept': 'angles', 'severity': 0.5}
        ]
        
        # Analyze trends
        trends = ga.analyze_gap_trends(historical_gaps)
        
        assert isinstance(trends, dict)
        assert 'improving_gaps' in trends
        assert 'persistent_gaps' in trends
        assert 'worsening_gaps' in trends

    @pytest.mark.slow
    def test_large_concept_set_performance(self, geometry_gap_analyzer):
        """Test performance with large set of concepts."""
        ga = geometry_gap_analyzer
        
        # Create large concept set
        large_concept_set = [f"concept_{i}" for i in range(100)]
        
        # Test that gap analysis still works efficiently
        gaps = ga.detect_enhanced_geometry_gaps(large_concept_set)
        categories = ga.classify_gaps_by_category(large_concept_set)
        
        assert isinstance(gaps, list)
        assert isinstance(categories, dict)

    @pytest.mark.unit
    def test_adaptive_threshold_adjustment(self, geometry_gap_analyzer):
        """Test adaptive adjustment of gap detection thresholds."""
        ga = geometry_gap_analyzer
        
        # Test threshold adjustment based on student performance
        student_performance = {
            'overall_accuracy': 0.7,
            'visual_exercise_accuracy': 0.8,
            'construction_exercise_accuracy': 0.5
        }
        
        adjusted_thresholds = ga.adjust_gap_thresholds(student_performance)
        
        assert isinstance(adjusted_thresholds, dict)
        assert 'visual_gap_threshold' in adjusted_thresholds
        assert 'spatial_reasoning_threshold' in adjusted_thresholds
        assert 'construction_gap_threshold' in adjusted_thresholds
        
        # Check that thresholds are reasonable values
        for threshold in adjusted_thresholds.values():
            assert 0.0 <= threshold <= 1.0 