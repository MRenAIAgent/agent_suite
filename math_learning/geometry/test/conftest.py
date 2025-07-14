"""
Shared test fixtures and configuration for geometry learning system tests.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Import geometry system components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from geometry_knowledge_graph import GeometryKnowledgeGraph
from geometry_learning_graph import GeometryLearningGraph
from geometry_gap_analyzer import GeometryGapAnalyzer
from geometry_recommender import GeometryRecommender
from geometry_learning_system import GeometryLearningSystem

# Import base classes for mocking
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from knowledge_graph.graph import KnowledgeGraph
from learning_graph.user_model import LearningGraph
from recommendation.gap_analyzer import GapAnalyzer
from recommendation.recommender import Recommender


@pytest.fixture
def sample_geometry_exercises():
    """Sample geometry exercises for testing."""
    return [
        {
            "id": "geo_001",
            "concept": "basic_shapes",
            "difficulty": 1,
            "grade_level": "K-5",
            "question": "How many sides does a triangle have?",
            "answer": "3",
            "category": "basic_geometry",
            "format_type": "multiple_choice",
            "visual_aids": True,
            "construction_required": False,
            "spatial_reasoning_level": 1,
            "learning_objectives": ["shape_identification"]
        },
        {
            "id": "geo_002", 
            "concept": "angles",
            "difficulty": 2,
            "grade_level": "6-8",
            "question": "What is the sum of angles in a triangle?",
            "answer": "180 degrees",
            "category": "angle_properties",
            "format_type": "free_response",
            "visual_aids": True,
            "construction_required": False,
            "spatial_reasoning_level": 2,
            "learning_objectives": ["angle_sum_theorem"]
        },
        {
            "id": "geo_003",
            "concept": "area_calculation",
            "difficulty": 3,
            "grade_level": "6-8", 
            "question": "Find the area of a rectangle with length 5 and width 3.",
            "answer": "15",
            "category": "area_perimeter",
            "format_type": "calculation",
            "visual_aids": True,
            "construction_required": False,
            "spatial_reasoning_level": 2,
            "learning_objectives": ["rectangle_area"]
        },
        {
            "id": "geo_004",
            "concept": "pythagorean_theorem",
            "difficulty": 4,
            "grade_level": "9-12",
            "question": "Find the hypotenuse of a right triangle with legs 3 and 4.",
            "answer": "5",
            "category": "triangles",
            "format_type": "calculation",
            "visual_aids": True,
            "construction_required": True,
            "spatial_reasoning_level": 3,
            "learning_objectives": ["pythagorean_theorem"]
        }
    ]


@pytest.fixture
def sample_student_data():
    """Sample student data for testing learning graphs."""
    return {
        "student_id": "test_student_001",
        "name": "Test Student",
        "grade_level": "7",
        "visual_learning_preference": 0.8,
        "spatial_reasoning_score": 0.7,
        "construction_familiarity": 0.6,
        "learning_style": "visual",
        "exercise_history": [
            {
                "exercise_id": "geo_001",
                "timestamp": "2024-01-01T10:00:00",
                "correct": True,
                "time_taken": 30,
                "attempts": 1,
                "visual_aids_used": True,
                "construction_used": False
            },
            {
                "exercise_id": "geo_002", 
                "timestamp": "2024-01-01T10:05:00",
                "correct": False,
                "time_taken": 120,
                "attempts": 2,
                "visual_aids_used": True,
                "construction_used": False
            }
        ]
    }


@pytest.fixture
def mock_knowledge_graph():
    """Mock knowledge graph for testing."""
    mock_kg = Mock(spec=KnowledgeGraph)
    mock_kg.get_concepts.return_value = [
        "basic_shapes", "angles", "triangles", "quadrilaterals", 
        "area_calculation", "perimeter", "pythagorean_theorem"
    ]
    mock_kg.get_prerequisites.return_value = {
        "angles": ["basic_shapes"],
        "triangles": ["basic_shapes", "angles"],
        "area_calculation": ["basic_shapes"],
        "pythagorean_theorem": ["triangles", "area_calculation"]
    }
    mock_kg.get_concept_difficulty.return_value = 2
    mock_kg.get_concept_metadata.return_value = {
        "category": "basic_geometry",
        "grade_level": "6-8"
    }
    return mock_kg


@pytest.fixture
def mock_learning_graph():
    """Mock learning graph for testing."""
    mock_lg = Mock(spec=LearningGraph)
    mock_lg.get_mastery_level.return_value = 0.7
    mock_lg.get_confidence_score.return_value = 0.8
    mock_lg.get_exercise_history.return_value = []
    mock_lg.get_learning_patterns.return_value = {
        "learning_velocity": 0.6,
        "retention_rate": 0.8,
        "preferred_difficulty": 2
    }
    return mock_lg


@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def geometry_knowledge_graph(temp_data_dir, sample_geometry_exercises):
    """Geometry knowledge graph instance for testing."""
    # Create a temporary exercise file
    exercise_file = os.path.join(temp_data_dir, "test_exercises.json")
    with open(exercise_file, 'w') as f:
        json.dump(sample_geometry_exercises, f)
    
    # Create geometry knowledge graph
    kg = GeometryKnowledgeGraph()
    kg.exercise_file = exercise_file
    return kg


@pytest.fixture
def geometry_learning_graph(sample_student_data):
    """Geometry learning graph instance for testing."""
    lg = GeometryLearningGraph(sample_student_data["student_id"])
    return lg


@pytest.fixture
def geometry_gap_analyzer(geometry_knowledge_graph, geometry_learning_graph):
    """Geometry gap analyzer instance for testing."""
    return GeometryGapAnalyzer(geometry_knowledge_graph, geometry_learning_graph)


@pytest.fixture
def geometry_recommender(geometry_knowledge_graph, geometry_learning_graph):
    """Geometry recommender instance for testing."""
    return GeometryRecommender(geometry_knowledge_graph, geometry_learning_graph)


@pytest.fixture
def geometry_learning_system(temp_data_dir, sample_geometry_exercises):
    """Complete geometry learning system for integration testing."""
    # Create temporary exercise file
    exercise_file = os.path.join(temp_data_dir, "test_exercises.json")
    with open(exercise_file, 'w') as f:
        json.dump(sample_geometry_exercises, f)
    
    system = GeometryLearningSystem(exercise_file=exercise_file)
    return system


@pytest.fixture
def sample_gap_analysis():
    """Sample gap analysis results for testing."""
    return {
        "knowledge_gaps": [
            {
                "concept": "angles",
                "gap_type": "conceptual",
                "severity": 0.8,
                "visual_component": True,
                "spatial_reasoning_required": True,
                "construction_skills_needed": False,
                "geometry_category": "angle_properties"
            }
        ],
        "learning_strategy_recommendations": [
            "Use visual aids for angle concepts",
            "Practice with interactive geometry tools",
            "Focus on spatial reasoning exercises"
        ],
        "priority_concepts": ["angles", "triangles"]
    }


@pytest.fixture
def sample_recommendations():
    """Sample exercise recommendations for testing."""
    return [
        {
            "exercise_id": "geo_002",
            "concept": "angles", 
            "confidence_score": 0.9,
            "learning_style_match": "visual",
            "visual_aids_recommended": True,
            "construction_tools_recommended": False,
            "spatial_reasoning_level": 2,
            "focus_areas": ["angle_measurement", "angle_properties"]
        }
    ]


# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data"
    ) 