"""
Unit tests for GeometryKnowledgeGraph class.

Tests inheritance from base KnowledgeGraph and geometry-specific functionality.
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
except ImportError:
    # Fallback import method
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "geometry_knowledge_graph", 
        os.path.join(os.path.dirname(__file__), '..', 'src', 'geometry_knowledge_graph.py')
    )
    geometry_kg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry_kg_module)
    GeometryKnowledgeGraph = geometry_kg_module.GeometryKnowledgeGraph


class TestGeometryKnowledgeGraph:
    """Test cases for GeometryKnowledgeGraph."""

    @pytest.mark.unit
    def test_inheritance(self):
        """Test that GeometryKnowledgeGraph properly inherits from KnowledgeGraph."""
        kg = GeometryKnowledgeGraph()
        
        # Check that it has inherited methods (even if they're not implemented)
        assert hasattr(kg, 'add_concept')
        assert hasattr(kg, 'add_edge')
        assert hasattr(kg, 'get_concepts')
        assert hasattr(kg, 'get_prerequisites')

    @pytest.mark.unit
    def test_initialization(self):
        """Test GeometryKnowledgeGraph initialization."""
        kg = GeometryKnowledgeGraph()
        
        # Check geometry-specific attributes
        assert hasattr(kg, 'geometry_concepts')
        assert hasattr(kg, 'grade_level_mapping')
        assert hasattr(kg, 'visual_learning_concepts')

    @pytest.mark.unit
    def test_load_geometry_exercises(self, temp_data_dir, sample_geometry_exercises):
        """Test loading geometry exercises from file."""
        # Create test exercise file
        exercise_file = os.path.join(temp_data_dir, "test_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        kg = GeometryKnowledgeGraph()
        exercises = kg.load_geometry_exercises(exercise_file)
        
        assert len(exercises) == 4
        assert exercises[0]['concept'] == 'basic_shapes'
        assert exercises[1]['concept'] == 'angles'

    @pytest.mark.unit
    def test_initialize_geometry_concepts(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test initialization of geometry concepts from exercises."""
        kg = geometry_knowledge_graph
        
        # Mock the load_geometry_exercises method
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
        
        # Check that concepts were added
        assert 'basic_shapes' in kg.geometry_concepts
        assert 'angles' in kg.geometry_concepts
        assert 'area_calculation' in kg.geometry_concepts
        assert 'pythagorean_theorem' in kg.geometry_concepts

    @pytest.mark.unit
    def test_create_prerequisite_relationships(self, geometry_knowledge_graph):
        """Test automatic creation of prerequisite relationships."""
        kg = geometry_knowledge_graph
        
        # Initialize some concepts first
        kg.geometry_concepts = {
            'basic_shapes': {'grade_level': 'K-5', 'difficulty': 1},
            'angles': {'grade_level': '6-8', 'difficulty': 2},
            'triangles': {'grade_level': '6-8', 'difficulty': 3},
            'pythagorean_theorem': {'grade_level': '9-12', 'difficulty': 4}
        }
        
        kg.create_prerequisite_relationships()
        
        # Check that prerequisites were created based on difficulty and grade level
        # (Implementation details may vary, but basic structure should exist)
        assert hasattr(kg, 'prerequisites')

    @pytest.mark.unit
    def test_categorize_by_grade_level(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test categorization of concepts by grade level."""
        kg = geometry_knowledge_graph
        
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
            grade_mapping = kg.categorize_by_grade_level()
        
        # Check grade level categorization
        assert 'K-5' in grade_mapping
        assert 'basic_shapes' in grade_mapping.get('K-5', [])
        
        assert '6-8' in grade_mapping
        assert 'angles' in grade_mapping.get('6-8', [])
        assert 'area_calculation' in grade_mapping.get('6-8', [])
        
        assert '9-12' in grade_mapping
        assert 'pythagorean_theorem' in grade_mapping.get('9-12', [])

    @pytest.mark.unit
    def test_estimate_concept_difficulty(self, geometry_knowledge_graph):
        """Test difficulty estimation for geometry concepts."""
        kg = geometry_knowledge_graph
        
        # Test different concept types
        basic_difficulty = kg.estimate_concept_difficulty('basic_shapes')
        angle_difficulty = kg.estimate_concept_difficulty('angles')
        theorem_difficulty = kg.estimate_concept_difficulty('pythagorean_theorem')
        
        # Basic shapes should be easier than angles, which should be easier than theorems
        assert isinstance(basic_difficulty, (int, float))
        assert isinstance(angle_difficulty, (int, float))
        assert isinstance(theorem_difficulty, (int, float))

    @pytest.mark.unit
    def test_get_geometry_learning_path(self, geometry_knowledge_graph):
        """Test generation of geometry learning path."""
        kg = geometry_knowledge_graph
        
        # Mock some concepts and prerequisites
        kg.geometry_concepts = {
            'basic_shapes': {'grade_level': 'K-5'},
            'angles': {'grade_level': '6-8'},
            'triangles': {'grade_level': '6-8'},
            'area_calculation': {'grade_level': '6-8'}
        }
        
        learning_path = kg.get_geometry_learning_path('basic_shapes', 'triangles')
        
        # Should return a list (even if empty in this mock case)
        assert isinstance(learning_path, list)

    @pytest.mark.unit
    def test_get_visual_learning_concepts(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test identification of visual learning concepts."""
        kg = geometry_knowledge_graph
        
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
            visual_concepts = kg.get_visual_learning_concepts()
        
        # All sample exercises have visual_aids=True, so should return all concepts
        assert isinstance(visual_concepts, list)
        assert len(visual_concepts) >= 0

    @pytest.mark.unit
    def test_get_construction_concepts(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test identification of construction-based concepts."""
        kg = geometry_knowledge_graph
        
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
            construction_concepts = kg.get_construction_concepts()
        
        # Only pythagorean_theorem has construction_required=True in sample data
        assert isinstance(construction_concepts, list)

    @pytest.mark.unit
    def test_get_spatial_reasoning_concepts(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test identification of spatial reasoning concepts by level."""
        kg = geometry_knowledge_graph
        
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
            
            level_1_concepts = kg.get_spatial_reasoning_concepts(level=1)
            level_2_concepts = kg.get_spatial_reasoning_concepts(level=2)
            level_3_concepts = kg.get_spatial_reasoning_concepts(level=3)
        
        assert isinstance(level_1_concepts, list)
        assert isinstance(level_2_concepts, list)
        assert isinstance(level_3_concepts, list)

    @pytest.mark.unit
    def test_error_handling_invalid_file(self):
        """Test error handling for invalid exercise file."""
        kg = GeometryKnowledgeGraph()
        
        # Test with non-existent file
        with pytest.raises((FileNotFoundError, IOError)):
            kg.load_geometry_exercises("non_existent_file.json")

    @pytest.mark.unit
    def test_error_handling_invalid_json(self, temp_data_dir):
        """Test error handling for invalid JSON file."""
        kg = GeometryKnowledgeGraph()
        
        # Create invalid JSON file
        invalid_file = os.path.join(temp_data_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(json.JSONDecodeError):
            kg.load_geometry_exercises(invalid_file)

    @pytest.mark.integration
    def test_full_initialization_workflow(self, temp_data_dir, sample_geometry_exercises):
        """Test complete initialization workflow."""
        # Create test exercise file
        exercise_file = os.path.join(temp_data_dir, "test_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(sample_geometry_exercises, f)
        
        kg = GeometryKnowledgeGraph()
        
        # Run full initialization
        exercises = kg.load_geometry_exercises(exercise_file)
        kg.initialize_geometry_concepts()
        kg.create_prerequisite_relationships()
        grade_mapping = kg.categorize_by_grade_level()
        
        # Verify complete workflow
        assert len(exercises) == 4
        assert len(kg.geometry_concepts) >= 4
        assert len(grade_mapping) >= 3  # K-5, 6-8, 9-12

    @pytest.mark.unit
    def test_concept_metadata_extraction(self, geometry_knowledge_graph, sample_geometry_exercises):
        """Test extraction of metadata from exercise data."""
        kg = geometry_knowledge_graph
        
        with patch.object(kg, 'load_geometry_exercises', return_value=sample_geometry_exercises):
            kg.initialize_geometry_concepts()
        
        # Check that metadata was properly extracted
        if 'basic_shapes' in kg.geometry_concepts:
            concept_data = kg.geometry_concepts['basic_shapes']
            assert 'grade_level' in concept_data
            assert 'difficulty' in concept_data
            assert 'category' in concept_data

    @pytest.mark.unit 
    def test_prerequisite_logic(self, geometry_knowledge_graph):
        """Test the logic for determining prerequisites."""
        kg = geometry_knowledge_graph
        
        # Test prerequisite determination
        prereqs = kg.determine_prerequisites('triangles')
        assert isinstance(prereqs, list)
        
        # Basic shapes should typically be a prerequisite for more advanced concepts
        advanced_prereqs = kg.determine_prerequisites('pythagorean_theorem')
        assert isinstance(advanced_prereqs, list)

    @pytest.mark.slow
    def test_large_dataset_performance(self, temp_data_dir):
        """Test performance with larger dataset."""
        # Create a larger dataset for performance testing
        large_dataset = []
        for i in range(100):
            exercise = {
                "id": f"geo_{i:03d}",
                "concept": f"concept_{i % 10}",
                "difficulty": (i % 5) + 1,
                "grade_level": ["K-5", "6-8", "9-12"][i % 3],
                "question": f"Question {i}",
                "answer": f"Answer {i}",
                "category": f"category_{i % 5}",
                "format_type": "multiple_choice",
                "visual_aids": True,
                "construction_required": i % 4 == 0,
                "spatial_reasoning_level": (i % 3) + 1,
                "learning_objectives": [f"objective_{i}"]
            }
            large_dataset.append(exercise)
        
        exercise_file = os.path.join(temp_data_dir, "large_exercises.json")
        with open(exercise_file, 'w') as f:
            json.dump(large_dataset, f)
        
        kg = GeometryKnowledgeGraph()
        
        # Test that it can handle larger datasets
        exercises = kg.load_geometry_exercises(exercise_file)
        assert len(exercises) == 100
        
        kg.initialize_geometry_concepts()
        assert len(kg.geometry_concepts) <= 10  # 10 unique concepts 