#!/usr/bin/env python3
"""
Comprehensive Concept Operations Test Suite

This test suite focuses specifically on concept-related operations including:
- Advanced concept creation and validation
- Concept metadata and properties
- Concept relationships and hierarchies
- Concept categorization and tagging
- Concept difficulty progression
- Concept prerequisite analysis
- Concept similarity and clustering
"""

import pytest
import tempfile
import os
import sys
import json
import time
from typing import Dict, List, Any, Optional, Set
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Knowledge graph imports
try:
    from math_learning.knowledge_graph.concept import Concept
    from math_learning.knowledge_graph.graph import KnowledgeGraph
    from math_learning.knowledge_graph.algebra_graph import AlgebraGraph
    from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend
    CONCEPT_COMPONENTS_AVAILABLE = True
except ImportError as e:
    CONCEPT_COMPONENTS_AVAILABLE = False
    print(f"Concept components not available: {e}")


class TestConceptCreationAndValidation:
    """Test suite for concept creation and validation."""
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_basic_concept_creation(self):
        """Test creating concepts with all required parameters."""
        concept = Concept(
            name="Linear Equations",
            description="Equations with variables to the first power",
            difficulty=3,  # Integer 1-5
            time_to_master=30,
            category="Algebra",
            concept_id="linear_eq_001"
        )
        
        assert concept.name == "Linear Equations"
        assert concept.description == "Equations with variables to the first power"
        assert concept.difficulty == 3
        assert concept.time_to_master == 30
        assert concept.category == "Algebra"
        assert concept.id == "linear_eq_001"  # Property is 'id'
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_with_optional_parameters(self):
        """Test creating concepts with optional parameters."""
        examples = [
            "2x + 3 = 7",
            "5x - 2 = 13",
            "3x + 4 = 2x + 9"
        ]
        
        concept = Concept(
            name="Linear Equations",
            description="First-degree equations",
            difficulty=3,
            time_to_master=30,
            category="Algebra",
            concept_id="linear_eq_002",
            examples=examples
        )
        
        assert concept.examples == examples
        assert len(concept.examples) == 3
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_difficulty_validation(self):
        """Test concept difficulty parameter validation."""
        # Valid difficulties (1-5)
        valid_difficulties = [1, 2, 3, 4, 5]
        
        for difficulty in valid_difficulties:
            concept = Concept(
                name="Test Concept",
                description="Test",
                difficulty=difficulty,
                time_to_master=10,
                category="Test",
                concept_id=f"test_{difficulty}"
            )
            assert concept.difficulty == difficulty
        
        # Test difficulty clamping
        concept_low = Concept(
            name="Low Difficulty",
            description="Test",
            difficulty=0,  # Should be clamped to 1
            time_to_master=10,
            category="Test",
            concept_id="test_low"
        )
        assert concept_low.difficulty == 1
        
        concept_high = Concept(
            name="High Difficulty",
            description="Test",
            difficulty=10,  # Should be clamped to 5
            time_to_master=10,
            category="Test",
            concept_id="test_high"
        )
        assert concept_high.difficulty == 5
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_time_to_master_validation(self):
        """Test time to master parameter validation."""
        # Valid times
        valid_times = [1, 5, 10, 30, 60, 120]
        
        for time_val in valid_times:
            concept = Concept(
                name="Test Concept",
                description="Test",
                difficulty=3,
                time_to_master=time_val,
                category="Test",
                concept_id=f"test_time_{time_val}"
            )
            assert concept.time_to_master == time_val
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_id_validation(self):
        """Test concept ID validation and uniqueness."""
        # Valid concept IDs
        valid_ids = [
            "simple_id",
            "concept_001",
            "algebra_linear_equations",
            "NS-01",
            "geometry_2d_shapes"
        ]
        
        for concept_id in valid_ids:
            concept = Concept(
                name="Test Concept",
                description="Test",
                difficulty=3,
                time_to_master=10,
                category="Test",
                concept_id=concept_id
            )
            assert concept.id == concept_id


class TestConceptMetadataAndProperties:
    """Test suite for concept metadata and advanced properties."""
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_with_rich_metadata(self):
        """Test concepts with rich metadata."""
        concept = Concept(
            name="Quadratic Equations",
            description="Second-degree polynomial equations",
            difficulty=4,
            time_to_master=45,
            category="Algebra",
            concept_id="quad_eq_001",
            examples=[
                "x² + 2x + 1 = 0",
                "2x² - 5x + 2 = 0",
                "x² - 4 = 0"
            ]
        )
        
        # Test metadata access
        assert len(concept.examples) == 3
        assert concept.category == "Algebra"
        assert concept.difficulty == 4
        
        # Test that examples contain expected patterns
        assert any("x²" in example for example in concept.examples)
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_categorization(self):
        """Test concept categorization functionality."""
        categories = [
            "Number Sense",
            "Arithmetic", 
            "Algebra",
            "Geometry",
            "Statistics",
            "Calculus"
        ]
        
        concepts = []
        for i, category in enumerate(categories):
            concept = Concept(
                name=f"{category} Concept",
                description=f"A concept in {category}",
                difficulty=1 + i,  # 1-5 range
                time_to_master=10 + (i * 5),
                category=category,
                concept_id=f"{category.lower().replace(' ', '_')}_001"
            )
            concepts.append(concept)
        
        # Test that all categories are preserved
        concept_categories = [c.category for c in concepts]
        assert set(concept_categories) == set(categories)
        
        # Test category-based filtering
        algebra_concepts = [c for c in concepts if c.category == "Algebra"]
        assert len(algebra_concepts) == 1
        assert algebra_concepts[0].name == "Algebra Concept"
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_difficulty_progression(self):
        """Test concept difficulty progression and ordering."""
        # Create concepts with progressive difficulty
        progression_concepts = []
        
        difficulty_levels = [
            (1, "Basic Addition"),
            (2, "Multi-digit Addition"),
            (3, "Addition with Decimals"),
            (4, "Addition Word Problems"),
            (5, "Complex Addition Scenarios")
        ]
        
        for difficulty, name in difficulty_levels:
            concept = Concept(
                name=name,
                description=f"Addition concept at difficulty {difficulty}",
                difficulty=difficulty,
                time_to_master=int(10 + difficulty * 5),
                category="Arithmetic",
                concept_id=f"add_{difficulty:02d}"
            )
            progression_concepts.append(concept)
        
        # Test difficulty ordering
        difficulties = [c.difficulty for c in progression_concepts]
        assert difficulties == sorted(difficulties)
        
        # Test time to master increases with difficulty
        times = [c.time_to_master for c in progression_concepts]
        assert times == sorted(times)
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_examples_validation(self):
        """Test validation and analysis of concept examples."""
        concept = Concept(
            name="Solving Linear Equations",
            description="Methods for solving first-degree equations",
            difficulty=3,
            time_to_master=30,
            category="Algebra",
            concept_id="solve_linear_001",
            examples=[
                "2x + 3 = 7 → x = 2",
                "5x - 4 = 16 → x = 4", 
                "3(x + 2) = 15 → x = 3",
                "2x + 1 = x + 5 → x = 4",
                "7 - 2x = 1 → x = 3"
            ]
        )
        
        # Test example count
        assert len(concept.examples) == 5
        
        # Test that examples follow expected patterns
        for example in concept.examples:
            assert "x" in example, "Each example should contain variable x"
            assert "=" in example, "Each example should contain equality"
        
        # Test that examples show progression
        examples_with_solutions = [ex for ex in concept.examples if "→" in ex]
        assert len(examples_with_solutions) == 5, "All examples should show solutions"


class TestConceptRelationshipsAndHierarchies:
    """Test suite for concept relationships and hierarchical structures."""
    
    @pytest.fixture
    def temp_graph_config(self):
        """Create temporary graph configuration."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        yield config
        
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_prerequisite_relationships(self, temp_graph_config):
        """Test prerequisite relationships between concepts."""
        kg = KnowledgeGraph(name="Prerequisite Test", config=temp_graph_config)
        
        # Create concept hierarchy: Numbers → Addition → Multiplication → Division
        concepts = [
            Concept("Numbers", "Basic number concepts", 1, 10, "Foundation", "numbers"),
            Concept("Addition", "Adding numbers", 2, 15, "Arithmetic", "addition"),
            Concept("Multiplication", "Multiplying numbers", 3, 20, "Arithmetic", "multiplication"),
            Concept("Division", "Dividing numbers", 4, 25, "Arithmetic", "division")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add prerequisite relationships
        kg.add_prerequisite("numbers", "addition")
        kg.add_prerequisite("addition", "multiplication")
        kg.add_prerequisite("multiplication", "division")
        
        # Test prerequisite queries
        addition_prereqs = kg.get_prerequisites("addition")
        assert len(addition_prereqs) == 1
        assert addition_prereqs[0].id == "numbers"
        
        division_prereqs = kg.get_prerequisites("division")
        assert len(division_prereqs) == 1
        assert division_prereqs[0].id == "multiplication"
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_dependency_chains(self, temp_graph_config):
        """Test complex dependency chains."""
        kg = KnowledgeGraph(name="Dependency Chain Test", config=temp_graph_config)
        
        # Create branching dependency structure
        #     Numbers
        #    /       \
        # Addition  Subtraction
        #    \       /
        #   Multiplication
        #        |
        #    Division
        
        concepts = [
            Concept("Numbers", "Number concepts", 1, 10, "Foundation", "numbers"),
            Concept("Addition", "Adding numbers", 2, 15, "Arithmetic", "addition"),
            Concept("Subtraction", "Subtracting numbers", 2, 15, "Arithmetic", "subtraction"),
            Concept("Multiplication", "Multiplying numbers", 3, 25, "Arithmetic", "multiplication"),
            Concept("Division", "Dividing numbers", 4, 30, "Arithmetic", "division")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add branching prerequisites
        kg.add_prerequisite("numbers", "addition")
        kg.add_prerequisite("numbers", "subtraction")
        kg.add_prerequisite("addition", "multiplication")
        kg.add_prerequisite("subtraction", "multiplication")
        kg.add_prerequisite("multiplication", "division")
        
        # Test that multiplication has two prerequisites
        mult_prereqs = kg.get_prerequisites("multiplication")
        prereq_ids = [p.id for p in mult_prereqs]
        assert "addition" in prereq_ids
        assert "subtraction" in prereq_ids
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_similarity_grouping(self, temp_graph_config):
        """Test grouping concepts by similarity."""
        kg = KnowledgeGraph(name="Similarity Test", config=temp_graph_config)
        
        # Create concepts in related groups
        arithmetic_concepts = [
            Concept("Basic Addition", "Simple addition", 1, 10, "Arithmetic", "add_basic"),
            Concept("Advanced Addition", "Complex addition", 3, 20, "Arithmetic", "add_advanced"),
            Concept("Basic Subtraction", "Simple subtraction", 1, 10, "Arithmetic", "sub_basic"),
            Concept("Advanced Subtraction", "Complex subtraction", 3, 20, "Arithmetic", "sub_advanced")
        ]
        
        algebra_concepts = [
            Concept("Linear Equations", "First-degree equations", 4, 30, "Algebra", "linear_eq"),
            Concept("Quadratic Equations", "Second-degree equations", 5, 40, "Algebra", "quad_eq"),
            Concept("Systems of Equations", "Multiple equations", 5, 45, "Algebra", "systems_eq")
        ]
        
        all_concepts = arithmetic_concepts + algebra_concepts
        
        for concept in all_concepts:
            kg.add_concept(concept)
        
        # Test category-based grouping
        all_stored_concepts = kg.get_all_concepts()
        arithmetic_group = [c for c in all_stored_concepts if c.category == "Arithmetic"]
        algebra_group = [c for c in all_stored_concepts if c.category == "Algebra"]
        
        assert len(arithmetic_group) == 4
        assert len(algebra_group) == 3
        
        # Test difficulty-based grouping
        basic_concepts = [c for c in all_stored_concepts if c.difficulty <= 3]
        advanced_concepts = [c for c in all_stored_concepts if c.difficulty >= 4]
        
        assert len(basic_concepts) >= 4  # All arithmetic concepts
        assert len(advanced_concepts) >= 3  # All algebra concepts
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_hierarchical_organization(self, temp_graph_config):
        """Test hierarchical organization of concepts."""
        kg = KnowledgeGraph(name="Hierarchy Test", config=temp_graph_config)
        
        # Create multi-level hierarchy
        hierarchy_levels = [
            # Level 1: Foundation
            [Concept("Counting", "Basic counting", 1, 5, "Foundation", "counting"),
             Concept("Number Recognition", "Recognizing numbers", 1, 8, "Foundation", "num_recog")],
            
            # Level 2: Basic Operations
            [Concept("Single Digit Addition", "1-digit addition", 2, 10, "Basic Ops", "add_1digit"),
             Concept("Single Digit Subtraction", "1-digit subtraction", 2, 10, "Basic Ops", "sub_1digit")],
            
            # Level 3: Advanced Operations
            [Concept("Multi-digit Addition", "Adding larger numbers", 3, 15, "Advanced Ops", "add_multi"),
             Concept("Multi-digit Subtraction", "Subtracting larger numbers", 3, 15, "Advanced Ops", "sub_multi")],
            
            # Level 4: Complex Operations
            [Concept("Word Problems", "Solving word problems", 4, 25, "Complex Ops", "word_problems"),
             Concept("Mixed Operations", "Combining operations", 4, 30, "Complex Ops", "mixed_ops")]
        ]
        
        # Add all concepts
        all_concepts = []
        for level in hierarchy_levels:
            all_concepts.extend(level)
            for concept in level:
                kg.add_concept(concept)
        
        # Add hierarchical relationships
        # Foundation → Basic Ops
        kg.add_prerequisite("counting", "add_1digit")
        kg.add_prerequisite("num_recog", "add_1digit")
        kg.add_prerequisite("counting", "sub_1digit")
        kg.add_prerequisite("num_recog", "sub_1digit")
        
        # Basic Ops → Advanced Ops
        kg.add_prerequisite("add_1digit", "add_multi")
        kg.add_prerequisite("sub_1digit", "sub_multi")
        
        # Advanced Ops → Complex Ops
        kg.add_prerequisite("add_multi", "word_problems")
        kg.add_prerequisite("sub_multi", "word_problems")
        kg.add_prerequisite("add_multi", "mixed_ops")
        kg.add_prerequisite("sub_multi", "mixed_ops")
        
        # Test hierarchical queries
        all_stored_concepts = kg.get_all_concepts()
        foundation_concepts = [c for c in all_stored_concepts if c.category == "Foundation"]
        complex_concepts = [c for c in all_stored_concepts if c.category == "Complex Ops"]
        
        assert len(foundation_concepts) == 2
        assert len(complex_concepts) == 2
        
        # Test that difficulty increases through hierarchy
        foundation_difficulties = [c.difficulty for c in foundation_concepts]
        complex_difficulties = [c.difficulty for c in complex_concepts]
        
        assert max(foundation_difficulties) < min(complex_difficulties)


class TestConceptPerformanceAndScalability:
    """Test suite for concept performance and scalability."""
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_large_scale_concept_creation(self):
        """Test creating large numbers of concepts efficiently."""
        start_time = time.time()
        
        concepts = []
        for i in range(100):  # Reduced from 1000 for faster testing
            concept = Concept(
                name=f"Concept {i:04d}",
                description=f"Auto-generated concept number {i}",
                difficulty=1 + (i % 5),  # 1-5 range
                time_to_master=5 + (i % 20),
                category=f"Category {i % 10}",
                concept_id=f"auto_{i:04d}",
                examples=[f"Example {j} for concept {i}" for j in range(3)]
            )
            concepts.append(concept)
        
        creation_time = time.time() - start_time
        
        # Should create concepts efficiently
        assert creation_time < 5.0, f"Concept creation took too long: {creation_time:.2f}s"
        assert len(concepts) == 100
        
        # Verify all concepts have correct properties
        assert all(c.name.startswith("Concept") for c in concepts)
        assert all(len(c.examples) == 3 for c in concepts)
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_serialization_performance(self):
        """Test performance of concept serialization."""
        # Create complex concepts
        concepts = []
        for i in range(50):  # Reduced for faster testing
            concept = Concept(
                name=f"Complex Concept {i}",
                description=f"A complex concept with rich metadata {i}",
                difficulty=3,
                time_to_master=30,
                category="Performance Test",
                concept_id=f"complex_{i:03d}",
                examples=[f"Example {j}: Complex operation {i}-{j}" for j in range(10)]
            )
            concepts.append(concept)
        
        # Test serialization performance
        start_time = time.time()
        
        serialized_concepts = []
        for concept in concepts:
            concept_dict = concept.to_dict()
            serialized_concepts.append(concept_dict)
        
        serialization_time = time.time() - start_time
        
        # Test deserialization performance
        start_time = time.time()
        
        deserialized_concepts = []
        for concept_dict in serialized_concepts:
            concept = Concept.from_dict(concept_dict)
            deserialized_concepts.append(concept)
        
        deserialization_time = time.time() - start_time
        
        # Performance assertions
        assert serialization_time < 2.0, f"Serialization took too long: {serialization_time:.2f}s"
        assert deserialization_time < 2.0, f"Deserialization took too long: {deserialization_time:.2f}s"
        
        # Verify data integrity
        assert len(deserialized_concepts) == 50
        for original, deserialized in zip(concepts, deserialized_concepts):
            assert original.name == deserialized.name
            assert original.id == deserialized.id
            assert len(original.examples) == len(deserialized.examples)
    
    @pytest.mark.skipif(not CONCEPT_COMPONENTS_AVAILABLE, reason="Concept components not available")
    def test_concept_comparison_performance(self):
        """Test performance of concept comparison operations."""
        # Create similar concepts for comparison
        base_concept = Concept(
            name="Base Concept",
            description="A base concept for comparison",
            difficulty=3,
            time_to_master=30,
            category="Test",
            concept_id="base_concept"
        )
        
        similar_concepts = []
        for i in range(100):  # Reduced from 1000 for faster testing
            concept = Concept(
                name=f"Similar Concept {i}",
                description="A similar concept for comparison",
                difficulty=3 + (i % 3),  # Slight variations
                time_to_master=30 + (i % 5),
                category="Test",
                concept_id=f"similar_{i:04d}"
            )
            similar_concepts.append(concept)
        
        # Test comparison performance
        start_time = time.time()
        
        equal_concepts = []
        for concept in similar_concepts:
            if concept.id == base_concept.id:  # Compare by ID
                equal_concepts.append(concept)
        
        comparison_time = time.time() - start_time
        
        # Should complete comparisons efficiently
        assert comparison_time < 1.0, f"Concept comparisons took too long: {comparison_time:.2f}s"
        
        # Verify comparison accuracy (none should be equal due to different IDs)
        assert len(equal_concepts) == 0


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 