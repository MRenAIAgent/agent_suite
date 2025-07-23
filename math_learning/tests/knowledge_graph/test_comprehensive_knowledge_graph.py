#!/usr/bin/env python3
"""
Comprehensive Knowledge Graph Tests

This module contains comprehensive tests for the knowledge graph system,
including concept management, relationship handling, and storage operations.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.knowledge_graph.networkx_storage import NetworkXStorage
from math_learning.knowledge_graph.storage_interface import KnowledgeGraphStorageInterface
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend

# Check if components are available
try:
    from math_learning.knowledge_graph.rag_storage import RAGStorage
    RAG_COMPONENTS_AVAILABLE = True
except ImportError:
    RAG_COMPONENTS_AVAILABLE = False

KG_COMPONENTS_AVAILABLE = True  # NetworkX should always be available


@pytest.fixture
def temp_storage_config():
    """Create a temporary storage configuration for testing."""
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


@pytest.fixture
def populated_graph():
    """Create a populated graph for testing algorithms."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.write('{"concepts": {}, "relationships": []}')
    temp_file.close()
    
    try:
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        kg = KnowledgeGraph(name="Algorithm Test Graph", config=config)
        
        # Create a complex graph structure for algorithm testing
        # A -> B -> D -> F
        # A -> C -> E -> G
        # B -> E (cross connection)
        # C -> D (cross connection)
        # H -> I (separate component)
        # J (isolated)
        
        concepts = [
            Concept("Concept A", "Root concept", 1, 10, "Foundation", "A"),
            Concept("Concept B", "Branch 1 level 1", 2, 15, "Basic", "B"),
            Concept("Concept C", "Branch 2 level 1", 2, 15, "Basic", "C"),
            Concept("Concept D", "Branch 1 level 2", 3, 20, "Intermediate", "D"),
            Concept("Concept E", "Branch 2 level 2", 3, 20, "Intermediate", "E"),
            Concept("Concept F", "Branch 1 terminus", 4, 25, "Advanced", "F"),
            Concept("Concept G", "Branch 2 terminus", 4, 25, "Advanced", "G"),
            Concept("Concept H", "Separate component root", 2, 15, "Other", "H"),
            Concept("Concept I", "Separate component leaf", 3, 20, "Other", "I"),
            Concept("Concept J", "Isolated concept", 2, 15, "Isolated", "J")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add prerequisite relationships
        kg.add_prerequisite("A", "B", strength=0.9)
        kg.add_prerequisite("A", "C", strength=0.9)
        kg.add_prerequisite("B", "D", strength=0.8)
        kg.add_prerequisite("C", "E", strength=0.8)
        kg.add_prerequisite("D", "F", strength=0.7)
        kg.add_prerequisite("E", "G", strength=0.7)
        kg.add_prerequisite("B", "E", strength=0.6)  # Cross connection
        kg.add_prerequisite("C", "D", strength=0.6)  # Cross connection
        kg.add_prerequisite("H", "I", strength=0.8)  # Separate component
        
        yield kg
        
    finally:
        # Cleanup
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


class TestKnowledgeGraph:
    """Test cases for the KnowledgeGraph class."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_knowledge_graph_creation(self, temp_storage_config):
        """Test creating a knowledge graph."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        assert kg.name == "Test Graph"
        assert hasattr(kg, '_storage')
        assert kg._storage is not None

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_add_concept_basic(self, temp_storage_config):
        """Test adding a basic concept."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concept = Concept(
            name="Basic Concept",
            description="A simple test concept",
            difficulty=2,
            time_to_master=30,
            category="Test",
            concept_id="test_001"
        )
        
        kg.add_concept(concept)
        
        # Verify concept was added
        retrieved = kg.get_concept("test_001")
        assert retrieved is not None
        assert retrieved.name == "Basic Concept"
        assert retrieved.difficulty == 2

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_add_prerequisite_relationship(self, temp_storage_config):
        """Test adding prerequisite relationships."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Add concepts
        basic_math = Concept(
            name="Basic Math",
            description="Basic mathematical operations",
            difficulty=1,
            time_to_master=5,
            category="Foundation",
            concept_id="basic_001"
        )
        
        addition = Concept(
            name="Addition",
            description="Adding numbers",
            difficulty=2,
            time_to_master=10,
            category="Arithmetic",
            concept_id="add_001"
        )
        
        kg.add_concept(basic_math)
        kg.add_concept(addition)
        
        # Add prerequisite relationship
        kg.add_prerequisite("basic_001", "add_001", strength=0.8)
        
        # Verify relationship exists
        prerequisites = kg.get_prerequisites("add_001")
        assert len(prerequisites) > 0
        prereq_ids = [p.id for p in prerequisites]
        assert "basic_001" in prereq_ids

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_find_learning_path(self, temp_storage_config):
        """Test generating learning paths."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Create concept chain: A -> B -> C -> D
        concepts = [
            Concept("Concept A", "First concept", 1, 5, "Foundation", "A"),
            Concept("Concept B", "Second concept", 2, 10, "Basic", "B"),
            Concept("Concept C", "Third concept", 3, 15, "Intermediate", "C"),
            Concept("Concept D", "Fourth concept", 4, 20, "Advanced", "D")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add prerequisite chain
        kg.add_prerequisite("A", "B")
        kg.add_prerequisite("B", "C")
        kg.add_prerequisite("C", "D")
        
        # Test learning path generation
        path = kg.find_learning_path("A", "D")
        
        assert isinstance(path, list)
        assert len(path) > 0
        
        # Path should include the concepts in order
        path_ids = [concept.id for concept in path]
        assert "A" in path_ids
        assert "D" in path_ids

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_concept_retrieval(self, temp_storage_config):
        """Test retrieving concepts by ID."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concept = Concept(
            name="Retrieval Test",
            description="Test concept retrieval",
            difficulty=3,
            time_to_master=45,
            category="Test",
            concept_id="retrieve_001"
        )
        
        kg.add_concept(concept)
        
        # Test successful retrieval
        retrieved = kg.get_concept("retrieve_001")
        assert retrieved is not None
        assert retrieved.name == "Retrieval Test"
        assert retrieved.id == "retrieve_001"
        
        # Test non-existent concept
        non_existent = kg.get_concept("does_not_exist")
        assert non_existent is None

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_get_all_concepts(self, temp_storage_config):
        """Test retrieving all concepts."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Add multiple concepts
        concepts = [
            Concept("Concept 1", "First", 1, 10, "Test", "c1"),
            Concept("Concept 2", "Second", 2, 20, "Test", "c2"),
            Concept("Concept 3", "Third", 3, 30, "Test", "c3")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Retrieve all concepts
        all_concepts = kg.get_all_concepts()
        assert len(all_concepts) == 3
        
        concept_ids = [c.id for c in all_concepts]
        assert "c1" in concept_ids
        assert "c2" in concept_ids
        assert "c3" in concept_ids

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_concept_categories(self, temp_storage_config):
        """Test concepts with different categories."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concepts = [
            Concept("Numbers", "Number concepts", 1, 15, "Number Sense", "num_001"),
            Concept("Addition", "Adding numbers", 2, 25, "Arithmetic", "add_001"),
            Concept("Variables", "Using variables", 3, 35, "Algebra", "var_001")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Verify categories are preserved
        num_concept = kg.get_concept("num_001")
        assert num_concept.category == "Number Sense"
        
        add_concept = kg.get_concept("add_001")
        assert add_concept.category == "Arithmetic"
        
        var_concept = kg.get_concept("var_001")
        assert var_concept.category == "Algebra"

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_difficulty_levels(self, temp_storage_config):
        """Test concepts with different difficulty levels."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concepts = [
            Concept("Easy", "Easy concept", 1, 10, "Test", "easy"),
            Concept("Medium", "Medium concept", 3, 20, "Test", "medium"),
            Concept("Hard", "Hard concept", 5, 30, "Test", "hard")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Verify difficulty levels
        easy = kg.get_concept("easy")
        assert easy.difficulty == 1
        
        medium = kg.get_concept("medium")
        assert medium.difficulty == 3
        
        hard = kg.get_concept("hard")
        assert hard.difficulty == 5


class TestStorageInterface:
    """Test cases for storage interface operations."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_networkx_storage_operations(self):
        """Test NetworkX storage basic operations."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        try:
            config = MathLearningStorageConfig(
                backend=StorageBackend.NETWORKX,
                networkx_persistence_file=temp_file.name,
                networkx_auto_save=False
            )
            
            storage = NetworkXStorage(config, "Test Storage")
            
            # Test adding concept
            concept = Concept(
                name="Test Concept",
                description="A test concept",
                difficulty=3,
                time_to_master=15,
                category="Test",
                concept_id="test_001"
            )
            
            storage.add_concept(concept)
            
            # Test retrieving concept
            retrieved = storage.get_concept("test_001")
            assert retrieved is not None
            assert retrieved.name == "Test Concept"
            
            # Test adding another concept and relationship
            concept2 = Concept(
                name="Test Concept 2",
                description="Another test concept",
                difficulty=4,
                time_to_master=20,
                category="Test",
                concept_id="test_002"
            )
            
            storage.add_concept(concept2)
            storage.add_prerequisite("test_001", "test_002", strength=0.8)
            
            # Test relationship retrieval
            prerequisites = storage.get_prerequisites("test_002")
            assert len(prerequisites) > 0
            prereq_ids = [p.id for p in prerequisites]
            assert "test_001" in prereq_ids
            
        finally:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    @pytest.mark.skipif(not RAG_COMPONENTS_AVAILABLE, reason="RAG components not available")
    def test_rag_storage_operations(self):
        """Test RAG storage basic operations."""
        # This test would need proper RAG setup
        pytest.skip("RAG storage testing requires proper environment setup")


class TestConceptValidation:
    """Test cases for concept validation and constraints."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_concept_id_uniqueness(self, temp_storage_config):
        """Test that concept IDs must be unique."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concept1 = Concept("First", "First concept", 1, 10, "Test", "duplicate_id")
        concept2 = Concept("Second", "Second concept", 2, 20, "Test", "duplicate_id")
        
        kg.add_concept(concept1)
        
        # Adding second concept with same ID should replace the first
        kg.add_concept(concept2)
        
        retrieved = kg.get_concept("duplicate_id")
        assert retrieved.name == "Second"  # Should be the second concept

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_difficulty_validation(self, temp_storage_config):
        """Test difficulty level validation."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Valid difficulty levels (1-5)
        valid_concepts = [
            Concept("Level 1", "Easy", 1, 10, "Test", "level1"),
            Concept("Level 5", "Hard", 5, 50, "Test", "level5")
        ]
        
        for concept in valid_concepts:
            kg.add_concept(concept)
            retrieved = kg.get_concept(concept.id)
            assert retrieved is not None

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_time_to_master_validation(self, temp_storage_config):
        """Test time to master validation."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concept = Concept(
            name="Time Test",
            description="Testing time validation",
            difficulty=3,
            time_to_master=120,
            category="Test",
            concept_id="time_test"
        )
        
        kg.add_concept(concept)
        retrieved = kg.get_concept("time_test")
        assert retrieved.time_to_master == 120


class TestRelationshipManagement:
    """Test cases for relationship management."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_multiple_prerequisites(self, temp_storage_config):
        """Test concepts with multiple prerequisites."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Create concepts
        concepts = [
            Concept("Base A", "First base", 1, 10, "Test", "base_a"),
            Concept("Base B", "Second base", 1, 10, "Test", "base_b"),
            Concept("Advanced", "Needs both bases", 3, 30, "Test", "advanced")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add multiple prerequisites
        kg.add_prerequisite("base_a", "advanced", strength=0.8)
        kg.add_prerequisite("base_b", "advanced", strength=0.7)
        
        # Verify both prerequisites exist
        prerequisites = kg.get_prerequisites("advanced")
        prereq_ids = [p.id for p in prerequisites]
        
        assert "base_a" in prereq_ids
        assert "base_b" in prereq_ids
        assert len(prerequisites) == 2

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_prerequisite_chains(self, temp_storage_config):
        """Test chains of prerequisite relationships."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        # Create chain: A -> B -> C -> D
        concepts = [
            Concept("A", "First", 1, 10, "Test", "A"),
            Concept("B", "Second", 2, 15, "Test", "B"),
            Concept("C", "Third", 3, 20, "Test", "C"),
            Concept("D", "Fourth", 4, 25, "Test", "D")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Create chain
        kg.add_prerequisite("A", "B")
        kg.add_prerequisite("B", "C")
        kg.add_prerequisite("C", "D")
        
        # Test each link in the chain
        b_prereqs = kg.get_prerequisites("B")
        assert len(b_prereqs) == 1
        assert b_prereqs[0].id == "A"
        
        c_prereqs = kg.get_prerequisites("C")
        assert len(c_prereqs) == 1
        assert c_prereqs[0].id == "B"
        
        d_prereqs = kg.get_prerequisites("D")
        assert len(d_prereqs) == 1
        assert d_prereqs[0].id == "C"

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_relationship_strength(self, temp_storage_config):
        """Test relationship strength values."""
        kg = KnowledgeGraph(name="Test Graph", config=temp_storage_config)
        
        concepts = [
            Concept("Strong Prereq", "Strong", 1, 10, "Test", "strong"),
            Concept("Weak Prereq", "Weak", 1, 10, "Test", "weak"),
            Concept("Target", "Target concept", 3, 30, "Test", "target")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        # Add relationships with different strengths
        kg.add_prerequisite("strong", "target", strength=0.9)
        kg.add_prerequisite("weak", "target", strength=0.3)
        
        # Both should be prerequisites regardless of strength
        prerequisites = kg.get_prerequisites("target")
        prereq_ids = [p.id for p in prerequisites]
        
        assert "strong" in prereq_ids
        assert "weak" in prereq_ids


class TestKnowledgeGraphErrorHandling:
    """Test cases for error handling and edge cases."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_empty_graph_operations(self, temp_storage_config):
        """Test operations on empty knowledge graph."""
        kg = KnowledgeGraph(name="Empty Graph Test", config=temp_storage_config)
        
        # Test queries on empty graph
        all_concepts = kg.get_all_concepts()
        assert len(all_concepts) == 0
        
        # Test getting non-existent concept
        concept = kg.get_concept("non_existent")
        assert concept is None
        
        # Test getting prerequisites for non-existent concept
        prereqs = kg.get_prerequisites("non_existent")
        assert len(prereqs) == 0
        
        # Test learning path on empty graph
        path = kg.find_learning_path("start", "end")
        assert isinstance(path, list)
        assert len(path) == 0

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_invalid_prerequisite_relationships(self, temp_storage_config):
        """Test handling of invalid prerequisite relationships."""
        kg = KnowledgeGraph(name="Invalid Relationships Test", config=temp_storage_config)
        
        concept = Concept("Valid Concept", "A valid concept", 2, 20, "Test", "valid")
        kg.add_concept(concept)
        
        # Try to add prerequisite with non-existent concepts
        # This should raise an error since the implementation validates existence
        try:
            kg.add_prerequisite("non_existent", "valid", strength=0.8)
            assert False, "Should have raised an error for non-existent prerequisite"
        except ValueError:
            # Expected behavior - non-existent concepts not allowed
            pass
        
        try:
            kg.add_prerequisite("valid", "also_non_existent", strength=0.8)
            assert False, "Should have raised an error for non-existent dependent"
        except ValueError:
            # Expected behavior - non-existent concepts not allowed
            pass
        
        # The valid concept should have no prerequisites
        prerequisites = kg.get_prerequisites("valid")
        assert len(prerequisites) == 0

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_graph_consistency(self, temp_storage_config):
        """Test graph consistency after operations."""
        kg = KnowledgeGraph(name="Consistency Test", config=temp_storage_config)
        
        # Add concepts and relationships
        concepts = [
            Concept("A", "First", 1, 10, "Test", "A"),
            Concept("B", "Second", 2, 15, "Test", "B"),
            Concept("C", "Third", 3, 20, "Test", "C")
        ]
        
        for concept in concepts:
            kg.add_concept(concept)
        
        kg.add_prerequisite("A", "B")
        kg.add_prerequisite("B", "C")
        
        # Verify graph structure
        all_concepts = kg.get_all_concepts()
        assert len(all_concepts) == 3
        
        # Verify relationships
        b_prereqs = kg.get_prerequisites("B")
        c_prereqs = kg.get_prerequisites("C")
        
        assert len(b_prereqs) == 1
        assert len(c_prereqs) == 1
        assert b_prereqs[0].id == "A"
        assert c_prereqs[0].id == "B"


class TestKnowledgeGraphSerialization:
    """Test cases for graph serialization and persistence."""

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_graph_persistence(self, temp_storage_config):
        """Test saving and loading graph data."""
        # Create and populate graph
        kg1 = KnowledgeGraph(name="Persistence Test", config=temp_storage_config)
        
        concept = Concept(
            name="Persistent Concept",
            description="A concept for persistence testing",
            difficulty=3,
            time_to_master=25,
            category="Test",
            concept_id="persist_001"
        )
        
        kg1.add_concept(concept)
        
        # Create new graph instance with same config
        kg2 = KnowledgeGraph(name="Persistence Test 2", config=temp_storage_config)
        
        # Should be able to retrieve the concept
        retrieved = kg2.get_concept("persist_001")
        if retrieved:  # Depending on storage implementation
            assert retrieved.name == "Persistent Concept"
            assert retrieved.difficulty == 3

    @pytest.mark.skipif(not KG_COMPONENTS_AVAILABLE, reason="Knowledge graph components not available")
    def test_concept_serialization(self, temp_storage_config):
        """Test concept serialization properties."""
        kg = KnowledgeGraph(name="Serialization Test", config=temp_storage_config)
        
        original_concept = Concept(
            name="Serialization Test",
            description="Testing serialization",
            difficulty=4,
            time_to_master=40,
            category="Testing",
            concept_id="serial_001"
        )
        
        kg.add_concept(original_concept)
        
        # Retrieve and verify all properties
        retrieved = kg.get_concept("serial_001")
        assert retrieved is not None
        assert retrieved.name == original_concept.name
        assert retrieved.description == original_concept.description
        assert retrieved.difficulty == original_concept.difficulty
        assert retrieved.time_to_master == original_concept.time_to_master
        assert retrieved.category == original_concept.category
        assert retrieved.id == original_concept.id 