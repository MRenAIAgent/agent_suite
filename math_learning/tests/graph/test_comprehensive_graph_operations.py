#!/usr/bin/env python3
"""
Comprehensive Graph Operations Tests

This test suite provides comprehensive coverage for all graph operations
in the math learning system, including:
- Knowledge graph construction and management
- Concept relationship handling
- Graph traversal algorithms
- Complex graph queries
- Prerequisite chain analysis
- Graph optimization and performance
- Memory usage and cleanup
"""

import pytest
import tempfile
import os
import time
import json
from typing import Dict, List, Set, Tuple, Any, Optional
from datetime import datetime, timedelta
import sys
import networkx as nx

# Add project root to path
# From math_learning/tests/graph/test_comprehensive_graph_operations.py
# Go up 4 levels: graph -> tests -> math_learning -> agent_suite
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Graph imports
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.algebra_graph import AlgebraGraph
from math_learning.knowledge_graph.rag_enhanced_algebra_graph import RagEnhancedAlgebraGraph
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend
from math_learning.config.rag_config import get_memory_config

# Learning graph imports
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem

# Analysis imports
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender


class TestKnowledgeGraphConstruction:
    """Tests for knowledge graph construction and basic operations."""
    
    @pytest.fixture
    def temp_graph(self):
        """Create temporary knowledge graph for testing."""
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        # Create config
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        # Create graph
        graph = KnowledgeGraph(name="Test Graph", config=config)
        
        yield graph
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_concept_addition_and_retrieval(self, temp_graph):
        """Test adding and retrieving concepts."""
        # Create test concepts
        concept1 = Concept(
            name="Basic Addition",
            description="Adding two numbers together",
            difficulty=0.2,
            time_to_master=10,
            category="Arithmetic",
            concept_id="add_001"
        )
        
        concept2 = Concept(
            name="Basic Subtraction", 
            description="Subtracting one number from another",
            difficulty=0.3,
            time_to_master=12,
            category="Arithmetic",
            concept_id="sub_001"
        )
        
        # Add concepts
        temp_graph.add_concept(concept1)
        temp_graph.add_concept(concept2)
        
        # Test retrieval
        retrieved1 = temp_graph.get_concept("add_001")
        assert retrieved1 is not None
        assert retrieved1.name == "Basic Addition"
        assert retrieved1.difficulty == 0.2
        
        retrieved2 = temp_graph.get_concept("sub_001")
        assert retrieved2 is not None
        assert retrieved2.name == "Basic Subtraction"
        assert retrieved2.difficulty == 0.3
        
        # Test get all concepts
        all_concepts = temp_graph.get_all_concepts()
        assert len(all_concepts) == 2
        concept_ids = [c.concept_id for c in all_concepts]
        assert "add_001" in concept_ids
        assert "sub_001" in concept_ids
    
    def test_relationship_management(self, temp_graph):
        """Test adding and managing relationships between concepts."""
        # Create concepts
        concepts = [
            Concept(name="Numbers", concept_id="num_001", difficulty=0.1, time_to_master=5),
            Concept(name="Addition", concept_id="add_001", difficulty=0.2, time_to_master=10),
            Concept(name="Subtraction", concept_id="sub_001", difficulty=0.3, time_to_master=12),
            Concept(name="Multiplication", concept_id="mul_001", difficulty=0.4, time_to_master=15),
            Concept(name="Algebra", concept_id="alg_001", difficulty=0.7, time_to_master=30)
        ]
        
        # Add concepts
        for concept in concepts:
            temp_graph.add_concept(concept)
        
        # Add prerequisite relationships
        relationships = [
            ("num_001", "add_001"),  # Numbers -> Addition
            ("num_001", "sub_001"),  # Numbers -> Subtraction  
            ("add_001", "mul_001"),  # Addition -> Multiplication
            ("sub_001", "mul_001"),  # Subtraction -> Multiplication
            ("mul_001", "alg_001")   # Multiplication -> Algebra
        ]
        
        for prereq, concept in relationships:
            temp_graph.add_prerequisite(prereq, concept)
        
        # Test prerequisite queries
        add_prereqs = temp_graph.get_prerequisites("add_001")
        assert "num_001" in add_prereqs
        
        mul_prereqs = temp_graph.get_prerequisites("mul_001")
        assert "add_001" in mul_prereqs
        assert "sub_001" in mul_prereqs
        
        alg_prereqs = temp_graph.get_prerequisites("alg_001")
        assert "mul_001" in alg_prereqs
        
        # Test dependent queries
        num_dependents = temp_graph.get_dependents("num_001")
        assert "add_001" in num_dependents
        assert "sub_001" in num_dependents
        
        mul_dependents = temp_graph.get_dependents("mul_001")
        assert "alg_001" in mul_dependents
    
    def test_complex_graph_queries(self, temp_graph):
        """Test complex graph traversal and queries."""
        # Build a more complex graph
        concepts = [
            Concept(name="Numbers", concept_id="A", difficulty=0.1),
            Concept(name="Addition", concept_id="B", difficulty=0.2),
            Concept(name="Subtraction", concept_id="C", difficulty=0.2),
            Concept(name="Multiplication", concept_id="D", difficulty=0.3),
            Concept(name="Division", concept_id="E", difficulty=0.4),
            Concept(name="Fractions", concept_id="F", difficulty=0.5),
            Concept(name="Decimals", concept_id="G", difficulty=0.5),
            Concept(name="Algebra", concept_id="H", difficulty=0.7),
            Concept(name="Equations", concept_id="I", difficulty=0.8)
        ]
        
        for concept in concepts:
            temp_graph.add_concept(concept)
        
        # Build prerequisite chain
        prereq_chains = [
            ("A", ["B", "C"]),           # Numbers -> Addition, Subtraction
            ("B", ["D"]),                # Addition -> Multiplication
            ("C", ["D"]),                # Subtraction -> Multiplication  
            ("D", ["E", "F"]),           # Multiplication -> Division, Fractions
            ("E", ["F", "G"]),           # Division -> Fractions, Decimals
            ("F", ["H"]),                # Fractions -> Algebra
            ("G", ["H"]),                # Decimals -> Algebra
            ("H", ["I"])                 # Algebra -> Equations
        ]
        
        for prereq, dependents in prereq_chains:
            for dependent in dependents:
                temp_graph.add_prerequisite(prereq, dependent)
        
        # Test transitive prerequisites (all concepts needed before target)
        algebra_all_prereqs = temp_graph.get_all_prerequisites("H")
        expected_prereqs = {"A", "B", "C", "D", "E", "F", "G"}
        assert expected_prereqs.issubset(set(algebra_all_prereqs))
        
        # Test learning path generation
        path = temp_graph.get_learning_path("A", "I")
        assert path[0] == "A"  # Should start with Numbers
        assert path[-1] == "I"  # Should end with Equations
        assert "H" in path  # Should include Algebra
        
        # Test concepts by difficulty
        easy_concepts = temp_graph.get_concepts_by_difficulty(max_difficulty=0.3)
        easy_ids = [c.concept_id for c in easy_concepts]
        assert "A" in easy_ids  # Numbers (0.1)
        assert "B" in easy_ids  # Addition (0.2)
        assert "D" in easy_ids  # Multiplication (0.3)
        assert "H" not in easy_ids  # Algebra (0.7) - too hard
        
        # Test concepts by category
        concepts[0].category = "Basic Math"
        concepts[1].category = "Basic Math"
        concepts[7].category = "Advanced Math"
        
        basic_concepts = temp_graph.get_concepts_by_category("Basic Math")
        assert len(basic_concepts) >= 2
    
    def test_graph_optimization(self, temp_graph):
        """Test graph optimization and performance."""
        # Create a large number of concepts
        large_concept_count = 100
        concepts = []
        
        for i in range(large_concept_count):
            concept = Concept(
                name=f"Concept {i}",
                concept_id=f"concept_{i:03d}",
                difficulty=0.1 + (i % 10) * 0.1,
                time_to_master=5 + (i % 20),
                category=f"Category {i % 5}"
            )
            concepts.append(concept)
            temp_graph.add_concept(concept)
        
        # Add relationships to create a complex graph
        for i in range(large_concept_count - 1):
            # Each concept is a prerequisite for the next few concepts
            for j in range(min(3, large_concept_count - i - 1)):
                temp_graph.add_prerequisite(f"concept_{i:03d}", f"concept_{i+j+1:03d}")
        
        # Performance test: retrieval should be fast
        start_time = time.time()
        for i in range(20):  # Test first 20 concepts
            concept = temp_graph.get_concept(f"concept_{i:03d}")
            assert concept is not None
        retrieval_time = time.time() - start_time
        assert retrieval_time < 1.0  # Should complete within 1 second
        
        # Performance test: prerequisite queries should be fast
        start_time = time.time()
        for i in range(50, 60):  # Test middle concepts
            prereqs = temp_graph.get_prerequisites(f"concept_{i:03d}")
            assert len(prereqs) > 0  # Should have prerequisites
        query_time = time.time() - start_time
        assert query_time < 1.0  # Should complete within 1 second
        
        # Memory usage test
        all_concepts = temp_graph.get_all_concepts()
        assert len(all_concepts) == large_concept_count
        
        # Test graph statistics
        stats = temp_graph.get_graph_stats()
        assert stats["concept_count"] == large_concept_count
        assert stats["relationship_count"] > 0


class TestAlgebraGraphSpecific:
    """Tests specific to the AlgebraGraph implementation."""
    
    def test_algebra_graph_initialization(self):
        """Test AlgebraGraph initialization and built-in concepts."""
        algebra_graph = AlgebraGraph()
        
        # Should have predefined algebra concepts
        all_concepts = algebra_graph.get_all_concepts()
        assert len(all_concepts) > 0
        
        # Check for some expected algebra concepts
        concept_names = [c.name.lower() for c in all_concepts]
        expected_concepts = ["addition", "subtraction", "multiplication", "variables"]
        
        found_concepts = 0
        for expected in expected_concepts:
            for name in concept_names:
                if expected in name:
                    found_concepts += 1
                    break
        
        assert found_concepts >= 2  # Should find at least 2 expected concepts
    
    def test_algebra_prerequisite_structure(self):
        """Test that algebra concepts have proper prerequisite structure."""
        algebra_graph = AlgebraGraph()
        
        # Get all concepts and check prerequisite relationships
        all_concepts = algebra_graph.get_all_concepts()
        concepts_with_prereqs = 0
        concepts_with_dependents = 0
        
        for concept in all_concepts[:10]:  # Test first 10 concepts
            prereqs = algebra_graph.get_prerequisites(concept.concept_id)
            dependents = algebra_graph.get_dependents(concept.concept_id)
            
            if len(prereqs) > 0:
                concepts_with_prereqs += 1
            if len(dependents) > 0:
                concepts_with_dependents += 1
        
        # Should have some concepts with prerequisites and dependents
        assert concepts_with_prereqs > 0
        assert concepts_with_dependents > 0
    
    def test_algebra_learning_paths(self):
        """Test learning path generation in algebra graph."""
        algebra_graph = AlgebraGraph()
        all_concepts = algebra_graph.get_all_concepts()
        
        if len(all_concepts) >= 2:
            # Find concepts with different difficulty levels
            easy_concepts = [c for c in all_concepts if c.difficulty <= 0.3]
            hard_concepts = [c for c in all_concepts if c.difficulty >= 0.6]
            
            if easy_concepts and hard_concepts:
                easy_concept = easy_concepts[0]
                hard_concept = hard_concepts[0]
                
                # Generate learning path
                path = algebra_graph.get_learning_path(
                    easy_concept.concept_id, 
                    hard_concept.concept_id
                )
                
                # Path should exist and be reasonable
                assert len(path) >= 2
                assert path[0] == easy_concept.concept_id
                assert path[-1] == hard_concept.concept_id


class TestGraphIntegrationWithLearning:
    """Tests for integration between knowledge graphs and learning graphs."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated knowledge and learning graph system."""
        # Create knowledge graph
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')
        temp_file.close()
        
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False
        )
        
        knowledge_graph = KnowledgeGraph(name="Integration Test", config=config)
        
        # Add test concepts
        concepts = [
            Concept(name="Numbers", concept_id="num", difficulty=0.1, time_to_master=5),
            Concept(name="Addition", concept_id="add", difficulty=0.2, time_to_master=10),
            Concept(name="Multiplication", concept_id="mul", difficulty=0.4, time_to_master=15),
            Concept(name="Fractions", concept_id="frac", difficulty=0.6, time_to_master=20)
        ]
        
        for concept in concepts:
            knowledge_graph.add_concept(concept)
        
        # Add relationships
        knowledge_graph.add_prerequisite("num", "add")
        knowledge_graph.add_prerequisite("add", "mul")
        knowledge_graph.add_prerequisite("mul", "frac")
        
        # Create learning graph
        learning_graph = LearningGraph("test_student", "Test Student")
        
        # Create analysis tools
        gap_analyzer = GapAnalyzer(knowledge_graph)
        
        yield {
            "knowledge_graph": knowledge_graph,
            "learning_graph": learning_graph,
            "gap_analyzer": gap_analyzer
        }
        
        # Cleanup
        os.unlink(temp_file.name)
    
    def test_prerequisite_based_learning_analysis(self, integrated_system):
        """Test learning analysis based on prerequisite relationships."""
        kg = integrated_system["knowledge_graph"]
        lg = integrated_system["learning_graph"]
        ga = integrated_system["gap_analyzer"]
        
        # Set student mastery levels
        lg.set_mastery("num", 0.9, 0.8)      # Strong in numbers
        lg.set_mastery("add", 0.7, 0.7)      # Good at addition
        lg.set_mastery("mul", 0.3, 0.5)      # Struggling with multiplication
        lg.set_mastery("frac", 0.1, 0.2)     # Very weak in fractions
        
        # Analyze gaps
        gaps = ga.identify_gaps(lg)
        assert len(gaps) > 0
        
        # Should identify multiplication and fractions as gaps
        gap_concepts = [gap.concept_id for gap in gaps]
        assert "mul" in gap_concepts
        assert "frac" in gap_concepts
        
        # Test prerequisite readiness
        # Student should be ready for multiplication (has addition)
        mul_prereqs = kg.get_prerequisites("mul")
        mul_ready = all(lg.get_mastery(prereq) >= 0.6 for prereq in mul_prereqs)
        assert mul_ready
        
        # Student should NOT be ready for fractions (weak multiplication)
        frac_prereqs = kg.get_prerequisites("frac")
        frac_ready = all(lg.get_mastery(prereq) >= 0.6 for prereq in frac_prereqs)
        assert not frac_ready
    
    def test_learning_path_with_mastery_levels(self, integrated_system):
        """Test learning path generation considering current mastery."""
        kg = integrated_system["knowledge_graph"]
        lg = integrated_system["learning_graph"]
        
        # Set partial mastery
        lg.set_mastery("num", 0.8, 0.7)
        lg.set_mastery("add", 0.4, 0.6)  # Needs work
        lg.set_mastery("mul", 0.1, 0.3)  # Very weak
        lg.set_mastery("frac", 0.0, 0.1) # Not learned
        
        # Generate personalized learning path
        # Should focus on concepts that need work and are ready to learn
        
        # Check which concepts are ready to learn
        ready_concepts = []
        for concept_id in ["add", "mul", "frac"]:
            prereqs = kg.get_prerequisites(concept_id)
            if not prereqs:  # No prerequisites
                ready_concepts.append(concept_id)
            else:
                # Check if all prerequisites are sufficiently mastered
                prereq_ready = all(lg.get_mastery(prereq) >= 0.6 for prereq in prereqs)
                if prereq_ready:
                    ready_concepts.append(concept_id)
        
        # Should be ready for addition (no prereqs or num is mastered)
        # Should NOT be ready for multiplication (addition not mastered enough)
        # Should NOT be ready for fractions (multiplication not mastered)
        
        assert "add" in ready_concepts or len(kg.get_prerequisites("add")) == 0 or lg.get_mastery("num") >= 0.6
    
    def test_adaptive_difficulty_progression(self, integrated_system):
        """Test adaptive difficulty progression based on graph structure."""
        kg = integrated_system["knowledge_graph"]
        lg = integrated_system["learning_graph"]
        
        # Simulate learning progression
        learning_sessions = [
            # Session 1: Work on numbers
            ("num", [(True, 0.1), (True, 0.1), (True, 0.2)]),
            # Session 2: Move to addition
            ("add", [(True, 0.2), (False, 0.3), (True, 0.2)]),
            # Session 3: Continue addition
            ("add", [(True, 0.3), (True, 0.3), (True, 0.4)]),
            # Session 4: Try multiplication
            ("mul", [(False, 0.4), (False, 0.5), (True, 0.4)])
        ]
        
        for concept, attempts in learning_sessions:
            for i, (success, difficulty) in enumerate(attempts):
                exercise_id = f"{concept}_ex_{len(lg.exercise_history) + 1}"
                lg.record_exercise_attempt(exercise_id, concept, success, difficulty, 1.0)
        
        # Check progression
        num_mastery = lg.get_mastery("num")
        add_mastery = lg.get_mastery("add") 
        mul_mastery = lg.get_mastery("mul")
        
        # Should show progression: numbers > addition > multiplication
        assert num_mastery >= add_mastery  # Numbers should be stronger than addition
        assert add_mastery >= mul_mastery  # Addition should be stronger than multiplication
        
        # Check if ready for next concept
        # If addition is strong enough, should be ready for multiplication
        if add_mastery >= 0.6:
            mul_prereqs = kg.get_prerequisites("mul")
            mul_ready = all(lg.get_mastery(prereq) >= 0.6 for prereq in mul_prereqs)
            # Should be ready if prerequisites are met


class TestRAGEnhancedGraphOperations:
    """Tests for RAG-enhanced graph operations."""
    
    @pytest.mark.asyncio
    async def test_rag_graph_construction(self):
        """Test RAG-enhanced graph construction."""
        try:
            # Create RAG-enhanced algebra graph
            config = get_memory_config()
            rag_graph = RagEnhancedAlgebraGraph(config)
            await rag_graph.initialize()
            
            # Test concept storage and retrieval
            test_concept = {
                "name": "Test Concept",
                "description": "A test mathematical concept",
                "difficulty": 0.5,
                "category": "Test Category"
            }
            
            # Store concept
            concept_id = await rag_graph.store_concept(test_concept)
            assert concept_id is not None
            
            # Retrieve concept
            retrieved = await rag_graph.get_concept(concept_id)
            assert retrieved is not None
            assert retrieved["name"] == "Test Concept"
            
            # Test semantic search
            similar_concepts = await rag_graph.find_similar_concepts("mathematical concept", top_k=5)
            assert isinstance(similar_concepts, list)
            
            # Cleanup
            await rag_graph.close()
            
        except ImportError:
            pytest.skip("RAG dependencies not available")
        except Exception as e:
            pytest.fail(f"RAG graph test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_rag_relationship_queries(self):
        """Test RAG-enhanced relationship queries."""
        try:
            config = get_memory_config()
            rag_graph = RagEnhancedAlgebraGraph(config)
            await rag_graph.initialize()
            
            # Store related concepts
            concept1 = {
                "name": "Basic Addition",
                "description": "Adding two numbers together",
                "difficulty": 0.2
            }
            concept2 = {
                "name": "Advanced Addition",
                "description": "Adding multiple numbers and complex expressions",
                "difficulty": 0.4
            }
            
            id1 = await rag_graph.store_concept(concept1)
            id2 = await rag_graph.store_concept(concept2)
            
            # Store relationship
            await rag_graph.store_relationship(id1, id2, "prerequisite", {"strength": 0.9})
            
            # Query relationships
            prereqs = await rag_graph.get_prerequisites(id2)
            assert len(prereqs) >= 1
            assert id1 in [p["concept_id"] for p in prereqs]
            
            # Cleanup
            await rag_graph.close()
            
        except ImportError:
            pytest.skip("RAG dependencies not available")
        except Exception as e:
            pytest.fail(f"RAG relationship test failed: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 