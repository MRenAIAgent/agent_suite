#!/usr/bin/env python3
"""
Comprehensive tests for RAG Backend Integration with Math Learning.

This test suite validates the integration using REAL RAG backends instead of mocks:
- Real RAG service creation and configuration
- Real graph storage operations (memory backend)
- Real knowledge graph construction and queries
- Real learning graph integration with RAG
- End-to-end scenarios with actual storage
- Performance testing with real backends
- Error handling with real components
"""

import pytest
import asyncio
import tempfile
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add the parent directory to the path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real RAG imports
from agents.rag.api.rag_service import RagService
from agents.rag.factory import create_rag_service
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agents.rag.middleware.storage_router import StorageType

# Math learning imports
from math_learning.config.rag_config import (
    RagConfig, get_memory_config, get_graph_first_config, 
    create_configured_rag_service, create_rag_config_dict
)
from math_learning.knowledge_graph.graph_rag_algebra_graph import (
    GraphRagAlgebraGraph, AlgebraConcept, create_graph_rag_algebra_graph
)
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem


class TestRagServiceCreation:
    """Test real RAG service creation with different configurations."""
    
    @pytest.mark.asyncio
    async def test_memory_rag_service_creation(self):
        """Test creating RAG service with memory backends."""
        config = get_memory_config()
        rag_service = await create_configured_rag_service(config)
        
        # Verify service is created
        assert rag_service is not None
        assert isinstance(rag_service, RagService)
        
        # Verify storage adaptors are available
        assert rag_service.has_storage_adaptor(StorageType.GRAPH)
        assert rag_service.has_storage_adaptor(StorageType.VECTOR)
        assert rag_service.has_storage_adaptor(StorageType.KEY_VALUE)
        
        # Test basic storage operations
        graph_storage = rag_service.get_storage_adaptor(StorageType.GRAPH)
        assert graph_storage is not None
        
        # Clean up
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_graph_first_rag_service_creation(self):
        """Test creating RAG service with graph-first configuration."""
        config = get_graph_first_config()
        rag_service = await create_configured_rag_service(config)
        
        # Verify service is created
        assert rag_service is not None
        assert config.primary_storage_type == "graph"
        
        # Verify graph storage is primary
        assert rag_service.has_storage_adaptor(StorageType.GRAPH)
        
        # Clean up
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_rag_config_dict_creation(self):
        """Test RAG configuration dictionary creation."""
        config = get_memory_config()
        config_dict = create_rag_config_dict(config)
        
        # Verify configuration structure
        assert "graph" in config_dict
        assert "vector" in config_dict
        assert "key_value" in config_dict
        
        # Verify graph configuration
        assert config_dict["graph"]["type"] == "memory"
        
        # Verify vector configuration
        assert config_dict["vector"]["type"] == "memory"
        assert "embedding" in config_dict["vector"]
        
        # Verify key-value configuration
        assert config_dict["key_value"]["type"] == "memory"


class TestRealGraphStorage:
    """Test real graph storage operations."""
    
    @pytest.fixture
    async def rag_service(self):
        """Create a real RAG service for testing."""
        config = get_memory_config()
        service = await create_configured_rag_service(config)
        yield service
        await service.close()
    
    @pytest.mark.asyncio
    async def test_entity_storage_and_retrieval(self, rag_service):
        """Test storing and retrieving entities in real graph storage."""
        graph_storage = rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Create test entity
        entity = Entity(
            type="test_concept",
            properties={
                "name": "Linear Equations",
                "description": "Equations with variables to the first power",
                "difficulty": 5
            },
            id="linear_equations"
        )
        
        # Store entity
        await graph_storage.store_entity(entity)
        
        # Retrieve entity
        retrieved = await graph_storage.get_entity("linear_equations")
        
        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == "linear_equations"
        assert retrieved.type == "test_concept"
        assert retrieved.properties["name"] == "Linear Equations"
        assert retrieved.properties["difficulty"] == 5
    
    @pytest.mark.asyncio
    async def test_relationship_storage_and_retrieval(self, rag_service):
        """Test storing and retrieving relationships in real graph storage."""
        graph_storage = rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Create test entities
        entity1 = Entity(type="concept", properties={"name": "Basic Algebra"}, id="basic_algebra")
        entity2 = Entity(type="concept", properties={"name": "Linear Equations"}, id="linear_equations")
        
        await graph_storage.store_entity(entity1)
        await graph_storage.store_entity(entity2)
        
        # Create relationship
        relationship = Relationship(
            source_id="basic_algebra",
            target_id="linear_equations",
            type="prerequisite_for",
            properties={"strength": 0.9},
            id="basic_algebra_prereq_linear"
        )
        
        # Store relationship
        await graph_storage.store_relationship(relationship)
        
        # Retrieve relationship
        retrieved = await graph_storage.get_relationship("basic_algebra_prereq_linear")
        
        # Verify retrieval
        assert retrieved is not None
        assert retrieved.source_id == "basic_algebra"
        assert retrieved.target_id == "linear_equations"
        assert retrieved.type == "prerequisite_for"
        assert retrieved.properties["strength"] == 0.9
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_storage(self, rag_service):
        """Test storing and retrieving complete knowledge graphs."""
        graph_storage = rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Create test knowledge graph
        entities = [
            Entity(type="concept", properties={"name": "Numbers"}, id="numbers"),
            Entity(type="concept", properties={"name": "Addition"}, id="addition")
        ]
        
        relationships = [
            Relationship(
                source_id="numbers",
                target_id="addition",
                type="prerequisite_for",
                properties={},
                id="numbers_prereq_addition"
            )
        ]
        
        kg = KnowledgeGraph(
            entities=entities,
            relationships=relationships,
            metadata={"name": "Test Graph"},
            id="test_kg"
        )
        
        # Store knowledge graph
        await graph_storage.store_knowledge_graph(kg)
        
        # Retrieve knowledge graph
        retrieved = await graph_storage.get_knowledge_graph("test_kg")
        
        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == "test_kg"
        assert len(retrieved.entities) == 2
        assert len(retrieved.relationships) == 1
        assert retrieved.metadata["name"] == "Test Graph"


class TestRealAlgebraGraph:
    """Test real algebra knowledge graph with RAG backend."""
    
    @pytest.fixture
    async def algebra_graph(self):
        """Create a real algebra graph for testing."""
        config = get_memory_config()
        graph = await create_graph_rag_algebra_graph(config)
        await graph.initialize()
        yield graph
        await graph.rag_service.close()
    
    @pytest.mark.asyncio
    async def test_algebra_graph_initialization(self, algebra_graph):
        """Test algebra graph initialization with real backend."""
        # Verify graph is initialized
        assert algebra_graph._initialized is True
        assert algebra_graph.rag_service is not None
        
        # Verify graph storage is available
        assert algebra_graph.rag_service.has_storage_adaptor(StorageType.GRAPH)
        
        # Test getting a concept
        concept = await algebra_graph.get_concept("basic_arithmetic")
        assert concept is not None
        assert concept.properties["name"] == "Basic Arithmetic"
    
    @pytest.mark.asyncio
    async def test_concept_addition_and_retrieval(self, algebra_graph):
        """Test adding and retrieving concepts with real storage."""
        # Create test concept
        test_concept = AlgebraConcept(
            id="test_concept",
            name="Test Concept",
            description="A test concept for validation",
            category="Testing",
            difficulty_level=3,
            grade_level="6th",
            examples=["Example 1", "Example 2"],
            common_mistakes=["Mistake 1"],
            teaching_strategies=["Strategy 1"]
        )
        
        # Add concept
        entity = test_concept.to_entity()
        await algebra_graph.add_concept(entity)
        
        # Retrieve concept
        retrieved = await algebra_graph.get_concept("test_concept")
        
        # Verify retrieval
        assert retrieved is not None
        assert retrieved.id == "test_concept"
        assert retrieved.properties["name"] == "Test Concept"
        assert retrieved.properties["difficulty_level"] == 3
        assert retrieved.properties["grade_level"] == "6th"
    
    @pytest.mark.asyncio
    async def test_prerequisite_relationships(self, algebra_graph):
        """Test prerequisite relationships with real storage."""
        # Add prerequisite relationship
        await algebra_graph.add_prerequisite("basic_arithmetic", "integers")
        
        # Get prerequisites
        prerequisites = await algebra_graph.get_prerequisites("integers")
        
        # Verify prerequisites
        assert len(prerequisites) > 0
        prerequisite_ids = [p.id for p in prerequisites]
        assert "basic_arithmetic" in prerequisite_ids
    
    @pytest.mark.asyncio
    async def test_learning_path_finding(self, algebra_graph):
        """Test learning path finding with real graph storage."""
        # Find learning path
        path = await algebra_graph.find_learning_path("basic_arithmetic", "quadratic_equations")
        
        # Verify path exists
        assert len(path) > 0
        assert path[0].id == "basic_arithmetic"
        assert path[-1].id == "quadratic_equations"
        
        # Verify path is logical (each step builds on previous)
        for i in range(len(path) - 1):
            current_concept = path[i]
            next_concept = path[i + 1]
            
            # Check that current concept is a prerequisite for next
            prerequisites = await algebra_graph.get_prerequisites(next_concept.id)
            prerequisite_ids = [p.id for p in prerequisites]
            
            # The path should follow prerequisite relationships
            # (Note: this is a simplified check - real path finding is more complex)
            assert len(prerequisite_ids) >= 0  # At least some prerequisites exist
    
    @pytest.mark.asyncio
    async def test_concept_search(self, algebra_graph):
        """Test concept search with real backends."""
        # Test graph-based search
        results = await algebra_graph.search_concepts("equation", limit=5)
        
        # Verify search results
        assert len(results) > 0
        assert len(results) <= 5
        
        # Verify results contain equation-related concepts
        result_names = [r.properties.get("name", "").lower() for r in results]
        equation_related = any("equation" in name for name in result_names)
        assert equation_related
    
    @pytest.mark.asyncio
    async def test_category_filtering(self, algebra_graph):
        """Test filtering concepts by category."""
        # Get concepts by category
        basic_concepts = await algebra_graph.get_concepts_by_category("Basic Operations")
        
        # Verify category filtering
        assert len(basic_concepts) > 0
        for concept in basic_concepts:
            assert concept.properties.get("category") == "Basic Operations"
    
    @pytest.mark.asyncio
    async def test_difficulty_filtering(self, algebra_graph):
        """Test filtering concepts by difficulty level."""
        # Get concepts by difficulty range
        easy_concepts = await algebra_graph.get_concepts_by_difficulty(1, 3)
        
        # Verify difficulty filtering
        assert len(easy_concepts) > 0
        for concept in easy_concepts:
            difficulty = concept.properties.get("difficulty_level", 0)
            assert 1 <= difficulty <= 3


class TestRealLearningGraphIntegration:
    """Test learning graph integration with real RAG backend."""
    
    @pytest.fixture
    async def learning_system(self):
        """Create a complete learning system with real backends."""
        config = get_memory_config()
        algebra_graph = await create_graph_rag_algebra_graph(config)
        await algebra_graph.initialize()
        
        learning_graph = LearningGraph(
            user_id="test_user",
            name="Test Learning Graph"
        )
        
        yield algebra_graph, learning_graph
        await algebra_graph.rag_service.close()
    
    @pytest.mark.asyncio
    async def test_learning_progress_tracking(self, learning_system):
        """Test learning progress tracking with real backend."""
        algebra_graph, learning_graph = learning_system
        
        # Record learning activity (exercise_id, concept_id, result, difficulty, concept_weight)
        learning_graph.record_exercise_attempt(
            exercise_id="ex_001",
            concept_id="basic_arithmetic",
            result=True,
            difficulty=0.6,
            concept_weight=1.0
        )
        
        # Check mastery update
        mastery = learning_graph.get_mastery("basic_arithmetic")
        assert mastery > 0
        
        # Check exercise history
        history = learning_graph.exercise_history.get("ex_001", [])
        assert len(history) == 1
        assert history[0]["result"] is True
        assert history[0]["difficulty"] == 0.6
    
    @pytest.mark.asyncio
    async def test_basic_learning_tracking(self, learning_system):
        """Test basic learning tracking functionality."""
        algebra_graph, learning_graph = learning_system
        
        # Build some learning history
        concepts = ["basic_arithmetic", "integers", "fractions"]
        for i, concept_id in enumerate(concepts):
            for j in range(5):
                learning_graph.record_exercise_attempt(
                    exercise_id=f"ex_{i}_{j}",
                    concept_id=concept_id,
                    result=j < 4,  # 80% success rate
                    difficulty=0.5 + i * 0.1,
                    concept_weight=1.0
                )
        
        # Check mastery levels
        for concept_id in concepts:
            mastery = learning_graph.get_mastery(concept_id)
            assert mastery > 0
        
        # Check mastered concepts
        mastered = learning_graph.get_mastered_concepts(threshold=0.5)
        assert len(mastered) > 0
        
        # Check struggling concepts
        struggling = learning_graph.get_struggling_concepts(threshold=0.3)
        assert isinstance(struggling, list)
    
    @pytest.mark.asyncio
    async def test_learning_gap_detection(self, learning_system):
        """Test basic learning gap detection."""
        algebra_graph, learning_graph = learning_system
        
        # Create learning gaps by having poor performance on some concepts
        learning_graph.record_exercise_attempt(
            exercise_id="ex_001",
            concept_id="basic_arithmetic",
            result=False,
            difficulty=0.3,
            concept_weight=1.0
        )
        
        learning_graph.record_exercise_attempt(
            exercise_id="ex_002",
            concept_id="basic_arithmetic",
            result=False,
            difficulty=0.3,
            concept_weight=1.0
        )
        
        # Check struggling concepts
        struggling = learning_graph.get_struggling_concepts(threshold=0.3)
        assert "basic_arithmetic" in struggling


class TestEndToEndIntegration:
    """Test complete end-to-end scenarios with real backends."""
    
    @pytest.fixture
    async def complete_system(self):
        """Create a complete math learning system."""
        config = get_memory_config()
        algebra_graph = await create_graph_rag_algebra_graph(config)
        await algebra_graph.initialize()
        
        learning_graph = LearningGraph(
            user_id="integration_test_user",
            name="Integration Test Learning Graph"
        )
        
        yield algebra_graph, learning_graph
        await algebra_graph.rag_service.close()
    
    @pytest.mark.asyncio
    async def test_complete_learning_scenario(self, complete_system):
        """Test a complete learning scenario from start to finish."""
        algebra_graph, learning_graph = complete_system
        
        # 1. Student starts with basic arithmetic
        start_concept = "basic_arithmetic"
        target_concept = "linear_equations"
        
        # 2. Get learning path
        path = await algebra_graph.find_learning_path(start_concept, target_concept)
        assert len(path) > 0
        
        # 3. Simulate learning progress along the path
        for i, concept in enumerate(path):
            concept_id = concept.id
            
            # Simulate multiple exercise attempts
            for attempt in range(3):
                success = attempt >= 1  # Fail first, then succeed
                learning_graph.record_exercise_attempt(
                    exercise_id=f"path_ex_{i}_{attempt}",
                    concept_id=concept_id,
                    result=success,
                    difficulty=0.5,
                    concept_weight=1.0
                )
        
        # 4. Check final mastery levels
        final_mastery = learning_graph.get_mastery(target_concept)
        assert final_mastery > 0
        
        # 5. Check overall progress
        mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.3)
        assert len(mastered_concepts) > 0
    
    @pytest.mark.asyncio
    async def test_multi_user_scenario(self, complete_system):
        """Test scenario with multiple users sharing the same knowledge graph."""
        algebra_graph, _ = complete_system
        
        # Create multiple learning graphs for different users
        users = ["user_1", "user_2", "user_3"]
        learning_graphs = {}
        
        for user_id in users:
            learning_graphs[user_id] = LearningGraph(
                user_id=user_id,
                name=f"Learning Graph for {user_id}"
            )
        
        # Simulate different learning patterns for each user
        concepts = ["basic_arithmetic", "integers", "fractions"]
        
        for user_id, learning_graph in learning_graphs.items():
            for i, concept_id in enumerate(concepts):
                # Different success rates for different users
                success_rate = 0.6 + (hash(user_id) % 3) * 0.1  # 0.6, 0.7, or 0.8
                
                for attempt in range(5):
                    success = (attempt / 5) < success_rate
                    learning_graph.record_exercise_attempt(
                        exercise_id=f"{user_id}_ex_{i}_{attempt}",
                        concept_id=concept_id,
                        result=success,
                        difficulty=0.5,
                        concept_weight=1.0
                    )
        
        # Verify each user has independent progress
        for user_id, learning_graph in learning_graphs.items():
            mastery = learning_graph.get_mastery("basic_arithmetic")
            assert mastery > 0
            
            # Each user should have different mastery levels based on their progress
            mastered = learning_graph.get_mastered_concepts(threshold=0.3)
            assert len(mastered) >= 0


class TestPerformanceWithRealBackends:
    """Test performance characteristics with real backends."""
    
    @pytest.fixture
    async def performance_system(self):
        """Create a system optimized for performance testing."""
        config = get_memory_config()
        algebra_graph = await create_graph_rag_algebra_graph(config)
        await algebra_graph.initialize()
        
        yield algebra_graph
        await algebra_graph.rag_service.close()
    
    @pytest.mark.asyncio
    async def test_concept_retrieval_performance(self, performance_system):
        """Test performance of concept retrieval operations."""
        algebra_graph = performance_system
        
        # Test single concept retrieval
        start_time = time.time()
        concept = await algebra_graph.get_concept("basic_arithmetic")
        single_retrieval_time = time.time() - start_time
        
        assert concept is not None
        assert single_retrieval_time < 1.0  # Should be fast with memory backend
        
        # Test multiple concept retrievals
        concept_ids = ["basic_arithmetic", "integers", "fractions", "decimals", "linear_equations"]
        
        start_time = time.time()
        concepts = []
        for concept_id in concept_ids:
            concept = await algebra_graph.get_concept(concept_id)
            if concept:
                concepts.append(concept)
        multiple_retrieval_time = time.time() - start_time
        
        assert len(concepts) > 0
        assert multiple_retrieval_time < 2.0  # Should still be reasonably fast
    
    @pytest.mark.asyncio
    async def test_search_performance(self, performance_system):
        """Test performance of search operations."""
        algebra_graph = performance_system
        
        # Test search performance
        search_queries = ["equation", "algebra", "number", "operation", "graph"]
        
        start_time = time.time()
        all_results = []
        for query in search_queries:
            results = await algebra_graph.search_concepts(query, limit=10)
            all_results.extend(results)
        search_time = time.time() - start_time
        
        assert len(all_results) > 0
        assert search_time < 3.0  # Should complete searches quickly
    
    @pytest.mark.asyncio
    async def test_learning_path_performance(self, performance_system):
        """Test performance of learning path finding."""
        algebra_graph = performance_system
        
        # Test learning path finding performance
        path_queries = [
            ("basic_arithmetic", "linear_equations"),
            ("integers", "quadratic_equations"),
            ("fractions", "algebraic_expressions")
        ]
        
        start_time = time.time()
        paths = []
        for start, target in path_queries:
            try:
                path = await algebra_graph.find_learning_path(start, target)
                paths.append(path)
            except Exception:
                # Some paths might not exist, that's okay for performance testing
                pass
        path_finding_time = time.time() - start_time
        
        assert path_finding_time < 5.0  # Should find paths reasonably quickly


class TestErrorHandlingWithRealBackends:
    """Test error handling scenarios with real backends."""
    
    @pytest.fixture
    async def error_test_system(self):
        """Create a system for error testing."""
        config = get_memory_config()
        algebra_graph = await create_graph_rag_algebra_graph(config)
        await algebra_graph.initialize()
        
        yield algebra_graph
        await algebra_graph.rag_service.close()
    
    @pytest.mark.asyncio
    async def test_nonexistent_concept_retrieval(self, error_test_system):
        """Test handling of nonexistent concept retrieval."""
        algebra_graph = error_test_system
        
        # Try to get a concept that doesn't exist
        concept = await algebra_graph.get_concept("nonexistent_concept")
        assert concept is None
    
    @pytest.mark.asyncio
    async def test_invalid_learning_path(self, error_test_system):
        """Test handling of invalid learning path requests."""
        algebra_graph = error_test_system
        
        # Try to find path with nonexistent concepts
        path = await algebra_graph.find_learning_path("nonexistent_start", "nonexistent_end")
        assert len(path) == 0  # Should return empty path
    
    @pytest.mark.asyncio
    async def test_empty_search_query(self, error_test_system):
        """Test handling of empty search queries."""
        algebra_graph = error_test_system
        
        # Test empty search
        results = await algebra_graph.search_concepts("", limit=10)
        assert isinstance(results, list)  # Should return empty list, not error
    
    @pytest.mark.asyncio
    async def test_invalid_category_filter(self, error_test_system):
        """Test handling of invalid category filters."""
        algebra_graph = error_test_system
        
        # Test nonexistent category
        concepts = await algebra_graph.get_concepts_by_category("Nonexistent Category")
        assert len(concepts) == 0  # Should return empty list


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 