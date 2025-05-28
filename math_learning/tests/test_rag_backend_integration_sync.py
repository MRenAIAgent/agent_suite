#!/usr/bin/env python3
"""
Synchronous tests for RAG Backend Integration with Math Learning.

This test suite validates the integration using REAL RAG backends with synchronous operations:
- Real RAG service creation and configuration
- Real graph storage operations (memory backend)
- Real knowledge graph construction and queries
- Real learning graph integration with RAG
- End-to-end scenarios with actual storage
"""

import pytest
import tempfile
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

# Add the parent directory to the path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agents.rag.api.rag_service import RagService
from agents.rag.storage.graph.memory_storage import MemoryGraphStorageAdaptor
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.middleware.storage_router import StorageType
from agents.rag.factory import create_rag_service
from math_learning.config.rag_config import RagConfig, get_memory_config, create_rag_config_dict, create_configured_rag_service
from math_learning.learning_graph.user_model import LearningGraph


async def create_simple_rag_service():
    """Create a simple RAG service with memory backends for testing."""
    config = {
        "graph": {"type": "memory"},
        "key_value": {"type": "memory"}
    }
    return await create_rag_service(config)


class TestSyncRagServiceCreation:
    """Test RAG service creation with real backends (synchronous)."""
    
    @pytest.mark.asyncio
    async def test_memory_rag_service_creation(self):
        """Test creating a RAG service with memory backends."""
        config = get_memory_config()
        
        # Verify config structure
        assert config.graph_storage_type == "memory"
        assert config.vector_storage_type == "memory"
        assert config.kv_storage_type == "memory"
        assert config.primary_storage_type == "graph"
        
        # Create RAG service using factory
        rag_service = await create_configured_rag_service(config)
        
        # Verify service creation
        assert rag_service is not None
        assert isinstance(rag_service, RagService)
        
        # Verify storage adaptors are registered
        assert rag_service.storage_adaptors is not None
        assert StorageType.GRAPH in rag_service.storage_adaptors
        
        # Clean up
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_graph_first_rag_service_creation(self):
        """Test creating a graph-first RAG service."""
        config = get_memory_config()
        
        # Verify it's configured as graph-first
        assert config.primary_storage_type == "graph"
        
        rag_service = await create_configured_rag_service(config)
        
        # Verify the graph storage is available
        assert StorageType.GRAPH in rag_service.storage_adaptors
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        assert isinstance(graph_adaptor, MemoryGraphStorageAdaptor)
        
        # Clean up
        await rag_service.close()
    
    def test_rag_config_dict_creation(self):
        """Test creating RAG configuration from dictionary."""
        config = get_memory_config()
        config_dict = create_rag_config_dict(config)
        
        # Verify config dictionary structure
        assert "graph" in config_dict
        assert config_dict["graph"]["type"] == "memory"
        
        # Vector storage should be present but may not be memory type
        # since the factory doesn't support memory vector storage
        assert "vector" in config_dict
        assert config_dict["vector"]["type"] in ["memory", "qdrant"]
        
        assert "key_value" in config_dict
        assert config_dict["key_value"]["type"] == "memory"
        
        # Verify embedding config is included
        assert "embedding" in config_dict["vector"]
        assert config_dict["vector"]["embedding"]["type"] == "dummy"


class TestSyncRealGraphStorage:
    """Test real graph storage operations (synchronous)."""
    
    @pytest.fixture
    async def rag_service(self):
        """Create a RAG service for testing."""
        service = await create_simple_rag_service()
        yield service
        await service.close()
    
    @pytest.mark.asyncio
    async def test_entity_storage_and_retrieval(self, rag_service):
        """Test storing and retrieving entities with real backend."""
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create test entity
        entity = Entity(
            id="test_concept",
            type="concept",
            properties={
                "name": "Test Concept",
                "description": "A test mathematical concept",
                "category": "Algebra",
                "difficulty": 0.5
            }
        )
        
        # Store entity
        await graph_storage.store_entity(entity)
        
        # Retrieve entity using query
        results = await graph_storage.query_graph("get_entities", {
            "type": "concept",
            "properties": {"name": "Test Concept"},
            "limit": 1
        })
        
        # Verify retrieval
        assert len(results) == 1
        retrieved = results[0]
        assert retrieved["id"] == "test_concept"
        assert retrieved["type"] == "concept"
        assert retrieved["properties"]["name"] == "Test Concept"
        assert retrieved["properties"]["category"] == "Algebra"
        assert retrieved["properties"]["difficulty"] == 0.5
    
    @pytest.mark.asyncio
    async def test_relationship_storage_and_retrieval(self, rag_service):
        """Test storing and retrieving relationships with real backend."""
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create test entities first
        entity1 = Entity(id="concept1", type="concept", properties={"name": "Basic Arithmetic"})
        entity2 = Entity(id="concept2", type="concept", properties={"name": "Integers"})
        
        await graph_storage.store_entity(entity1)
        await graph_storage.store_entity(entity2)
        
        # Create test relationship
        relationship = Relationship(
            id="prereq_1",
            type="prerequisite",
            source_id="concept1",
            target_id="concept2",
            properties={
                "strength": 0.8,
                "description": "Basic arithmetic is prerequisite for integers"
            }
        )
        
        # Store relationship
        await graph_storage.store_relationship(relationship)
        
        # Retrieve relationship using query
        results = await graph_storage.query_graph("get_relationships", {
            "source_id": "concept1",
            "type": "prerequisite",
            "limit": 1
        })
        
        # Verify retrieval
        assert len(results) == 1
        retrieved = results[0]
        assert retrieved["id"] == "prereq_1"
        assert retrieved["type"] == "prerequisite"
        assert retrieved["source_id"] == "concept1"
        assert retrieved["target_id"] == "concept2"
        assert retrieved["properties"]["strength"] == 0.8
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_storage(self, rag_service):
        """Test storing a complete knowledge graph structure."""
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create multiple entities
        concepts = [
            Entity(id="basic_arithmetic", type="concept", properties={"name": "Basic Arithmetic", "category": "Number Sense"}),
            Entity(id="integers", type="concept", properties={"name": "Integers", "category": "Number Sense"}),
            Entity(id="fractions", type="concept", properties={"name": "Fractions", "category": "Number Sense"}),
            Entity(id="linear_equations", type="concept", properties={"name": "Linear Equations", "category": "Algebra"})
        ]
        
        # Store all concepts
        for concept in concepts:
            await graph_storage.store_entity(concept)
        
        # Create relationships
        relationships = [
            Relationship(id="rel1", type="prerequisite", source_id="basic_arithmetic", target_id="integers", properties={"strength": 0.9}),
            Relationship(id="rel2", type="prerequisite", source_id="basic_arithmetic", target_id="fractions", properties={"strength": 0.8}),
            Relationship(id="rel3", type="prerequisite", source_id="integers", target_id="linear_equations", properties={"strength": 0.7}),
            Relationship(id="rel4", type="prerequisite", source_id="fractions", target_id="linear_equations", properties={"strength": 0.6})
        ]
        
        # Store all relationships
        for rel in relationships:
            await graph_storage.store_relationship(rel)
        
        # Verify all entities can be retrieved
        for concept in concepts:
            results = await graph_storage.query_graph("get_entities", {
                "type": "concept",
                "properties": {"name": concept.properties["name"]},
                "limit": 1
            })
            assert len(results) == 1
            assert results[0]["id"] == concept.id
        
        # Verify all relationships can be retrieved
        for rel in relationships:
            results = await graph_storage.query_graph("get_relationships", {
                "source_id": rel.source_id,
                "type": "prerequisite",
                "limit": 10
            })
            # Find the specific relationship in the results
            found = False
            for result in results:
                if result["id"] == rel.id:
                    found = True
                    break
            assert found


class TestSyncLearningGraphIntegration:
    """Test learning graph integration with real RAG backend (synchronous)."""
    
    @pytest.fixture
    def learning_system(self):
        """Create a learning system with RAG backend."""
        # Create learning graph (no RAG dependency for basic LearningGraph)
        learning_graph = LearningGraph(
            user_id="test_user",
            name="Test Learning Graph"
        )
        
        yield learning_graph
    
    def test_learning_progress_tracking(self, learning_system):
        """Test learning progress tracking with real backend."""
        learning_graph = learning_system
        
        # Record learning activity
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
    
    def test_basic_learning_tracking(self, learning_system):
        """Test basic learning tracking functionality."""
        learning_graph = learning_system
        
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
    
    def test_learning_gap_detection(self, learning_system):
        """Test basic learning gap detection."""
        learning_graph = learning_system
        
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


class TestSyncEndToEndIntegration:
    """Test complete end-to-end scenarios with real backends (synchronous)."""
    
    @pytest.fixture
    async def complete_system(self):
        """Create a complete system with RAG backend and learning components."""
        # Create RAG service
        rag_service = await create_simple_rag_service()
        
        # Create learning graph
        learning_graph = LearningGraph(
            user_id="complete_test_user",
            name="Complete Test Learning Graph"
        )
        
        yield {
            "rag_service": rag_service,
            "learning_graph": learning_graph
        }
        
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_rag_service_and_learning_integration(self, complete_system):
        """Test integration between RAG service and learning components."""
        rag_service = complete_system["rag_service"]
        learning_graph = complete_system["learning_graph"]
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Store some concepts in RAG
        concepts = [
            Entity(id="basic_arithmetic", type="concept", properties={"name": "Basic Arithmetic", "category": "Number Sense"}),
            Entity(id="integers", type="concept", properties={"name": "Integers", "category": "Number Sense"})
        ]
        
        for concept in concepts:
            await graph_storage.store_entity(concept)
        
        # Record learning activity
        learning_graph.record_exercise_attempt(
            exercise_id="integration_ex_001",
            concept_id="basic_arithmetic",
            result=True,
            difficulty=0.5,
            concept_weight=1.0
        )
        
        # Verify learning data
        mastery = learning_graph.get_mastery("basic_arithmetic")
        assert mastery > 0
        
        # Verify RAG data
        results = await graph_storage.query_graph("get_entities", {
            "type": "concept",
            "properties": {"name": "Basic Arithmetic"},
            "limit": 1
        })
        assert len(results) == 1
        assert results[0]["id"] == "basic_arithmetic"
    
    @pytest.mark.asyncio
    async def test_multi_user_scenario(self, complete_system):
        """Test multi-user scenario with shared knowledge graph."""
        rag_service = complete_system["rag_service"]
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create shared knowledge in RAG
        shared_concepts = [
            Entity(id="algebra_basics", type="concept", properties={"name": "Algebra Basics", "category": "Algebra"}),
            Entity(id="linear_equations", type="concept", properties={"name": "Linear Equations", "category": "Algebra"})
        ]
        
        for concept in shared_concepts:
            await graph_storage.store_entity(concept)
        
        # Create multiple user learning graphs
        users = ["user1", "user2", "user3"]
        learning_graphs = {}
        
        for user in users:
            learning_graphs[user] = LearningGraph(
                user_id=user,
                name=f"{user} Learning Graph"
            )
        
        # Simulate different learning patterns
        for i, user in enumerate(users):
            lg = learning_graphs[user]
            
            # Different success rates for different users
            success_rate = 0.5 + (i * 0.2)  # 0.5, 0.7, 0.9
            
            for j in range(5):
                success = (j / 5) < success_rate
                lg.record_exercise_attempt(
                    exercise_id=f"{user}_ex_{j}",
                    concept_id="algebra_basics",
                    result=success,
                    difficulty=0.6,
                    concept_weight=1.0
                )
        
        # Verify different mastery levels
        masteries = [learning_graphs[user].get_mastery("algebra_basics") for user in users]
        
        # Should have different mastery levels
        assert len(set(masteries)) > 1  # Not all the same
        assert all(m >= 0 for m in masteries)  # All non-negative


class TestSyncPerformanceWithRealBackends:
    """Test performance characteristics with real backends (synchronous)."""
    
    @pytest.fixture
    async def performance_system(self):
        """Create a system for performance testing."""
        # Create RAG service
        rag_service = await create_simple_rag_service()
        
        # Create learning graph
        learning_graph = LearningGraph(
            user_id="performance_test_user",
            name="Performance Test Learning Graph"
        )
        
        yield {
            "rag_service": rag_service,
            "learning_graph": learning_graph
        }
        
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_concept_retrieval_performance(self, performance_system):
        """Test performance of concept retrieval operations."""
        rag_service = performance_system["rag_service"]
        learning_graph = performance_system["learning_graph"]
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Pre-populate with test data
        concepts = ["basic_arithmetic", "integers", "fractions", "decimals", "linear_equations"]
        for concept_id in concepts:
            entity = Entity(
                id=concept_id,
                type="concept",
                properties={
                    "name": concept_id.replace("_", " ").title(),
                    "category": "Math",
                    "difficulty": 0.5
                }
            )
            await graph_storage.store_entity(entity)
        
        # Test retrieval performance
        import time
        start_time = time.time()
        
        for concept_id in concepts:
            results = await graph_storage.query_graph("get_entities", {
                "type": "concept",
                "properties": {"name": concept_id.replace("_", " ").title()},
                "limit": 1
            })
            assert len(results) == 1
        
        end_time = time.time()
        retrieval_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert retrieval_time < 5.0  # 5 seconds for 5 retrievals
    
    @pytest.mark.asyncio
    async def test_storage_performance(self, performance_system):
        """Test performance of storage operations."""
        rag_service = performance_system["rag_service"]
        learning_graph = performance_system["learning_graph"]
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Test storage performance
        import time
        start_time = time.time()
        
        # Store multiple entities
        for i in range(20):
            entity = Entity(
                id=f"perf_concept_{i}",
                type="concept",
                properties={
                    "name": f"Performance Concept {i}",
                    "category": "Performance Test",
                    "difficulty": 0.5
                }
            )
            await graph_storage.store_entity(entity)
        
        end_time = time.time()
        storage_time = end_time - start_time
        
        # Should complete within reasonable time
        assert storage_time < 10.0  # 10 seconds for 20 storage operations


class TestSyncErrorHandlingWithRealBackends:
    """Test error handling scenarios with real backends (synchronous)."""
    
    @pytest.fixture
    async def error_test_system(self):
        """Create a system for error handling tests."""
        rag_service = await create_simple_rag_service()
        yield rag_service
        await rag_service.close()
    
    @pytest.mark.asyncio
    async def test_nonexistent_entity_retrieval(self, error_test_system):
        """Test handling of nonexistent entity retrieval."""
        rag_service = error_test_system
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Try to get an entity that doesn't exist
        results = await graph_storage.query_graph("get_entities", {
            "type": "concept",
            "properties": {"name": "Nonexistent Concept"},
            "limit": 1
        })
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_nonexistent_relationship_retrieval(self, error_test_system):
        """Test handling of nonexistent relationship retrieval."""
        rag_service = error_test_system
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Try to get a relationship that doesn't exist
        results = await graph_storage.query_graph("get_relationships", {
            "source_id": "nonexistent_source",
            "type": "prerequisite",
            "limit": 1
        })
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_duplicate_entity_storage(self, error_test_system):
        """Test handling of duplicate entity storage."""
        rag_service = error_test_system
        graph_storage = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Store an entity
        entity1 = Entity(id="duplicate_test", type="concept", properties={"name": "First"})
        await graph_storage.store_entity(entity1)
        
        # Store another entity with the same ID (should overwrite)
        entity2 = Entity(id="duplicate_test", type="concept", properties={"name": "Second"})
        await graph_storage.store_entity(entity2)
        
        # Retrieve and verify it was overwritten
        results = await graph_storage.query_graph("get_entities", {
            "type": "concept",
            "properties": {"name": "Second"},
            "limit": 1
        })
        assert len(results) == 1
        assert results[0]["properties"]["name"] == "Second"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"]) 