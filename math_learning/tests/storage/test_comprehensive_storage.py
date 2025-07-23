#!/usr/bin/env python3
"""
Comprehensive Storage Backend Tests

This test suite provides comprehensive coverage for all storage backends
used in the math learning system, including:
- Memory-based storage (for testing)
- Neo4j graph database storage
- Qdrant vector storage
- Redis key-value storage
- RAG integration storage
- Performance and scalability testing
- Error handling and edge cases
"""

import pytest
import asyncio
import tempfile
import os
import time
import json
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import sys

# Add project root to path
# From math_learning/tests/storage/test_comprehensive_storage.py
# Go up 4 levels: storage -> tests -> math_learning -> agent_suite
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Storage imports
from agents.rag.storage.graph.memory_storage import MemoryGraphStorageAdaptor
from agents.rag.storage.key_value.memory_storage import MemoryKeyValueStorage
from agents.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor
from agents.rag.storage.graph.neo4j_storage import Neo4jGraphStorageAdaptor
from agents.rag.storage.key_value.redis_storage import RedisKeyValueStorageAdaptor
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agents.rag.models.context import ContextItem
from agents.rag.models.document import Document, DocumentChunk
from agents.rag.middleware.storage_router import StorageType

# Math learning imports
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.concept import Concept
from math_learning.config.rag_config import get_memory_config, get_real_backends_config
from math_learning.scripts.utils.database_usage_checker import DatabaseUsageChecker


class TestMemoryStorage:
    """Comprehensive tests for memory-based storage backends."""
    
    @pytest.fixture
    def graph_storage(self):
        """Create memory graph storage."""
        return MemoryGraphStorageAdaptor()
    
    @pytest.fixture
    def kv_storage(self):
        """Create memory key-value storage."""
        return MemoryKeyValueStorage()
    
    def test_graph_storage_basic_operations(self, graph_storage):
        """Test basic graph storage operations."""
        # Test entity storage
        entity = Entity(
            type="math_concept",
            properties={
                "name": "Basic Arithmetic",
                "difficulty": 0.3,
                "category": "Elementary Math"
            }
        )
        
        # Store entity
        entity_id = asyncio.run(graph_storage.store_entity(entity))
        assert entity_id == entity.id
        assert entity_id in graph_storage.entities
        
        # Retrieve entity
        retrieved = asyncio.run(graph_storage.get_entity(entity_id))
        assert retrieved is not None
        assert retrieved.properties["name"] == "Basic Arithmetic"
        assert retrieved.properties["difficulty"] == 0.3
        
        # Test relationship storage
        entity2 = Entity(
            type="math_concept",
            properties={"name": "Addition", "difficulty": 0.2}
        )
        entity2_id = asyncio.run(graph_storage.store_entity(entity2))
        
        relationship = Relationship(
            source_id=entity2_id,
            target_id=entity_id,
            type="prerequisite",
            properties={"strength": 0.9}
        )
        
        rel_id = asyncio.run(graph_storage.store_relationship(relationship))
        assert rel_id == relationship.id
        assert rel_id in graph_storage.relationships
        
        # Test graph queries
        neighbors = asyncio.run(graph_storage.query_graph(
            "get_neighbors",
            {"entity_id": entity_id, "direction": "incoming"}
        ))
        assert len(neighbors) == 1
        assert neighbors[0]["entity_id"] == entity2_id
    
    def test_graph_storage_search_functionality(self, graph_storage):
        """Test graph storage search capabilities."""
        # Create multiple entities
        entities = []
        for i in range(5):
            entity = Entity(
                type="math_concept",
                properties={
                    "name": f"Concept {i}",
                    "description": f"Mathematical concept about {['numbers', 'equations', 'geometry', 'algebra', 'calculus'][i]}",
                    "difficulty": i * 0.2,
                    "category": "Mathematics"
                }
            )
            entity_id = asyncio.run(graph_storage.store_entity(entity))
            entities.append((entity_id, entity))
        
        # Test text search
        results = asyncio.run(graph_storage.search("algebra"))
        assert len(results) > 0
        found_algebra = False
        for result in results:
            if "algebra" in result.get("description", "").lower():
                found_algebra = True
                break
        assert found_algebra
        
        # Test structured search
        structured_results = asyncio.run(graph_storage.search({
            "type": "math_concept",
            "properties.difficulty": {"$lt": 0.5}
        }))
        assert len(structured_results) >= 2  # Should find concepts 0, 1, 2
    
    def test_kv_storage_operations(self, kv_storage):
        """Test key-value storage operations."""
        # Test basic set/get
        key = "test_key"
        value = {"data": "test_value", "timestamp": time.time()}
        
        result_id = asyncio.run(kv_storage.set(key, value))
        assert result_id == key
        
        retrieved = asyncio.run(kv_storage.get(key))
        assert retrieved == value
        
        # Test context item storage
        context_item = ContextItem(
            key="learning_progress_001",
            value={
                "user_id": "student_001",
                "concept": "arithmetic",
                "mastery": 0.75,
                "exercises_completed": 10
            },
            metadata={"type": "progress", "timestamp": time.time()}
        )
        
        item_id = asyncio.run(kv_storage.store_context_item(context_item))
        assert item_id == context_item.id
        
        # Test get all context
        all_context = asyncio.run(kv_storage.get_all_context())
        assert len(all_context) == 1
        assert all_context[0].key == "learning_progress_001"
        assert all_context[0].value["mastery"] == 0.75
        
        # Test prefix search
        asyncio.run(kv_storage.set("progress_001", {"user": "A"}))
        asyncio.run(kv_storage.set("progress_002", {"user": "B"}))
        asyncio.run(kv_storage.set("settings_001", {"theme": "dark"}))
        
        progress_items = asyncio.run(kv_storage.get_by_prefix("progress_"))
        assert len(progress_items) == 2
        assert "progress_001" in progress_items
        assert "progress_002" in progress_items
        assert "settings_001" not in progress_items
    
    def test_storage_performance(self, graph_storage, kv_storage):
        """Test storage performance with larger datasets."""
        # Performance test for graph storage
        start_time = time.time()
        entity_ids = []
        
        # Store 100 entities
        for i in range(100):
            entity = Entity(
                type="test_entity",
                properties={
                    "name": f"Entity {i}",
                    "value": i,
                    "category": f"Category {i % 10}"
                }
            )
            entity_id = asyncio.run(graph_storage.store_entity(entity))
            entity_ids.append(entity_id)
        
        storage_time = time.time() - start_time
        assert storage_time < 5.0  # Should complete within 5 seconds
        
        # Performance test for retrieval
        start_time = time.time()
        for entity_id in entity_ids[:20]:  # Test first 20
            retrieved = asyncio.run(graph_storage.get_entity(entity_id))
            assert retrieved is not None
        
        retrieval_time = time.time() - start_time
        assert retrieval_time < 1.0  # Should complete within 1 second
        
        # Performance test for key-value storage
        start_time = time.time()
        for i in range(100):
            asyncio.run(kv_storage.set(f"key_{i}", {"data": f"value_{i}", "index": i}))
        
        kv_storage_time = time.time() - start_time
        assert kv_storage_time < 2.0  # Should complete within 2 seconds
    
    def test_storage_cleanup(self, graph_storage, kv_storage):
        """Test storage cleanup and resource management."""
        # Add some data
        entity = Entity(type="temp", properties={"name": "temp"})
        entity_id = asyncio.run(graph_storage.store_entity(entity))
        
        asyncio.run(kv_storage.set("temp_key", "temp_value"))
        
        # Test cleanup
        asyncio.run(graph_storage.close())
        asyncio.run(kv_storage.close())
        
        # Verify cleanup doesn't cause errors
        assert True  # If we get here, cleanup worked


class TestRealDatabaseStorage:
    """Tests for real database storage backends."""
    
    @pytest.fixture
    def database_checker(self):
        """Create database usage checker."""
        return DatabaseUsageChecker()
    
    @pytest.mark.asyncio
    async def test_neo4j_graph_storage(self, database_checker):
        """Test Neo4j graph storage if available."""
        # Check if Neo4j is available
        neo4j_available = await database_checker.check_neo4j()
        if not neo4j_available:
            pytest.skip("Neo4j not available")
        
        try:
            # Create Neo4j storage
            storage = Neo4jGraphStorageAdaptor(
                uri="bolt://localhost:7687",
                username="neo4j",
                password="neo4j",
                namespace="test_math_learning"
            )
            
            # Test entity storage
            entity = Entity(
                type="math_concept",
                properties={
                    "name": "Neo4j Test Concept",
                    "difficulty": 0.5,
                    "test_id": "neo4j_test"
                }
            )
            
            entity_id = await storage.store_entity(entity)
            assert entity_id == entity.id
            
            # Test entity retrieval
            retrieved = await storage.get_entity(entity_id)
            assert retrieved is not None
            assert retrieved.properties["name"] == "Neo4j Test Concept"
            
            # Test relationship storage
            entity2 = Entity(
                type="math_concept",
                properties={
                    "name": "Prerequisite Concept",
                    "test_id": "neo4j_test"
                }
            )
            entity2_id = await storage.store_entity(entity2)
            
            relationship = Relationship(
                source_id=entity2_id,
                target_id=entity_id,
                type="prerequisite",
                properties={"strength": 0.8, "test_id": "neo4j_test"}
            )
            
            rel_id = await storage.store_relationship(relationship)
            assert rel_id == relationship.id
            
            # Test graph queries
            neighbors = await storage.query_graph(
                "get_neighbors",
                {"entity_id": entity_id, "direction": "incoming"}
            )
            assert len(neighbors) >= 1
            
            # Cleanup test data
            await storage.execute_cypher(
                "MATCH (n {test_id: 'neo4j_test'}) DETACH DELETE n"
            )
            
        except Exception as e:
            pytest.fail(f"Neo4j test failed: {e}")
        finally:
            if 'storage' in locals():
                await storage.close()
    
    @pytest.mark.asyncio
    async def test_qdrant_vector_storage(self, database_checker):
        """Test Qdrant vector storage if available."""
        # Check if Qdrant is available
        qdrant_available = await database_checker.check_qdrant()
        if not qdrant_available:
            pytest.skip("Qdrant not available")
        
        try:
            # Create Qdrant storage
            storage = QdrantStorageAdaptor(
                host="localhost",
                port=6333,
                collection_name="test_math_learning"
            )
            
            # Test document storage
            document = Document(
                content="This is a test document about basic arithmetic operations including addition and subtraction.",
                metadata={
                    "concept": "arithmetic",
                    "difficulty": 0.3,
                    "test_id": "qdrant_test"
                }
            )
            
            doc_id = await storage.store_document(document)
            assert doc_id is not None
            
            # Test similarity search
            results = await storage.search_similar(
                query="addition and subtraction",
                limit=5,
                threshold=0.1
            )
            assert len(results) >= 1
            
            # Find our test document
            found_test_doc = False
            for result in results:
                if result.metadata.get("test_id") == "qdrant_test":
                    found_test_doc = True
                    assert result.metadata["concept"] == "arithmetic"
                    break
            assert found_test_doc
            
            # Test batch operations
            batch_docs = [
                Document(
                    content=f"Test document {i} about mathematical concept number {i}",
                    metadata={"test_id": "qdrant_test", "doc_num": i}
                )
                for i in range(5)
            ]
            
            batch_ids = await storage.store_batch(
                documents=[doc.content for doc in batch_docs],
                metadatas=[doc.metadata for doc in batch_docs]
            )
            assert len(batch_ids) == 5
            
            # Cleanup test data
            # Note: Qdrant cleanup would require deleting by filter
            # For now, we'll leave test data (it's in a test collection)
            
        except Exception as e:
            pytest.fail(f"Qdrant test failed: {e}")
        finally:
            if 'storage' in locals():
                await storage.close()


class TestIntegratedStorageScenarios:
    """Tests for integrated storage scenarios combining multiple backends."""
    
    @pytest.mark.asyncio
    async def test_learning_progress_storage_integration(self):
        """Test storing learning progress across multiple storage backends."""
        # Create storage backends
        graph_storage = MemoryGraphStorageAdaptor()
        kv_storage = MemoryKeyValueStorage()
        
        # Store math concepts in graph storage
        concepts = {
            "arithmetic": Entity(
                type="math_concept",
                properties={
                    "name": "Basic Arithmetic",
                    "description": "Addition, subtraction, multiplication, division",
                    "difficulty": 0.3,
                    "category": "Elementary Math"
                }
            ),
            "algebra": Entity(
                type="math_concept",
                properties={
                    "name": "Basic Algebra",
                    "description": "Working with variables and simple equations",
                    "difficulty": 0.6,
                    "category": "Middle School Math"
                }
            )
        }
        
        concept_ids = {}
        for key, concept in concepts.items():
            concept_id = await graph_storage.store_entity(concept)
            concept_ids[key] = concept_id
        
        # Store prerequisite relationship
        prereq_rel = Relationship(
            source_id=concept_ids["arithmetic"],
            target_id=concept_ids["algebra"],
            type="prerequisite",
            properties={"strength": 0.9}
        )
        await graph_storage.store_relationship(prereq_rel)
        
        # Create learning graph and record progress
        learning_graph = LearningGraph("test_student", "Test Student")
        
        # Simulate learning sessions
        learning_sessions = [
            ("arithmetic", [(True, 0.3), (True, 0.4), (False, 0.5), (True, 0.3)]),
            ("algebra", [(False, 0.6), (False, 0.7), (True, 0.5), (True, 0.6)])
        ]
        
        for concept, attempts in learning_sessions:
            for i, (success, difficulty) in enumerate(attempts):
                exercise_id = f"{concept}_ex_{i+1}"
                learning_graph.record_exercise_attempt(
                    exercise_id, concept, success, difficulty, 1.0
                )
        
        # Store learning progress in key-value storage
        for concept in ["arithmetic", "algebra"]:
            progress_item = ContextItem(
                key=f"progress_{concept}_test_student",
                value={
                    "user_id": "test_student",
                    "concept": concept,
                    "mastery": learning_graph.get_mastery(concept),
                    "confidence": learning_graph.get_confidence(concept),
                    "exercises_completed": len([
                        ex for ex in learning_graph.exercise_history 
                        if any(attempt["concept_id"] == concept 
                              for attempt in learning_graph.exercise_history[ex])
                    ]),
                    "timestamp": time.time()
                },
                metadata={"type": "learning_progress", "user": "test_student"}
            )
            await kv_storage.store_context_item(progress_item)
        
        # Verify integrated data
        # Check that concepts are stored in graph
        arithmetic_concept = await graph_storage.get_entity(concept_ids["arithmetic"])
        assert arithmetic_concept.properties["name"] == "Basic Arithmetic"
        
        algebra_concept = await graph_storage.get_entity(concept_ids["algebra"])
        assert algebra_concept.properties["name"] == "Basic Algebra"
        
        # Check prerequisite relationship
        neighbors = await graph_storage.query_graph(
            "get_neighbors",
            {"entity_id": concept_ids["algebra"], "direction": "incoming"}
        )
        assert len(neighbors) == 1
        assert neighbors[0]["entity_id"] == concept_ids["arithmetic"]
        
        # Check learning progress in key-value storage
        all_progress = await kv_storage.get_all_context()
        assert len(all_progress) == 2
        
        progress_by_concept = {item.value["concept"]: item for item in all_progress}
        
        # Verify arithmetic progress
        arithmetic_progress = progress_by_concept["arithmetic"]
        assert arithmetic_progress.value["mastery"] > 0.0
        assert arithmetic_progress.value["exercises_completed"] == 4
        
        # Verify algebra progress
        algebra_progress = progress_by_concept["algebra"]
        assert algebra_progress.value["mastery"] > 0.0
        assert algebra_progress.value["exercises_completed"] == 4
        
        # Test cross-storage queries
        # Get concepts that are prerequisites for algebra
        prereq_neighbors = await graph_storage.query_graph(
            "get_neighbors",
            {"entity_id": concept_ids["algebra"], "direction": "incoming", "relationship_type": "prerequisite"}
        )
        
        # Check progress for prerequisite concepts
        for neighbor in prereq_neighbors:
            prereq_id = neighbor["entity_id"]
            if prereq_id == concept_ids["arithmetic"]:
                # Get progress for this prerequisite
                progress_key = f"progress_arithmetic_test_student"
                progress = await kv_storage.get(progress_key)
                # Note: get() returns the raw value, not the ContextItem
                # So we need to search through all context items
                arithmetic_context = None
                for item in all_progress:
                    if item.value["concept"] == "arithmetic":
                        arithmetic_context = item
                        break
                
                assert arithmetic_context is not None
                assert arithmetic_context.value["mastery"] > 0.0
        
        # Cleanup
        await graph_storage.close()
        await kv_storage.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_storage_operations(self):
        """Test concurrent storage operations for thread safety."""
        graph_storage = MemoryGraphStorageAdaptor()
        kv_storage = MemoryKeyValueStorage()
        
        # Define concurrent operations
        async def store_entities(start_idx, count):
            """Store entities concurrently."""
            entity_ids = []
            for i in range(start_idx, start_idx + count):
                entity = Entity(
                    type="concurrent_test",
                    properties={
                        "name": f"Entity {i}",
                        "thread": start_idx,
                        "index": i
                    }
                )
                entity_id = await graph_storage.store_entity(entity)
                entity_ids.append(entity_id)
            return entity_ids
        
        async def store_kv_pairs(start_idx, count):
            """Store key-value pairs concurrently."""
            for i in range(start_idx, start_idx + count):
                await kv_storage.set(
                    f"concurrent_key_{i}",
                    {"value": f"data_{i}", "thread": start_idx, "index": i}
                )
        
        # Run concurrent operations
        tasks = []
        for i in range(0, 50, 10):  # 5 tasks, 10 items each
            tasks.append(store_entities(i, 10))
            tasks.append(store_kv_pairs(i, 10))
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Check for exceptions
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Verify all data was stored correctly
        # Check entities
        all_entities = []
        for entity_id, entity in graph_storage.entities.items():
            if entity.type == "concurrent_test":
                all_entities.append(entity)
        
        assert len(all_entities) == 50
        
        # Check key-value pairs
        concurrent_keys = []
        for key in graph_storage.data.keys() if hasattr(graph_storage, 'data') else kv_storage.data.keys():
            if key.startswith("concurrent_key_"):
                concurrent_keys.append(key)
        
        # Note: MemoryKeyValueStorage stores in self.data
        concurrent_kv_count = len([k for k in kv_storage.data.keys() if k.startswith("concurrent_key_")])
        assert concurrent_kv_count == 50
        
        # Performance check - concurrent operations should be reasonably fast
        assert end_time - start_time < 5.0
        
        # Cleanup
        await graph_storage.close()
        await kv_storage.close()


class TestStorageErrorHandling:
    """Tests for storage error handling and edge cases."""
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data."""
        graph_storage = MemoryGraphStorageAdaptor()
        kv_storage = MemoryKeyValueStorage()
        
        # Test invalid entity data
        with pytest.raises((ValueError, TypeError, AttributeError)):
            invalid_entity = Entity(type=None, properties=None)
            asyncio.run(graph_storage.store_entity(invalid_entity))
        
        # Test invalid relationship data
        with pytest.raises((ValueError, TypeError, AttributeError)):
            invalid_relationship = Relationship(
                source_id="",  # Empty source
                target_id="nonexistent",
                type="invalid"
            )
            asyncio.run(graph_storage.store_relationship(invalid_relationship))
        
        # Test invalid key-value data
        # Note: Most key-value stores are flexible, so we test edge cases
        result = asyncio.run(kv_storage.set("", "empty_key_test"))  # Empty key
        assert result == ""  # Should handle gracefully
        
        result = asyncio.run(kv_storage.set("test_none", None))  # None value
        assert result == "test_none"  # Should handle gracefully
        
        retrieved = asyncio.run(kv_storage.get("test_none"))
        assert retrieved is None  # Should retrieve None value correctly
    
    def test_storage_limits(self):
        """Test storage behavior at limits."""
        graph_storage = MemoryGraphStorageAdaptor()
        kv_storage = MemoryKeyValueStorage()
        
        # Test large data storage
        large_properties = {f"prop_{i}": f"value_{i}" * 100 for i in range(100)}
        large_entity = Entity(
            type="large_test",
            properties=large_properties
        )
        
        entity_id = asyncio.run(graph_storage.store_entity(large_entity))
        assert entity_id is not None
        
        retrieved = asyncio.run(graph_storage.get_entity(entity_id))
        assert retrieved is not None
        assert len(retrieved.properties) == 100
        
        # Test large key-value storage
        large_value = {"data": "x" * 10000}  # 10KB value
        result = asyncio.run(kv_storage.set("large_value", large_value))
        assert result == "large_value"
        
        retrieved_large = asyncio.run(kv_storage.get("large_value"))
        assert len(retrieved_large["data"]) == 10000
    
    def test_cleanup_after_errors(self):
        """Test cleanup behavior after errors."""
        graph_storage = MemoryGraphStorageAdaptor()
        kv_storage = MemoryKeyValueStorage()
        
        # Store some valid data first
        valid_entity = Entity(
            type="valid",
            properties={"name": "Valid Entity"}
        )
        valid_id = asyncio.run(graph_storage.store_entity(valid_entity))
        
        asyncio.run(kv_storage.set("valid_key", "valid_value"))
        
        # Try to cause an error and then cleanup
        try:
            # This might cause an error depending on implementation
            asyncio.run(graph_storage.store_entity("invalid_entity"))
        except:
            pass  # Expected to fail
        
        # Verify valid data is still accessible after error
        retrieved = asyncio.run(graph_storage.get_entity(valid_id))
        assert retrieved is not None
        assert retrieved.properties["name"] == "Valid Entity"
        
        retrieved_kv = asyncio.run(kv_storage.get("valid_key"))
        assert retrieved_kv == "valid_value"
        
        # Test cleanup
        asyncio.run(graph_storage.close())
        asyncio.run(kv_storage.close())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"]) 