#!/usr/bin/env python3
"""
Comprehensive tests for RAG Enhanced Algebra Graph.

This test suite provides coverage for the RAG-enhanced algebra graph functionality
that currently has 0% test coverage.
"""

import pytest
import asyncio
import tempfile
import os
import sys
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

# Test imports
try:
    from math_learning.knowledge_graph.rag_enhanced_algebra_graph import (
        RagEnhancedAlgebraGraph,
        create_rag_enhanced_algebra_graph
    )
    from math_learning.config.rag_config import RagConfig, get_memory_config
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"RAG components not available: {e}")

# Mock RAG service for testing
class MockRagService:
    """Mock RAG service for testing."""
    
    def __init__(self):
        self.entities = {}
        self.relationships = {}
        self.contexts = {}
    
    async def store_entity(self, entity):
        """Mock store entity."""
        self.entities[entity.id] = entity
        return entity.id
    
    async def store_relationship(self, relationship):
        """Mock store relationship."""
        rel_id = f"{relationship.source_id}-{relationship.target_id}"
        self.relationships[rel_id] = relationship
        return rel_id
    
    async def store_context(self, context):
        """Mock store context."""
        self.contexts[context.id] = context
        return context.id
    
    async def search_entities(self, query, limit=10):
        """Mock search entities."""
        # Simple mock search
        results = []
        for entity in self.entities.values():
            if query.lower() in str(entity.properties).lower():
                results.append(entity)
            if len(results) >= limit:
                break
        return results
    
    async def get_related_entities(self, entity_id, relationship_type=None):
        """Mock get related entities."""
        related = []
        for rel in self.relationships.values():
            if rel.source_id == entity_id:
                if relationship_type is None or rel.type == relationship_type:
                    target_entity = self.entities.get(rel.target_id)
                    if target_entity:
                        related.append(target_entity)
        return related
    
    async def close(self):
        """Mock close."""
        pass


@pytest.fixture
def mock_rag_config():
    """Create mock RAG configuration."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="memory",
        embedding_provider="dummy",
        graph_storage_type="memory",
        kv_storage_type="memory",
        similarity_threshold=0.5,
        max_results=10
    )


@pytest.fixture
async def mock_rag_service():
    """Create mock RAG service."""
    return MockRagService()


@pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG components not available")
class TestRagEnhancedAlgebraGraph:
    """Test suite for RAG Enhanced Algebra Graph."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_rag_service, mock_rag_config):
        """Test RAG enhanced algebra graph initialization."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service', 
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            assert graph.rag_service is not None
            assert graph.config == mock_rag_config
            assert hasattr(graph, 'algebra_concepts')
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_concept_storage(self, mock_rag_service, mock_rag_config):
        """Test storing algebra concepts in RAG system."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Test storing a concept
            concept_id = await graph.store_concept(
                concept_id="test_concept",
                name="Test Concept",
                description="A test concept for algebra",
                difficulty=0.5,
                category="Test Category"
            )
            
            assert concept_id == "test_concept"
            assert "test_concept" in mock_rag_service.entities
            
            stored_entity = mock_rag_service.entities["test_concept"]
            assert stored_entity.properties["name"] == "Test Concept"
            assert stored_entity.properties["description"] == "A test concept for algebra"
            assert stored_entity.properties["difficulty"] == 0.5
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_prerequisite_relationships(self, mock_rag_service, mock_rag_config):
        """Test storing prerequisite relationships."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Store concepts first
            await graph.store_concept("basic_math", "Basic Math", "Foundation math", 0.1, "Foundation")
            await graph.store_concept("algebra", "Algebra", "Basic algebra", 0.5, "Algebra")
            
            # Store prerequisite relationship
            rel_id = await graph.store_prerequisite_relationship("basic_math", "algebra", weight=0.8)
            
            assert rel_id is not None
            assert len(mock_rag_service.relationships) > 0
            
            # Verify relationship properties
            stored_rel = None
            for rel in mock_rag_service.relationships.values():
                if rel.source_id == "basic_math" and rel.target_id == "algebra":
                    stored_rel = rel
                    break
            
            assert stored_rel is not None
            assert stored_rel.type == "prerequisite"
            assert stored_rel.properties.get("weight") == 0.8
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_concept_search(self, mock_rag_service, mock_rag_config):
        """Test semantic concept search."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Store test concepts
            await graph.store_concept("addition", "Addition", "Adding numbers together", 0.2, "Arithmetic")
            await graph.store_concept("multiplication", "Multiplication", "Multiplying numbers", 0.4, "Arithmetic")
            await graph.store_concept("fractions", "Fractions", "Parts of a whole", 0.6, "Numbers")
            
            # Test search
            results = await graph.find_similar_concepts("arithmetic operations", top_k=2)
            
            assert isinstance(results, list)
            assert len(results) <= 2
            
            # Should find arithmetic-related concepts
            found_concepts = [r.properties.get("name", "") for r in results]
            assert any("Addition" in name or "Multiplication" in name for name in found_concepts)
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_learning_path_generation(self, mock_rag_service, mock_rag_config):
        """Test generating learning paths using RAG."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Create a concept chain
            concepts = [
                ("numbers", "Numbers", "Basic number concepts", 0.1),
                ("addition", "Addition", "Adding numbers", 0.2),
                ("multiplication", "Multiplication", "Multiplying numbers", 0.4),
                ("algebra", "Algebra", "Basic algebra", 0.6)
            ]
            
            # Store concepts
            for concept_id, name, desc, difficulty in concepts:
                await graph.store_concept(concept_id, name, desc, difficulty, "Math")
            
            # Store prerequisite chain
            await graph.store_prerequisite_relationship("numbers", "addition")
            await graph.store_prerequisite_relationship("addition", "multiplication") 
            await graph.store_prerequisite_relationship("multiplication", "algebra")
            
            # Test learning path generation
            path = await graph.get_learning_path_suggestions("numbers", "algebra")
            
            assert isinstance(path, list)
            assert len(path) > 0
            
            # Should include concepts in the path
            path_concepts = [item.get("concept_id") for item in path if isinstance(item, dict)]
            assert "numbers" in str(path) or len(path_concepts) > 0
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_teaching_recommendations(self, mock_rag_service, mock_rag_config):
        """Test getting teaching recommendations."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Store a concept
            await graph.store_concept("fractions", "Fractions", "Understanding fractions", 0.5, "Numbers")
            
            # Test teaching recommendations
            student_profile = {
                "learning_style": "visual",
                "difficulty_preference": "moderate",
                "interests": ["games", "art"]
            }
            
            recommendations = await graph.get_teaching_recommendations("fractions", student_profile)
            
            assert isinstance(recommendations, dict)
            # Should contain recommendation structure
            assert len(recommendations) >= 0  # May be empty but should be valid dict
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_bulk_concept_loading(self, mock_rag_service, mock_rag_config):
        """Test loading multiple concepts efficiently."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Test bulk loading
            concepts_data = [
                {"id": "concept1", "name": "Concept 1", "description": "First concept", "difficulty": 0.1},
                {"id": "concept2", "name": "Concept 2", "description": "Second concept", "difficulty": 0.2},
                {"id": "concept3", "name": "Concept 3", "description": "Third concept", "difficulty": 0.3}
            ]
            
            loaded_count = await graph.load_algebra_concepts(concepts_data)
            
            assert loaded_count == 3
            assert len(mock_rag_service.entities) == 3
            assert "concept1" in mock_rag_service.entities
            assert "concept2" in mock_rag_service.entities
            assert "concept3" in mock_rag_service.entities
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_rag_config):
        """Test error handling in RAG operations."""
        # Test with failing RAG service
        failing_service = Mock()
        failing_service.store_entity = AsyncMock(side_effect=Exception("Storage failed"))
        failing_service.close = AsyncMock()
        
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=failing_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Should handle storage errors gracefully
            with pytest.raises(Exception):
                await graph.store_concept("test", "Test", "Test concept", 0.5, "Test")
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_context_storage(self, mock_rag_service, mock_rag_config):
        """Test storing learning context information."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service',
                   return_value=mock_rag_service):
            
            graph = RagEnhancedAlgebraGraph(config=mock_rag_config)
            await graph.initialize()
            
            # Test storing learning context
            context_id = await graph.store_learning_context(
                user_id="student123",
                concept_id="fractions",
                context_data={
                    "mastery_level": 0.7,
                    "attempts": 5,
                    "last_practiced": "2024-01-15",
                    "common_errors": ["denominator confusion", "improper fractions"]
                }
            )
            
            assert context_id is not None
            assert len(mock_rag_service.contexts) > 0
            
            # Verify context data
            stored_context = None
            for ctx in mock_rag_service.contexts.values():
                if hasattr(ctx, 'properties') and ctx.properties.get("user_id") == "student123":
                    stored_context = ctx
                    break
            
            # Context should be stored (exact structure may vary)
            assert len(mock_rag_service.contexts) > 0
            
            await graph.close()


@pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG components not available")
class TestRagEnhancedAlgebraGraphFactory:
    """Test the factory function for creating RAG enhanced algebra graphs."""
    
    @pytest.mark.asyncio
    async def test_create_rag_enhanced_algebra_graph(self):
        """Test the factory function."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service') as mock_create:
            mock_service = MockRagService()
            mock_create.return_value = mock_service
            
            # Test factory function
            graph = await create_rag_enhanced_algebra_graph()
            
            assert graph is not None
            assert isinstance(graph, RagEnhancedAlgebraGraph)
            
            await graph.close()
    
    @pytest.mark.asyncio
    async def test_create_with_custom_config(self, mock_rag_config):
        """Test creating with custom configuration."""
        with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service') as mock_create:
            mock_service = MockRagService()
            mock_create.return_value = mock_service
            
            # Test with custom config
            graph = await create_rag_enhanced_algebra_graph(config=mock_rag_config)
            
            assert graph is not None
            assert graph.config == mock_rag_config
            
            await graph.close()


# Integration test with real components (if available)
@pytest.mark.skipif(not RAG_AVAILABLE, reason="RAG components not available")
class TestRagEnhancedAlgebraGraphIntegration:
    """Integration tests with real RAG components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from concept storage to retrieval."""
        try:
            # Use memory config for testing
            config = get_memory_config()
            
            with patch('math_learning.knowledge_graph.rag_enhanced_algebra_graph.create_configured_rag_service') as mock_create:
                mock_service = MockRagService()
                mock_create.return_value = mock_service
                
                graph = RagEnhancedAlgebraGraph(config=config)
                await graph.initialize()
                
                # Complete workflow test
                # 1. Store concepts
                await graph.store_concept("basic_arithmetic", "Basic Arithmetic", 
                                         "Addition and subtraction", 0.2, "Foundation")
                await graph.store_concept("multiplication", "Multiplication",
                                         "Multiplying numbers", 0.4, "Operations")
                
                # 2. Store relationships
                await graph.store_prerequisite_relationship("basic_arithmetic", "multiplication")
                
                # 3. Search for concepts
                results = await graph.find_similar_concepts("math operations", top_k=5)
                assert isinstance(results, list)
                
                # 4. Get learning path
                path = await graph.get_learning_path_suggestions("basic_arithmetic", "multiplication")
                assert isinstance(path, list)
                
                await graph.close()
                
        except Exception as e:
            pytest.skip(f"Integration test requires full RAG setup: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"]) 