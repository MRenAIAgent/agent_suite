#!/usr/bin/env python3
"""
Direct tests for storage backends using real databases.

This test suite directly tests the Qdrant vector storage and Neo4j graph storage
implementations to ensure they work correctly with real database instances.

Requirements:
- Qdrant running on localhost:6333
- Neo4j running on localhost:7687 with neo4j/password
"""

import asyncio
import sys
import os
import time
import traceback
import pytest
import uuid
from typing import Dict, List, Any
import numpy as np

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rag.storage.graph.neo4j_storage import Neo4jGraphStorageAdaptor
from agents.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.document import Document, DocumentChunk
from agents.rag.utils.embedding_provider import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
from math_learning.scripts.utils.database_usage_checker import DatabaseUsageChecker


@pytest.fixture(scope="session")
async def usage_checker():
    """Fixture to provide database usage checking."""
    return DatabaseUsageChecker()


@pytest.fixture(scope="session", autouse=True)
async def check_database_usage(usage_checker):
    """Check database usage before and after all tests."""
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE BEFORE STORAGE BACKEND TESTS")
    print("="*80)
    
    # Check usage before tests
    usage_before = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_before, "Database Usage BEFORE Storage Tests")
    
    yield  # Run all tests
    
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE AFTER STORAGE BACKEND TESTS")
    print("="*80)
    
    # Check usage after tests
    usage_after = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_after, "Database Usage AFTER Storage Tests")
    
    # Calculate and show the difference
    usage_checker.print_usage_comparison(usage_before, usage_after, "Storage Backend Tests Impact")


@pytest.fixture(scope="session")
async def qdrant_storage():
    """Fixture to provide QdrantStorage instance."""
    try:
        # Try to use SentenceTransformer for embeddings
        embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        print("✅ Using SentenceTransformer for embeddings")
    except (ImportError, Exception) as e:
        print(f"⚠️ Could not create SentenceTransformer provider: {e}")
        print("✅ Using DummyEmbeddingProvider as fallback")
        embedding_provider = DummyEmbeddingProvider(vector_size=384)
    
    storage = QdrantStorageAdaptor(
        collection_name="test_storage_backend",
        vector_size=384,
        embedding_provider=embedding_provider,
        host="localhost",
        port=6333
    )
    yield storage
    # Note: QdrantStorageAdaptor doesn't have a clear() method


@pytest.fixture(scope="session")
async def neo4j_storage():
    """Fixture to provide Neo4jStorage instance."""
    storage = Neo4jGraphStorageAdaptor(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        namespace="test_storage_backend"
    )
    yield storage
    # Cleanup after tests
    storage.close()


async def test_qdrant_storage(qdrant_storage):
    """Test Qdrant vector storage functionality."""
    print("\n🔍 Testing Qdrant vector storage...")
    
    # Create test documents
    documents = [
        Document(
            content="Addition is the mathematical operation of combining numbers",
            metadata={"concept": "addition", "difficulty": 0.2},
            id=str(uuid.uuid4())
        ),
        Document(
            content="Multiplication is repeated addition",
            metadata={"concept": "multiplication", "difficulty": 0.4},
            id=str(uuid.uuid4())
        ),
        Document(
            content="Fractions represent parts of a whole",
            metadata={"concept": "fractions", "difficulty": 0.6},
            id=str(uuid.uuid4())
        )
    ]
    
    # Store documents
    stored_ids = []
    for doc in documents:
        doc_id = await qdrant_storage.store_document(doc)
        stored_ids.append(doc_id)
        print(f"  ✅ Stored document: {doc.metadata['concept']} (ID: {doc_id})")
    
    # Test vector search
    search_results = await qdrant_storage.search(
        query="mathematical operations with numbers",
        top_k=2
    )
    print(f"  🔍 Found {len(search_results)} similar documents")
    return True


async def test_neo4j_storage(neo4j_storage):
    """Test Neo4j graph storage functionality."""
    print("\n🕸️ Testing Neo4j graph storage...")
    
    # Create test entities
    entities = [
        {
            "id": "addition",
            "type": "MathConcept",
            "name": "Addition",
            "description": "Basic arithmetic operation",
            "properties": {"difficulty": 0.2, "grade_level": 1}
        },
        {
            "id": "multiplication", 
            "type": "MathConcept",
            "name": "Multiplication",
            "description": "Repeated addition operation",
            "properties": {"difficulty": 0.4, "grade_level": 3}
        }
    ]
    
    # Store entities
    for entity_data in entities:
        entity_id = await neo4j_storage.store(entity_data)
        print(f"  ✅ Stored entity: {entity_data['name']} (ID: {entity_id})")
    
    # Create relationship
    relationship_data = {
        "source_id": "addition",
        "target_id": "multiplication", 
        "type": "prerequisite",
        "properties": {"strength": 0.8}
    }
    
    rel_id = await neo4j_storage.store(relationship_data)
    print(f"  🔗 Created relationship: addition -> multiplication (ID: {rel_id})")
    
    # Test graph query
    query_results = await neo4j_storage.query_graph(
        query="""
        MATCH (a:Entity {id: $entity_id})-[r]->(b:Entity)
        RETURN a.name as source, type(r) as relationship, b.name as target
        """,
        parameters={"entity_id": "addition"}
    )
    print(f"  🔍 Found {len(query_results)} graph relationships")
    return True


async def test_cross_system_integration(qdrant_storage, neo4j_storage):
    """Test integration between Qdrant and Neo4j storage systems."""
    print("\n🔄 Testing cross-system integration...")
    
    # Store the same concept in both systems
    concept_id = "algebra_basics"
    
    # Store in Qdrant as a document
    doc = Document(
        content="Algebra involves solving equations with variables and unknown values",
        metadata={"concept": concept_id, "difficulty": 0.7, "grade_level": 8},
        id=str(uuid.uuid4())
    )
    vector_id = await qdrant_storage.store_document(doc)
    print(f"  ✅ Stored in Qdrant: {concept_id} (Vector ID: {vector_id})")
    
    # Store in Neo4j as an entity
    entity_data = {
        "id": concept_id,
        "type": "MathConcept",
        "name": "Algebra Basics",
        "description": "Introduction to algebraic concepts",
        "properties": {"difficulty": 0.7, "grade_level": 8, "vector_id": vector_id}
    }
    graph_id = await neo4j_storage.store(entity_data)
    print(f"  ✅ Stored in Neo4j: {concept_id} (Graph ID: {graph_id})")
    
    # Test cross-system query: find related concepts in graph, then search vectors
    related_query = await neo4j_storage.query_graph(
        query="""
        MATCH (concept:Entity {id: $concept_id})
        RETURN concept.name as name, concept.properties as properties
        """,
        parameters={"concept_id": concept_id}
    )
    
    if related_query:
        print(f"  🔍 Found concept in graph: {related_query[0].get('name', 'Unknown')}")
        
        # Search for similar concepts in vector space
        similar_docs = await qdrant_storage.search(
            query="mathematical equations and variables",
            top_k=3
        )
        print(f"  🔍 Found {len(similar_docs)} similar documents in vector space")
    
    return True


async def main():
    """Run all storage backend tests."""
    print("🚀 Starting Storage Backend Tests with Real Databases")
    print("="*80)
    
    # Check database availability
    print("🔍 Checking database availability...")
    usage_checker = DatabaseUsageChecker()
    
    # Check Neo4j
    neo4j_usage = await usage_checker.check_neo4j_usage()
    neo4j_available = neo4j_usage.get('status') == 'connected'
    
    # Check Qdrant
    qdrant_usage = await usage_checker.check_qdrant_usage()
    qdrant_available = qdrant_usage.get('status') == 'connected'
    
    if not qdrant_available:
        print("❌ Qdrant is not available. Please start it with: docker-compose up -d qdrant")
        return False
    
    if not neo4j_available:
        print("❌ Neo4j is not available. Please start it with: docker-compose up -d neo4j")
        return False
    
    print("✅ Both databases are available")
    
    # Run the tests
    test_results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "errors": []
    }
    
    # Initialize storage instances
    try:
        # Try to use SentenceTransformer for embeddings
        embedding_provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
        print("✅ Using SentenceTransformer for embeddings")
    except (ImportError, Exception) as e:
        print(f"⚠️ Could not create SentenceTransformer provider: {e}")
        print("✅ Using DummyEmbeddingProvider as fallback")
        embedding_provider = DummyEmbeddingProvider(vector_size=384)
    
    qdrant_storage = QdrantStorageAdaptor(
        collection_name="test_storage_backend",
        vector_size=384,
        embedding_provider=embedding_provider,
        host="localhost",
        port=6333
    )
    
    neo4j_storage = Neo4jGraphStorageAdaptor(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        namespace="test_storage_backend"
    )
    
    try:
        # Test functions to run
        tests = [
            ("test_qdrant_storage", test_qdrant_storage, [qdrant_storage]),
            ("test_neo4j_storage", test_neo4j_storage, [neo4j_storage]),
            ("test_cross_system_integration", test_cross_system_integration, [qdrant_storage, neo4j_storage])
        ]
        
        for test_name, test_func, args in tests:
            test_results["total"] += 1
            print(f"\n🧪 Running {test_name}...")
            
            try:
                # Call the test function with its arguments
                await test_func(*args)
                
                test_results["passed"] += 1
                print(f"✅ {test_name} PASSED")
                
            except Exception as e:
                test_results["failed"] += 1
                error_msg = f"{test_name} FAILED: {str(e)}"
                test_results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                print(f"   Traceback: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up test data...")
        try:
            # Note: Storage adaptors don't have clear() methods
            await neo4j_storage.close()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
    # Print final results
    print("\n" + "="*80)
    print("📊 STORAGE BACKEND TEST RESULTS")
    print("="*80)
    print(f"Total tests: {test_results['total']}")
    print(f"Passed: {test_results['passed']}")
    print(f"Failed: {test_results['failed']}")
    
    if test_results["failed"] > 0:
        print("\n❌ Failed tests:")
        for error in test_results["errors"]:
            print(f"  - {error}")
    
    success_rate = (test_results["passed"] / test_results["total"]) * 100 if test_results["total"] > 0 else 0
    print(f"\n🎯 Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Storage backend tests completed successfully!")
        return True
    else:
        print("⚠️ Some storage backend tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1) 