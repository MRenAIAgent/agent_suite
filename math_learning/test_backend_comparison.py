#!/usr/bin/env python3
"""
Compare memory backends vs real backends for the math learning system.

This test demonstrates the difference between using memory-based storage
(for testing/development) and real database backends (for production).
"""

import asyncio
import sys
import time
from typing import Dict, Any
import os

# Add correct paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # This should allow importing 'agents'
sys.path.insert(0, current_dir)

from config.rag_config import get_memory_config, get_real_backends_config, create_configured_rag_service
# Import only the LearningGraph class directly to avoid circular imports
from learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem

def check_real_backends_available():
    """Check if real backends (Neo4j and Redis) are available."""
    try:
        import neo4j
        import redis
        
        # Test Neo4j connection
        driver = neo4j.GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        
        # Test Redis connection
        client = redis.Redis(host="localhost", port=6379)
        client.ping()
        client.close()
        
        return True
    except Exception:
        return False

async def test_memory_backends():
    """Test the system with memory backends."""
    print("🧪 Testing with Memory Backends")
    print("----------------------------------------")
    
    start_time = time.time()
    try:
        # Create RAG service with memory backends
        config = get_memory_config()
        rag_service = await create_configured_rag_service(config)
        print("  ✅ RAG service created")
        
        # Test basic operations
        # Create some test entities
        concept1 = Entity(
            id="basic_arithmetic",
            type="concept",
            properties={"name": "Basic Arithmetic", "difficulty": 1}
        )
        
        concept2 = Entity(
            id="fractions",
            type="concept", 
            properties={"name": "Fractions", "difficulty": 3}
        )
        
        # Store entities
        await rag_service.store_knowledge(concept1)
        await rag_service.store_knowledge(concept2)
        print("  ✅ Stored test entities")
        
        # Create a relationship
        prerequisite = Relationship(
            id="prereq_1",
            type="prerequisite",
            source_id="basic_arithmetic",
            target_id="fractions",
            properties={"strength": 0.9}
        )
        
        await rag_service.store_knowledge(prerequisite)
        print("  ✅ Stored test relationship")
        
        # Store some context
        context = ContextItem(
            key="user_progress",
            value={"user_id": "test_user", "mastery": {"basic_arithmetic": 0.8}}
        )
        
        await rag_service.store_context(context)
        print("  ✅ Stored test context")
        
        # Test retrieval
        results = await rag_service.retrieve("arithmetic", top_k=5)
        print(f"  ✅ Retrieved {len(results)} results")
        
        # Test context retrieval
        contexts = await rag_service.retrieve_context()
        print(f"  ✅ Retrieved {len(contexts)} context items")
        
        # Clean up
        await rag_service.close()
        print("  ✅ Service closed successfully")
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ Completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Test failed: {e}")
        print(f"  ⏱️ Failed after {elapsed:.2f} seconds")
        return False

async def test_real_backends():
    """Test the system with real backends (Neo4j + Redis)."""
    print("🧪 Testing with Real Backends")
    print("----------------------------------------")
    
    if not check_real_backends_available():
        print("  ❌ Real backends not available")
        return False
    
    start_time = time.time()
    try:
        # Create RAG service with real backends
        config = get_real_backends_config()
        rag_service = await create_configured_rag_service(config)
        print("  ✅ RAG service created")
        
        # Test basic operations (same as memory test)
        # Create some test entities
        concept1 = Entity(
            id="test_basic_arithmetic",
            type="concept",
            properties={"name": "Basic Arithmetic", "difficulty": 1}
        )
        
        concept2 = Entity(
            id="test_fractions",
            type="concept", 
            properties={"name": "Fractions", "difficulty": 3}
        )
        
        # Store entities
        await rag_service.store_knowledge(concept1)
        await rag_service.store_knowledge(concept2)
        print("  ✅ Stored test entities")
        
        # Create a relationship
        prerequisite = Relationship(
            id="test_prereq_1",
            type="prerequisite",
            source_id="test_basic_arithmetic",
            target_id="test_fractions",
            properties={"strength": 0.9}
        )
        
        await rag_service.store_knowledge(prerequisite)
        print("  ✅ Stored test relationship")
        
        # Store some context
        context = ContextItem(
            key="test_user_progress",
            value={"user_id": "test_user", "mastery": {"basic_arithmetic": 0.8}}
        )
        
        await rag_service.store_context(context)
        print("  ✅ Stored test context")
        
        # Test retrieval
        results = await rag_service.retrieve("arithmetic", top_k=5)
        print(f"  ✅ Retrieved {len(results)} results")
        
        # Test context retrieval
        contexts = await rag_service.retrieve_context()
        print(f"  ✅ Retrieved {len(contexts)} context items")
        
        # Clean up test data
        await rag_service.delete("test_basic_arithmetic")
        await rag_service.delete("test_fractions")
        await rag_service.delete("test_prereq_1")
        print("  ✅ Cleaned up test data")
        
        # Close service
        await rag_service.close()
        print("  ✅ Service closed successfully")
        
        elapsed = time.time() - start_time
        print(f"  ⏱️ Completed in {elapsed:.2f} seconds")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ Test failed: {e}")
        print(f"  ⏱️ Failed after {elapsed:.2f} seconds")
        return False

async def main():
    """Main test function."""
    print("🔄 Backend Comparison Test")
    print("=" * 50)
    
    # Check if real backends are available
    real_backends_available = check_real_backends_available()
    
    # Test 1: Memory backends (always available)
    print("\n📝 Test 1: Memory Backends (Development/Testing)")
    memory_success = await test_memory_backends()
    
    # Test 2: Real backends (if available)
    if real_backends_available:
        print("\n🏢 Test 2: Real Backends (Production)")
        real_success = await test_real_backends()
    else:
        print("\n🏢 Test 2: Real Backends (Production)")
        print("  ❌ Real backends not available")
        print("  📝 To test with real backends:")
        print("    1. Install dependencies: pip install neo4j redis")
        print("    2. Start Neo4j: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        print("    3. Start Redis: docker run -p 6379:6379 redis:latest")
        real_success = None
    
    # Print summary
    print("\n📊 Test Summary")
    print("-" * 20)
    print(f"Memory backends: {'✅ PASSED' if memory_success else '❌ FAILED'}")
    if real_success is not None:
        print(f"Real backends: {'✅ PASSED' if real_success else '❌ FAILED'}")
    else:
        print("Real backends: ⏭️ SKIPPED")
    
    # Print comparison
    print("\n🔍 Backend Comparison")
    print("-" * 25)
    print("Memory Backends:")
    print("  ✅ Fast and lightweight")
    print("  ✅ No external dependencies")
    print("  ✅ Perfect for testing/development")
    print("  ❌ Data doesn't persist between runs")
    print("  ❌ Limited scalability")
    print()
    print("Real Backends (Neo4j + Redis):")
    print("  ✅ Data persists between runs")
    print("  ✅ Production-ready scalability")
    print("  ✅ Advanced query capabilities")
    print("  ✅ Better performance for large datasets")
    print("  ❌ Requires external services")
    print("  ❌ More complex setup")
    print()
    print("💡 Recommendation:")
    print("  - Use memory backends for development and testing")
    print("  - Use real backends for production deployments")
    print("  - The math learning system works with both!")
    
    # Final result
    if memory_success and (real_success is None or real_success):
        print("\n🎉 SUCCESS: Backend comparison completed successfully!")
        return True
    else:
        print("\n💥 FAILURE: Backend comparison failed!")
        return False

if __name__ == "__main__":
    asyncio.run(main()) 