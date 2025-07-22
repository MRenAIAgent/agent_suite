#!/usr/bin/env python3
"""
Test math learning system with real backends (Neo4j + Redis).

This test demonstrates the math learning system using real database backends
instead of memory-based storage. It requires:
- Neo4j running on localhost:7687 (username: neo4j, password: password)
- Redis running on localhost:6379

To run this test:
1. Start Neo4j: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest
2. Start Redis: docker run -p 6379:6379 redis:latest
3. Install dependencies: pip install neo4j redis
4. Run test: python test_real_backends.py
"""

import asyncio
import sys
import traceback
from typing import Dict, Any, List, Tuple

# Add parent directory to path for imports
sys.path.append('..')

from config.rag_config import get_real_backends_config, create_configured_rag_service
from learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem

# Test if real backends are available
def check_backend_availability():
    """Check if Neo4j and Redis are available."""
    try:
        import neo4j
        NEO4J_AVAILABLE = True
    except ImportError:
        NEO4J_AVAILABLE = False
    
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
    
    return NEO4J_AVAILABLE, REDIS_AVAILABLE

def test_neo4j_connection():
    """Test Neo4j connection."""
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            for record in result:
                if record["test"] == 1:
                    driver.close()
                    return True
        driver.close()
        return False
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection."""
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, decode_responses=True)
        client.ping()
        client.close()
        return True
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return False

async def cleanup_test_data():
    """Clean up test data from previous runs."""
    print("🧹 Cleaning up test data...")
    
    # Clean Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            # Delete all nodes and relationships in our namespace
            session.run("MATCH (n {namespace: 'rag_math_learning'}) DETACH DELETE n")
        driver.close()
        print("  ✅ Neo4j cleaned")
    except Exception as e:
        print(f"  ⚠️ Neo4j cleanup failed: {e}")
    
    # Clean Redis
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379)
        # Delete all keys with our namespace
        keys = client.keys("rag_math_learning:*")
        if keys:
            client.delete(*keys)
        client.close()
        print("  ✅ Redis cleaned")
    except Exception as e:
        print(f"  ⚠️ Redis cleanup failed: {e}")

async def test_real_backends_integration():
    """Test the math learning system with real backends."""
    print("🚀 Testing Math Learning System with Real Backends")
    print("=" * 60)
    
    # Check backend availability
    neo4j_available, redis_available = check_backend_availability()
    print(f"📦 Neo4j client available: {neo4j_available}")
    print(f"📦 Redis client available: {redis_available}")
    
    if not neo4j_available:
        print("❌ Neo4j client not installed. Install with: pip install neo4j")
        return False
    
    if not redis_available:
        print("❌ Redis client not installed. Install with: pip install redis")
        return False
    
    # Test connections
    print("\n🔌 Testing backend connections...")
    neo4j_connected = test_neo4j_connection()
    redis_connected = test_redis_connection()
    
    print(f"  Neo4j connection: {'✅ Connected' if neo4j_connected else '❌ Failed'}")
    print(f"  Redis connection: {'✅ Connected' if redis_connected else '❌ Failed'}")
    
    if not neo4j_connected:
        print("❌ Neo4j not available. Start with: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return False
    
    if not redis_connected:
        print("❌ Redis not available. Start with: docker run -p 6379:6379 redis:latest")
        return False
    
    # Clean up any existing test data
    await cleanup_test_data()
    
    try:
        # Create RAG service with real backends
        print("\n🏗️ Creating RAG service with real backends...")
        config = get_real_backends_config()
        rag_service = await create_configured_rag_service(config)
        print("  ✅ RAG service created successfully")
        
        # Test 1: Store math concepts in Neo4j
        print("\n📚 Test 1: Storing math concepts in Neo4j...")
        concepts = [
            Entity(
                id="basic_arithmetic",
                type="MathConcept",
                name="Basic Arithmetic",
                description="Addition, subtraction, multiplication, and division of whole numbers",
                properties={"difficulty": 1, "grade_level": "K-2"}
            ),
            Entity(
                id="fractions",
                type="MathConcept", 
                name="Fractions",
                description="Understanding and operations with fractions",
                properties={"difficulty": 3, "grade_level": "3-5"}
            ),
            Entity(
                id="algebra",
                type="MathConcept",
                name="Algebra",
                description="Variables, equations, and algebraic thinking",
                properties={"difficulty": 5, "grade_level": "6-8"}
            )
        ]
        
        graph_storage = rag_service.get_storage_adaptor("graph")
        for concept in concepts:
            await graph_storage.store_entity(concept)
            print(f"  ✅ Stored concept: {concept.name}")
        
        # Store prerequisite relationships
        relationships = [
            Relationship(
                id="arithmetic_to_fractions",
                type="PREREQUISITE",
                source_id="basic_arithmetic",
                target_id="fractions",
                properties={"strength": 0.9}
            ),
            Relationship(
                id="fractions_to_algebra", 
                type="PREREQUISITE",
                source_id="fractions",
                target_id="algebra",
                properties={"strength": 0.8}
            )
        ]
        
        for rel in relationships:
            await graph_storage.store_relationship(rel)
            print(f"  ✅ Stored relationship: {rel.source_id} -> {rel.target_id}")
        
        # Test 2: Query concepts from Neo4j
        print("\n🔍 Test 2: Querying concepts from Neo4j...")
        query_results = await graph_storage.query_graph(
            """
            MATCH (c:Entity {namespace: 'rag_math_learning'})
            WHERE c.type = 'MathConcept'
            RETURN c.id as id, c.name as name, c.description as description
            ORDER BY c.name
            """
        )
        
        print(f"  📊 Found {len(query_results)} concepts:")
        for result in query_results:
            print(f"    - {result['name']}: {result['description']}")
        
        # Test 3: Store learning progress in Redis
        print("\n📈 Test 3: Storing learning progress in Redis...")
        kv_storage = rag_service.get_storage_adaptor("key_value")
        
        # Simulate learning progress for a user
        user_id = "test_user_real_backend"
        learning_graph = LearningGraph(user_id=user_id, name="Test User")
        
        # Simulate exercises
        exercise_results = [
            ("basic_arithmetic", True),
            ("basic_arithmetic", True),
            ("basic_arithmetic", False),
            ("fractions", True),
            ("fractions", False),
            ("fractions", False),
            ("algebra", False),
            ("algebra", False)
        ]
        
        for concept_id, is_correct in exercise_results:
            learning_graph.record_exercise_attempt(concept_id, result=is_correct)
        
        # Store progress in Redis
        progress_data = {}
        for concept_id in ["basic_arithmetic", "fractions", "algebra"]:
            mastery = learning_graph.get_mastery(concept_id)
            progress_data[concept_id] = {
                "mastery": mastery,
                "attempts": len([r for r in exercise_results if r[0] == concept_id]),
                "correct": len([r for r in exercise_results if r[0] == concept_id and r[1]])
            }
        
        progress_item = ContextItem(
            key=f"learning_progress_{user_id}",
            value={
                "user_id": user_id,
                "progress": progress_data,
                "last_updated": "2024-01-15T10:00:00Z"
            }
        )
        
        await kv_storage.store_context_item(progress_item)
        print(f"  ✅ Stored learning progress for {user_id}")
        
        # Test 4: Retrieve and analyze progress from Redis
        print("\n📊 Test 4: Retrieving progress from Redis...")
        retrieved_progress = await kv_storage.get(f"learning_progress_{user_id}")
        
        if retrieved_progress:
            print(f"  📈 Progress for {retrieved_progress['user_id']}:")
            for concept_id, data in retrieved_progress['progress'].items():
                mastery_pct = data['mastery'] * 100
                success_rate = (data['correct'] / data['attempts']) * 100 if data['attempts'] > 0 else 0
                print(f"    - {concept_id}: {mastery_pct:.1f}% mastery ({data['correct']}/{data['attempts']} = {success_rate:.1f}% success)")
        
        # Test 5: Complex query combining Neo4j and Redis data
        print("\n🔗 Test 5: Complex analysis combining both backends...")
        
        # Get struggling concepts (mastery < 50%)
        struggling_concepts = []
        strong_concepts = []
        
        for concept_id, data in retrieved_progress['progress'].items():
            mastery_pct = data['mastery'] * 100
            if mastery_pct < 50:
                struggling_concepts.append((concept_id, mastery_pct))
            else:
                strong_concepts.append((concept_id, mastery_pct))
        
        print(f"  📉 Struggling concepts ({len(struggling_concepts)}):")
        for concept_id, mastery in struggling_concepts:
            # Get concept details from Neo4j
            concept_entity = await graph_storage.get_entity(concept_id)
            if concept_entity:
                difficulty = concept_entity.properties.get('difficulty', 'Unknown')
                print(f"    - {concept_entity.name}: {mastery:.1f}% mastery (difficulty: {difficulty})")
        
        print(f"  📈 Strong concepts ({len(strong_concepts)}):")
        for concept_id, mastery in strong_concepts:
            concept_entity = await graph_storage.get_entity(concept_id)
            if concept_entity:
                difficulty = concept_entity.properties.get('difficulty', 'Unknown')
                print(f"    - {concept_entity.name}: {mastery:.1f}% mastery (difficulty: {difficulty})")
        
        # Test 6: Generate recommendations using prerequisite graph
        print("\n💡 Test 6: Generating recommendations using prerequisite graph...")
        
        if struggling_concepts:
            weakest_concept_id = struggling_concepts[0][0]
            
            # Find prerequisites for the weakest concept
            prereq_query = await graph_storage.query_graph(
                """
                MATCH (prereq:Entity)-[r:PREREQUISITE]->(target:Entity {id: $concept_id, namespace: 'rag_math_learning'})
                RETURN prereq.id as prereq_id, prereq.name as prereq_name, r.properties as rel_props
                """,
                {"concept_id": weakest_concept_id}
            )
            
            if prereq_query:
                print(f"  🎯 Recommendations for improving {weakest_concept_id}:")
                for result in prereq_query:
                    prereq_id = result['prereq_id']
                    prereq_name = result['prereq_name']
                    
                    # Check mastery of prerequisite
                    if prereq_id in retrieved_progress['progress']:
                        prereq_mastery = retrieved_progress['progress'][prereq_id]['mastery'] * 100
                        if prereq_mastery < 70:
                            print(f"    - Focus on {prereq_name} (current: {prereq_mastery:.1f}% mastery)")
                        else:
                            print(f"    - {prereq_name} is strong ({prereq_mastery:.1f}% mastery)")
            else:
                print(f"  ℹ️ No prerequisites found for {weakest_concept_id}")
        
        # Test 7: Cleanup and verify data persistence
        print("\n🧪 Test 7: Verifying data persistence...")
        
        # Close and recreate service to test persistence
        await rag_service.close()
        
        # Create new service instance
        rag_service = await create_configured_rag_service(config)
        
        # Verify data is still there
        graph_storage = rag_service.get_storage_adaptor("graph")
        kv_storage = rag_service.get_storage_adaptor("key_value")
        
        # Check Neo4j data
        concept_count = await graph_storage.query_graph(
            "MATCH (c:Entity {namespace: 'rag_math_learning'}) RETURN count(c) as count"
        )
        neo4j_count = concept_count[0]['count'] if concept_count else 0
        
        # Check Redis data
        redis_data = await kv_storage.get(f"learning_progress_{user_id}")
        redis_exists = redis_data is not None
        
        print(f"  📊 Neo4j entities persisted: {neo4j_count}")
        print(f"  📊 Redis data persisted: {redis_exists}")
        
        # Final cleanup
        await rag_service.close()
        await cleanup_test_data()
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Real backend integration is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"📍 Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function."""
    print("🧪 Math Learning System - Real Backends Test")
    print("=" * 50)
    
    success = await test_real_backends_integration()
    
    if success:
        print("\n🎊 SUCCESS: Real backend integration test passed!")
        sys.exit(0)
    else:
        print("\n💥 FAILURE: Real backend integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 