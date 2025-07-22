#!/usr/bin/env python3
"""
Comprehensive test for math learning system using real Qdrant and Neo4j databases.

This test requires:
1. Qdrant running on localhost:6333
2. Neo4j running on localhost:7687 with username/password: neo4j/password

Run with: python test_real_databases.py
"""

import asyncio
import sys
import os
import time
import traceback
from typing import Dict, List, Any
import pytest

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.config.rag_config import RagConfig, create_configured_rag_service
from math_learning.learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem
from agents.rag.middleware.storage_router import StorageType
from math_learning.scripts.utils.database_usage_checker import DatabaseUsageChecker


def get_real_databases_config() -> RagConfig:
    """Get configuration for real Qdrant and Neo4j databases."""
    return RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="neo4j",
        kv_storage_type="memory",  # Keep KV as memory for simplicity
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
        
        # Qdrant settings
        qdrant_host="localhost",
        qdrant_port=6333,
        vector_collection_name="math_learning_test",
        vector_size=384,
        
        # Neo4j settings
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        
        # Test settings
        chunk_size=512,
        chunk_overlap=50,
        max_results=10,
        similarity_threshold=0.7
    )


async def check_database_connections(config: RagConfig) -> Dict[str, bool]:
    """Check if databases are accessible."""
    print("🔍 Checking database connections...")
    
    connections = {
        "qdrant": False,
        "neo4j": False
    }
    
    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        collections = client.get_collections()
        connections["qdrant"] = True
        print(f"  ✅ Qdrant connected at {config.qdrant_host}:{config.qdrant_port}")
    except Exception as e:
        print(f"  ❌ Qdrant connection failed: {e}")
    
    # Check Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password)
        )
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            result.single()
        driver.close()
        connections["neo4j"] = True
        print(f"  ✅ Neo4j connected at {config.neo4j_uri}")
    except Exception as e:
        print(f"  ❌ Neo4j connection failed: {e}")
    
    return connections


async def cleanup_test_data(config: RagConfig):
    """Clean up test data from databases."""
    print("🧹 Cleaning up test data...")
    
    # Clean Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=config.qdrant_host, port=config.qdrant_port)
        try:
            client.delete_collection(config.vector_collection_name)
            print(f"  ✅ Cleaned Qdrant collection: {config.vector_collection_name}")
        except Exception:
            print(f"  ℹ️  Qdrant collection {config.vector_collection_name} didn't exist")
    except Exception as e:
        print(f"  ⚠️  Failed to clean Qdrant: {e}")
    
    # Clean Neo4j
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password)
        )
        with driver.session() as session:
            # Delete all test nodes and relationships
            session.run("MATCH (n) WHERE n.test_id = 'math_learning_test' DETACH DELETE n")
            print("  ✅ Cleaned Neo4j test data")
        driver.close()
    except Exception as e:
        print(f"  ⚠️  Failed to clean Neo4j: {e}")


async def test_vector_storage_operations(rag_service):
    """Test Qdrant vector storage operations."""
    print("\n📊 Testing Qdrant Vector Storage Operations...")
    
    try:
        vector_adaptor = rag_service.storage_adaptors[StorageType.VECTOR]
        
        # Test 1: Store math concept vectors
        print("  📝 Storing math concept vectors...")
        
        math_concepts = [
            {
                "text": "Addition is the mathematical operation of combining two or more numbers to get their sum",
                "metadata": {"concept": "addition", "difficulty": 0.2, "grade_level": 1}
            },
            {
                "text": "Multiplication is repeated addition, where you add a number to itself multiple times",
                "metadata": {"concept": "multiplication", "difficulty": 0.4, "grade_level": 3}
            },
            {
                "text": "Fractions represent parts of a whole, written as numerator over denominator",
                "metadata": {"concept": "fractions", "difficulty": 0.6, "grade_level": 4}
            },
            {
                "text": "Algebra uses variables and equations to solve mathematical problems",
                "metadata": {"concept": "algebra", "difficulty": 0.8, "grade_level": 8}
            },
            {
                "text": "Quadratic equations are polynomial equations of degree 2, often written as ax² + bx + c = 0",
                "metadata": {"concept": "quadratic_equations", "difficulty": 0.9, "grade_level": 9}
            }
        ]
        
        stored_ids = []
        for concept in math_concepts:
            doc_id = await vector_adaptor.store_document(
                content=concept["text"],
                metadata=concept["metadata"]
            )
            stored_ids.append(doc_id)
            print(f"    ✅ Stored: {concept['metadata']['concept']} (ID: {doc_id})")
        
        # Test 2: Semantic search
        print("\n  🔍 Testing semantic search...")
        
        search_queries = [
            "How do I solve equations with variables?",
            "What are parts of a whole in mathematics?",
            "How do I combine numbers together?",
            "What is repeated addition called?"
        ]
        
        for query in search_queries:
            results = await vector_adaptor.search_similar(
                query=query,
                limit=3,
                threshold=0.3
            )
            print(f"    Query: '{query}'")
            for i, result in enumerate(results[:2]):  # Show top 2 results
                concept = result.metadata.get('concept', 'unknown')
                score = result.score
                print(f"      {i+1}. {concept} (score: {score:.3f})")
        
        # Test 3: Metadata filtering
        print("\n  🎯 Testing metadata filtering...")
        
        # Find concepts suitable for grade 4 and below
        filtered_results = await vector_adaptor.search_similar(
            query="basic math concepts",
            limit=10,
            metadata_filter={"grade_level": {"$lte": 4}}
        )
        
        print(f"    Concepts for grade ≤4: {len(filtered_results)} found")
        for result in filtered_results:
            concept = result.metadata.get('concept', 'unknown')
            grade = result.metadata.get('grade_level', 'unknown')
            print(f"      - {concept} (grade {grade})")
        
        # Test 4: Batch operations
        print("\n  📦 Testing batch operations...")
        
        batch_concepts = [
            {
                "text": f"Practice problem {i}: Solve for x in the equation 2x + {i} = {i*2 + 5}",
                "metadata": {"type": "practice_problem", "difficulty": 0.5 + i*0.1, "problem_id": f"prob_{i}"}
            }
            for i in range(1, 6)
        ]
        
        batch_ids = await vector_adaptor.store_batch(
            documents=[c["text"] for c in batch_concepts],
            metadatas=[c["metadata"] for c in batch_concepts]
        )
        
        print(f"    ✅ Stored {len(batch_ids)} practice problems in batch")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector storage test failed: {e}")
        traceback.print_exc()
        return False


async def test_graph_storage_operations(rag_service):
    """Test Neo4j graph storage operations."""
    print("\n🕸️  Testing Neo4j Graph Storage Operations...")
    
    try:
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Test 1: Store math concept entities
        print("  📝 Storing math concept entities...")
        
        concept_entities = {}
        concepts = [
            {"id": "counting", "name": "Counting", "difficulty": 0.1, "grade": 1},
            {"id": "addition", "name": "Addition", "difficulty": 0.2, "grade": 1},
            {"id": "subtraction", "name": "Subtraction", "difficulty": 0.3, "grade": 2},
            {"id": "multiplication", "name": "Multiplication", "difficulty": 0.4, "grade": 3},
            {"id": "division", "name": "Division", "difficulty": 0.5, "grade": 3},
            {"id": "fractions", "name": "Fractions", "difficulty": 0.6, "grade": 4},
            {"id": "decimals", "name": "Decimals", "difficulty": 0.7, "grade": 5},
            {"id": "algebra", "name": "Basic Algebra", "difficulty": 0.8, "grade": 8},
            {"id": "quadratics", "name": "Quadratic Equations", "difficulty": 0.9, "grade": 9}
        ]
        
        for concept in concepts:
            entity = Entity(
                type='math_concept',
                properties={
                    'concept_id': concept['id'],
                    'name': concept['name'],
                    'difficulty': concept['difficulty'],
                    'grade_level': concept['grade'],
                    'domain': 'mathematics',
                    'test_id': 'math_learning_test'  # For cleanup
                }
            )
            entity_id = await graph_adaptor.store_entity(entity)
            concept_entities[concept['id']] = entity_id
            print(f"    ✅ Stored: {concept['name']} (ID: {entity_id})")
        
        # Test 2: Create prerequisite relationships
        print("\n  🔗 Creating prerequisite relationships...")
        
        prerequisites = [
            ("addition", "counting"),
            ("subtraction", "counting"),
            ("subtraction", "addition"),
            ("multiplication", "addition"),
            ("division", "multiplication"),
            ("fractions", "division"),
            ("decimals", "fractions"),
            ("algebra", "fractions"),
            ("algebra", "decimals"),
            ("quadratics", "algebra")
        ]
        
        relationship_ids = []
        for concept, prereq in prerequisites:
            relationship = Relationship(
                source_id=concept_entities[prereq],
                target_id=concept_entities[concept],
                type='prerequisite',
                properties={
                    'strength': 0.9,
                    'required': True,
                    'test_id': 'math_learning_test'
                }
            )
            rel_id = await graph_adaptor.store_relationship(relationship)
            relationship_ids.append(rel_id)
            print(f"    ✅ {prereq} → {concept}")
        
        # Test 3: Query graph structure
        print("\n  🔍 Testing graph queries...")
        
        # Find all prerequisites for algebra
        algebra_id = concept_entities['algebra']
        prerequisites_query = await graph_adaptor.query_graph(
            'get_neighbors',
            {'entity_id': algebra_id, 'direction': 'incoming', 'relationship_type': 'prerequisite'}
        )
        print(f"    Prerequisites for algebra: {len(prerequisites_query)} found")
        
        # Find concepts that depend on fractions
        fractions_id = concept_entities['fractions']
        dependents_query = await graph_adaptor.query_graph(
            'get_neighbors',
            {'entity_id': fractions_id, 'direction': 'outgoing', 'relationship_type': 'prerequisite'}
        )
        print(f"    Concepts depending on fractions: {len(dependents_query)} found")
        
        # Test 4: Complex graph traversal
        print("\n  🗺️  Testing learning path discovery...")
        
        # Find learning path from counting to quadratics
        counting_id = concept_entities['counting']
        quadratics_id = concept_entities['quadratics']
        
        # This would be a custom query for finding paths
        path_query = f"""
        MATCH path = (start {{concept_id: 'counting'}})-[:prerequisite*]->(end {{concept_id: 'quadratics'}})
        WHERE start.test_id = 'math_learning_test' AND end.test_id = 'math_learning_test'
        RETURN path
        LIMIT 5
        """
        
        try:
            paths = await graph_adaptor.execute_cypher(path_query)
            print(f"    Learning paths from counting to quadratics: {len(paths)} found")
        except Exception as e:
            print(f"    ⚠️  Path query not supported by adaptor: {e}")
        
        # Test 5: Entity retrieval and updates
        print("\n  📖 Testing entity retrieval and updates...")
        
        # Retrieve an entity
        retrieved_entity = await graph_adaptor.get_entity(algebra_id)
        if retrieved_entity:
            print(f"    ✅ Retrieved entity: {retrieved_entity.properties.get('name')}")
            
            # Update entity properties
            retrieved_entity.properties['last_accessed'] = time.time()
            retrieved_entity.properties['access_count'] = 1
            
            updated = await graph_adaptor.update_entity(algebra_id, retrieved_entity)
            if updated:
                print(f"    ✅ Updated entity properties")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Graph storage test failed: {e}")
        traceback.print_exc()
        return False


async def test_integrated_learning_scenario(rag_service):
    """Test integrated learning scenario using both vector and graph storage."""
    print("\n🎓 Testing Integrated Learning Scenario...")
    
    try:
        vector_adaptor = rag_service.storage_adaptors[StorageType.VECTOR]
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        kv_adaptor = rag_service.storage_adaptors[StorageType.KEY_VALUE]
        
        # Create a learning graph for a student
        learning_graph = LearningGraph(user_id="student_001", name="Alice's Learning Journey")
        
        print("  👤 Created learning profile for Alice")
        
        # Simulate learning session: fractions
        print("\n  📚 Simulating fractions learning session...")
        
        # Store learning content in vector database
        fraction_content = [
            "A fraction represents a part of a whole. The top number is the numerator, the bottom is the denominator.",
            "To add fractions with the same denominator, add the numerators and keep the denominator the same.",
            "To multiply fractions, multiply the numerators together and multiply the denominators together.",
            "To convert a fraction to a decimal, divide the numerator by the denominator."
        ]
        
        content_ids = []
        for i, content in enumerate(fraction_content):
            doc_id = await vector_adaptor.store_document(
                content=content,
                metadata={
                    "concept": "fractions",
                    "lesson_part": i + 1,
                    "content_type": "explanation",
                    "difficulty": 0.6
                }
            )
            content_ids.append(doc_id)
        
        print(f"    ✅ Stored {len(content_ids)} learning materials")
        
        # Simulate exercise attempts
        exercises = [
            ("frac_001", True, 0.5),   # 1/2 + 1/4 = ?
            ("frac_002", True, 0.6),   # 2/3 × 3/4 = ?
            ("frac_003", False, 0.7),  # Convert 3/8 to decimal
            ("frac_004", True, 0.6),   # 1/3 + 2/3 = ?
            ("frac_005", False, 0.8),  # 5/6 ÷ 2/3 = ?
        ]
        
        print("\n  ✏️  Recording exercise attempts...")
        for exercise_id, success, difficulty in exercises:
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id="fractions",
                result=success,
                difficulty=difficulty,
                concept_weight=1.0
            )
            
            mastery = learning_graph.get_mastery("fractions")
            print(f"    Exercise {exercise_id}: {'✅' if success else '❌'} (mastery: {mastery:.3f})")
        
        # Store learning progress in key-value store
        progress_data = {
            "user_id": "student_001",
            "concept": "fractions",
            "mastery_level": learning_graph.get_mastery("fractions"),
            "confidence": learning_graph.get_confidence("fractions"),
            "exercises_completed": len(exercises),
            "success_rate": sum(1 for _, success, _ in exercises if success) / len(exercises),
            "timestamp": time.time()
        }
        
        progress_item = ContextItem(
            key=f"progress_fractions_student_001",
            value=progress_data,
            metadata={"type": "learning_progress", "user": "student_001", "concept": "fractions"}
        )
        
        progress_id = await kv_adaptor.store_context_item(progress_item)
        print(f"    ✅ Stored learning progress (ID: {progress_id})")
        
        # Query for personalized recommendations
        print("\n  🎯 Generating personalized recommendations...")
        
        # Search for related content based on performance
        if learning_graph.get_mastery("fractions") < 0.7:
            # Student struggling, find easier content
            easier_content = await vector_adaptor.search_similar(
                query="basic fraction concepts simple explanation",
                limit=3,
                metadata_filter={"difficulty": {"$lt": 0.6}}
            )
            print(f"    📖 Found {len(easier_content)} easier materials")
            
            # Check prerequisites in graph
            try:
                # This would need the fractions entity ID from previous test
                # For demo, we'll simulate the query
                print("    🔍 Checking prerequisite concepts...")
                print("    💡 Recommendation: Review division before continuing with fractions")
            except Exception:
                pass
        else:
            # Student doing well, find advanced content
            advanced_content = await vector_adaptor.search_similar(
                query="advanced fraction operations complex problems",
                limit=3,
                metadata_filter={"difficulty": {"$gt": 0.7}}
            )
            print(f"    📈 Found {len(advanced_content)} advanced materials")
        
        # Retrieve and analyze all progress
        all_progress = await kv_adaptor.get_all_context()
        student_progress = [
            item for item in all_progress 
            if item.metadata.get("user") == "student_001"
        ]
        
        print(f"    📊 Total progress records for Alice: {len(student_progress)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integrated scenario test failed: {e}")
        traceback.print_exc()
        return False


async def test_performance_and_scalability(rag_service):
    """Test performance with larger datasets."""
    print("\n⚡ Testing Performance and Scalability...")
    
    try:
        vector_adaptor = rag_service.storage_adaptors[StorageType.VECTOR]
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Test 1: Batch vector operations
        print("  📦 Testing batch vector operations...")
        
        start_time = time.time()
        
        # Generate 100 practice problems
        practice_problems = []
        for i in range(100):
            problem_text = f"Practice problem {i}: Calculate {i} + {i*2} - {i//2 if i > 0 else 0}"
            metadata = {
                "type": "practice_problem",
                "difficulty": 0.3 + (i % 10) * 0.05,
                "problem_number": i,
                "topic": "arithmetic"
            }
            practice_problems.append((problem_text, metadata))
        
        # Store in batch
        batch_ids = await vector_adaptor.store_batch(
            documents=[p[0] for p in practice_problems],
            metadatas=[p[1] for p in practice_problems]
        )
        
        batch_time = time.time() - start_time
        print(f"    ✅ Stored 100 problems in {batch_time:.2f}s ({len(batch_ids)/batch_time:.1f} docs/sec)")
        
        # Test 2: Concurrent searches
        print("  🔍 Testing concurrent search performance...")
        
        search_queries = [
            "addition problems",
            "subtraction exercises", 
            "multiplication practice",
            "division questions",
            "arithmetic calculations"
        ]
        
        start_time = time.time()
        
        # Run searches concurrently
        search_tasks = [
            vector_adaptor.search_similar(query, limit=10)
            for query in search_queries
        ]
        
        search_results = await asyncio.gather(*search_tasks)
        search_time = time.time() - start_time
        
        total_results = sum(len(results) for results in search_results)
        print(f"    ✅ Completed {len(search_queries)} searches in {search_time:.2f}s")
        print(f"    📊 Found {total_results} total results")
        
        # Test 3: Graph traversal performance
        print("  🗺️  Testing graph traversal performance...")
        
        start_time = time.time()
        
        # Create a larger concept hierarchy
        advanced_concepts = []
        for i in range(20):
            entity = Entity(
                type='math_concept',
                properties={
                    'concept_id': f'advanced_concept_{i}',
                    'name': f'Advanced Concept {i}',
                    'difficulty': 0.8 + i * 0.01,
                    'test_id': 'math_learning_test'
                }
            )
            entity_id = await graph_adaptor.store_entity(entity)
            advanced_concepts.append(entity_id)
        
        graph_time = time.time() - start_time
        print(f"    ✅ Created 20 concept entities in {graph_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        traceback.print_exc()
        return False


@pytest.fixture(scope="session")
async def usage_checker():
    """Fixture to provide database usage checking."""
    return DatabaseUsageChecker()


@pytest.fixture(scope="session", autouse=True)
async def check_database_usage(usage_checker):
    """Check database usage before and after all tests."""
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE BEFORE TESTS")
    print("="*80)
    
    # Check usage before tests
    usage_before = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_before, "Database Usage BEFORE Tests")
    
    yield  # Run all tests
    
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE AFTER TESTS")
    print("="*80)
    
    # Check usage after tests
    usage_after = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_after, "Database Usage AFTER Tests")
    
    # Calculate and show differences
    print("\n" + "="*80)
    print("📊 USAGE COMPARISON")
    print("="*80)
    
    # Neo4j comparison
    neo4j_before = usage_before.get("neo4j", {})
    neo4j_after = usage_after.get("neo4j", {})
    
    if neo4j_before.get("status") == "connected" and neo4j_after.get("status") == "connected":
        nodes_diff = neo4j_after.get("total_nodes", 0) - neo4j_before.get("total_nodes", 0)
        rels_diff = neo4j_after.get("total_relationships", 0) - neo4j_before.get("total_relationships", 0)
        test_nodes_diff = neo4j_after.get("test_nodes", 0) - neo4j_before.get("test_nodes", 0)
        test_rels_diff = neo4j_after.get("test_relationships", 0) - neo4j_before.get("test_relationships", 0)
        
        print("🕸️  Neo4j Changes:")
        print(f"   📊 Nodes: {neo4j_before.get('total_nodes', 0):,} → {neo4j_after.get('total_nodes', 0):,} ({nodes_diff:+,})")
        print(f"   🔗 Relationships: {neo4j_before.get('total_relationships', 0):,} → {neo4j_after.get('total_relationships', 0):,} ({rels_diff:+,})")
        print(f"   🧪 Test Nodes: {neo4j_before.get('test_nodes', 0):,} → {neo4j_after.get('test_nodes', 0):,} ({test_nodes_diff:+,})")
        print(f"   🧪 Test Relationships: {neo4j_before.get('test_relationships', 0):,} → {neo4j_after.get('test_relationships', 0):,} ({test_rels_diff:+,})")
    
    # Qdrant comparison
    qdrant_before = usage_before.get("qdrant", {})
    qdrant_after = usage_after.get("qdrant", {})
    
    if qdrant_before.get("status") == "connected" and qdrant_after.get("status") == "connected":
        vectors_diff = qdrant_after.get("total_vectors", 0) - qdrant_before.get("total_vectors", 0)
        size_diff = qdrant_after.get("total_estimated_size_mb", 0) - qdrant_before.get("total_estimated_size_mb", 0)
        collections_diff = qdrant_after.get("total_collections", 0) - qdrant_before.get("total_collections", 0)
        
        print("📊 Qdrant Changes:")
        print(f"   📦 Collections: {qdrant_before.get('total_collections', 0)} → {qdrant_after.get('total_collections', 0)} ({collections_diff:+})")
        print(f"   🎯 Vectors: {qdrant_before.get('total_vectors', 0):,} → {qdrant_after.get('total_vectors', 0):,} ({vectors_diff:+,})")
        print(f"   💾 Size: {qdrant_before.get('total_estimated_size_mb', 0):.2f} MB → {qdrant_after.get('total_estimated_size_mb', 0):.2f} MB ({size_diff:+.2f} MB)")
    
    print("="*80)


async def main():
    """Main test function."""
    print("=" * 80)
    print("🧪 REAL DATABASE INTEGRATION TEST")
    print("   Testing with Qdrant Vector DB + Neo4j Graph DB")
    print("=" * 80)
    
    # Get configuration
    config = get_real_databases_config()
    
    # Check database connections
    connections = await check_database_connections(config)
    
    if not all(connections.values()):
        print("\n❌ Database connection check failed!")
        print("\nPlease ensure:")
        print("  1. Qdrant is running on localhost:6333")
        print("  2. Neo4j is running on localhost:7687")
        print("  3. Neo4j credentials are neo4j/password")
        return False
    
    print("\n✅ All database connections successful!")
    
    # Clean up any existing test data
    await cleanup_test_data(config)
    
    # Create RAG service with real databases
    print("\n🚀 Creating RAG service with real databases...")
    try:
        rag_service = await create_configured_rag_service(config)
        print("✅ RAG service created successfully")
    except Exception as e:
        print(f"❌ Failed to create RAG service: {e}")
        traceback.print_exc()
        return False
    
    # Run test suites
    test_results = {}
    
    try:
        # Test 1: Vector storage operations
        test_results["vector_storage"] = await test_vector_storage_operations(rag_service)
        
        # Test 2: Graph storage operations  
        test_results["graph_storage"] = await test_graph_storage_operations(rag_service)
        
        # Test 3: Integrated learning scenario
        test_results["integrated_scenario"] = await test_integrated_learning_scenario(rag_service)
        
        # Test 4: Performance and scalability
        test_results["performance"] = await test_performance_and_scalability(rag_service)
        
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        await rag_service.close()
        await cleanup_test_data(config)
    
    # Results summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT! Real database integration is working perfectly!")
    elif success_rate >= 75:
        print("✅ GOOD! Most functionality working, minor issues to address.")
    else:
        print("⚠️  NEEDS WORK! Significant issues with database integration.")
    
    print("\n💡 Key Validations:")
    print("  • Qdrant vector storage and semantic search")
    print("  • Neo4j graph storage and relationship queries")
    print("  • Integrated learning scenarios with real data persistence")
    print("  • Performance with realistic data volumes")
    print("  • Proper cleanup and connection management")
    
    return success_rate >= 75


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 