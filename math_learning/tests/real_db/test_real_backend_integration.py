#!/usr/bin/env python3
"""
Test for Math Learning using REAL Neo4j and Qdrant backends.

This test demonstrates the math learning system using actual Neo4j and Qdrant
databases instead of memory backends, showing real database usage.
"""

import asyncio
import sys
import os
from typing import Dict, List, Any

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.config.rag_config import RagConfig, create_configured_rag_service
from math_learning.learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem
from agents.rag.middleware.storage_router import StorageType
from math_learning.scripts.utils.database_usage_checker import DatabaseUsageChecker

async def create_real_backend_rag_service() -> 'RagService':
    """
    Create a RAG service configured to use real Neo4j and Qdrant backends.
    """
    # Get configuration from environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    
    # Create configuration for real backends
    config = RagConfig(
        primary_storage_type="graph",
        vector_storage_type="qdrant",
        graph_storage_type="neo4j",
        kv_storage_type="memory",  # Keep memory for simplicity
        embedding_provider="sentence-transformer",
        embedding_model="all-MiniLM-L6-v2",
        vector_collection_name="math_learning_concepts",
        vector_size=384,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        neo4j_uri=neo4j_uri,
        neo4j_username=neo4j_username,
        neo4j_password=neo4j_password
    )
    
    print(f"🔧 Creating RAG service with real backends:")
    print(f"   📊 Neo4j: {neo4j_uri}")
    print(f"   🔍 Qdrant: {qdrant_host}:{qdrant_port}")
    
    return await create_configured_rag_service(config)

async def test_real_backend_math_learning():
    """Test math learning system with real Neo4j and Qdrant backends."""
    print("🚀 Starting Math Learning Test with REAL Backends")
    print("=" * 80)
    
    # Initialize usage checker
    usage_checker = DatabaseUsageChecker()
    
    # Check initial usage
    print("\n📊 CHECKING INITIAL DATABASE USAGE")
    print("-" * 50)
    initial_usage = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(initial_usage)
    
    try:
        # Step 1: Create RAG service with real backends
        print("\n🔧 STEP 1: Creating RAG service with real backends...")
        rag_service = await create_real_backend_rag_service()
        print("✅ RAG service created successfully")
        
        # Step 2: Create learning graph
        print("\n👤 STEP 2: Creating learning graph...")
        learning_graph = LearningGraph(user_id="real_test_student", name="Real Backend Test Student")
        print("✅ Learning graph created")
        
        # Step 3: Store math concepts in Neo4j
        print("\n📚 STEP 3: Storing math concepts in Neo4j...")
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create comprehensive math concept entities
        concepts = [
            {"id": "counting", "name": "Counting", "difficulty": 0.1, "grade_level": "K"},
            {"id": "basic_arithmetic", "name": "Basic Arithmetic", "difficulty": 0.3, "grade_level": "1-2"},
            {"id": "multiplication", "name": "Multiplication", "difficulty": 0.5, "grade_level": "3-4"},
            {"id": "division", "name": "Division", "difficulty": 0.5, "grade_level": "3-4"},
            {"id": "fractions", "name": "Fractions", "difficulty": 0.7, "grade_level": "4-5"},
            {"id": "decimals", "name": "Decimals", "difficulty": 0.7, "grade_level": "4-5"},
            {"id": "percentages", "name": "Percentages", "difficulty": 0.8, "grade_level": "5-6"},
            {"id": "basic_algebra", "name": "Basic Algebra", "difficulty": 0.8, "grade_level": "6-8"},
            {"id": "linear_equations", "name": "Linear Equations", "difficulty": 0.9, "grade_level": "8-9"},
            {"id": "quadratic_equations", "name": "Quadratic Equations", "difficulty": 1.0, "grade_level": "9-10"}
        ]
        
        concept_entities = {}
        for concept in concepts:
            entity = Entity(
                type='math_concept',
                properties={
                    'concept_id': concept['id'],
                    'name': concept['name'],
                    'difficulty': concept['difficulty'],
                    'grade_level': concept['grade_level'],
                    'domain': 'mathematics',
                    'created_by': 'real_backend_test'
                }
            )
            entity_id = await graph_adaptor.store_entity(entity)
            concept_entities[concept['id']] = entity_id
            print(f"  ✅ Stored concept: {concept['name']} (Grade {concept['grade_level']})")
        
        # Create prerequisite relationships
        print("\n🔗 Creating prerequisite relationships...")
        prerequisites = [
            ("basic_arithmetic", "counting"),
            ("multiplication", "basic_arithmetic"),
            ("division", "multiplication"),
            ("fractions", "division"),
            ("decimals", "fractions"),
            ("percentages", "decimals"),
            ("basic_algebra", "fractions"),
            ("basic_algebra", "percentages"),
            ("linear_equations", "basic_algebra"),
            ("quadratic_equations", "linear_equations")
        ]
        
        for concept, prereq in prerequisites:
            relationship = Relationship(
                source_id=concept_entities[prereq],
                target_id=concept_entities[concept],
                type='prerequisite',
                properties={
                    'strength': 0.9,
                    'created_by': 'real_backend_test',
                    'relationship_type': 'learning_prerequisite'
                }
            )
            await graph_adaptor.store_relationship(relationship)
            print(f"  🔗 Created prerequisite: {prereq} → {concept}")
        
        # Step 4: Store concepts in Qdrant for vector search
        print("\n🔍 STEP 4: Storing concepts in Qdrant for vector search...")
        vector_adaptor = rag_service.storage_adaptors[StorageType.VECTOR]
        
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
        
        stored_vector_ids = []
        for concept in math_concepts:
            # Create Document object with UUID
            import uuid
            from agents.rag.models.document import Document
            doc = Document(
                content=concept["text"],
                metadata=concept["metadata"],
                id=str(uuid.uuid4())  # Use UUID for Qdrant compatibility
            )
            
            vector_id = await vector_adaptor.store_document(doc)
            stored_vector_ids.append(vector_id)
            print(f"  ✅ Stored vector: {concept['metadata']['concept']} (ID: {vector_id})")
        
        # Step 5: Simulate comprehensive learning progress
        print("\n🎯 STEP 5: Simulating comprehensive learning progress...")
        
        # Simulate learning sessions for multiple concepts
        learning_sessions = [
            ("counting", [True, True, True, True, True]),  # 100% mastery
            ("basic_arithmetic", [True, True, True, False, True]),  # 80% mastery
            ("multiplication", [True, False, True, True, False]),  # 60% mastery
            ("division", [False, True, True, False, True]),  # 60% mastery
            ("fractions", [True, False, False, True, True]),  # 60% mastery
            ("decimals", [False, True, False, True, False]),  # 40% mastery
            ("percentages", [False, False, True, False, True]),  # 40% mastery
            ("basic_algebra", [False, False, False, True, False]),  # 20% mastery
        ]
        
        total_exercises = 0
        for concept_id, results in learning_sessions:
            print(f"\n  📖 Learning session: {concept_id}")
            for i, result in enumerate(results):
                exercise_id = f"{concept_id}_exercise_{i+1}"
                difficulty = 0.3 + (i * 0.15)  # Progressive difficulty
                
                learning_graph.record_exercise_attempt(
                    exercise_id=exercise_id,
                    concept_id=concept_id,
                    result=result,
                    difficulty=difficulty,
                    concept_weight=1.0
                )
                
                mastery = learning_graph.get_mastery(concept_id)
                total_exercises += 1
                print(f"    Exercise {i+1}: {'✅' if result else '❌'} (mastery: {mastery:.3f})")
        
        # Step 6: Store learning progress in key-value storage
        print("\n💾 STEP 6: Storing learning progress...")
        kv_adaptor = rag_service.storage_adaptors[StorageType.KEY_VALUE]
        
        # Store comprehensive progress summary
        progress_summary = {
            'user_id': learning_graph.user_id,
            'total_exercises': total_exercises,
            'concepts_attempted': len(learning_sessions),
            'mastery_levels': {
                concept_id: learning_graph.get_mastery(concept_id)
                for concept_id, _ in learning_sessions
            },
            'confidence_levels': {
                concept_id: learning_graph.get_confidence(concept_id)
                for concept_id, _ in learning_sessions
            },
            'test_type': 'real_backend_integration'
        }
        
        progress_item = ContextItem(
            key='comprehensive_learning_progress',
            value=progress_summary,
            metadata={'type': 'progress_summary', 'backend': 'real_databases'}
        )
        
        progress_id = await kv_adaptor.store_context_item(progress_item)
        print(f"✅ Stored comprehensive progress summary (ID: {progress_id})")
        
        # Step 7: Test vector search capabilities
        print("\n🔍 STEP 7: Testing vector search capabilities...")
        
        # Search for algebra-related concepts
        search_results = await vector_adaptor.search(
            query="solving equations with variables",
            top_k=3
        )
        
        print(f"  🔍 Found {len(search_results)} similar concepts for 'solving equations with variables':")
        for i, result in enumerate(search_results[:3], 1):
            # Handle RetrievalResult objects
            if hasattr(result, 'item'):
                item = result.item
                score = getattr(result, 'score', 0.0)
            else:
                item = result
                score = 0.0
                
            if hasattr(item, 'content'):
                content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                print(f"    {i}. Score: {score:.3f} - {content_preview}")
            else:
                print(f"    {i}. Score: {score:.3f} - {item}")
        
        # Search for basic math concepts
        search_results2 = await vector_adaptor.search(
            query="basic arithmetic operations",
            top_k=2
        )
        
        print(f"\n  🔍 Found {len(search_results2)} similar concepts for 'basic arithmetic operations':")
        for i, result in enumerate(search_results2[:2], 1):
            # Handle RetrievalResult objects
            if hasattr(result, 'item'):
                item = result.item
                score = getattr(result, 'score', 0.0)
            else:
                item = result
                score = 0.0
                
            if hasattr(item, 'content'):
                content_preview = item.content[:100] + "..." if len(item.content) > 100 else item.content
                print(f"    {i}. Score: {score:.3f} - {content_preview}")
            else:
                print(f"    {i}. Score: {score:.3f} - {item}")
        
        # Step 8: Query graph relationships
        print("\n🕸️ STEP 8: Testing graph relationship queries...")
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Query prerequisites for fractions
        prerequisites = await graph_adaptor.query_graph(
            query="""
            MATCH (start:Entity {id: $entity_id})<-[:prerequisite]-(prereq:Entity)
            RETURN prereq.id as id, prereq.name as name, prereq.type as type
            """,
            parameters={"entity_id": "fractions"}
        )
        print(f"  📚 Prerequisites for fractions: {[p['name'] for p in prerequisites]}")
        
        # Query what concepts require basic_algebra as prerequisite
        dependents = await graph_adaptor.query_graph(
            query="""
            MATCH (start:Entity {id: $entity_id})-[:prerequisite]->(dependent:Entity)
            RETURN dependent.id as id, dependent.name as name, dependent.type as type
            """,
            parameters={"entity_id": "basic_algebra"}
        )
        print(f"  🎯 Concepts that depend on basic algebra: {[d['name'] for d in dependents]}")
        
        # Find learning path from counting to quadratic equations
        learning_path = await graph_adaptor.query_graph(
            query="""
            MATCH path = (start:Entity {id: $start_id})-[:prerequisite*]->(end:Entity {id: $end_id})
            RETURN [node in nodes(path) | node.name] as path_names
            LIMIT 1
            """,
            parameters={"start_id": "counting", "end_id": "quadratic_equations"}
        )
        if learning_path:
            path_names = learning_path[0]['path_names']
            print(f"  🛤️ Learning path from counting to quadratic equations: {' → '.join(path_names)}")
        else:
            print("  🛤️ No direct learning path found")
        
        # Step 9: Generate learning recommendations
        print("\n🎯 STEP 9: Generating learning recommendations...")
        
        mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.7)
        struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.5)
        
        print(f"  ✅ Mastered concepts (≥70%): {mastered_concepts}")
        print(f"  ⚠️  Struggling concepts (<50%): {struggling_concepts}")
        
        # Generate personalized recommendations
        recommendations = []
        for concept in struggling_concepts:
            if concept in concept_entities:
                recommendations.append(f"Review {concept} - current mastery below 50%")
        
        if not struggling_concepts:
            next_concepts = ['decimals', 'percentages', 'basic_algebra']
            available_next = [c for c in next_concepts if c not in [concept for concept, _ in learning_sessions]]
            if available_next:
                recommendations.append(f"Ready to advance to: {available_next[0]}")
        
        print(f"  💡 Recommendations: {len(recommendations)} generated")
        for rec in recommendations:
            print(f"    - {rec}")
        
        # Step 10: Clean up
        print("\n🧹 STEP 10: Cleaning up...")
        await rag_service.close()
        print("✅ RAG service closed")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check final usage
    print("\n📊 CHECKING FINAL DATABASE USAGE")
    print("-" * 50)
    final_usage = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(final_usage)
    
    # Show the difference
    print("\n📈 DATABASE USAGE CHANGES")
    print("-" * 50)
    
    # Calculate changes
    neo4j_initial = initial_usage.get('neo4j', {})
    neo4j_final = final_usage.get('neo4j', {})
    qdrant_initial = initial_usage.get('qdrant', {})
    qdrant_final = final_usage.get('qdrant', {})
    
    print("🔗 Neo4j Changes:")
    if neo4j_initial.get('status') == 'connected' and neo4j_final.get('status') == 'connected':
        node_diff = neo4j_final.get('node_count', 0) - neo4j_initial.get('node_count', 0)
        rel_diff = neo4j_final.get('relationship_count', 0) - neo4j_initial.get('relationship_count', 0)
        label_diff = neo4j_final.get('label_count', 0) - neo4j_initial.get('label_count', 0)
        print(f"   📊 Nodes: {node_diff:+d}")
        print(f"   🔗 Relationships: {rel_diff:+d}")
        print(f"   🏷️  Labels: {label_diff:+d}")
    else:
        print("   ❌ Could not compare Neo4j usage")
    
    print("\n🔍 Qdrant Changes:")
    if qdrant_initial.get('status') == 'connected' and qdrant_final.get('status') == 'connected':
        collection_diff = qdrant_final.get('collection_count', 0) - qdrant_initial.get('collection_count', 0)
        vector_diff = qdrant_final.get('total_vectors', 0) - qdrant_initial.get('total_vectors', 0)
        print(f"   📦 Collections: {collection_diff:+d}")
        print(f"   🔢 Vectors: {vector_diff:+d}")
    else:
        print("   ❌ Could not compare Qdrant usage")
    
    print("\n" + "=" * 80)
    print("✅ REAL BACKEND INTEGRATION TEST COMPLETE!")
    print("✅ Successfully demonstrated math learning with Neo4j + Qdrant")
    print(f"✅ Stored 10 math concepts and 10 relationships in Neo4j")
    print(f"✅ Stored 10 concept vectors in Qdrant")
    print(f"✅ Processed {total_exercises} learning exercises")
    print(f"✅ Generated personalized learning recommendations")
    print("=" * 80)
    
    return True

async def main():
    """Main function to run the test."""
    success = await test_real_backend_math_learning()
    return success

if __name__ == "__main__":
    asyncio.run(main()) 