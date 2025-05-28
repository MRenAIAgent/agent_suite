#!/usr/bin/env python3
"""Comprehensive integration test for math learning system with RAG backend."""

import asyncio
import sys
import os

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.config.rag_config import get_memory_config, create_configured_rag_service
from math_learning.learning_graph.user_model import LearningGraph
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem
from agents.rag.middleware.storage_router import StorageType

async def test_full_integration():
    """Test the complete integration of math learning system with RAG backend."""
    print("=== Math Learning System + RAG Integration Test ===\\n")
    
    # Step 1: Create RAG service
    print("1. Creating RAG service...")
    try:
        config = get_memory_config()
        rag_service = await create_configured_rag_service(config)
        print("✓ RAG service created successfully")
    except Exception as e:
        print(f"✗ Failed to create RAG service: {e}")
        return False
    
    # Step 2: Create learning graph
    print("\\n2. Creating learning graph...")
    learning_graph = LearningGraph(user_id="test_student", name="Test Student Learning Graph")
    print("✓ Learning graph created")
    
    # Step 3: Store math concepts in RAG knowledge graph
    print("\\n3. Storing math concepts in RAG knowledge graph...")
    try:
        graph_adaptor = rag_service.storage_adaptors[StorageType.GRAPH]
        
        # Create math concept entities
        concepts = [
            {"id": "basic_arithmetic", "name": "Basic Arithmetic", "difficulty": 0.3},
            {"id": "integers", "name": "Integer Operations", "difficulty": 0.5},
            {"id": "fractions", "name": "Fractions", "difficulty": 0.7},
            {"id": "algebra", "name": "Basic Algebra", "difficulty": 0.8}
        ]
        
        concept_entities = {}
        for concept in concepts:
            entity = Entity(
                type='math_concept',
                properties={
                    'concept_id': concept['id'],
                    'name': concept['name'],
                    'difficulty': concept['difficulty'],
                    'domain': 'mathematics'
                }
            )
            entity_id = await graph_adaptor.store_entity(entity)
            concept_entities[concept['id']] = entity_id
            print(f"  ✓ Stored concept: {concept['name']}")
        
        # Create prerequisite relationships
        prerequisites = [
            ("integers", "basic_arithmetic"),
            ("fractions", "basic_arithmetic"),
            ("algebra", "integers"),
            ("algebra", "fractions")
        ]
        
        for concept, prereq in prerequisites:
            relationship = Relationship(
                source_id=concept_entities[prereq],
                target_id=concept_entities[concept],
                type='prerequisite',
                properties={'strength': 0.9}
            )
            await graph_adaptor.store_relationship(relationship)
            print(f"  ✓ Created prerequisite: {prereq} → {concept}")
            
    except Exception as e:
        print(f"✗ Failed to store concepts: {e}")
        await rag_service.close()
        return False
    
    # Step 4: Simulate learning progress
    print("\\n4. Simulating learning progress...")
    
    # Simulate exercises for each concept
    exercises = [
        # Basic arithmetic exercises
        ("basic_arithmetic", [True, True, True, True, False]),  # 80% success
        # Integer exercises  
        ("integers", [True, True, False, True, True]),  # 80% success
        # Fraction exercises
        ("fractions", [True, False, True, True, True]),  # 80% success
        # Algebra exercises
        ("algebra", [False, True, True, False, True]),  # 60% success
    ]
    
    for concept_id, results in exercises:
        print(f"\\n  Learning {concept_id}:")
        for i, result in enumerate(results):
            exercise_id = f"{concept_id}_ex_{i+1}"
            difficulty = 0.4 + (i * 0.1)  # Increasing difficulty
            
            learning_graph.record_exercise_attempt(
                exercise_id=exercise_id,
                concept_id=concept_id,
                result=result,
                difficulty=difficulty,
                concept_weight=1.0
            )
            
            mastery = learning_graph.get_mastery(concept_id)
            print(f"    Exercise {i+1}: {'✓' if result else '✗'} (mastery: {mastery:.3f})")
    
    # Step 5: Store learning progress in RAG context
    print("\\n5. Storing learning progress in RAG context...")
    try:
        kv_adaptor = rag_service.storage_adaptors[StorageType.KEY_VALUE]
        
        # Store overall progress
        progress_item = ContextItem(
            key='learning_progress',
            value={
                'user_id': learning_graph.user_id,
                'mastery_levels': {
                    concept_id: learning_graph.get_mastery(concept_id)
                    for concept_id in ['basic_arithmetic', 'integers', 'fractions', 'algebra']
                },
                'total_exercises': len(learning_graph.exercise_history)
            },
            metadata={'type': 'progress_summary', 'timestamp': 'current'}
        )
        
        progress_id = await kv_adaptor.store_context_item(progress_item)
        print(f"✓ Stored progress summary with ID: {progress_id}")
        
        # Store individual concept progress
        for concept_id in ['basic_arithmetic', 'integers', 'fractions', 'algebra']:
            concept_item = ContextItem(
                key=f'concept_progress_{concept_id}',
                value={
                    'concept_id': concept_id,
                    'mastery': learning_graph.get_mastery(concept_id),
                    'confidence': learning_graph.get_confidence(concept_id),
                    'exercises_completed': sum(1 for ex_id, history in learning_graph.exercise_history.items() 
                                             if any(h['concept_id'] == concept_id for h in history))
                },
                metadata={'type': 'concept_progress', 'concept': concept_id}
            )
            
            await kv_adaptor.store_context_item(concept_item)
            print(f"  ✓ Stored progress for {concept_id}")
            
    except Exception as e:
        print(f"✗ Failed to store progress: {e}")
        await rag_service.close()
        return False
    
    # Step 6: Query and analyze learning data
    print("\\n6. Analyzing learning data...")
    
    # Analyze mastery levels
    mastered_concepts = learning_graph.get_mastered_concepts(threshold=0.5)
    struggling_concepts = learning_graph.get_struggling_concepts(threshold=0.4)
    
    print(f"  Mastered concepts (≥50%): {mastered_concepts}")
    print(f"  Struggling concepts (<40%): {struggling_concepts}")
    
    # Query RAG for concept relationships
    try:
        # Find prerequisites for struggling concepts
        for concept in struggling_concepts:
            if concept in concept_entities:
                neighbors = await graph_adaptor.query_graph(
                    'get_neighbors', 
                    {'entity_id': concept_entities[concept], 'direction': 'incoming'}
                )
                print(f"  Prerequisites for {concept}: {len(neighbors)} found")
        
        # Get all stored context
        all_context = await kv_adaptor.get_all_context()
        print(f"  Total context items stored: {len(all_context)}")
        
    except Exception as e:
        print(f"✗ Failed to query data: {e}")
        await rag_service.close()
        return False
    
    # Step 7: Generate recommendations
    print("\\n7. Generating learning recommendations...")
    
    recommendations = []
    
    # Recommend reviewing prerequisites for struggling concepts
    for concept in struggling_concepts:
        if concept == 'algebra':
            recommendations.append(f"Review prerequisites: integers and fractions before continuing with algebra")
        elif concept in ['integers', 'fractions']:
            recommendations.append(f"Review basic arithmetic before advancing to {concept}")
    
    # Recommend next concepts for mastered ones
    if 'basic_arithmetic' in mastered_concepts and 'integers' not in mastered_concepts:
        recommendations.append("Ready to start learning integers")
    
    if all(c in mastered_concepts for c in ['basic_arithmetic', 'integers', 'fractions']):
        recommendations.append("Ready to start learning algebra")
    
    for rec in recommendations:
        print(f"  📚 {rec}")
    
    # Step 8: Clean up
    print("\\n8. Cleaning up...")
    await rag_service.close()
    print("✓ RAG service closed")
    
    print("\\n=== Integration Test Complete ===")
    print("✓ Successfully demonstrated math learning system with RAG backend integration!")
    print(f"✓ Processed {len(exercises)} concept areas with {sum(len(results) for _, results in exercises)} total exercises")
    print(f"✓ Stored {len(concept_entities)} concepts and {len(prerequisites)} relationships in knowledge graph")
    print(f"✓ Generated {len(recommendations)} personalized learning recommendations")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_full_integration())
    sys.exit(0 if success else 1) 