#!/usr/bin/env python3
"""
Simple integration test for the math learning system with memory backends.
This test demonstrates the core functionality without requiring external services.
"""

import asyncio
import sys
import os

# Add the parent directory to the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.rag_config import create_simple_rag_service
from knowledge_graph.graph import KnowledgeGraph
from learning_graph.user_model import LearningGraph
from learning_graph.personalized_learning import PersonalizedLearningSystem
from agents.rag.api.entity import Entity
from agents.rag.api.context_item import ContextItem


async def create_test_knowledge_graph():
    """Create a simple knowledge graph for testing."""
    kg = KnowledgeGraph()
    
    # Add basic math concepts
    concepts = [
        ("basic_arithmetic", "Basic arithmetic operations"),
        ("integers", "Working with integers"),
        ("fractions", "Understanding fractions"),
        ("algebra", "Basic algebra concepts")
    ]
    
    for concept_id, description in concepts:
        kg.add_concept(concept_id, description)
    
    # Add prerequisite relationships
    kg.add_prerequisite("integers", "basic_arithmetic")
    kg.add_prerequisite("fractions", "basic_arithmetic")
    kg.add_prerequisite("algebra", "integers")
    
    return kg


async def test_simple_integration():
    """Test the simple integration of all components."""
    print("🧮 Simple Math Learning System Integration Test")
    print("=" * 60)
    
    try:
        # 1. Create RAG service with memory backends
        print("1. Creating RAG service with memory backends...")
        rag_service = await create_simple_rag_service()
        print("   ✅ RAG service created successfully")
        
        # 2. Create knowledge graph
        print("2. Creating knowledge graph...")
        kg = await create_test_knowledge_graph()
        print(f"   ✅ Knowledge graph created with {len(kg.concepts)} concepts")
        
        # 3. Store concepts in RAG system
        print("3. Storing concepts in RAG system...")
        for concept_id, concept in kg.concepts.items():
            entity = Entity(
                id=concept_id,
                type="concept",
                properties={
                    "name": concept.name,
                    "description": concept.description,
                    "difficulty": concept.difficulty
                }
            )
            await rag_service.store_entity(entity)
        print(f"   ✅ Stored {len(kg.concepts)} concepts in RAG system")
        
        # 4. Create learning graph for a user
        print("4. Creating learning graph for user...")
        lg = LearningGraph("test_user", "Test User")
        
        # Simulate some learning exercises
        exercises = [
            ("basic_arithmetic", True, 0.9),
            ("basic_arithmetic", True, 0.8),
            ("integers", True, 0.7),
            ("integers", False, 0.3),
            ("fractions", False, 0.2),
            ("fractions", True, 0.6),
        ]
        
        for concept, success, confidence in exercises:
            lg.record_exercise_attempt(concept, success, confidence)
        
        print(f"   ✅ Recorded {len(exercises)} exercise attempts")
        
        # 5. Store learning progress in RAG context
        print("5. Storing learning progress in RAG context...")
        for concept_id in kg.concepts:
            mastery = lg.get_mastery(concept_id)
            context_item = ContextItem(
                key=f"mastery_{concept_id}",
                value={
                    "user_id": lg.user_id,
                    "concept_id": concept_id,
                    "mastery_level": mastery,
                    "timestamp": "2024-01-01T00:00:00Z"
                }
            )
            await rag_service.store_context_item(context_item)
        
        print("   ✅ Learning progress stored in RAG context")
        
        # 6. Create personalized learning system
        print("6. Creating personalized learning system...")
        pls = PersonalizedLearningSystem(kg, rag_service)
        print("   ✅ Personalized learning system created")
        
        # 7. Analyze learning patterns
        print("7. Analyzing learning patterns...")
        patterns = pls.analyze_learning_patterns(lg)
        print(f"   ✅ Detected {len(patterns)} learning patterns:")
        for pattern_type, success_rate in patterns.items():
            print(f"      - {pattern_type}: {success_rate:.1%} success rate")
        
        # 8. Generate personalized recommendations
        print("8. Generating personalized recommendations...")
        recommendations = pls.generate_adaptive_recommendations(lg)
        print(f"   ✅ Generated {len(recommendations)} recommendations:")
        for rec in recommendations:
            print(f"      - {rec.type}: {rec.content}")
        
        # 9. Show mastery levels
        print("9. Current mastery levels:")
        for concept_id in kg.concepts:
            mastery = lg.get_mastery(concept_id)
            print(f"   - {concept_id}: {mastery:.1%}")
        
        # 10. Cleanup
        print("10. Cleaning up...")
        await rag_service.close()
        print("    ✅ Resources cleaned up")
        
        print("\n🎉 Simple integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_simple_integration()
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 