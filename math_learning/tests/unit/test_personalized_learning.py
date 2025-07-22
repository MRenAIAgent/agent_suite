#!/usr/bin/env python3
"""
Test the PersonalizedLearningSystem with advanced pattern analysis and adaptive recommendations.
"""

import asyncio
import sys
import os
from typing import Dict, List

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from config.rag_config import get_memory_config, create_configured_rag_service, RagConfig, create_simple_rag_service
from learning_graph.user_model import LearningGraph
from learning_graph.personalized_learning import PersonalizedLearningSystem
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept import Concept
from agents.rag.models.knowledge import Entity, Relationship
from agents.rag.models.context import ContextItem


async def create_test_knowledge_graph() -> KnowledgeGraph:
    """Create a test knowledge graph with math concepts."""
    kg = KnowledgeGraph("Test Math Knowledge Graph")
    
    # Create concepts using the correct Concept constructor
    concepts = [
        Concept("Basic Arithmetic", "Addition, subtraction, multiplication, division", 
                difficulty=1, concept_id="basic_arithmetic"),
        Concept("Integers", "Positive and negative whole numbers", 
                difficulty=2, concept_id="integers"),
        Concept("Fractions", "Parts of a whole number", 
                difficulty=3, concept_id="fractions"),
        Concept("Decimals", "Numbers with decimal points", 
                difficulty=3, concept_id="decimals"),
        Concept("Percentages", "Parts per hundred", 
                difficulty=3, concept_id="percentages"),
        Concept("Basic Algebra", "Variables and simple equations", 
                difficulty=4, concept_id="algebra"),
        Concept("Linear Equations", "Equations with one variable", 
                difficulty=4, concept_id="linear_equations"),
        Concept("Quadratic Equations", "Second-degree polynomial equations", 
                difficulty=5, concept_id="quadratic_equations")
    ]
    
    # Add concepts to knowledge graph
    for concept in concepts:
        kg.add_concept(concept)
    
    # Add prerequisite relationships
    kg.add_prerequisite("basic_arithmetic", "integers")
    kg.add_prerequisite("integers", "fractions")
    kg.add_prerequisite("fractions", "decimals")
    kg.add_prerequisite("decimals", "percentages")
    kg.add_prerequisite("integers", "algebra")
    kg.add_prerequisite("algebra", "linear_equations")
    kg.add_prerequisite("linear_equations", "quadratic_equations")
    
    return kg


async def test_learning_pattern_analysis():
    """Test the learning pattern analysis capabilities."""
    print("🧠 Testing Learning Pattern Analysis")
    print("----------------------------------------")
    
    try:
        # Create RAG service with simple memory configuration
        rag_service = await create_simple_rag_service()
        
        # Create knowledge graph
        kg = await create_test_knowledge_graph()
        
        # Create learning graph with some exercise history
        lg = LearningGraph("test_user", "Test User")
        
        # Simulate learning history with patterns
        exercises = [
            ("basic_arithmetic", True, 0.9),
            ("basic_arithmetic", True, 0.8),
            ("integers", True, 0.7),
            ("integers", False, 0.3),  # Struggle with integers
            ("fractions", False, 0.2),  # Major struggle with fractions
            ("fractions", False, 0.1),
            ("algebra", True, 0.8),  # Surprisingly good at algebra
        ]
        
        for concept_id, success, confidence in exercises:
            lg.record_exercise_attempt(concept_id, success, confidence)
        
        # Create personalized learning system
        pls = PersonalizedLearningSystem(kg, rag_service)
        
        # Analyze learning patterns
        patterns = pls.analyze_learning_patterns(lg)
        
        print(f"  ✅ Detected {len(patterns)} learning patterns")
        for pattern in list(patterns.values())[:3]:  # Show top 3
            print(f"    - {pattern.concept_id}: success rate {pattern.success_rate:.1%}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def test_weakness_detection():
    """Test the weakness detection capabilities."""
    print("🔍 Testing Weakness Detection")
    print("----------------------------------------")
    
    try:
        # Create RAG service with simple memory configuration
        rag_service = await create_simple_rag_service()
        
        # Create knowledge graph
        kg = await create_test_knowledge_graph()
        
        # Create learning graph with clear weaknesses
        lg = LearningGraph("test_user2", "Test User 2")
        
        # Simulate exercises showing weakness in fractions
        exercises = [
            ("basic_arithmetic", True, 0.9),
            ("integers", True, 0.8),
            ("fractions", False, 0.2),
            ("fractions", False, 0.1),
            ("fractions", False, 0.3),
            ("decimals", False, 0.2),  # Related weakness
        ]
        
        for concept_id, success, confidence in exercises:
            lg.record_exercise_attempt(concept_id, success, confidence)
        
        # Create personalized learning system
        pls = PersonalizedLearningSystem(kg, rag_service)
        
        # Detect weaknesses
        weaknesses = pls.detect_weaknesses(lg)
        
        print(f"  ✅ Detected {len(weaknesses)} knowledge gaps")
        for weakness in weaknesses[:3]:
            print(f"    - {weakness.concept_id}: {weakness.severity:.1%} severity")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def test_personalized_path_generation():
    """Test personalized learning path generation."""
    print("🛤️  Testing Personalized Path Generation")
    print("----------------------------------------")
    
    try:
        # Create RAG service with simple memory configuration
        rag_service = await create_simple_rag_service()
        
        # Create knowledge graph
        kg = await create_test_knowledge_graph()
        
        # Create learning graph
        lg = LearningGraph("test_user3", "Test User 3")
        
        # Simulate partial mastery
        exercises = [
            ("basic_arithmetic", True, 0.9),
            ("integers", True, 0.7),
            ("fractions", False, 0.4),  # Partial understanding
        ]
        
        for concept_id, success, confidence in exercises:
            lg.record_exercise_attempt(concept_id, success, confidence)
        
        # Create personalized learning system
        pls = PersonalizedLearningSystem(kg, rag_service)
        
        # Generate personalized path
        path = pls.generate_personalized_path(lg, "algebra", max_concepts=5)
        
        print(f"  ✅ Generated learning path with {len(path)} concepts")
        for i, step in enumerate(path[:3], 1):
            print(f"    {i}. {step[0]} (priority: {step[1]:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def test_adaptive_recommendations():
    """Test adaptive recommendation generation."""
    print("🎯 Testing Adaptive Recommendations")
    print("----------------------------------------")
    
    try:
        # Create RAG service with simple memory configuration
        rag_service = await create_simple_rag_service()
        
        # Create knowledge graph
        kg = await create_test_knowledge_graph()
        
        # Create learning graph
        lg = LearningGraph("test_user4", "Test User 4")
        
        # Simulate recent struggles
        exercises = [
            ("fractions", False, 0.2),
            ("fractions", False, 0.3),
            ("decimals", False, 0.1),
        ]
        
        for concept_id, success, confidence in exercises:
            lg.record_exercise_attempt(concept_id, success, confidence)
        
        # Create personalized learning system
        pls = PersonalizedLearningSystem(kg, rag_service)
        
        # Generate adaptive recommendations
        recommendations = pls.get_adaptive_recommendations(lg, num_recommendations=3)
        
        print(f"  ✅ Generated {len(recommendations)} adaptive recommendations")
        for rec in recommendations:
            print(f"    - {rec['type']}: {rec['title']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def test_learning_insights():
    """Test learning insights generation."""
    print("💡 Testing Learning Insights")
    print("----------------------------------------")
    
    try:
        # Create RAG service with simple memory configuration
        rag_service = await create_simple_rag_service()
        
        # Create knowledge graph
        kg = await create_test_knowledge_graph()
        
        # Create learning graph with diverse history
        lg = LearningGraph("test_user5", "Test User 5")
        
        # Simulate varied performance
        exercises = [
            ("basic_arithmetic", True, 0.9),
            ("integers", True, 0.8),
            ("fractions", False, 0.3),
            ("algebra", True, 0.7),  # Skip ahead
            ("decimals", False, 0.2),
        ]
        
        for concept_id, success, confidence in exercises:
            lg.record_exercise_attempt(concept_id, success, confidence)
        
        # Create personalized learning system
        pls = PersonalizedLearningSystem(kg, rag_service)
        
        # Generate insights
        insights = pls.generate_learning_insights(lg)
        
        print(f"  ✅ Generated {len(insights)} learning insights")
        for insight in insights[:3]:
            print(f"    - {insight.insight_type}: {insight.title}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


async def main():
    """Run all personalized learning tests."""
    print("🎓 PERSONALIZED LEARNING SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_learning_pattern_analysis,
        test_weakness_detection,
        test_personalized_path_generation,
        test_adaptive_recommendations,
        test_learning_insights
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100
    
    print("📊 FINAL RESULTS")
    print("=" * 30)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"🎯 Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! The PersonalizedLearningSystem is working correctly.")
    else:
        print("\n⚠️  Some tests failed - review the output above")


if __name__ == "__main__":
    asyncio.run(main()) 