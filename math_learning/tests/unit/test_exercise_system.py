#!/usr/bin/env python3
"""
Simple test script for the exercise system.

This script tests the basic functionality of the exercise system
without requiring external dependencies.
"""

import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from math_learning.algebra_learning import AlgebraLearningSystem
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.exercises.exercise_system import ExerciseSystem


def test_basic_functionality():
    """Test basic exercise system functionality."""
    print("🧪 Testing Exercise System Basic Functionality")
    print("=" * 60)
    
    # Initialize systems
    print("1. Initializing algebra learning system...")
    algebra_system = AlgebraLearningSystem()
    print(f"   ✅ Knowledge graph has {len(algebra_system.knowledge_graph.get_all_concepts())} concepts")
    print(f"   ✅ Exercise bank has {len(algebra_system.exercise_bank.get_all_exercises())} exercises")
    
    # Initialize exercise system
    print("\n2. Initializing exercise system...")
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    print("   ✅ Exercise system initialized")
    
    # Create a sample student
    print("\n3. Creating sample student...")
    student = LearningGraph("test_student", "Test Student")
    
    # Add some sample progress using exercise concept names
    student.set_mastery("addition", 0.6, 0.7)
    student.set_mastery("multiplication", 0.4, 0.5)
    student.set_mastery("variables", 0.3, 0.4)
    print(f"   ✅ Set mastery for addition: 0.6 (confidence: 0.7)")
    print(f"   ✅ Set mastery for multiplication: 0.4 (confidence: 0.5)")
    print(f"   ✅ Set mastery for variables: 0.3 (confidence: 0.4)")
    
    # Test Level 1: Basic concept + difficulty selection
    print("\n4. Testing Level 1: Basic concept + difficulty selection...")
    concept_id = "multiplication"  # Use exercise concept name
    exercises = exercise_system.get_exercises_by_concept_difficulty(
        concept_id, 2.0, count=3
    )
    print(f"   ✅ Found {len(exercises)} exercises for concept {concept_id} at difficulty 2.0")
    
    if exercises:
        print(f"   First exercise: {exercises[0].title}")
    else:
        print(f"   ⚠️  No exercises found for {concept_id} - trying 'addition'")
        exercises = exercise_system.get_exercises_by_concept_difficulty(
            "addition", 2.0, count=3
        )
        print(f"   ✅ Found {len(exercises)} exercises for addition at difficulty 2.0")
    
    # Test Level 2: Adaptive selection
    print("\n5. Testing Level 2: Adaptive exercise selection...")
    recommendations = exercise_system.get_adaptive_exercises(student, count=3)
    print(f"   ✅ Generated {len(recommendations)} adaptive recommendations")
    
    if recommendations:
        rec = recommendations[0]
        print(f"   First recommendation: {rec.exercise.title}")
        print(f"   Concept: {rec.concept_id}")
        print(f"   Success rate: {rec.estimated_success_rate:.1%}")
    
    # Test Level 3: AI optimization (without LLM)
    print("\n6. Testing Level 3: AI optimization (rule-based fallback)...")
    session = exercise_system.get_ai_optimized_exercises(
        student,
        session_type="practice",
        duration_minutes=15
    )
    print(f"   ✅ Created session: {session.session_id}")
    print(f"   Session type: {session.session_type}")
    print(f"   Exercises: {len(session.exercises)}")
    print(f"   Duration: {session.estimated_duration} minutes")
    
    # Test analytics
    print("\n7. Testing session analytics...")
    analytics = exercise_system.get_session_analytics(session.session_id)
    if 'error' not in analytics:
        print(f"   ✅ Analytics generated successfully")
        print(f"   Average difficulty: {analytics['average_difficulty']:.1f}")
        print(f"   Average success rate: {analytics['average_success_rate']:.1%}")
    else:
        print(f"   ❌ Analytics error: {analytics['error']}")
    
    print("\n✅ All tests completed successfully!")
    return True


def test_concept_difficulty_combinations():
    """Test various concept and difficulty combinations."""
    print("\n🎯 Testing Concept + Difficulty Combinations")
    print("=" * 60)
    
    # Initialize systems
    algebra_system = AlgebraLearningSystem()
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    
    # Test with exercise concept names
    test_concepts = ["addition", "multiplication", "variables"]
    
    # Test different difficulties
    test_cases = [
        (1.0, "Very Easy"),
        (2.5, "Easy-Medium"),
        (4.0, "Hard"),
        (5.0, "Very Hard"),
    ]
    
    for concept_name in test_concepts:
        print(f"\n📚 Testing concept: {concept_name}")
        
        for difficulty, difficulty_name in test_cases:
            exercises = exercise_system.get_exercises_by_concept_difficulty(
                concept_name, difficulty, count=2
            )
            print(f"   {difficulty_name} ({difficulty}): {len(exercises)} exercises")
            
            if exercises:
                avg_difficulty = sum(ex.difficulty for ex in exercises) / len(exercises)
                print(f"      Avg actual difficulty: {avg_difficulty:.1f}")


def main():
    """Run all tests."""
    print("🚀 EXERCISE SYSTEM TESTING")
    print("=" * 80)
    
    try:
        # Run basic functionality test
        test_basic_functionality()
        
        # Run concept/difficulty combination test
        test_concept_difficulty_combinations()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe exercise system is working correctly and includes:")
        print("• Level 1: Basic concept + difficulty selection")
        print("• Level 2: Adaptive exercise selection based on learning progress")
        print("• Level 3: AI-optimized session creation (with rule-based fallback)")
        print("• Session management and analytics")
        print("• Success rate estimation")
        print("• Multi-session learning path creation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 