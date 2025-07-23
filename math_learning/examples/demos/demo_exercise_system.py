#!/usr/bin/env python3
"""
Exercise System Demo

This demo showcases the comprehensive exercise system with:
1. Basic concept + difficulty selection
2. Adaptive exercise selection based on learning progress
3. AI-powered learning path optimization
4. Session management and analytics
"""

import json
import random
from datetime import datetime
from typing import List, Dict, Any

from .algebra_learning import AlgebraLearningSystem
from .learning_graph.user_model import LearningGraph
from .exercises.exercise_system import ExerciseSystem
from .exercises.exercise_selector import DifficultyLevel

# Try to import LLM
try:
    from ..llm.openai.openai_llm import OpenAILLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    OpenAILLM = None


def create_sample_student() -> LearningGraph:
    """Create a sample student with some learning progress."""
    print("📚 Creating sample student learning profile...")
    
    student = LearningGraph("demo_student_2024", "Alex Johnson")
    
    # Simulate some learning progress
    progress_data = [
        # (concept_id, mastery, confidence)
        ("NS-01", 0.9, 0.8),  # Counting - well mastered
        ("NS-02", 0.8, 0.7),  # Place value - good
        ("NS-03", 0.7, 0.6),  # Commutative property - decent
        ("NS-06", 0.4, 0.5),  # Multiplication facts - struggling
        ("NS-07", 0.3, 0.4),  # Division facts - struggling more
        ("AT-01", 0.2, 0.3),  # Variables - just starting
        ("AT-02", 0.1, 0.2),  # Expressions - very new
    ]
    
    for concept_id, mastery, confidence in progress_data:
        student.set_mastery(concept_id, mastery, confidence)
        print(f"  {concept_id}: Mastery={mastery:.1f}, Confidence={confidence:.1f}")
    
    return student


def demo_level_1_basic_selection():
    """Demo Level 1: Basic concept + difficulty selection."""
    print("\n" + "="*80)
    print("🎯 LEVEL 1 DEMO: Basic Concept + Difficulty Selection")
    print("="*80)
    
    # Initialize system
    algebra_system = AlgebraLearningSystem()
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    
    # Test different concept/difficulty combinations
    test_cases = [
        ("NS-06", 2.0, "Multiplication facts at easy level"),
        ("NS-06", 4.0, "Multiplication facts at hard level"),
        ("AT-01", 1.5, "Variables at very easy level"),
        ("AT-01", 3.0, "Variables at medium level"),
    ]
    
    for concept_id, difficulty, description in test_cases:
        print(f"\n🔍 Testing: {description}")
        exercises = exercise_system.get_exercises_by_concept_difficulty(
            concept_id, difficulty, count=3
        )
        
        print(f"   Found {len(exercises)} exercises:")
        for i, exercise in enumerate(exercises, 1):
            print(f"   {i}. {exercise.title} (Difficulty: {exercise.difficulty})")
            print(f"      Problem: {exercise.problem}")
            print(f"      Solution: {exercise.solution}")


def demo_level_2_adaptive_selection():
    """Demo Level 2: Adaptive selection with learning graph."""
    print("\n" + "="*80)
    print("🧠 LEVEL 2 DEMO: Adaptive Exercise Selection")
    print("="*80)
    
    # Initialize system
    algebra_system = AlgebraLearningSystem()
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    
    # Create sample student
    student = create_sample_student()
    
    print("\n🎯 Getting adaptive exercise recommendations...")
    recommendations = exercise_system.get_adaptive_exercises(
        student, count=5
    )
    
    print(f"\nGenerated {len(recommendations)} adaptive recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.exercise.title}")
        print(f"   Concept: {rec.concept_id}")
        print(f"   Difficulty: {rec.difficulty_score:.1f}/5.0")
        print(f"   Confidence: {rec.confidence_score:.1f}")
        print(f"   Success Rate: {rec.estimated_success_rate:.1%}")
        print(f"   Reason: {rec.reason}")
        print(f"   Problem: {rec.exercise.problem}")
    
    # Test with specific concepts
    print("\n🎯 Testing with specific struggling concepts...")
    struggling_concepts = student.get_struggling_concepts(threshold=0.4)
    print(f"Struggling concepts: {struggling_concepts}")
    
    if struggling_concepts:
        targeted_recommendations = exercise_system.get_adaptive_exercises(
            student, target_concepts=struggling_concepts[:2], count=3
        )
        
        print(f"\nTargeted recommendations for struggling concepts:")
        for i, rec in enumerate(targeted_recommendations, 1):
            print(f"{i}. {rec.exercise.title} (Concept: {rec.concept_id})")


def demo_level_3_ai_optimization():
    """Demo Level 3: AI-powered optimization."""
    print("\n" + "="*80)
    print("🤖 LEVEL 3 DEMO: AI-Powered Exercise Optimization")
    print("="*80)
    
    # Initialize system
    algebra_system = AlgebraLearningSystem()
    
    # Try to initialize with LLM
    llm = None
    if LLM_AVAILABLE:
        try:
            llm = OpenAILLM(model="gpt-3.5-turbo")
            print("✅ AI/LLM integration available")
        except Exception as e:
            print(f"⚠️  AI/LLM not available: {e}")
    else:
        print("⚠️  AI/LLM not available - will use rule-based fallback")
    
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph,
        llm
    )
    
    # Create sample student
    student = create_sample_student()
    
    # Test different session types
    session_types = [
        ("practice", 20, "Regular practice session"),
        ("assessment", 15, "Quick assessment"),
        ("review", 25, "Review session"),
        ("challenge", 30, "Challenge session"),
    ]
    
    for session_type, duration, description in session_types:
        print(f"\n🎯 Creating {description} ({duration} minutes)...")
        
        session = exercise_system.get_ai_optimized_exercises(
            student,
            session_type=session_type,
            duration_minutes=duration,
            learning_goals=[f"Improve {session_type} skills"]
        )
        
        print(f"   Session ID: {session.session_id}")
        print(f"   Type: {session.session_type}")
        print(f"   Exercises: {len(session.exercises)}")
        print(f"   Estimated Duration: {session.estimated_duration} minutes")
        print(f"   Target Concepts: {session.target_concepts}")
        
        # Show first 2 exercises
        for i, rec in enumerate(session.exercises[:2], 1):
            print(f"   {i}. {rec.exercise.title} (Success Rate: {rec.estimated_success_rate:.1%})")
        
        # Get session analytics
        analytics = exercise_system.get_session_analytics(session.session_id)
        print(f"   Analytics: Avg Difficulty={analytics['average_difficulty']:.1f}, "
              f"Avg Success Rate={analytics['average_success_rate']:.1%}")


def demo_learning_path_sessions():
    """Demo learning path creation."""
    print("\n" + "="*80)
    print("🛤️  LEARNING PATH DEMO: Multi-Session Learning Journey")
    print("="*80)
    
    # Initialize system
    algebra_system = AlgebraLearningSystem()
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    
    # Create sample student
    student = create_sample_student()
    
    # Create learning path towards a target concept
    target_concept = "AT-02"  # Expressions
    print(f"🎯 Creating learning path towards: {target_concept}")
    
    sessions = exercise_system.create_learning_path_session(
        student, target_concept, session_count=3
    )
    
    print(f"\nCreated {len(sessions)} learning sessions:")
    for i, session in enumerate(sessions, 1):
        print(f"\n📅 Session {i}: {session.session_id}")
        print(f"   Type: {session.session_type}")
        print(f"   Duration: {session.estimated_duration} minutes")
        print(f"   Concepts: {session.target_concepts}")
        print(f"   Exercises: {len(session.exercises)}")
        print(f"   Objectives: {session.learning_objectives}")
        
        # Show exercise progression
        for j, rec in enumerate(session.exercises[:2], 1):
            print(f"   {j}. {rec.exercise.title} (Concept: {rec.concept_id})")


def demo_comprehensive_analytics():
    """Demo comprehensive analytics and reporting."""
    print("\n" + "="*80)
    print("📊 ANALYTICS DEMO: Session Analytics and Insights")
    print("="*80)
    
    # Initialize system
    algebra_system = AlgebraLearningSystem()
    exercise_system = ExerciseSystem(
        algebra_system.exercise_bank,
        algebra_system.knowledge_graph
    )
    
    # Create sample student
    student = create_sample_student()
    
    # Create a practice session
    session = exercise_system.get_ai_optimized_exercises(
        student,
        session_type="practice",
        duration_minutes=25
    )
    
    # Get comprehensive analytics
    analytics = exercise_system.get_session_analytics(session.session_id)
    
    print(f"📈 Session Analytics for: {session.session_id}")
    print(f"   Total Exercises: {analytics['total_exercises']}")
    print(f"   Estimated Duration: {analytics['estimated_duration']} minutes")
    print(f"   Average Success Rate: {analytics['average_success_rate']:.1%}")
    print(f"   Average Difficulty: {analytics['average_difficulty']:.1f}/5.0")
    
    print(f"\n🎯 Difficulty Distribution:")
    for difficulty, count in analytics['difficulty_distribution'].items():
        print(f"   {difficulty}: {count} exercises")
    
    print(f"\n📚 Concept Distribution:")
    for concept, count in analytics['concept_distribution'].items():
        print(f"   {concept}: {count} exercises")
    
    print(f"\n🎯 Learning Objectives:")
    for objective in analytics['learning_objectives']:
        print(f"   • {objective}")


def main():
    """Run the complete exercise system demo."""
    print("🚀 COMPREHENSIVE EXERCISE SYSTEM DEMO")
    print("=" * 80)
    print("This demo showcases a 4-level exercise system:")
    print("1. Basic concept + difficulty selection")
    print("2. Adaptive selection based on learning progress")
    print("3. AI-powered optimization")
    print("4. Learning path creation and analytics")
    
    try:
        # Run all demos
        demo_level_1_basic_selection()
        demo_level_2_adaptive_selection()
        demo_level_3_ai_optimization()
        demo_learning_path_sessions()
        demo_comprehensive_analytics()
        
        print("\n" + "="*80)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nKey Features Demonstrated:")
        print("• Static exercise selection by concept and difficulty")
        print("• Adaptive difficulty based on student mastery and confidence")
        print("• AI-powered session optimization (with fallback)")
        print("• Multi-session learning path creation")
        print("• Comprehensive session analytics")
        print("• Success rate prediction")
        print("• Intelligent exercise scoring and ranking")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 