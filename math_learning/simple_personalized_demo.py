#!/usr/bin/env python3
"""
Simple Personalized Learning System Demo

This demo showcases the core personalization features without requiring visualization libraries.
"""

import json
import random
from datetime import datetime, timedelta

from .learning_graph.user_model import LearningGraph
from .learning_graph.personalized_learning import PersonalizedLearningSystem
from .knowledge_graph.graph import KnowledgeGraph
from .recommendation.gap_analyzer import GapAnalyzer
from .algebra_learning import AlgebraLearningSystem


def simulate_learning_session(algebra_system, learning_graph, num_exercises=10):
    """Simulate a learning session with exercises."""
    print(f"\n=== Simulating Learning Session ({num_exercises} exercises) ===")
    
    # Get some actual concepts from the algebra system
    all_concepts = algebra_system.knowledge_graph.get_all_concepts()
    available_concepts = [c for c in all_concepts if c.difficulty <= 3]  # Focus on easier concepts
    
    if not available_concepts:
        print("No suitable concepts found for simulation")
        return
    
    exercises_attempted = 0
    
    for i in range(num_exercises):
        # Pick a random concept
        concept = random.choice(available_concepts)
        concept_id = concept.id
        
        print(f"\nExercise {i+1}: Working on {concept.name} (ID: {concept_id})")
        
        # Try to get exercises for this concept
        exercises = algebra_system.exercise_bank.get_exercises_for_concept(concept_id)
        
        if not exercises:
            # If no exercises found, create a simulated exercise result
            print(f"  No exercises found for {concept.name}, simulating...")
            # Simulate performance based on current mastery
            current_mastery = learning_graph.get_mastery(concept_id)
            success_probability = max(0.1, current_mastery + random.uniform(-0.3, 0.3))
            success = random.random() < success_probability
            
            # Record the exercise attempt with difficulty and weight
            exercise_id = f"sim_{concept_id}_{i}"
            difficulty = random.uniform(0.3, 0.8)  # Random difficulty
            concept_weight = 1.0  # Full weight for concept
            learning_graph.record_exercise_attempt(exercise_id, concept_id, success, difficulty, concept_weight)
            
            print(f"  Simulated exercise: {'Success' if success else 'Failed'}")
            exercises_attempted += 1
        else:
            # Use actual exercise
            exercise = random.choice(exercises)
            print(f"  Exercise: {exercise.title}")
            
            # Simulate performance
            current_mastery = learning_graph.get_mastery(concept_id)
            # Adjust success probability based on exercise difficulty vs mastery
            difficulty_factor = (6 - exercise.difficulty) / 5.0  # Higher difficulty = lower success
            success_probability = current_mastery * difficulty_factor + random.uniform(-0.2, 0.2)
            success_probability = max(0.05, min(0.95, success_probability))  # Clamp between 5% and 95%
            
            success = random.random() < success_probability
            
            # Record the exercise attempt with difficulty and weight
            exercise_id = f"real_{concept_id}_{i}"
            difficulty = random.uniform(0.4, 0.9)  # Slightly higher difficulty for real exercises
            concept_weight = 1.0  # Full weight for concept
            learning_graph.record_exercise_attempt(exercise_id, concept_id, success, difficulty, concept_weight)
            
            print(f"  Result: {'Success' if success else 'Failed'} (Mastery: {learning_graph.get_mastery(concept_id):.2f})")
            exercises_attempted += 1
    
    print(f"\nSimulation complete! {exercises_attempted} exercises attempted.")


def demonstrate_learning_patterns(personalized_system: PersonalizedLearningSystem,
                                learning_graph: LearningGraph) -> None:
    """Demonstrate learning pattern analysis."""
    print("\n" + "="*60)
    print("🧠 LEARNING PATTERN ANALYSIS")
    print("="*60)
    
    patterns = personalized_system.analyze_learning_patterns(learning_graph)
    
    print(f"📈 Analyzed patterns for {len(patterns)} concepts:")
    
    for concept_id, pattern in list(patterns.items())[:5]:  # Show first 5
        print(f"\n🎯 Concept: {concept_id}")
        print(f"   • Attempts: {pattern.attempts}")
        print(f"   • Success Rate: {pattern.success_rate:.2%}")
        print(f"   • Learning Velocity: {pattern.learning_velocity:.3f}")
        print(f"   • Retention Rate: {pattern.retention_rate:.2%}")
        print(f"   • Mistake Patterns: {', '.join(pattern.mistake_patterns) if pattern.mistake_patterns else 'None identified'}")


def demonstrate_weakness_detection(personalized_system: PersonalizedLearningSystem,
                                 learning_graph: LearningGraph) -> None:
    """Demonstrate weakness detection and analysis."""
    print("\n" + "="*60)
    print("🔍 WEAKNESS DETECTION & ANALYSIS")
    print("="*60)
    
    weaknesses = personalized_system.detect_weaknesses(learning_graph)
    
    if not weaknesses:
        print("🎉 No significant weaknesses detected!")
        return
    
    print(f"⚠️  Detected {len(weaknesses)} areas needing attention:")
    
    for i, weakness in enumerate(weaknesses[:3], 1):  # Show top 3
        print(f"\n{i}. 🎯 Concept: {weakness.concept_id}")
        print(f"   • Type: {weakness.weakness_type.title()}")
        print(f"   • Severity: {weakness.severity:.2f}/1.0")
        print(f"   • Confidence: {weakness.confidence_level:.2%}")
        print(f"   • Related Struggling Concepts: {', '.join(weakness.related_concepts[:3])}")
        print(f"   • Suggested Interventions:")
        for intervention in weakness.suggested_interventions[:2]:
            print(f"     - {intervention}")


def demonstrate_personalized_recommendations(personalized_system: PersonalizedLearningSystem,
                                           learning_graph: LearningGraph) -> None:
    """Demonstrate adaptive recommendations."""
    print("\n" + "="*60)
    print("🎯 PERSONALIZED RECOMMENDATIONS")
    print("="*60)
    
    recommendations = personalized_system.get_adaptive_recommendations(learning_graph, num_recommendations=5)
    
    print(f"💡 Generated {len(recommendations)} personalized recommendations:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. 📚 {rec['type'].title()}: {rec['concept_id']}")
        print(f"   • Priority: {rec['priority']:.2f}")
        print(f"   • Reason: {rec['reason']}")
        if 'estimated_time' in rec:
            print(f"   • Estimated Time: {rec['estimated_time']:.0f} minutes")


def demonstrate_learning_insights(personalized_system: PersonalizedLearningSystem,
                                learning_graph: LearningGraph) -> None:
    """Demonstrate learning insights generation."""
    print("\n" + "="*60)
    print("💡 LEARNING INSIGHTS")
    print("="*60)
    
    insights = personalized_system.generate_learning_insights(learning_graph)
    
    print(f"🔍 Generated {len(insights)} actionable insights:")
    
    for i, insight in enumerate(insights[:4], 1):  # Show first 4
        print(f"\n{i}. {insight.insight_type.upper()}: {insight.title}")
        print(f"   📝 {insight.description}")
        print(f"   🎯 Confidence: {insight.confidence:.2%}")
        if insight.actionable_steps:
            print(f"   📋 Action Steps:")
            for step in insight.actionable_steps[:2]:
                print(f"      • {step}")


def demonstrate_personalized_learning_path(personalized_system: PersonalizedLearningSystem,
                                         algebra_system: AlgebraLearningSystem,
                                         learning_graph: LearningGraph) -> None:
    """Demonstrate personalized learning path generation."""
    print("\n" + "="*60)
    print("🛤️  PERSONALIZED LEARNING PATH")
    print("="*60)
    
    # Find a target concept that the user hasn't mastered yet
    target_concept = None
    for concept in algebra_system.knowledge_graph.get_all_concepts():
        if concept.id not in learning_graph.concept_mastery or learning_graph.concept_mastery[concept.id]['mastery'] < 0.7:
            if concept.difficulty <= 3:  # Choose a reasonable target
                target_concept = concept
                break
    
    if not target_concept:
        print("🎉 User has mastered all available concepts!")
        return
    
    print(f"🎯 Target Concept: {target_concept.name} (ID: {target_concept.id})")
    print(f"   Difficulty: {target_concept.difficulty}/5")
    print(f"   Category: {target_concept.category}")
    
    # Generate personalized learning path
    path = personalized_system.generate_personalized_path(learning_graph, target_concept.id, max_concepts=8)
    
    if not path:
        print("❌ Could not generate a learning path for this concept.")
        return
    
    print(f"\n📚 Recommended Learning Path ({len(path)} concepts):")
    
    for i, (concept_id, priority) in enumerate(path, 1):
        concept = algebra_system.knowledge_graph.get_concept(concept_id)
        mastery = learning_graph.get_mastery(concept_id)
        
        print(f"\n{i}. {concept.name}")
        print(f"   • Priority Score: {priority:.2f}")
        print(f"   • Current Mastery: {mastery:.1%}")
        print(f"   • Difficulty: {concept.difficulty}/5")
        print(f"   • Category: {concept.category}")


def main():
    """Main demo function."""
    print("🚀 PERSONALIZED MATH LEARNING SYSTEM DEMO")
    print("="*60)
    
    # Initialize the algebra learning system
    print("🔧 Initializing Algebra Learning System...")
    algebra_system = AlgebraLearningSystem()
    
    # Create a personalized learning system
    personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
    
    # Create a sample user
    user_id = "demo_user_001"
    learning_graph = LearningGraph(user_id)
    
    # Simulate learning history
    simulate_learning_session(algebra_system, learning_graph, num_exercises=15)
    
    # Demonstrate various features
    demonstrate_learning_patterns(personalized_system, learning_graph)
    demonstrate_weakness_detection(personalized_system, learning_graph)
    demonstrate_personalized_recommendations(personalized_system, learning_graph)
    demonstrate_learning_insights(personalized_system, learning_graph)
    demonstrate_personalized_learning_path(personalized_system, algebra_system, learning_graph)
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE!")
    print("="*60)
    print("The personalized learning system successfully:")
    print("✅ Analyzed learning patterns")
    print("✅ Detected weaknesses and provided interventions")
    print("✅ Generated adaptive recommendations")
    print("✅ Provided actionable learning insights")
    print("✅ Created personalized learning paths")
    print("\nThis system can now be integrated into educational platforms")
    print("to provide truly personalized learning experiences! 🎓")


if __name__ == "__main__":
    main() 