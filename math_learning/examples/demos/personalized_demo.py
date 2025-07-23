#!/usr/bin/env python3
"""
Personalized Learning System Demo

This demo showcases the advanced personalization features of the math learning system.
"""

import json
import random
from datetime import datetime, timedelta

from .learning_graph.user_model import LearningGraph
from .learning_graph.personalized_learning import PersonalizedLearningSystem
from .learning_graph.analytics_dashboard import LearningAnalyticsDashboard
from .knowledge_graph.graph import KnowledgeGraph
from .recommendation.gap_analyzer import GapAnalyzer
from .algebra_learning import AlgebraLearningSystem


def simulate_user_learning_history(learning_graph: LearningGraph, 
                                 algebra_system: AlgebraLearningSystem,
                                 num_sessions: int = 20) -> None:
    """
    Simulate a realistic learning history for a user.
    
    Args:
        learning_graph: The user's learning graph to populate
        algebra_system: The algebra learning system
        num_sessions: Number of learning sessions to simulate
    """
    print(f"🎯 Simulating {num_sessions} learning sessions...")
    
    # Get some basic concepts to start with
    all_concepts = algebra_system.knowledge_graph.get_all_concepts()
    basic_concepts = [c for c in all_concepts if c.difficulty <= 2]
    
    # Simulate learning sessions
    for session in range(num_sessions):
        # Choose 2-4 concepts to work on this session
        session_concepts = random.sample(basic_concepts, min(3, len(basic_concepts)))
        
        for concept in session_concepts:
            # Get exercises for this concept
            exercises = algebra_system.exercise_bank.get_exercises_for_concept(concept.id)
            if not exercises:
                continue
                
            # Simulate doing 2-5 exercises
            num_exercises = random.randint(2, 5)
            selected_exercises = random.sample(exercises, min(num_exercises, len(exercises)))
            
            for exercise in selected_exercises:
                # Simulate performance based on concept difficulty and user "ability"
                # Some users are better at certain types of concepts
                success_probability = 0.7
                
                # Adjust based on concept category
                if concept.category == "Number Sense":
                    success_probability += 0.1
                elif concept.category == "Algebra":
                    success_probability -= 0.1
                    
                # Adjust based on difficulty
                success_probability -= (concept.difficulty - 1) * 0.15
                
                # Add some randomness
                success_probability += random.uniform(-0.2, 0.2)
                success_probability = max(0.1, min(0.9, success_probability))
                
                # Determine if exercise was successful
                success = random.random() < success_probability
                
                # Record the exercise attempt
                learning_graph.record_exercise_attempt(exercise.id, concept.id, success)
                
                # Update mastery based on performance
                learning_graph.update_mastery(
                    concept.id, 
                    success, 
                    exercise.difficulty / 5.0,  # Normalize difficulty
                    0.8  # Concept weight
                )
    
    print(f"✅ Simulation complete! User attempted {len(learning_graph.exercise_history)} exercises")
    print(f"📊 Mastery data for {len(learning_graph.concept_mastery)} concepts")


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


def demonstrate_analytics_dashboard(dashboard: LearningAnalyticsDashboard,
                                  learning_graph: LearningGraph) -> None:
    """Demonstrate comprehensive analytics."""
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ANALYTICS DASHBOARD")
    print("="*60)
    
    # Generate progress report
    progress_report = dashboard.generate_progress_report(learning_graph)
    
    print("📈 PROGRESS METRICS:")
    metrics = progress_report['progress_metrics']
    print(f"   • Concepts Attempted: {metrics['total_concepts_attempted']}")
    print(f"   • Concepts Mastered: {metrics['concepts_mastered']}")
    print(f"   • Overall Mastery Score: {metrics['overall_mastery_score']:.2%}")
    print(f"   • Learning Velocity: {metrics['learning_velocity']:.3f}")
    print(f"   • Retention Rate: {metrics['retention_rate']:.2%}")
    
    print("\n📚 CATEGORY ANALYSIS:")
    for category in progress_report['category_analysis'][:3]:
        print(f"   • {category['category_name']}: {category['strength_level'].title()} "
              f"(Avg Mastery: {category['avg_mastery']:.2%})")
    
    print("\n🎯 MASTERY DISTRIBUTION:")
    distribution = progress_report['mastery_distribution']
    for level, count in distribution.items():
        print(f"   • {level}: {count} concepts")
    
    # Generate weakness analysis
    weakness_analysis = dashboard.generate_weakness_analysis(learning_graph)
    
    print(f"\n⚠️  WEAKNESS ANALYSIS:")
    print(f"   • Total Weaknesses: {weakness_analysis['total_weaknesses']}")
    print(f"   • Estimated Improvement Time: {weakness_analysis['estimated_improvement_time'].get('total_weeks', 0):.1f} weeks")
    
    # Track learning journey
    journey = dashboard.track_learning_journey(learning_graph, days_back=30)
    
    print(f"\n🚀 LEARNING JOURNEY (Last 30 days):")
    print(f"   • Momentum Score: {journey['momentum_score']:.2f}")
    if journey['learning_streaks']:
        print(f"   • Current Streak: {journey['learning_streaks'].get('current_streak', 0)} days")
        print(f"   • Longest Streak: {journey['learning_streaks'].get('longest_streak', 0)} days")


def demonstrate_personalized_learning_path(personalized_system: PersonalizedLearningSystem,
                                         algebra_system: AlgebraLearningSystem,
                                         learning_graph: LearningGraph) -> None:
    """Demonstrate personalized learning path generation."""
    print("\n" + "="*60)
    print("🛤️  PERSONALIZED LEARNING PATH")
    print("="*60)
    
    # Find a target concept that's not yet mastered
    all_concepts = algebra_system.knowledge_graph.get_all_concepts()
    target_concepts = [c for c in all_concepts if learning_graph.get_mastery(c.id) < 0.7]
    
    if not target_concepts:
        print("🎉 All concepts mastered! No learning path needed.")
        return
    
    # Choose a moderately difficult target
    target_concept = random.choice([c for c in target_concepts if c.difficulty >= 3])
    
    print(f"🎯 Target Concept: {target_concept.name} (ID: {target_concept.id})")
    print(f"   Difficulty: {target_concept.difficulty}/5")
    print(f"   Category: {target_concept.category}")
    
    # Generate personalized path
    path = personalized_system.generate_personalized_path(
        learning_graph, 
        target_concept.id, 
        max_concepts=8
    )
    
    print(f"\n📋 Personalized Learning Path ({len(path)} concepts):")
    for i, (concept_id, priority) in enumerate(path, 1):
        concept = algebra_system.knowledge_graph.get_concept(concept_id)
        current_mastery = learning_graph.get_mastery(concept_id)
        print(f"   {i}. {concept.name}")
        print(f"      • Current Mastery: {current_mastery:.2%}")
        print(f"      • Priority Score: {priority:.2f}")
        print(f"      • Difficulty: {concept.difficulty}/5")


def main():
    """Main demonstration function."""
    print("🎓 PERSONALIZED MATH LEARNING SYSTEM DEMONSTRATION")
    print("="*70)
    
    # Initialize the system
    print("🔧 Initializing algebra learning system...")
    algebra_system = AlgebraLearningSystem()
    
    print("🔧 Setting up personalized learning components...")
    personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
    analytics_dashboard = LearningAnalyticsDashboard(algebra_system.knowledge_graph)
    
    # Create a sample user
    print("👤 Creating sample user learning graph...")
    user_graph = LearningGraph(user_id="demo_user_001", name="Demo Student")
    
    # Simulate learning history
    simulate_user_learning_history(user_graph, algebra_system, num_sessions=25)
    
    # Demonstrate all capabilities
    demonstrate_learning_patterns(personalized_system, user_graph)
    demonstrate_weakness_detection(personalized_system, user_graph)
    demonstrate_personalized_recommendations(personalized_system, user_graph)
    demonstrate_learning_insights(personalized_system, user_graph)
    demonstrate_analytics_dashboard(analytics_dashboard, user_graph)
    demonstrate_personalized_learning_path(personalized_system, algebra_system, user_graph)
    
    print("\n" + "="*70)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\n📋 SYSTEM CAPABILITIES DEMONSTRATED:")
    print("   ✅ Learning Graph Creation & Management")
    print("   ✅ Learning Pattern Analysis")
    print("   ✅ Weakness Detection & Classification")
    print("   ✅ Personalized Recommendations")
    print("   ✅ Learning Insights Generation")
    print("   ✅ Comprehensive Analytics Dashboard")
    print("   ✅ Personalized Learning Path Generation")
    print("   ✅ Progress Tracking & Metrics")
    print("   ✅ Category-based Performance Analysis")
    print("   ✅ Retention & Velocity Tracking")
    
    print(f"\n📊 FINAL STATS:")
    print(f"   • Total Concepts in Knowledge Graph: {len(algebra_system.knowledge_graph.get_all_concepts())}")
    print(f"   • Total Exercises Available: {len(algebra_system.exercise_bank.get_all_exercises())}")
    print(f"   • User's Concepts Attempted: {len(user_graph.concept_mastery)}")
    print(f"   • User's Exercise History: {len(user_graph.exercise_history)} exercises")
    
    # Save the demo data
    print(f"\n💾 Saving demo data...")
    user_graph.save_to_file("demo_user_learning_graph.json")
    
    # Generate and save comprehensive report
    report = analytics_dashboard.generate_progress_report(user_graph)
    with open("demo_user_progress_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("   ✅ Saved demo_user_learning_graph.json")
    print("   ✅ Saved demo_user_progress_report.json")


if __name__ == "__main__":
    main() 