#!/usr/bin/env python3
"""
Comprehensive Learning Graph Demo - Shows detailed user understanding over time
"""

import random
import json
from datetime import datetime, timedelta
from pprint import pprint

from .learning_graph.user_model import LearningGraph
from .learning_graph.personalized_learning import PersonalizedLearningSystem
from .algebra_learning import AlgebraLearningSystem


def create_realistic_user_journey():
    """Create a realistic user learning journey over time."""
    
    print("🚀 COMPREHENSIVE LEARNING GRAPH DEMONSTRATION")
    print("=" * 80)
    
    # Initialize systems
    algebra_system = AlgebraLearningSystem()
    learning_graph = LearningGraph("student_alex_2024")
    personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
    
    # Get concepts for simulation
    concepts = algebra_system.knowledge_graph.get_all_concepts()
    basic_concepts = [c for c in concepts if c.difficulty <= 2][:8]  # Focus on basic concepts
    
    print(f"👤 Student: {learning_graph.user_id}")
    print(f"📚 Working with {len(basic_concepts)} basic concepts")
    print()
    
    # Simulate learning over multiple sessions
    sessions = [
        {"name": "Week 1 - Introduction", "exercises": 20, "success_bias": 0.3},
        {"name": "Week 2 - Building Foundation", "exercises": 25, "success_bias": 0.5},
        {"name": "Week 3 - Gaining Confidence", "exercises": 30, "success_bias": 0.7},
        {"name": "Week 4 - Mastery Development", "exercises": 35, "success_bias": 0.8},
    ]
    
    for session_num, session in enumerate(sessions, 1):
        print(f"📅 {session['name']}")
        print("-" * 50)
        
        session_start = datetime.now() - timedelta(weeks=4-session_num)
        
        for exercise_num in range(session['exercises']):
            # Select concept (bias toward earlier concepts in later sessions)
            if session_num <= 2:
                concept = random.choice(basic_concepts[:4])  # Focus on first 4 concepts
            else:
                concept = random.choice(basic_concepts)  # All concepts available
            
            # Calculate success probability based on current mastery and session bias
            current_mastery = learning_graph.get_mastery(concept.id)
            base_probability = session['success_bias']
            mastery_boost = current_mastery * 0.3  # Mastery helps success
            success_probability = min(0.95, base_probability + mastery_boost)
            
            # Determine success
            success = random.random() < success_probability
            
            # Create exercise with realistic difficulty progression
            difficulty = min(0.9, 0.3 + (session_num - 1) * 0.15 + random.uniform(-0.1, 0.1))
            concept_weight = 1.0
            
            # Record the exercise
            exercise_id = f"session_{session_num}_ex_{exercise_num}_{concept.id}"
            learning_graph.record_exercise_attempt(exercise_id, concept.id, success, difficulty, concept_weight)
            
            # Show progress every 10 exercises
            if exercise_num % 10 == 9:
                print(f"  Exercise {exercise_num + 1}: {concept.name} - {'✅' if success else '❌'} "
                      f"(Mastery: {learning_graph.get_mastery(concept.id):.2f})")
        
        print(f"  Session complete: {session['exercises']} exercises")
        print()
    
    return learning_graph, personalized_system


def analyze_user_understanding(learning_graph, personalized_system):
    """Analyze and display comprehensive user understanding."""
    
    print("🧠 COMPREHENSIVE USER UNDERSTANDING ANALYSIS")
    print("=" * 80)
    
    # 1. Overall Learning Statistics
    print("📊 OVERALL LEARNING STATISTICS")
    print("-" * 40)
    
    total_exercises = sum(len(attempts) for attempts in learning_graph.exercise_history.values())
    total_concepts = len(learning_graph.concept_mastery)
    
    # Calculate overall success rate
    total_successes = 0
    for attempts in learning_graph.exercise_history.values():
        total_successes += sum(1 for attempt in attempts if attempt['result'])
    
    overall_success_rate = total_successes / total_exercises if total_exercises > 0 else 0
    
    # Calculate average mastery
    avg_mastery = sum(learning_graph.get_mastery(cid) for cid in learning_graph.concept_mastery.keys()) / total_concepts if total_concepts > 0 else 0
    
    print(f"📈 Total Exercises Completed: {total_exercises}")
    print(f"🎯 Concepts Encountered: {total_concepts}")
    print(f"✅ Overall Success Rate: {overall_success_rate:.1%}")
    print(f"🏆 Average Mastery Level: {avg_mastery:.1%}")
    print()
    
    # 2. Concept Mastery Breakdown
    print("🎯 CONCEPT MASTERY BREAKDOWN")
    print("-" * 40)
    
    concept_data = []
    for concept_id, mastery_data in learning_graph.concept_mastery.items():
        mastery = mastery_data['mastery']
        confidence = mastery_data['confidence']
        
        # Count exercises for this concept
        exercise_count = sum(
            len([a for a in attempts if a['concept_id'] == concept_id])
            for attempts in learning_graph.exercise_history.values()
        )
        
        concept_data.append({
            'id': concept_id,
            'mastery': mastery,
            'confidence': confidence,
            'exercises': exercise_count
        })
    
    # Sort by mastery level
    concept_data.sort(key=lambda x: x['mastery'], reverse=True)
    
    for i, data in enumerate(concept_data[:10], 1):  # Show top 10
        mastery_bar = "█" * int(data['mastery'] * 20) + "░" * (20 - int(data['mastery'] * 20))
        print(f"{i:2d}. {data['id']:<15} {mastery_bar} {data['mastery']:.1%} "
              f"(Confidence: {data['confidence']:.1%}, Exercises: {data['exercises']})")
    
    print()
    
    # 3. Learning Patterns Analysis
    print("🔍 LEARNING PATTERNS ANALYSIS")
    print("-" * 40)
    
    patterns = personalized_system.analyze_learning_patterns(learning_graph)
    
    if patterns:
        # Calculate overall learning metrics
        avg_learning_velocity = sum(p.learning_velocity for p in patterns.values()) / len(patterns)
        avg_retention_rate = sum(p.retention_rate for p in patterns.values()) / len(patterns)
        avg_success_rate = sum(p.success_rate for p in patterns.values()) / len(patterns)
        
        print(f"🚀 Average Learning Velocity: {avg_learning_velocity:.2f}/1.0")
        print(f"🧠 Average Retention Rate: {avg_retention_rate:.2f}/1.0")
        print(f"🎯 Average Success Rate: {avg_success_rate:.1%}")
        print()
        
        # Show detailed patterns for top concepts
        print("📈 DETAILED LEARNING PATTERNS (Top 5 concepts):")
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1].attempts, reverse=True)
        
        for concept_id, pattern in sorted_patterns[:5]:
            print(f"\n🎯 {concept_id}:")
            print(f"   📊 Attempts: {pattern.attempts}")
            print(f"   ✅ Success Rate: {pattern.success_rate:.1%}")
            print(f"   🚀 Learning Velocity: {pattern.learning_velocity:.2f}")
            print(f"   🧠 Retention Rate: {pattern.retention_rate:.2f}")
            if pattern.mistake_patterns:
                print(f"   ⚠️  Mistake Patterns: {', '.join(pattern.mistake_patterns)}")
    else:
        print("No learning patterns available yet.")
    
    print()
    
    # 4. Weakness Analysis
    print("⚠️  WEAKNESS ANALYSIS")
    print("-" * 40)
    
    weaknesses = personalized_system.detect_weaknesses(learning_graph)
    
    if weaknesses:
        print(f"Detected {len(weaknesses)} areas needing attention:")
        
        for i, weakness in enumerate(weaknesses[:5], 1):  # Show top 5 weaknesses
            print(f"\n{i}. 🎯 {weakness.concept_id}")
            print(f"   Type: {weakness.weakness_type.title()}")
            print(f"   Severity: {weakness.severity:.2f}/1.0")
            print(f"   Confidence: {weakness.confidence_level:.1%}")
            if weakness.related_concepts:
                print(f"   Related Issues: {', '.join(weakness.related_concepts[:3])}")
            print(f"   Interventions: {weakness.suggested_interventions[0]}")
    else:
        print("No significant weaknesses detected! 🎉")
    
    print()
    
    # 5. Learning Trajectory Over Time
    print("📈 LEARNING TRAJECTORY OVER TIME")
    print("-" * 40)
    
    # Analyze learning progression
    weekly_progress = {}
    for exercise_id, attempts in learning_graph.exercise_history.items():
        for attempt in attempts:
            # Parse timestamp and group by week
            timestamp = datetime.fromisoformat(attempt['timestamp'])
            week_key = timestamp.strftime("%Y-W%U")
            
            if week_key not in weekly_progress:
                weekly_progress[week_key] = {'total': 0, 'success': 0}
            
            weekly_progress[week_key]['total'] += 1
            if attempt['result']:
                weekly_progress[week_key]['success'] += 1
    
    print("Weekly Performance Trend:")
    for week, data in sorted(weekly_progress.items()):
        success_rate = data['success'] / data['total'] if data['total'] > 0 else 0
        bar = "█" * int(success_rate * 20) + "░" * (20 - int(success_rate * 20))
        print(f"  {week}: {bar} {success_rate:.1%} ({data['success']}/{data['total']})")
    
    print()
    
    # 6. Personalized Recommendations
    print("💡 PERSONALIZED RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = personalized_system.get_adaptive_recommendations(learning_graph)
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. 📚 {rec['type'].replace('_', ' ').title()}: {rec['concept_id']}")
            print(f"   Priority: {rec['priority']:.2f}")
            print(f"   Reason: {rec['reason']}")
            print()
    else:
        print("No specific recommendations at this time.")


def simulate_learning_session(learning_graph, personalized_system, concept_id, session_name, 
                             num_exercises=5, base_success_rate=0.6, improvement_rate=0.1):
    """Simulate a focused learning session on a specific concept."""
    print(f"\n📚 {session_name}")
    print("-" * 40)
    
    for i in range(num_exercises):
        # Simulate improvement over the session
        success_rate = min(0.95, base_success_rate + (i * improvement_rate))
        success = random.random() < success_rate
        
        exercise_id = f"{concept_id}_session_{session_name.lower().replace(' ', '_')}_{i+1}"
        difficulty = random.uniform(0.4, 0.8)
        
        learning_graph.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)
        
        status = "✅ Correct" if success else "❌ Incorrect"
        print(f"  Exercise {i+1}: {status} (difficulty: {difficulty:.2f})")
    
    # Show current mastery
    mastery = learning_graph.get_concept_mastery(concept_id)
    print(f"  📊 Current mastery: {mastery:.1%}")


def show_user_understanding(learning_graph, personalized_system):
    """Show comprehensive user understanding and insights."""
    print("\n🧠 USER UNDERSTANDING ANALYSIS")
    print("=" * 80)
    
    # 1. Overall Learning Profile
    print("\n1️⃣ LEARNING PROFILE")
    print("-" * 30)
    
    total_exercises = sum(len(attempts) for attempts in learning_graph.exercise_history.values())
    total_concepts = len(learning_graph.concept_mastery)
    avg_mastery = sum(learning_graph.concept_mastery.values()) / len(learning_graph.concept_mastery) if learning_graph.concept_mastery else 0
    
    print(f"📈 Total exercises completed: {total_exercises}")
    print(f"🎯 Concepts encountered: {total_concepts}")
    print(f"🏆 Average mastery level: {avg_mastery:.1%}")
    
    # 2. Learning Patterns
    print("\n2️⃣ LEARNING PATTERNS")
    print("-" * 30)
    
    patterns = personalized_system.analyze_learning_patterns(learning_graph)
    for concept_id, pattern in patterns.items():
        print(f"\n🔍 {concept_id}:")
        print(f"  • Attempts: {pattern['total_attempts']}")
        print(f"  • Success rate: {pattern['success_rate']:.1%}")
        print(f"  • Learning velocity: {pattern['learning_velocity']:.3f}")
        print(f"  • Retention rate: {pattern['retention_rate']:.1%}")
        if pattern['mistake_patterns']:
            print(f"  • Common mistakes: {', '.join(pattern['mistake_patterns'])}")
    
    # 3. Weakness Analysis
    print("\n3️⃣ WEAKNESS ANALYSIS")
    print("-" * 30)
    
    weaknesses = personalized_system.detect_weaknesses(learning_graph)
    if weaknesses:
        for weakness in weaknesses:
            severity_emoji = "🔴" if weakness['severity'] == 'high' else "🟡" if weakness['severity'] == 'medium' else "🟢"
            print(f"{severity_emoji} {weakness['concept_id']} ({weakness['type']} - {weakness['severity']})")
            print(f"   Reason: {weakness['reason']}")
    else:
        print("🎉 No significant weaknesses detected!")
    
    # 4. Learning Insights
    print("\n4️⃣ LEARNING INSIGHTS")
    print("-" * 30)
    
    insights = personalized_system.generate_learning_insights(learning_graph)
    for insight in insights:
        insight_emoji = "💡" if insight['type'] == 'strength' else "⚠️" if insight['type'] == 'weakness' else "📈"
        print(f"{insight_emoji} {insight['insight']}")
        if insight.get('recommendation'):
            print(f"   💭 Recommendation: {insight['recommendation']}")
    
    # 5. Adaptive Recommendations
    print("\n5️⃣ ADAPTIVE RECOMMENDATIONS")
    print("-" * 30)
    
    recommendations = personalized_system.get_adaptive_recommendations(learning_graph)
    
    if recommendations.get('weakness_remediation'):
        print("\n🔧 Weakness Remediation:")
        for rec in recommendations['weakness_remediation']:
            print(f"  • {rec['concept_id']}: {rec['reason']}")
    
    if recommendations.get('ready_to_learn'):
        print("\n🚀 Ready to Learn:")
        for rec in recommendations['ready_to_learn']:
            print(f"  • {rec['concept_id']}: {rec['reason']}")
    
    if recommendations.get('review_concepts'):
        print("\n🔄 Review Recommendations:")
        for rec in recommendations['review_concepts']:
            print(f"  • {rec['concept_id']}: {rec['reason']}")


def show_mastery_progression(learning_graph):
    """Show how mastery has progressed over time."""
    print("\n📊 MASTERY PROGRESSION")
    print("=" * 80)
    
    print("\nConcept Mastery Levels:")
    for concept_id, mastery in sorted(learning_graph.concept_mastery.items()):
        mastery_bar = "█" * int(mastery * 20) + "░" * (20 - int(mastery * 20))
        print(f"{concept_id:25} [{mastery_bar}] {mastery:.1%}")


def main():
    """Run the comprehensive learning graph demonstration."""
    
    # Create realistic user journey
    learning_graph, personalized_system = create_realistic_user_journey()
    
    # Analyze user understanding
    analyze_user_understanding(learning_graph, personalized_system)
    
    print("🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 80)
    print("This demonstration shows how the learning graph captures:")
    print("✅ Exercise history and performance trends")
    print("✅ Concept mastery progression over time")
    print("✅ Learning patterns and velocities")
    print("✅ Weakness detection and intervention suggestions")
    print("✅ Personalized recommendations based on user data")
    print("✅ Long-term learning trajectory analysis")
    print()
    print("The learning graph provides a comprehensive foundation for")
    print("understanding users and personalizing their learning experience! 🎓")


if __name__ == "__main__":
    main() 