#!/usr/bin/env python3
"""
Detailed Learning Graph Example - Shows a complete user learning profile
"""

import json
import random
from datetime import datetime, timedelta
from pprint import pprint

from .learning_graph.user_model import LearningGraph
from .learning_graph.personalized_learning import PersonalizedLearningSystem
from .algebra_learning import AlgebraLearningSystem


def create_comprehensive_learning_example():
    """Create a comprehensive learning graph example with realistic data."""
    
    print("🚀 COMPREHENSIVE LEARNING GRAPH EXAMPLE")
    print("="*80)
    
    # Initialize systems
    algebra_system = AlgebraLearningSystem()
    learning_graph = LearningGraph("student_sarah_2024")
    personalized_system = PersonalizedLearningSystem(algebra_system.knowledge_graph)
    
    # Get some concepts to work with
    concepts = algebra_system.knowledge_graph.get_all_concepts()
    selected_concepts = [c for c in concepts if c.difficulty <= 4][:8]
    
    print(f"👩‍🎓 Creating learning profile for Sarah...")
    print(f"📚 Working with {len(selected_concepts)} concepts")
    
    # Simulate 6 weeks of learning
    start_date = datetime.now() - timedelta(days=42)
    
    for week in range(6):
        print(f"   Week {week + 1}: ", end="")
        week_start = start_date + timedelta(days=week * 7)
        
        # Each week, practice 3-5 concepts with varying intensity
        week_concepts = random.sample(selected_concepts, min(5, len(selected_concepts)))
        exercises_this_week = 0
        
        for concept in week_concepts:
            # Number of exercises varies by week and concept difficulty
            base_exercises = max(2, 8 - concept.difficulty)
            num_exercises = random.randint(base_exercises, base_exercises + 3)
            
            for ex_num in range(num_exercises):
                # Exercise timestamp within the week
                exercise_date = week_start + timedelta(
                    days=random.randint(0, 6),
                    hours=random.randint(9, 17),
                    minutes=random.randint(0, 59)
                )
                
                # Success probability improves over time and varies by concept
                base_success = 0.3 + (week * 0.1)  # Improvement over time
                concept_modifier = (5 - concept.difficulty) * 0.1  # Easier concepts = higher success
                current_mastery = learning_graph.get_mastery(concept.id)
                mastery_boost = current_mastery * 0.3  # Previous mastery helps
                
                success_prob = min(0.95, base_success + concept_modifier + mastery_boost)
                success = random.random() < success_prob
                
                # Record the exercise
                exercise_id = f"ex_{concept.id}_w{week}_{ex_num}"
                learning_graph.record_exercise_attempt(exercise_id, concept.id, success)
                
                # Set realistic timestamp
                if learning_graph.exercise_history[exercise_id]:
                    learning_graph.exercise_history[exercise_id][-1]['timestamp'] = exercise_date.isoformat()
                
                exercises_this_week += 1
        
        print(f"{exercises_this_week} exercises")
    
    return learning_graph, personalized_system, algebra_system


def display_comprehensive_learning_profile(learning_graph, personalized_system, algebra_system):
    """Display a comprehensive learning profile."""
    
    print(f"\n📊 COMPREHENSIVE LEARNING PROFILE")
    print("="*80)
    print(f"👤 Student: {learning_graph.user_id}")
    
    # Overall Statistics
    total_exercises = sum(len(attempts) for attempts in learning_graph.exercise_history.values())
    total_correct = sum(
        sum(1 for attempt in attempts if attempt['result'])
        for attempts in learning_graph.exercise_history.values()
    )
    overall_success_rate = total_correct / total_exercises if total_exercises > 0 else 0
    
    print(f"\n📈 OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Total Exercises: {total_exercises}")
    print(f"Total Correct: {total_correct}")
    print(f"Success Rate: {overall_success_rate:.1%}")
    print(f"Concepts Practiced: {len(set(attempt['concept_id'] for attempts in learning_graph.exercise_history.values() for attempt in attempts))}")
    
    # Concept Performance Analysis
    print(f"\n🎯 CONCEPT PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    concept_stats = {}
    for exercise_id, attempts in learning_graph.exercise_history.items():
        for attempt in attempts:
            concept_id = attempt['concept_id']
            if concept_id not in concept_stats:
                concept_stats[concept_id] = {
                    'attempts': 0, 'correct': 0, 'first_attempt': None, 'last_attempt': None
                }
            
            concept_stats[concept_id]['attempts'] += 1
            if attempt['result']:
                concept_stats[concept_id]['correct'] += 1
            
            attempt_time = datetime.fromisoformat(attempt['timestamp'])
            if (concept_stats[concept_id]['first_attempt'] is None or 
                attempt_time < concept_stats[concept_id]['first_attempt']):
                concept_stats[concept_id]['first_attempt'] = attempt_time
            
            if (concept_stats[concept_id]['last_attempt'] is None or 
                attempt_time > concept_stats[concept_id]['last_attempt']):
                concept_stats[concept_id]['last_attempt'] = attempt_time
    
    # Sort by performance
    sorted_concepts = sorted(concept_stats.items(), 
                           key=lambda x: x[1]['correct'] / x[1]['attempts'], 
                           reverse=True)
    
    for concept_id, stats in sorted_concepts:
        success_rate = stats['correct'] / stats['attempts']
        current_mastery = learning_graph.get_mastery(concept_id)
        concept = algebra_system.knowledge_graph.get_concept(concept_id)
        
        # Performance indicator
        if success_rate >= 0.8:
            indicator = "🟢 Strong"
        elif success_rate >= 0.6:
            indicator = "🟡 Developing"
        else:
            indicator = "🔴 Struggling"
        
        print(f"{indicator} {concept.name} ({concept_id})")
        print(f"   📊 {stats['attempts']} attempts, {success_rate:.1%} success")
        print(f"   🎯 Current mastery: {current_mastery:.1%}")
        print(f"   📅 Practice span: {(stats['last_attempt'] - stats['first_attempt']).days} days")
        print()
    
    # Learning Patterns Analysis
    print(f"\n🧠 LEARNING PATTERNS & INSIGHTS")
    print("-" * 40)
    
    # Analyze learning patterns
    patterns = personalized_system.analyze_learning_patterns(learning_graph)
    weaknesses = personalized_system.detect_weaknesses(learning_graph)
    recommendations = personalized_system.get_adaptive_recommendations(learning_graph)
    
    # Calculate overall metrics from individual patterns
    if patterns:
        avg_learning_velocity = sum(p.learning_velocity for p in patterns.values()) / len(patterns)
        avg_retention_rate = sum(p.retention_rate for p in patterns.values()) / len(patterns)
        
        print(f"📊 Average Learning Velocity: {avg_learning_velocity:.3f}")
        print(f"🎯 Average Retention Rate: {avg_retention_rate:.1%}")
    else:
        print(f"📊 No learning patterns available yet")
    
    print(f"⚠️  Detected Weaknesses: {len(weaknesses)}")
    
    if weaknesses:
        print(f"\n🔍 TOP WEAKNESSES:")
        for weakness in weaknesses[:3]:
            concept = algebra_system.knowledge_graph.get_concept(weakness.concept_id)
            print(f"   • {concept.name}: {weakness.weakness_type} (severity: {weakness.severity:.2f})")
            if weakness.suggested_interventions:
                print(f"     💡 Suggestion: {weakness.suggested_interventions[0]}")
    
    print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:4], 1):
        concept = algebra_system.knowledge_graph.get_concept(rec['concept_id'])
        print(f"   {i}. {rec['title']}")
        print(f"      📝 {rec['description']}")
        print(f"      🎯 Priority: {rec['priority']:.2f}")
        if 'reason' in rec:
            print(f"      💭 Reason: {rec['reason']}")
        print()
    
    # Weekly Progress
    print(f"\n📅 WEEKLY LEARNING PROGRESSION")
    print("-" * 40)
    
    # Group exercises by week
    weekly_data = {}
    for exercise_id, attempts in learning_graph.exercise_history.items():
        for attempt in attempts:
            attempt_date = datetime.fromisoformat(attempt['timestamp'])
            week_start = attempt_date - timedelta(days=attempt_date.weekday())
            week_key = week_start.strftime("%Y-%m-%d")
            
            if week_key not in weekly_data:
                weekly_data[week_key] = {'total': 0, 'correct': 0, 'concepts': set()}
            
            weekly_data[week_key]['total'] += 1
            weekly_data[week_key]['concepts'].add(attempt['concept_id'])
            if attempt['result']:
                weekly_data[week_key]['correct'] += 1
    
    for week_start, data in sorted(weekly_data.items()):
        success_rate = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"Week of {week_start}: {data['total']} exercises, "
              f"{success_rate:.1%} success, {len(data['concepts'])} concepts")


def main():
    """Main demonstration function."""
    
    # Create comprehensive example
    learning_graph, personalized_system, algebra_system = create_comprehensive_learning_example()
    
    # Display the comprehensive profile
    display_comprehensive_learning_profile(learning_graph, personalized_system, algebra_system)
    
    print(f"\n✨ SUMMARY")
    print("="*80)
    print("This learning graph demonstrates:")
    print("📊 Detailed exercise history with timestamps")
    print("🎯 Concept mastery tracking over time") 
    print("🧠 Learning pattern analysis")
    print("⚠️  Weakness detection and classification")
    print("💡 Personalized recommendations")
    print("📈 Progress tracking and analytics")
    print("\nThis rich data enables truly personalized learning experiences! 🎓")


if __name__ == "__main__":
    main() 