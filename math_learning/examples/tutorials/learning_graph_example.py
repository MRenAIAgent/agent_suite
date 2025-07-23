#!/usr/bin/env python3
"""
Learning Graph Example - Shows detailed structure of a user's learning graph
"""

import json
import random
from datetime import datetime, timedelta
from pprint import pprint

from .learning_graph.user_model import LearningGraph
from .algebra_learning import AlgebraLearningSystem


def create_realistic_learning_graph():
    """Create a learning graph with realistic user data."""
    
    # Initialize algebra system
    algebra_system = AlgebraLearningSystem()
    
    # Create a user learning graph
    user_id = "student_alice_2024"
    learning_graph = LearningGraph(user_id)
    
    # Get some concepts to work with
    concepts = algebra_system.knowledge_graph.get_all_concepts()
    basic_concepts = [c for c in concepts if c.difficulty <= 2][:10]
    
    print("🎓 Creating realistic learning graph for Alice...")
    print(f"Working with {len(basic_concepts)} basic concepts")
    
    # Simulate learning over several weeks
    base_date = datetime.now() - timedelta(days=30)
    
    for day in range(30):
        current_date = base_date + timedelta(days=day)
        
        # Alice studies 2-4 concepts per day
        daily_concepts = random.sample(basic_concepts, min(random.randint(2, 4), len(basic_concepts)))
        
        for concept in daily_concepts:
            # Simulate 1-3 exercises per concept per day
            num_exercises = random.randint(1, 3)
            
            for ex_num in range(num_exercises):
                exercise_id = f"ex_{concept.id}_{day}_{ex_num}"
                
                # Alice's performance improves over time
                # Early days: 40-60% success rate
                # Later days: 70-90% success rate
                progress_factor = min(1.0, day / 20.0)  # Improves over 20 days
                base_success_rate = 0.4 + (progress_factor * 0.4)  # 40% to 80%
                
                # Add some randomness and concept difficulty
                difficulty_penalty = (concept.difficulty - 1) * 0.1
                success_rate = base_success_rate - difficulty_penalty + random.uniform(-0.2, 0.2)
                success_rate = max(0.1, min(0.95, success_rate))
                
                success = random.random() < success_rate
                
                # Record the exercise with timestamp
                learning_graph.record_exercise_attempt(exercise_id, concept.id, success)
                
                # Manually set timestamp for realistic progression
                if learning_graph.exercise_history[exercise_id]:
                    learning_graph.exercise_history[exercise_id][-1]['timestamp'] = current_date.isoformat()
    
    return learning_graph, algebra_system


def display_learning_graph_structure(learning_graph: LearningGraph):
    """Display the detailed structure of a learning graph."""
    
    print("\n" + "="*80)
    print("📊 LEARNING GRAPH DETAILED STRUCTURE")
    print("="*80)
    
    print(f"👤 User ID: {learning_graph.user_id}")
    print(f"📝 Name: {learning_graph.name}")
    
    # Concept Mastery Overview
    print(f"\n📈 CONCEPT MASTERY ({len(learning_graph.concept_mastery)} concepts)")
    print("-" * 60)
    
    for concept_id, mastery_data in sorted(learning_graph.concept_mastery.items()):
        print(f"\n🎯 Concept: {concept_id}")
        print(f"   Mastery Level: {mastery_data['mastery']:.3f}")
        print(f"   Confidence: {mastery_data['confidence']:.3f}")
        print(f"   Last Practiced: {mastery_data['last_practiced']}")
        
        # Show history if available
        if 'history' in mastery_data and mastery_data['history']:
            print(f"   📊 History ({len(mastery_data['history'])} entries):")
            for i, hist in enumerate(mastery_data['history'][-3:]):  # Show last 3
                print(f"      {i+1}. {hist['timestamp']}: Mastery={hist['mastery']:.3f}, Confidence={hist['confidence']:.3f}")
    
    # Exercise History Overview
    print(f"\n📝 EXERCISE HISTORY ({len(learning_graph.exercise_history)} exercises)")
    print("-" * 60)
    
    # Group exercises by concept
    concept_exercises = {}
    total_attempts = 0
    total_correct = 0
    
    for exercise_id, attempts in learning_graph.exercise_history.items():
        total_attempts += len(attempts)
        for attempt in attempts:
            concept_id = attempt['concept_id']
            if concept_id not in concept_exercises:
                concept_exercises[concept_id] = {'attempts': 0, 'correct': 0, 'exercises': []}
            
            concept_exercises[concept_id]['attempts'] += 1
            concept_exercises[concept_id]['exercises'].append(exercise_id)
            if attempt['result']:
                concept_exercises[concept_id]['correct'] += 1
                total_correct += 1
    
    print(f"📊 Overall Statistics:")
    print(f"   Total Attempts: {total_attempts}")
    print(f"   Total Correct: {total_correct}")
    print(f"   Overall Success Rate: {total_correct/total_attempts:.1%}")
    
    print(f"\n📋 By Concept:")
    for concept_id, stats in sorted(concept_exercises.items()):
        success_rate = stats['correct'] / stats['attempts'] if stats['attempts'] > 0 else 0
        print(f"   {concept_id}:")
        print(f"      Attempts: {stats['attempts']}, Correct: {stats['correct']}")
        print(f"      Success Rate: {success_rate:.1%}")
        print(f"      Unique Exercises: {len(set(stats['exercises']))}")
    
    # Show some actual exercise attempts
    print(f"\n🔍 SAMPLE EXERCISE ATTEMPTS")
    print("-" * 60)
    
    sample_exercises = list(learning_graph.exercise_history.items())[:3]
    for exercise_id, attempts in sample_exercises:
        print(f"\n📝 Exercise: {exercise_id}")
        for i, attempt in enumerate(attempts):
            result_emoji = "✅" if attempt['result'] else "❌"
            print(f"   Attempt {i+1}: {result_emoji} {attempt['result']} "
                  f"(Concept: {attempt['concept_id']}, Time: {attempt['timestamp']})")


def display_learning_graph_json(learning_graph: LearningGraph):
    """Display the learning graph as JSON for technical inspection."""
    
    print(f"\n🔧 LEARNING GRAPH JSON STRUCTURE")
    print("-" * 60)
    
    # Convert to a serializable format
    graph_data = {
        'user_id': learning_graph.user_id,
        'name': learning_graph.name,
        'concept_mastery': {},
        'exercise_history': {},
    }
    
    # Convert concept mastery
    for concept_id, mastery_data in learning_graph.concept_mastery.items():
        graph_data['concept_mastery'][concept_id] = {
            'mastery': mastery_data['mastery'],
            'confidence': mastery_data['confidence'],
            'last_practiced': mastery_data['last_practiced'],
            'history': mastery_data.get('history', [])
        }
    
    # Convert exercise history (sample only)
    sample_exercises = dict(list(learning_graph.exercise_history.items())[:2])
    for exercise_id, attempts in sample_exercises.items():
        graph_data['exercise_history'][exercise_id] = attempts
    
    print("Sample JSON structure (first 2 exercises only):")
    print(json.dumps(graph_data, indent=2))


def analyze_learning_progression(learning_graph: LearningGraph):
    """Analyze learning progression over time."""
    
    print(f"\n📈 LEARNING PROGRESSION ANALYSIS")
    print("-" * 60)
    
    # Collect all exercise attempts with timestamps
    all_attempts = []
    for exercise_id, attempts in learning_graph.exercise_history.items():
        for attempt in attempts:
            # Parse ISO format timestamp
            timestamp_str = attempt['timestamp']
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = timestamp_str
            
            all_attempts.append({
                'timestamp': timestamp,
                'concept_id': attempt['concept_id'],
                'result': attempt['result'],
                'exercise_id': exercise_id
            })
    
    if not all_attempts:
        print("No exercise attempts found.")
        return
    
    # Sort by timestamp
    all_attempts.sort(key=lambda x: x['timestamp'])
    
    # Analyze by week
    first_date = all_attempts[0]['timestamp']
    weekly_stats = {}
    
    for attempt in all_attempts:
        week_num = (attempt['timestamp'] - first_date).days // 7
        if week_num not in weekly_stats:
            weekly_stats[week_num] = {'total': 0, 'correct': 0, 'concepts': set()}
        
        weekly_stats[week_num]['total'] += 1
        weekly_stats[week_num]['concepts'].add(attempt['concept_id'])
        if attempt['result']:
            weekly_stats[week_num]['correct'] += 1
    
    print(f"📅 Weekly Progress:")
    for week, stats in sorted(weekly_stats.items()):
        success_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   Week {week + 1}: {stats['total']} attempts, "
              f"{success_rate:.1%} success rate, "
              f"{len(stats['concepts'])} concepts practiced")
    
    # Show concept mastery progression
    print(f"\n🎯 Current Concept Mastery:")
    if learning_graph.concept_mastery:
        for concept_id, mastery_data in sorted(learning_graph.concept_mastery.items()):
            print(f"   {concept_id}: {mastery_data['mastery']:.3f} mastery, "
                  f"{mastery_data['confidence']:.3f} confidence")
    else:
        print("   No concept mastery data available.")


def main():
    """Main function to demonstrate learning graph structure."""
    
    print("🚀 LEARNING GRAPH STRUCTURE DEMONSTRATION")
    print("="*80)
    
    # Create a realistic learning graph
    learning_graph, algebra_system = create_realistic_learning_graph()
    
    # Display various aspects of the learning graph
    display_learning_graph_structure(learning_graph)
    analyze_learning_progression(learning_graph)
    display_learning_graph_json(learning_graph)
    
    print("\n" + "="*80)
    print("✨ This shows how the learning graph captures:")
    print("   📊 Mastery levels and confidence for each concept")
    print("   📝 Detailed exercise history with timestamps")
    print("   📈 Learning progression over time")
    print("   🎯 Individual learning patterns and preferences")
    print("   🔄 Historical mastery changes")
    print("\nThis rich data enables personalized recommendations! 🎓")


if __name__ == "__main__":
    main() 