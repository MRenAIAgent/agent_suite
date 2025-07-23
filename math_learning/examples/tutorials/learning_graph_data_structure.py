#!/usr/bin/env python3
"""
Learning Graph Data Structure Example - Shows the actual internal data structure
"""

import json
import random
from datetime import datetime, timedelta
from pprint import pprint

from .learning_graph.user_model import LearningGraph
from .algebra_learning import AlgebraLearningSystem


def create_sample_learning_graph():
    """Create a sample learning graph with realistic data."""
    
    print("🔍 LEARNING GRAPH DATA STRUCTURE EXAMPLE")
    print("="*80)
    
    # Initialize systems
    algebra_system = AlgebraLearningSystem()
    learning_graph = LearningGraph("demo_student_2024")
    
    # Get a few concepts to work with
    concepts = algebra_system.knowledge_graph.get_all_concepts()
    sample_concepts = concepts[:3]  # Just use first 3 concepts
    
    print(f"📚 Working with concepts: {[c.name for c in sample_concepts]}")
    
    # Simulate some learning activity
    start_date = datetime.now() - timedelta(days=14)
    
    for i, concept in enumerate(sample_concepts):
        print(f"\n📝 Simulating exercises for {concept.name}...")
        
        # Create several exercises for this concept
        for ex_num in range(5):
            exercise_date = start_date + timedelta(days=i*2 + ex_num)
            success = random.random() < (0.5 + i * 0.2)  # Improving success rate
            
            exercise_id = f"exercise_{concept.id}_{ex_num}"
            learning_graph.record_exercise_attempt(exercise_id, concept.id, success)
            
            # Set realistic timestamp
            if learning_graph.exercise_history[exercise_id]:
                learning_graph.exercise_history[exercise_id][-1]['timestamp'] = exercise_date.isoformat()
            
            print(f"   Exercise {ex_num + 1}: {'✅' if success else '❌'}")
    
    return learning_graph, algebra_system


def display_raw_data_structure(learning_graph, algebra_system):
    """Display the raw data structure of the learning graph."""
    
    print(f"\n🔧 RAW LEARNING GRAPH DATA STRUCTURE")
    print("="*80)
    
    print(f"👤 User ID: {learning_graph.user_id}")
    print(f"📝 Name: {learning_graph.name}")
    
    # Show concept mastery structure
    print(f"\n📊 CONCEPT MASTERY DATA:")
    print("-" * 40)
    if learning_graph.concept_mastery:
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            concept = algebra_system.knowledge_graph.get_concept(concept_id)
            print(f"\n🎯 {concept.name} ({concept_id}):")
            print(f"   Raw data: {mastery_data}")
    else:
        print("   No concept mastery data recorded")
    
    # Show exercise history structure
    print(f"\n📝 EXERCISE HISTORY DATA:")
    print("-" * 40)
    for exercise_id, attempts in list(learning_graph.exercise_history.items())[:3]:
        print(f"\n📋 {exercise_id}:")
        for i, attempt in enumerate(attempts):
            print(f"   Attempt {i+1}: {attempt}")
    
    if len(learning_graph.exercise_history) > 3:
        print(f"\n   ... and {len(learning_graph.exercise_history) - 3} more exercises")
    
    # Show methods available
    print(f"\n🛠️  AVAILABLE METHODS:")
    print("-" * 40)
    methods = [method for method in dir(learning_graph) if not method.startswith('_')]
    for method in methods:
        print(f"   • {method}")
    
    # Test some methods
    print(f"\n🧪 METHOD TESTING:")
    print("-" * 40)
    
    if learning_graph.exercise_history:
        # Get a concept that has exercises
        sample_concept_id = None
        for exercise_id, attempts in learning_graph.exercise_history.items():
            if attempts:
                sample_concept_id = attempts[0]['concept_id']
                break
        
        if sample_concept_id:
            concept = algebra_system.knowledge_graph.get_concept(sample_concept_id)
            print(f"Testing methods with concept: {concept.name} ({sample_concept_id})")
            
            mastery = learning_graph.get_mastery(sample_concept_id)
            confidence = learning_graph.get_confidence(sample_concept_id)
            
            print(f"   get_mastery('{sample_concept_id}'): {mastery}")
            print(f"   get_confidence('{sample_concept_id}'): {confidence}")
            
            # Show what these methods actually return
            print(f"\n   Mastery type: {type(mastery)}")
            print(f"   Confidence type: {type(confidence)}")


def display_json_structure(learning_graph):
    """Display the learning graph as JSON."""
    
    print(f"\n📄 JSON REPRESENTATION:")
    print("-" * 40)
    
    # Create a JSON-serializable representation
    json_data = {
        'user_id': learning_graph.user_id,
        'name': learning_graph.name,
        'concept_mastery': dict(learning_graph.concept_mastery),
        'exercise_history': {}
    }
    
    # Convert exercise history (limit to first 2 exercises for readability)
    sample_exercises = dict(list(learning_graph.exercise_history.items())[:2])
    for exercise_id, attempts in sample_exercises.items():
        json_data['exercise_history'][exercise_id] = attempts
    
    print(json.dumps(json_data, indent=2))


def analyze_data_patterns(learning_graph):
    """Analyze patterns in the learning graph data."""
    
    print(f"\n📈 DATA PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Exercise statistics
    total_exercises = len(learning_graph.exercise_history)
    total_attempts = sum(len(attempts) for attempts in learning_graph.exercise_history.values())
    
    print(f"📊 Exercise Statistics:")
    print(f"   Unique exercises: {total_exercises}")
    print(f"   Total attempts: {total_attempts}")
    
    # Concept coverage
    concepts_practiced = set()
    for attempts in learning_graph.exercise_history.values():
        for attempt in attempts:
            concepts_practiced.add(attempt['concept_id'])
    
    print(f"   Concepts practiced: {len(concepts_practiced)}")
    print(f"   Concept IDs: {list(concepts_practiced)}")
    
    # Success rates by concept
    concept_stats = {}
    for attempts in learning_graph.exercise_history.values():
        for attempt in attempts:
            concept_id = attempt['concept_id']
            if concept_id not in concept_stats:
                concept_stats[concept_id] = {'total': 0, 'correct': 0}
            
            concept_stats[concept_id]['total'] += 1
            if attempt['result']:
                concept_stats[concept_id]['correct'] += 1
    
    print(f"\n📋 Success Rates by Concept:")
    for concept_id, stats in concept_stats.items():
        success_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {concept_id}: {stats['correct']}/{stats['total']} = {success_rate:.1%}")
    
    # Timeline analysis
    all_timestamps = []
    for attempts in learning_graph.exercise_history.values():
        for attempt in attempts:
            all_timestamps.append(datetime.fromisoformat(attempt['timestamp']))
    
    if all_timestamps:
        all_timestamps.sort()
        print(f"\n📅 Timeline:")
        print(f"   First exercise: {all_timestamps[0].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Last exercise: {all_timestamps[-1].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Learning span: {(all_timestamps[-1] - all_timestamps[0]).days} days")


def main():
    """Main demonstration function."""
    
    # Create sample learning graph
    learning_graph, algebra_system = create_sample_learning_graph()
    
    # Display the raw data structure
    display_raw_data_structure(learning_graph, algebra_system)
    
    # Display JSON representation
    display_json_structure(learning_graph)
    
    # Analyze data patterns
    analyze_data_patterns(learning_graph)
    
    print(f"\n✨ SUMMARY")
    print("="*80)
    print("This example shows the actual internal data structure of a learning graph:")
    print("📊 How concept mastery is stored (if at all)")
    print("📝 How exercise history is structured")
    print("🔧 What methods are available")
    print("📄 JSON representation of the data")
    print("📈 How to analyze patterns in the data")
    print("\nThis is the foundation for all personalized learning features! 🎓")


if __name__ == "__main__":
    main() 