#!/usr/bin/env python3
"""
Debug script for the exercise bank.

This script identifies and fixes issues with the exercise bank's ability to find 
exercises for specific concepts.
"""

import os
import sys
from typing import Dict, List

from exercises.exercise import Exercise
from exercises.exercise_bank import ExerciseBank


def debug_exercise_bank():
    """Debug the exercise bank functionality."""
    print("=== Exercise Bank Debug ===")
    
    # Define concept IDs
    concept_ids = {
        "Angles": "angles-id",
        "Triangles": "triangles-id"
    }
    
    # Create test exercises
    bank = ExerciseBank(name="Debug Exercise Bank")
    
    # Create an angles exercise
    angles_ex = Exercise(
        title="Test Angles Exercise",
        problem="Test problem for angles",
        solution="Test solution",
        difficulty=2,
        exercise_id="angles-ex1"
    )
    angles_ex.add_concept_relationship(
        concept_id=concept_ids["Angles"],
        weight=0.9,
        relationship_type="primary"
    )
    
    # Create a triangles exercise
    triangles_ex = Exercise(
        title="Test Triangles Exercise",
        problem="Test problem for triangles",
        solution="Test solution",
        difficulty=2,
        exercise_id="triangles-ex1"
    )
    triangles_ex.add_concept_relationship(
        concept_id=concept_ids["Triangles"],
        weight=0.9,
        relationship_type="primary"
    )
    
    # Add exercises to bank
    print("\n1. Adding exercises to bank")
    bank.add_exercise(angles_ex)
    bank.add_exercise(triangles_ex)
    
    # Debug concept-to-exercises mapping
    print("\n2. Concept-to-exercises mapping after adding exercises:")
    for concept_name, concept_id in concept_ids.items():
        exercises_ids = bank.concept_to_exercises.get(concept_id, [])
        print(f"  {concept_name} ({concept_id}): {exercises_ids}")
    
    # Debug get_exercises_for_concept
    print("\n3. Testing get_exercises_for_concept:")
    for concept_name, concept_id in concept_ids.items():
        exercises = bank.get_exercises_for_concept(concept_id)
        print(f"  {concept_name}: Found {len(exercises)} exercises")
        for ex in exercises:
            print(f"    - {ex.title} (ID: {ex.id})")
            print(f"      Concept relationships: {ex.concept_relationships}")
    
    # Debug select_exercises
    print("\n4. Testing select_exercises:")
    for concept_name, concept_id in concept_ids.items():
        for mastery in [0.0, 0.5, 0.9]:
            print(f"  {concept_name} with mastery {mastery:.1f}:")
            selected = bank.select_exercises(concept_id, mastery, count=1)
            print(f"    Found {len(selected)} exercises")
            for ex in selected:
                print(f"    - {ex.title} (ID: {ex.id})")
    
    # Try manually setting concept_to_exercises
    print("\n5. Manually setting concept_to_exercises:")
    bank.concept_to_exercises = {
        concept_ids["Angles"]: ["angles-ex1"],
        concept_ids["Triangles"]: ["triangles-ex1"]
    }
    
    # Debug get_exercises_for_concept after manual setting
    print("\n6. Testing get_exercises_for_concept after manual update:")
    for concept_name, concept_id in concept_ids.items():
        exercises = bank.get_exercises_for_concept(concept_id)
        print(f"  {concept_name}: Found {len(exercises)} exercises")
        for ex in exercises:
            print(f"    - {ex.title} (ID: {ex.id})")
    
    # Debug issue with get_all_concept_ids
    print("\n7. Testing exercise.get_all_concept_ids():")
    for ex in [angles_ex, triangles_ex]:
        print(f"  {ex.title}:")
        print(f"    Concept relationships: {ex.concept_relationships}")
        print(f"    get_all_concept_ids(): {ex.get_all_concept_ids()}")
    
    # Fix issue with add_exercise method to update concept_to_exercises correctly
    print("\n8. Demonstrating correct behavior of add_exercise:")
    fixed_bank = ExerciseBank(name="Fixed Exercise Bank")
    
    # Add the exercises to the fixed bank
    fixed_bank.add_exercise(angles_ex)
    fixed_bank.add_exercise(triangles_ex)
    
    # Print the concept-to-exercises mapping
    print("  concept_to_exercises mapping:")
    for concept_name, concept_id in concept_ids.items():
        exercise_ids = fixed_bank.concept_to_exercises.get(concept_id, [])
        print(f"    {concept_name} ({concept_id}): {exercise_ids}")


if __name__ == "__main__":
    debug_exercise_bank() 