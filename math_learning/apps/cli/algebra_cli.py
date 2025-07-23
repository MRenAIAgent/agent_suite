"""
Algebra Learning System CLI

This module provides a command-line interface for interacting with the 
K-12 algebra learning system.
"""

import os
import argparse
import json
from typing import List, Dict, Any

from algebra_learning import AlgebraLearningSystem
from knowledge_graph.concept import Concept
from exercises.exercise import Exercise


def display_concept(concept: Concept) -> None:
    """Display information about a concept."""
    print(f"\n{'=' * 80}")
    print(f"CONCEPT: {concept.name}")
    print(f"{'=' * 80}")
    print(f"Category: {concept.category}")
    print(f"Difficulty: {concept.difficulty}/5")
    print(f"Time to Master: {concept.time_to_master} minutes")
    print(f"\nDescription:")
    print(f"{concept.description}")
    
    if concept.examples:
        print("\nExamples:")
        for example in concept.examples:
            print(f"- {example}")
            
    print(f"\nID: {concept.id}")
    print(f"{'=' * 80}\n")


def display_exercise(exercise: Exercise) -> None:
    """Display an exercise."""
    print(f"\n{'-' * 80}")
    print(f"EXERCISE: {exercise.title}")
    print(f"{'-' * 80}")
    print(f"Difficulty: {exercise.difficulty}/5 | Time: {exercise.estimated_time} min | Type: {exercise.format_type}")
    print(f"\nProblem:")
    print(f"{exercise.problem}")
    
    # Input for solution
    input("\nPress Enter to view solution...")
    
    print(f"\nSolution:")
    print(f"{exercise.solution}")
    
    if exercise.hints:
        print("\nHints:")
        for i, hint in enumerate(exercise.hints, 1):
            print(f"{i}. {hint}")
            
    print(f"{'-' * 80}\n")


def list_concepts(system: AlgebraLearningSystem, filter_category: str = None, 
                 filter_difficulty: int = None) -> None:
    """List all concepts, optionally filtered by category or difficulty."""
    concepts = system.knowledge_graph.get_all_concepts()
    
    # Apply filters
    if filter_category:
        concepts = [c for c in concepts if filter_category.lower() in c.category.lower()]
    if filter_difficulty:
        concepts = [c for c in concepts if c.difficulty == filter_difficulty]
    
    # Sort by difficulty, then by name
    concepts.sort(key=lambda c: (c.difficulty, c.name))
    
    # Group by difficulty level
    difficulty_groups = {1: [], 2: [], 3: [], 4: [], 5: []}
    for concept in concepts:
        difficulty_groups[concept.difficulty].append(concept)
    
    # Display grouped by difficulty
    for difficulty in range(1, 6):
        group_concepts = difficulty_groups[difficulty]
        if not group_concepts:
            continue
            
        print(f"\nDifficulty Level {difficulty}:")
        print(f"{'-' * 50}")
        
        for concept in group_concepts:
            print(f"- {concept.name} ({concept.category})")
    
    print(f"\nTotal: {len(concepts)} concepts")


def explore_concept(system: AlgebraLearningSystem, concept_name: str) -> None:
    """Explore a concept in detail."""
    # Find the concept
    concept = system.get_concept_by_name(concept_name)
    if not concept:
        print(f"Concept '{concept_name}' not found.")
        return
    
    # Display concept details
    display_concept(concept)
    
    # Get prerequisites and dependents
    prerequisites = system.knowledge_graph.get_prerequisites(concept.id)
    dependents = system.knowledge_graph.get_dependent_concepts(concept.id)
    
    # Display relationships
    if prerequisites:
        print("Prerequisites:")
        for prereq in prerequisites:
            print(f"- {prereq.name} (Difficulty: {prereq.difficulty})")
    else:
        print("No prerequisites.")
    
    if dependents:
        print("\nConcepts that depend on this:")
        for dep in dependents:
            print(f"- {dep.name} (Difficulty: {dep.difficulty})")
    else:
        print("\nNo dependent concepts.")
    
    # Get exercises
    exercises = system.exercise_bank.get_exercises_for_concept(concept.id)
    print(f"\nExercises: {len(exercises)} available")
    
    # Display exercises if requested
    if exercises:
        view_exercises = input("\nView exercises? (y/n): ").lower().startswith('y')
        if view_exercises:
            for i, exercise in enumerate(exercises, 1):
                print(f"\nExercise {i}/{len(exercises)}")
                display_exercise(exercise)
                
                if i < len(exercises):
                    continue_viewing = input("Continue to next exercise? (y/n): ").lower().startswith('y')
                    if not continue_viewing:
                        break


def practice_concept(system: AlgebraLearningSystem, concept_name: str) -> None:
    """Practice exercises for a specific concept."""
    # Find the concept
    concept = system.get_concept_by_name(concept_name)
    if not concept:
        print(f"Concept '{concept_name}' not found.")
        return
    
    # Get exercises
    exercises = system.exercise_bank.get_exercises_for_concept(concept.id)
    if not exercises:
        print(f"No exercises available for '{concept.name}'.")
        return
    
    # Sort by difficulty
    exercises.sort(key=lambda ex: ex.difficulty)
    
    print(f"\nPracticing {concept.name}")
    print(f"{len(exercises)} exercises available")
    
    # Practice loop
    correct = 0
    for i, exercise in enumerate(exercises, 1):
        print(f"\nExercise {i}/{len(exercises)}")
        print(f"{'-' * 50}")
        print(f"{exercise.title} (Difficulty: {exercise.difficulty}/5)")
        print(f"\n{exercise.problem}")
        
        # Show hints if requested
        if exercise.hints:
            show_hint = input("\nNeed a hint? (y/n): ").lower().startswith('y')
            if show_hint:
                for j, hint in enumerate(exercise.hints, 1):
                    print(f"Hint {j}: {hint}")
                    if j < len(exercise.hints):
                        more_hints = input("Show next hint? (y/n): ").lower().startswith('y')
                        if not more_hints:
                            break
        
        # Get user's answer
        user_answer = input("\nYour answer: ")
        
        # Show solution
        print(f"\nCorrect solution: {exercise.solution}")
        
        # Ask user if they were correct
        was_correct = input("Did you get it right? (y/n): ").lower().startswith('y')
        if was_correct:
            correct += 1
        
        if i < len(exercises):
            continue_practice = input("\nContinue to next exercise? (y/n): ").lower().startswith('y')
            if not continue_practice:
                break
    
    # Show practice summary
    print(f"\nPractice Summary:")
    print(f"Completed: {i} out of {len(exercises)} exercises")
    print(f"Correct: {correct} ({correct/i*100:.1f}%)")


def show_learning_path(system: AlgebraLearningSystem, target_concept: str) -> None:
    """Show a learning path to reach a target concept."""
    # Find the concept
    concept = system.get_concept_by_name(target_concept)
    if not concept:
        print(f"Target concept '{target_concept}' not found.")
        return
    
    # Generate learning path
    learning_path = system.get_learning_path(concept.id)
    
    print(f"\nLearning Path for {concept.name}:")
    print(f"{'=' * 80}")
    
    if not learning_path:
        print("No learning path found.")
        return
    
    # Display learning path
    for i, step_concept in enumerate(learning_path, 1):
        exercises = system.exercise_bank.get_exercises_for_concept(step_concept.id)
        print(f"{i}. {step_concept.name}")
        print(f"   Difficulty: {step_concept.difficulty}/5 | Time: {step_concept.time_to_master} min")
        print(f"   Exercises: {len(exercises)} available")
        
        # Only show full details for the last few concepts
        if i >= len(learning_path) - 3 or len(learning_path) <= 5:
            print(f"   Description: {step_concept.description}")
            if step_concept.examples:
                print(f"   Example: {step_concept.examples[0]}")
        
        print()
    
    print(f"Total Learning Path Length: {len(learning_path)} concepts")
    print(f"Estimated Time to Master: {sum(c.time_to_master for c in learning_path)} minutes")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="K-12 Algebra Learning System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List concepts command
    list_parser = subparsers.add_parser("list", help="List all concepts")
    list_parser.add_argument("--category", "-c", help="Filter by category")
    list_parser.add_argument("--difficulty", "-d", type=int, choices=range(1, 6),
                           help="Filter by difficulty (1-5)")
    
    # Explore concept command
    explore_parser = subparsers.add_parser("explore", help="Explore a concept in detail")
    explore_parser.add_argument("concept", help="Concept name to explore")
    
    # Practice concept command
    practice_parser = subparsers.add_parser("practice", help="Practice exercises for a concept")
    practice_parser.add_argument("concept", help="Concept name to practice")
    
    # Learning path command
    path_parser = subparsers.add_parser("path", help="Show learning path to a target concept")
    path_parser.add_argument("concept", help="Target concept name")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize the system
    print("Initializing K-12 Algebra Learning System...")
    system = AlgebraLearningSystem()
    print(f"Loaded {len(system.knowledge_graph.concepts)} concepts and " +
          f"{len(system.exercise_bank.get_all_exercises())} exercises")
    
    # Execute command
    if args.command == "list":
        list_concepts(system, args.category, args.difficulty)
    elif args.command == "explore":
        explore_concept(system, args.concept)
    elif args.command == "practice":
        practice_concept(system, args.concept)
    elif args.command == "path":
        show_learning_path(system, args.concept)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 