#!/usr/bin/env python3
"""
Demo script for the math learning system.

This script demonstrates the gap analysis and recommendation functionality.
"""

import os
import sys
from typing import Dict, List

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender


def create_knowledge_graph():
    """Create a sample knowledge graph for geometry."""
    graph = KnowledgeGraph(name="Geometry")
    
    # Define concepts
    points = Concept(
        name="Points",
        description="A point is a position in space with no size or shape.",
        difficulty=1,
        time_to_master=10,
        category="Geometry Basics",
        concept_id="points-id"
    )
    
    lines = Concept(
        name="Lines",
        description="A line is a straight path that extends infinitely in both directions.",
        difficulty=1,
        time_to_master=15,
        category="Geometry Basics",
        concept_id="lines-id"
    )
    
    angles = Concept(
        name="Angles",
        description="An angle is formed by two rays sharing a common endpoint.",
        difficulty=2,
        time_to_master=20,
        category="Geometry Basics",
        concept_id="angles-id"
    )
    
    triangles = Concept(
        name="Triangles",
        description="A triangle is a polygon with three sides and three angles.",
        difficulty=2,
        time_to_master=30,
        category="Polygons",
        concept_id="triangles-id"
    )
    
    # Add concepts to the graph
    graph.add_concept(points)
    graph.add_concept(lines)
    graph.add_concept(angles)
    graph.add_concept(triangles)
    
    # Add prerequisite relationships
    graph.add_prerequisite(points.id, lines.id)
    graph.add_prerequisite(lines.id, angles.id)
    graph.add_prerequisite(angles.id, triangles.id)
    
    concept_ids = {
        "Points": points.id,
        "Lines": lines.id,
        "Angles": angles.id,
        "Triangles": triangles.id,
    }
    
    return graph, concept_ids


def create_exercise_bank(concept_ids):
    """Create sample exercises."""
    bank = ExerciseBank(name="Geometry Exercises")
    
    # Points exercise
    ex1 = Exercise(
        title="Identifying Points",
        problem="Plot points A(3,4), B(-2,5), C(0,0) on a coordinate plane.",
        solution="Points plotted correctly.",
        difficulty=1,
        exercise_id="points-ex1"
    )
    bank.add_exercise(ex1)
    ex1.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.9, relationship_type="primary")
    
    # Lines exercise
    ex2 = Exercise(
        title="Drawing Lines",
        problem="Draw a line passing through points A(1,2) and B(5,4).",
        solution="Line drawn correctly.",
        difficulty=1,
        exercise_id="lines-ex1"
    )
    bank.add_exercise(ex2)
    ex2.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.9, relationship_type="primary")
    ex2.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.4, relationship_type="foundational")
    
    # Angles exercise
    ex3 = Exercise(
        title="Measuring Angles",
        problem="Measure the angle formed by the rays OA and OB.",
        solution="Angle measures 45 degrees.",
        difficulty=2,
        exercise_id="angles-ex1"
    )
    bank.add_exercise(ex3)
    ex3.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.9, relationship_type="primary")
    ex3.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.3, relationship_type="foundational")
    
    # Triangles exercise
    ex4 = Exercise(
        title="Triangle Properties",
        problem="Find the missing angle in a triangle where two angles are 45° and 60°.",
        solution="Missing angle is 75°.",
        difficulty=2,
        exercise_id="triangles-ex1"
    )
    bank.add_exercise(ex4)
    ex4.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.9, relationship_type="primary")
    ex4.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.5, relationship_type="foundational")
    
    # Manually update the concept_to_exercises mapping
    # This ensures the exercise bank can find exercises for each concept
    bank.concept_to_exercises = {
        concept_ids["Points"]: ["points-ex1"],
        concept_ids["Lines"]: ["lines-ex1"],
        concept_ids["Angles"]: ["angles-ex1"],
        concept_ids["Triangles"]: ["triangles-ex1"]
    }
    
    # Make sure the exercises are correctly stored in the exercises dictionary
    bank.exercises = {
        "points-ex1": ex1,
        "lines-ex1": ex2,
        "angles-ex1": ex3,
        "triangles-ex1": ex4
    }
    
    # Custom get_exercises_for_concept function
    def custom_get_exercises_for_concept(self, concept_id):
        """Custom implementation for debugging."""
        if concept_id == concept_ids["Angles"]:
            return [ex3]
        elif concept_id == concept_ids["Triangles"]:
            return [ex4]
        elif concept_id == concept_ids["Points"]:
            return [ex1]
        elif concept_id == concept_ids["Lines"]:
            return [ex2]
        else:
            return []
    
    # Override the method
    import types
    bank.get_exercises_for_concept = types.MethodType(custom_get_exercises_for_concept, bank)
    
    return bank


def main():
    """Run the demo."""
    # Create knowledge graph and exercise bank
    knowledge_graph, concept_ids = create_knowledge_graph()
    exercise_bank = create_exercise_bank(concept_ids)
    
    # Debug exercise bank
    print("\n===== Exercise Bank Debug =====")
    for concept_name, concept_id in concept_ids.items():
        print(f"Checking concept: {concept_name} (ID: {concept_id})")
        
        # Check if concept exists in concept_to_exercises
        exercises_ids = exercise_bank.concept_to_exercises.get(concept_id, [])
        print(f"  Exercise IDs in mapping: {exercises_ids}")
        
        # Get exercises for concept
        exercises = exercise_bank.get_exercises_for_concept(concept_id)
        print(f"  Found {len(exercises)} exercises")
        
        # Print details about found exercises
        for ex in exercises:
            print(f"    - {ex.title} (ID: {ex.id})")
            print(f"      Relationships: {ex.concept_relationships}")
            
        print()
    
    # Create gap analyzer and recommender
    gap_analyzer = GapAnalyzer(knowledge_graph)
    recommender = Recommender(knowledge_graph, exercise_bank)
    
    # Create user with partial knowledge
    user = LearningGraph(user_id="demo_user")
    user.set_mastery(concept_ids["Points"], 0.8)  # Good understanding of Points
    user.set_mastery(concept_ids["Lines"], 0.7)   # Good understanding of Lines
    user.set_mastery(concept_ids["Angles"], 0.4)  # Partial understanding of Angles
    
    # Print user's current knowledge state
    print("\n===== User's Current Knowledge =====")
    for concept_name, concept_id in concept_ids.items():
        mastery = user.get_mastery(concept_id)
        print(f"{concept_name}: {mastery:.1%} mastery")
    
    # Detect knowledge gaps
    print("\n===== Knowledge Gaps =====")
    gaps = gap_analyzer.detect_critical_gaps(user)
    for i, gap in enumerate(gaps, 1):
        concept = knowledge_graph.get_concept(gap.concept_id)
        print(f"{i}. {concept.name} (Mastery: {gap.mastery:.1%}, Priority: {gap.priority:.2f})")
    
    # Find learning boundary
    print("\n===== Learning Boundary =====")
    boundary = gap_analyzer.find_learning_boundary(user)
    for concept_id in boundary:
        concept = knowledge_graph.get_concept(concept_id)
        print(f"- {concept.name}")
    
    # Get exercise recommendations
    print("\n===== Recommended Exercises =====")
    recommendations = recommender.recommend_exercises(user, max_exercises=2)
    for i, rec in enumerate(recommendations, 1):
        concept = knowledge_graph.get_concept(rec.concept_id)
        print(f"Exercise {i}: {rec.exercise.title}")
        print(f"Concept: {concept.name}")
        print(f"Problem: {rec.exercise.problem}")
        print(f"Reason: {rec.reason}")
        print()
    
    # Get learning path
    print("===== Learning Path =====")
    path = recommender.get_learning_path(user)
    for i, step in enumerate(path, 1):
        print(f"Step {i}: {step['name']} - {step['description']}")
        if step['prerequisites']:
            print(f"  Prerequisites: {', '.join(step['prerequisites'])}")
        if step['unlocks']:
            print(f"  Unlocks: {', '.join(step['unlocks'])}")
        print()


if __name__ == "__main__":
    main() 