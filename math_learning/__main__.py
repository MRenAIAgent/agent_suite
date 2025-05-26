"""
Main module for the math learning system.

This module provides a simple demo of the math learning system.
"""

import os
import argparse
import json
import sys
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender


def create_geometry_knowledge_graph() -> KnowledgeGraph:
    """Create a knowledge graph for basic geometry."""
    graph = KnowledgeGraph(name="Basic Geometry")
    
    # Define concepts
    points = Concept(
        name="Points",
        description="A point is a position in space with no size or shape.",
        difficulty=1,
        time_to_master=10,
        category="Geometry Basics",
        examples=["Point A is represented as a dot."]
    )
    
    lines = Concept(
        name="Lines",
        description="A line is a straight path that extends infinitely in both directions.",
        difficulty=1,
        time_to_master=15,
        category="Geometry Basics",
        examples=["Line AB is represented by a straight line passing through points A and B."]
    )
    
    angles = Concept(
        name="Angles",
        description="An angle is formed by two rays sharing a common endpoint.",
        difficulty=2,
        time_to_master=20,
        category="Geometry Basics",
        examples=["Angle ABC measures 45 degrees."]
    )
    
    triangles = Concept(
        name="Triangles",
        description="A triangle is a polygon with three sides and three angles.",
        difficulty=2,
        time_to_master=30,
        category="Polygons",
        examples=["Triangle ABC has angles that sum to 180 degrees."]
    )
    
    right_triangles = Concept(
        name="Right Triangles",
        description="A right triangle has one angle that measures 90 degrees.",
        difficulty=3,
        time_to_master=40,
        category="Polygons",
        examples=["The Pythagorean theorem applies to right triangles."]
    )
    
    pythagorean_theorem = Concept(
        name="Pythagorean Theorem",
        description="In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
        difficulty=3,
        time_to_master=45,
        category="Theorems",
        examples=["a² + b² = c² where c is the hypotenuse."]
    )
    
    circles = Concept(
        name="Circles",
        description="A circle is the set of all points equidistant from a fixed point called the center.",
        difficulty=3,
        time_to_master=35,
        category="Curved Shapes",
        examples=["The circumference of a circle is 2πr."]
    )
    
    perimeter = Concept(
        name="Perimeter",
        description="The perimeter is the total length of the boundary of a two-dimensional shape.",
        difficulty=2,
        time_to_master=25,
        category="Measurement",
        examples=["The perimeter of a rectangle is 2(length + width)."]
    )
    
    area = Concept(
        name="Area",
        description="The area is the amount of space inside a two-dimensional shape.",
        difficulty=2,
        time_to_master=30,
        category="Measurement",
        examples=["The area of a rectangle is length × width."]
    )
    
    # Add concepts to the graph
    graph.add_concept(points)
    graph.add_concept(lines)
    graph.add_concept(angles)
    graph.add_concept(triangles)
    graph.add_concept(right_triangles)
    graph.add_concept(pythagorean_theorem)
    graph.add_concept(circles)
    graph.add_concept(perimeter)
    graph.add_concept(area)
    
    # Add prerequisite relationships
    graph.add_prerequisite(points.id, lines.id)
    graph.add_prerequisite(lines.id, angles.id)
    graph.add_prerequisite(angles.id, triangles.id)
    graph.add_prerequisite(triangles.id, right_triangles.id)
    graph.add_prerequisite(right_triangles.id, pythagorean_theorem.id)
    graph.add_prerequisite(points.id, circles.id)
    graph.add_prerequisite(lines.id, perimeter.id)
    graph.add_prerequisite(triangles.id, area.id)
    
    # Add related relationships
    graph.add_related(angles.id, circles.id)
    graph.add_related(perimeter.id, area.id)
    
    return graph, {
        "Points": points.id,
        "Lines": lines.id,
        "Angles": angles.id,
        "Triangles": triangles.id,
        "Right Triangles": right_triangles.id,
        "Pythagorean Theorem": pythagorean_theorem.id,
        "Circles": circles.id,
        "Perimeter": perimeter.id,
        "Area": area.id
    }


def create_geometry_exercises(concept_ids: Dict[str, str]) -> ExerciseBank:
    """Create exercises for basic geometry."""
    bank = ExerciseBank(name="Basic Geometry Exercises")
    
    # Points exercises
    ex1 = Exercise(
        title="Identifying Points",
        problem="Plot the following points on a coordinate plane: A(3, 4), B(-2, 5), C(0, 0).",
        solution="Points should be plotted at the coordinates specified.",
        difficulty=1,
        estimated_time=5,
        format_type="graphing"
    )
    
    # Lines exercises
    ex2 = Exercise(
        title="Drawing Lines",
        problem="Draw a line passing through the points A(1, 2) and B(5, 4).",
        solution="Draw a straight line connecting the two points.",
        difficulty=1,
        estimated_time=5,
        format_type="graphing"
    )
    
    # Angles exercises
    ex3 = Exercise(
        title="Measuring Angles",
        problem="Using a protractor, measure the angle formed by the rays OA and OB.",
        solution="The angle measures 45 degrees.",
        difficulty=2,
        estimated_time=7,
        format_type="measurement"
    )
    
    # Triangles exercises
    ex4 = Exercise(
        title="Triangle Properties",
        problem="Find the missing angle in a triangle where two angles are 45° and 60°.",
        solution="The missing angle is 75° because the sum of angles in a triangle is 180°. 180° - 45° - 60° = 75°",
        difficulty=2,
        estimated_time=8,
        format_type="calculation"
    )
    
    # Right triangles exercises
    ex5 = Exercise(
        title="Identifying Right Triangles",
        problem="Which of the following triangles is a right triangle? A(3,4,5), B(5,6,7), C(8,15,17)",
        solution="Triangles A(3,4,5) and C(8,15,17) are right triangles because they satisfy the Pythagorean theorem.",
        difficulty=3,
        estimated_time=10,
        format_type="multiple-choice"
    )
    
    # Pythagorean theorem exercises
    ex6 = Exercise(
        title="Applying Pythagorean Theorem",
        problem="Find the hypotenuse of a right triangle with legs of length 6 and 8.",
        solution="Using the Pythagorean theorem: c² = 6² + 8² = 36 + 64 = 100, so c = 10.",
        difficulty=3,
        estimated_time=10,
        format_type="calculation"
    )
    
    # Circles exercises
    ex7 = Exercise(
        title="Circle Properties",
        problem="Find the circumference of a circle with radius 5 cm.",
        solution="The circumference is 2πr = 2π × 5 = 10π ≈ 31.4 cm.",
        difficulty=2,
        estimated_time=7,
        format_type="calculation"
    )
    
    # Perimeter exercises
    ex8 = Exercise(
        title="Finding Perimeter",
        problem="Calculate the perimeter of a rectangle with length 8 cm and width 5 cm.",
        solution="The perimeter is 2(l + w) = 2(8 + 5) = 2(13) = 26 cm.",
        difficulty=2,
        estimated_time=6,
        format_type="calculation"
    )
    
    # Area exercises
    ex9 = Exercise(
        title="Calculating Area",
        problem="Find the area of a triangle with base 10 cm and height 6 cm.",
        solution="The area is (1/2) × base × height = (1/2) × 10 × 6 = 30 cm².",
        difficulty=2,
        estimated_time=8,
        format_type="calculation"
    )
    
    # Add exercises to bank and link to concepts
    # Points exercise
    bank.add_exercise(ex1)
    ex1.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.9, relationship_type="primary")
    
    # Lines exercise
    bank.add_exercise(ex2)
    ex2.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.9, relationship_type="primary")
    ex2.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.4, relationship_type="foundational")
    
    # Angles exercise
    bank.add_exercise(ex3)
    ex3.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.9, relationship_type="primary")
    ex3.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.3, relationship_type="foundational")
    
    # Triangles exercise
    bank.add_exercise(ex4)
    ex4.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.9, relationship_type="primary")
    ex4.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.5, relationship_type="foundational")
    
    # Right triangles exercise
    bank.add_exercise(ex5)
    ex5.add_concept_relationship(concept_id=concept_ids["Right Triangles"], weight=0.9, relationship_type="primary")
    ex5.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.4, relationship_type="foundational")
    
    # Pythagorean theorem exercise
    bank.add_exercise(ex6)
    ex6.add_concept_relationship(concept_id=concept_ids["Pythagorean Theorem"], weight=0.9, relationship_type="primary")
    ex6.add_concept_relationship(concept_id=concept_ids["Right Triangles"], weight=0.5, relationship_type="foundational")
    
    # Circles exercise
    bank.add_exercise(ex7)
    ex7.add_concept_relationship(concept_id=concept_ids["Circles"], weight=0.9, relationship_type="primary")
    
    # Perimeter exercise
    bank.add_exercise(ex8)
    ex8.add_concept_relationship(concept_id=concept_ids["Perimeter"], weight=0.9, relationship_type="primary")
    
    # Area exercise
    bank.add_exercise(ex9)
    ex9.add_concept_relationship(concept_id=concept_ids["Area"], weight=0.9, relationship_type="primary")
    ex9.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.4, relationship_type="foundational")
    
    return bank


def simulate_user_learning(knowledge_graph: KnowledgeGraph,
                         exercise_bank: ExerciseBank,
                         concept_ids: Dict[str, str],
                         output_dir: str = "output") -> None:
    """Simulate a user learning geometry."""
    # Create learning graph for user
    learning_graph = LearningGraph(user_id="user1", name="John's Geometry Learning")
    
    # Initialize with some prior knowledge
    learning_graph.set_mastery(concept_ids["Points"], 0.9)  # Already knows points well
    learning_graph.set_mastery(concept_ids["Lines"], 0.7)   # Familiar with lines
    learning_graph.set_mastery(concept_ids["Angles"], 0.4)  # Some understanding of angles
    
    # Create recommender
    recommender = Recommender(knowledge_graph, exercise_bank)
    
    # Initial assessment
    print("\n=== Initial Learning State ===")
    for concept in knowledge_graph.get_all_concepts():
        mastery = learning_graph.get_mastery(concept.id)
        print(f"{concept.name}: {mastery:.1%} mastery")
    
    # Analyze gaps
    gap_analyzer = GapAnalyzer(knowledge_graph)
    gaps = gap_analyzer.detect_critical_gaps(learning_graph)
    
    print("\n=== Knowledge Gaps ===")
    for gap in gaps[:3]:
        concept = knowledge_graph.get_concept(gap.concept_id)
        if concept:
            print(f"{concept.name}: {gap.mastery:.1%} mastery, priority: {gap.priority:.2f}")
    
    # Recommend exercises
    recommendations = recommender.recommend_exercises(learning_graph, max_exercises=3)
    
    print("\n=== Recommended Exercises ===")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.exercise.title} - {rec.reason}")
        print(f"   Problem: {rec.exercise.problem}")
    
    # Simulate user completing an exercise
    if recommendations:
        print("\n=== Simulating Exercise Completion ===")
        rec = recommendations[0]
        concept_id = rec.concept_id
        exercise = rec.exercise
        
        # Let's say the user got it right
        result = True
        print(f"Completed exercise: {exercise.title}")
        print(f"Result: {'Correct' if result else 'Incorrect'}")
        
        # Record the attempt
        learning_graph.record_exercise_attempt(exercise.id, concept_id, result)
        
        # Update mastery
        old_mastery = learning_graph.get_mastery(concept_id)
        new_mastery = learning_graph.update_mastery(
            concept_id, 
            exercise_result=result,
            exercise_difficulty=exercise.difficulty / 5.0,  # Convert to 0-1 scale
            concept_weight=exercise.get_concept_weight(concept_id)
        )
        
        print(f"Mastery of {knowledge_graph.get_concept(concept_id).name} updated: {old_mastery:.1%} -> {new_mastery:.1%}")
    
    # Get recommended learning path
    path = recommender.get_learning_path(learning_graph, max_concepts=3)
    
    print("\n=== Recommended Learning Path ===")
    for i, step in enumerate(path, 1):
        print(f"{i}. {step['name']}")
        print(f"   Prerequisites: {', '.join(step['prerequisites']) if step['prerequisites'] else 'None'}")
        print(f"   Unlocks: {', '.join(step['unlocks']) if step['unlocks'] else 'None'}")
    
    # Save results
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if output_dir:
        knowledge_graph.save_to_file(os.path.join(output_dir, "knowledge_graph.json"))
        learning_graph.save_to_file(os.path.join(output_dir, "learning_graph.json"))
        
        # Save exercises
        with open(os.path.join(output_dir, "exercises.json"), 'w') as f:
            json.dump({
                "exercises": [ex.to_dict() for ex in exercise_bank.get_all_exercises()]
            }, f, indent=2)
            
        # Save recommendations
        with open(os.path.join(output_dir, "recommendations.json"), 'w') as f:
            json.dump({
                "gaps": [gap.to_dict() for gap in gaps],
                "recommendations": [rec.to_dict() for rec in recommendations],
                "learning_path": path
            }, f, indent=2)
            

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Math Learning System Demo")
    parser.add_argument("--output", "-o", default="output", help="Output directory for results")
    args = parser.parse_args()
    
    print("=== Math Learning System Demo ===")
    print("Creating geometry knowledge graph...")
    knowledge_graph, concept_ids = create_geometry_knowledge_graph()
    
    print("Creating exercise bank...")
    exercise_bank = create_geometry_exercises(concept_ids)
    
    print("Simulating user learning...")
    simulate_user_learning(knowledge_graph, exercise_bank, concept_ids, args.output)
    
    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main() 