"""
Test script for the algebra learning system.

This script directly creates a small knowledge graph and exercise bank
without relying on the more complex importing structure.
"""

import uuid
import os
import json
import networkx as nx
from typing import Dict, List, Optional, Set, Any, Tuple


# Simplified Concept class
class Concept:
    def __init__(
        self,
        name: str,
        description: str,
        difficulty: int = 3,
        time_to_master: int = 60,
        category: str = "",
        concept_id: Optional[str] = None,
        examples: Optional[List[str]] = None,
    ):
        self.id = concept_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.difficulty = max(1, min(5, difficulty))  # Ensure 1-5 range
        self.time_to_master = time_to_master
        self.category = category
        self.examples = examples or []
        
        # Relationships
        self.prerequisites: Set[str] = set()
        self.dependents: Set[str] = set()
        self.related: Set[str] = set()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "difficulty": self.difficulty,
            "time_to_master": self.time_to_master,
            "category": self.category,
            "examples": self.examples,
            "prerequisites": list(self.prerequisites),
            "dependents": list(self.dependents),
            "related": list(self.related)
        }


# Simplified Knowledge Graph class
class KnowledgeGraph:
    def __init__(self, name: str = ""):
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        self.graph = nx.DiGraph()  # Directed graph
        
    def add_concept(self, concept: Concept) -> None:
        self.concepts[concept.id] = concept
        self.graph.add_node(concept.id, data=concept.to_dict())
        
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        if prerequisite_id not in self.concepts or concept_id not in self.concepts:
            raise ValueError(f"Both concepts must exist in the knowledge graph. Missing: {prerequisite_id if prerequisite_id not in self.concepts else concept_id}")
            
        self.graph.add_edge(prerequisite_id, concept_id, 
                            type="prerequisite", 
                            strength=strength)
        
        # Update concept relationship sets
        self.concepts[prerequisite_id].dependents.add(concept_id)
        self.concepts[concept_id].prerequisites.add(prerequisite_id)


# Simplified Exercise class
class Exercise:
    def __init__(
        self,
        title: str,
        problem: str,
        solution: str,
        difficulty: int = 3,
        estimated_time: int = 5,
        format_type: str = "free-response",
        exercise_id: Optional[str] = None,
        hints: Optional[List[str]] = None,
    ):
        self.id = exercise_id or str(uuid.uuid4())
        self.title = title
        self.problem = problem
        self.solution = solution
        self.difficulty = max(1, min(5, difficulty))  # Ensure 1-5 range
        self.estimated_time = estimated_time
        self.format_type = format_type
        self.hints = hints or []
        
        # Concept relationships
        self.concept_relationships: List[Dict[str, Any]] = []
        
    def add_concept_relationship(
        self,
        concept_id: str,
        weight: float = 0.8,
        relationship_type: str = "primary"
    ) -> None:
        self.concept_relationships.append({
            "concept_id": concept_id,
            "weight": max(0.0, min(1.0, weight)),  # Ensure 0.0-1.0 range
            "type": relationship_type
        })
    
    def get_all_concept_ids(self) -> List[str]:
        return [rel["concept_id"] for rel in self.concept_relationships]


# Simplified Exercise Bank class
class ExerciseBank:
    def __init__(self, name: str = ""):
        self.name = name
        self.exercises: Dict[str, Exercise] = {}
        self.concept_to_exercises: Dict[str, List[str]] = {}
        
    def add_exercise(self, exercise: Exercise) -> None:
        self.exercises[exercise.id] = exercise
        
        # Update concept-to-exercises mapping
        for concept_id in exercise.get_all_concept_ids():
            if concept_id not in self.concept_to_exercises:
                self.concept_to_exercises[concept_id] = []
            self.concept_to_exercises[concept_id].append(exercise.id)
    
    def get_all_exercises(self) -> List[Exercise]:
        return list(self.exercises.values())
        
    def get_exercises_for_concept(self, concept_id: str) -> List[Exercise]:
        exercise_ids = self.concept_to_exercises.get(concept_id, [])
        return [self.exercises[ex_id] for ex_id in exercise_ids if ex_id in self.exercises]


def build_mini_algebra_system():
    """Build a mini algebra knowledge graph and exercise bank for testing."""
    
    # Create knowledge graph
    graph = KnowledgeGraph(name="Mini Algebra KG")
    
    # Create some concepts with fixed IDs for easy reference
    addition = Concept(
        name="Addition",
        description="Combining two or more numbers to find their sum.",
        difficulty=1,
        time_to_master=60,
        category="Operations",
        concept_id="addition",
        examples=["2 + 3 = 5", "If you have 4 apples and get 3 more, how many do you have?"]
    )
    
    subtraction = Concept(
        name="Subtraction",
        description="Taking one number away from another to find the difference.",
        difficulty=1, 
        time_to_master=60,
        category="Operations",
        concept_id="subtraction",
        examples=["5 - 2 = 3", "If you have 7 candies and eat 3, how many are left?"]
    )
    
    variables = Concept(
        name="Variables and Expressions",
        description="Using letters to represent unknown quantities and forming expressions.",
        difficulty=3,
        time_to_master=120,
        category="Algebra",
        concept_id="variables",
        examples=["x + 5", "3y - 2", "If a pencil costs x dollars, how much do 3 pencils cost?"]
    )
    
    # Add concepts to graph
    graph.add_concept(addition)
    graph.add_concept(subtraction)
    graph.add_concept(variables)
    
    # Add relationships
    graph.add_prerequisite("addition", "subtraction")
    graph.add_prerequisite("addition", "variables")
    graph.add_prerequisite("subtraction", "variables")
    
    # Create exercise bank
    bank = ExerciseBank(name="Mini Algebra Exercises")
    
    # Create exercises
    addition_ex = Exercise(
        title="Simple Addition",
        problem="What is 3 + 5?",
        solution="8",
        difficulty=1,
        estimated_time=1,
        format_type="free-response",
        hints=["Count 3 and then count 5 more", "Use your fingers to help"]
    )
    addition_ex.add_concept_relationship("addition", weight=1.0, relationship_type="primary")
    bank.add_exercise(addition_ex)
    
    subtraction_ex = Exercise(
        title="Simple Subtraction",
        problem="What is 8 - 3?",
        solution="5",
        difficulty=1,
        estimated_time=1,
        format_type="free-response",
        hints=["Start with 8 and count backwards 3 times", "Use your fingers to help"]
    )
    subtraction_ex.add_concept_relationship("subtraction", weight=1.0, relationship_type="primary")
    subtraction_ex.add_concept_relationship("addition", weight=0.3, relationship_type="foundational")
    bank.add_exercise(subtraction_ex)
    
    variables_ex = Exercise(
        title="Evaluate an Expression",
        problem="Evaluate the expression 3x + 2 when x = 4.",
        solution="14",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["Substitute x = 4 into the expression", "Calculate 3 × 4 + 2"]
    )
    variables_ex.add_concept_relationship("variables", weight=1.0, relationship_type="primary")
    variables_ex.add_concept_relationship("addition", weight=0.5, relationship_type="foundational")
    bank.add_exercise(variables_ex)
    
    return graph, bank


def main():
    """Main test function."""
    print("Building mini algebra learning system...")
    graph, bank = build_mini_algebra_system()
    
    # Print knowledge graph info
    print(f"\nKnowledge Graph: {len(graph.concepts)} concepts")
    for concept_id, concept in graph.concepts.items():
        prereqs = concept.prerequisites
        deps = concept.dependents
        print(f"- {concept.name} (ID: {concept.id})")
        print(f"  Prerequisites: {list(prereqs) or 'None'}")
        print(f"  Dependents: {list(deps) or 'None'}")
    
    # Print exercise bank info
    print(f"\nExercise Bank: {len(bank.exercises)} exercises")
    for exercise in bank.get_all_exercises():
        concept_ids = exercise.get_all_concept_ids()
        concept_names = [graph.concepts[cid].name for cid in concept_ids if cid in graph.concepts]
        print(f"- {exercise.title} (Difficulty: {exercise.difficulty})")
        print(f"  Related concepts: {concept_names}")
    
    # Test getting exercises for concepts
    print("\nExercises for Addition:")
    for ex in bank.get_exercises_for_concept("addition"):
        print(f"- {ex.title}")
    
    print("\nExercises for Variables:")
    for ex in bank.get_exercises_for_concept("variables"):
        print(f"- {ex.title}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main() 