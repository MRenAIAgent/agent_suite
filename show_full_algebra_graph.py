#!/usr/bin/env python3
"""
Display the full algebra knowledge graph structure.

This script builds a comprehensive algebra knowledge graph with all K-12 concepts
from elementary to high school levels.
"""

import uuid
from typing import Dict, List, Optional, Set, Any


# Concept class (simplified version of knowledge_graph/concept.py)
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


# Knowledge Graph class (simplified version of knowledge_graph/graph.py)
class KnowledgeGraph:
    def __init__(self, name: str = ""):
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        
    def add_concept(self, concept: Concept) -> None:
        self.concepts[concept.id] = concept
        
    def add_prerequisite(self, prerequisite_id: str, concept_id: str) -> None:
        if prerequisite_id not in self.concepts or concept_id not in self.concepts:
            print(f"Warning: Cannot add prerequisite relationship between {prerequisite_id} -> {concept_id}")
            return
            
        self.concepts[prerequisite_id].dependents.add(concept_id)
        self.concepts[concept_id].prerequisites.add(prerequisite_id)
        
    def add_related(self, concept_id1: str, concept_id2: str) -> None:
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            print(f"Warning: Cannot add related relationship between {concept_id1} <-> {concept_id2}")
            return
            
        self.concepts[concept_id1].related.add(concept_id2)
        self.concepts[concept_id2].related.add(concept_id1)


def build_full_algebra_graph() -> KnowledgeGraph:
    """Build the complete K-12 algebra knowledge graph."""
    
    # Initialize the knowledge graph
    graph = KnowledgeGraph(name="K-12 Algebra")
    
    # Elementary School Algebra Concepts (Grades K-5)
    elementary_concepts = {
        # Kindergarten - Grade 2
        "counting": Concept(
            name="Counting",
            description="Counting objects and understanding numbers represent quantities.",
            difficulty=1,
            time_to_master=30,
            category="Number Sense",
            concept_id="counting",
            examples=["Count objects up to 20", "Count by 2s, 5s, and 10s"]
        ),
        "addition": Concept(
            name="Addition",
            description="Combining two or more numbers to find their sum.",
            difficulty=1,
            time_to_master=60,
            category="Operations",
            concept_id="addition",
            examples=["2 + 3 = 5", "If you have 4 apples and get 3 more, how many do you have?"]
        ),
        "subtraction": Concept(
            name="Subtraction",
            description="Taking one number away from another to find the difference.",
            difficulty=1, 
            time_to_master=60,
            category="Operations",
            concept_id="subtraction",
            examples=["5 - 2 = 3", "If you have 7 candies and eat 3, how many are left?"]
        ),
        
        # Grades 3-5
        "multiplication": Concept(
            name="Multiplication",
            description="Repeated addition of the same number.",
            difficulty=2,
            time_to_master=90,
            category="Operations",
            concept_id="multiplication",
            examples=["3 × 4 = 12", "If you have 5 bags with 3 marbles in each, how many marbles do you have?"]
        ),
        "division": Concept(
            name="Division",
            description="Splitting a number into equal parts or groups.",
            difficulty=2,
            time_to_master=90,
            category="Operations",
            concept_id="division",
            examples=["12 ÷ 3 = 4", "If you share 15 cookies equally among 5 friends, how many does each friend get?"]
        ),
        "fractions_intro": Concept(
            name="Introduction to Fractions",
            description="Understanding parts of a whole and representing them as fractions.",
            difficulty=2,
            time_to_master=120,
            category="Number Sense",
            concept_id="fractions_intro",
            examples=["1/2 of a pizza", "3/4 of a rectangle"]
        ),
        "patterns": Concept(
            name="Patterns and Sequences",
            description="Identifying and continuing number patterns and sequences.",
            difficulty=2,
            time_to_master=60,
            category="Algebra Readiness",
            concept_id="patterns",
            examples=["2, 4, 6, 8, ...", "Complete the pattern: 3, 6, 9, __, __"]
        ),
        "place_value": Concept(
            name="Place Value",
            description="Understanding the value of digits based on their position in a number.",
            difficulty=2,
            time_to_master=60,
            category="Number Sense",
            concept_id="place_value",
            examples=["In 352, the 5 represents 50", "Write 4,025 in expanded form"]
        ),
    }
    
    # Middle School Algebra Concepts (Grades 6-8)
    middle_concepts = {
        "variables": Concept(
            name="Variables and Expressions",
            description="Using letters to represent unknown quantities and forming expressions.",
            difficulty=3,
            time_to_master=120,
            category="Algebra",
            concept_id="variables",
            examples=["x + 5", "3y - 2", "If a pencil costs x dollars, how much do 3 pencils cost?"]
        ),
        "equations_one_step": Concept(
            name="One-Step Equations",
            description="Solving equations that require a single operation to isolate the variable.",
            difficulty=3,
            time_to_master=90,
            category="Algebra",
            concept_id="equations_one_step",
            examples=["x + 3 = 8", "2y = 10"]
        ),
        "equations_two_step": Concept(
            name="Two-Step Equations",
            description="Solving equations that require two operations to isolate the variable.",
            difficulty=3,
            time_to_master=120,
            category="Algebra",
            concept_id="equations_two_step",
            examples=["2x + 3 = 11", "3y - 5 = 10"]
        ),
        "inequalities": Concept(
            name="Inequalities",
            description="Mathematical statements showing the relationship between unequal expressions.",
            difficulty=3,
            time_to_master=120,
            category="Algebra",
            concept_id="inequalities",
            examples=["x > 5", "3y - 2 ≤ 10"]
        ),
        "coordinate_plane": Concept(
            name="Coordinate Plane",
            description="A plane with a coordinate system for specifying points using ordered pairs.",
            difficulty=3,
            time_to_master=90,
            category="Algebra",
            concept_id="coordinate_plane",
            examples=["Plot the point (3, -2)", "Find the coordinates of point P"]
        ),
        "proportional_relationships": Concept(
            name="Proportional Relationships",
            description="Relationships where the ratio of two quantities remains constant.",
            difficulty=3,
            time_to_master=120,
            category="Algebra",
            concept_id="proportional_relationships",
            examples=["If 3 apples cost $1.50, how much do 7 apples cost?", "y = kx"]
        ),
        "linear_functions": Concept(
            name="Linear Functions",
            description="Functions that graph as a straight line, often in the form y = mx + b.",
            difficulty=4,
            time_to_master=150,
            category="Algebra",
            concept_id="linear_functions",
            examples=["y = 2x + 3", "Find the slope and y-intercept of y = -x + 5"]
        ),
        "systems_linear": Concept(
            name="Systems of Linear Equations",
            description="Two or more linear equations that are solved simultaneously.",
            difficulty=4,
            time_to_master=180,
            category="Algebra",
            concept_id="systems_linear",
            examples=["Solve: y = 2x + 1 and y = -x + 7", "Find the point of intersection"]
        ),
    }
    
    # High School Algebra Concepts (Grades 9-12)
    high_concepts = {
        "exponents_roots": Concept(
            name="Exponents and Roots",
            description="Working with powers and roots of numbers.",
            difficulty=4,
            time_to_master=120,
            category="Algebra",
            concept_id="exponents_roots",
            examples=["Simplify: (2³)²", "Solve for x: x² = 25"]
        ),
        "polynomials": Concept(
            name="Polynomials",
            description="Expressions with variables raised to whole number powers.",
            difficulty=4,
            time_to_master=180,
            category="Algebra",
            concept_id="polynomials",
            examples=["3x² + 2x - 5", "Factor: x² - 9"]
        ),
        "factoring": Concept(
            name="Factoring",
            description="Breaking down a polynomial into simpler expressions that multiply to give the original.",
            difficulty=4,
            time_to_master=210,
            category="Algebra",
            concept_id="factoring",
            examples=["Factor: x² + 5x + 6", "Factor: 2x² - 8"]
        ),
        "quadratic_functions": Concept(
            name="Quadratic Functions",
            description="Functions in the form f(x) = ax² + bx + c where a ≠ 0.",
            difficulty=4,
            time_to_master=210,
            category="Algebra",
            concept_id="quadratic_functions",
            examples=["f(x) = x² - 4x + 3", "Find the vertex of y = 2x² + 4x - 1"]
        ),
        "quadratic_formula": Concept(
            name="Quadratic Formula",
            description="Formula to solve quadratic equations: x = (-b ± √(b² - 4ac)) / 2a.",
            difficulty=4,
            time_to_master=150,
            category="Algebra",
            concept_id="quadratic_formula",
            examples=["Solve: 2x² - 5x + 2 = 0", "Find the roots of y = x² - 6x + 8"]
        ),
        "rational_expressions": Concept(
            name="Rational Expressions",
            description="Fractions where the numerator and denominator are polynomials.",
            difficulty=5,
            time_to_master=240,
            category="Algebra",
            concept_id="rational_expressions",
            examples=["Simplify: (x² - 4)/(x - 2)", "Add: x/(x+1) + 2/(x-1)"]
        ),
        "exponential_functions": Concept(
            name="Exponential Functions",
            description="Functions in the form f(x) = ab^x where a and b are constants and b > 0.",
            difficulty=4,
            time_to_master=180,
            category="Algebra",
            concept_id="exponential_functions",
            examples=["y = 3(2^x)", "A population grows by 5% annually"]
        ),
        "logarithmic_functions": Concept(
            name="Logarithmic Functions",
            description="The inverse of exponential functions, expressing the power to which a base must be raised.",
            difficulty=5,
            time_to_master=240,
            category="Algebra",
            concept_id="logarithmic_functions",
            examples=["y = log₂(x)", "Solve: log(x) = 3"]
        ),
        "function_transformations": Concept(
            name="Function Transformations",
            description="Shifting, stretching, or reflecting the graph of a function.",
            difficulty=4,
            time_to_master=180,
            category="Algebra",
            concept_id="function_transformations",
            examples=["How does the graph of y = f(x) + 3 compare to y = f(x)?", "Graph y = |x - 2| + 1"]
        ),
    }
    
    # Add all concepts to the graph
    for concept_dict in [elementary_concepts, middle_concepts, high_concepts]:
        for concept_id, concept in concept_dict.items():
            graph.add_concept(concept)
    
    # Add prerequisite relationships - Elementary School
    graph.add_prerequisite("counting", "addition")
    graph.add_prerequisite("counting", "subtraction")
    graph.add_prerequisite("addition", "subtraction")
    graph.add_prerequisite("addition", "multiplication")
    graph.add_prerequisite("subtraction", "division")
    graph.add_prerequisite("multiplication", "division")
    graph.add_prerequisite("division", "fractions_intro")
    graph.add_prerequisite("place_value", "addition")
    graph.add_prerequisite("place_value", "subtraction")
    
    # Add related relationships - Elementary School
    graph.add_related("addition", "patterns")
    graph.add_related("multiplication", "patterns")
    
    # Add prerequisite relationships - Middle School
    graph.add_prerequisite("fractions_intro", "variables")
    graph.add_prerequisite("patterns", "variables")
    graph.add_prerequisite("variables", "equations_one_step")
    graph.add_prerequisite("equations_one_step", "equations_two_step")
    graph.add_prerequisite("equations_one_step", "inequalities")
    graph.add_prerequisite("variables", "coordinate_plane")
    graph.add_prerequisite("fractions_intro", "proportional_relationships")
    graph.add_prerequisite("multiplication", "proportional_relationships")
    graph.add_prerequisite("division", "proportional_relationships")
    graph.add_prerequisite("coordinate_plane", "linear_functions")
    graph.add_prerequisite("equations_two_step", "linear_functions")
    graph.add_prerequisite("linear_functions", "systems_linear")
    
    # Add prerequisite relationships - High School
    graph.add_prerequisite("multiplication", "exponents_roots")
    graph.add_prerequisite("variables", "exponents_roots")
    graph.add_prerequisite("variables", "polynomials")
    graph.add_prerequisite("exponents_roots", "polynomials")
    graph.add_prerequisite("polynomials", "factoring")
    graph.add_prerequisite("factoring", "quadratic_functions")
    graph.add_prerequisite("linear_functions", "quadratic_functions")
    graph.add_prerequisite("quadratic_functions", "quadratic_formula")
    graph.add_prerequisite("factoring", "rational_expressions")
    graph.add_prerequisite("division", "rational_expressions")
    graph.add_prerequisite("exponents_roots", "exponential_functions")
    graph.add_prerequisite("exponential_functions", "logarithmic_functions")
    graph.add_prerequisite("linear_functions", "function_transformations")
    graph.add_prerequisite("quadratic_functions", "function_transformations")
    
    # Add some additional related relationships
    graph.add_related("linear_functions", "inequalities")
    graph.add_related("exponents_roots", "quadratic_formula")
    graph.add_related("factoring", "quadratic_formula")
    graph.add_related("rational_expressions", "logarithmic_functions")
    
    return graph


def display_concepts_by_difficulty(graph: KnowledgeGraph) -> None:
    """Display concepts grouped by difficulty level."""
    # Group concepts by difficulty
    concepts_by_difficulty = {1: [], 2: [], 3: [], 4: [], 5: []}
    for concept_id, concept in graph.concepts.items():
        concepts_by_difficulty[concept.difficulty].append(concept)
    
    print("\nALL CONCEPTS BY DIFFICULTY LEVEL:")
    for difficulty in range(1, 6):
        concepts = concepts_by_difficulty[difficulty]
        if concepts:
            print(f"\nDifficulty Level {difficulty}/5:")
            print("-" * 30)
            for concept in sorted(concepts, key=lambda x: x.name):
                print(f"- {concept.name} (Category: {concept.category})")
                if concept.examples:
                    print(f"  Example: {concept.examples[0]}")


def display_concept_relationships(graph: KnowledgeGraph) -> None:
    """Display prerequisite and related relationships for all concepts."""
    print("\nCONCEPT RELATIONSHIPS:")
    print("=" * 30)
    
    for concept_id, concept in sorted(graph.concepts.items(), key=lambda x: x[1].name):
        print(f"\n{concept.name} (Difficulty: {concept.difficulty}/5)")
        
        # Show prerequisites
        if concept.prerequisites:
            prereq_names = [graph.concepts[p].name for p in concept.prerequisites if p in graph.concepts]
            print(f"  Prerequisites: {', '.join(prereq_names)}")
        else:
            print("  Prerequisites: None")
        
        # Show dependent concepts
        if concept.dependents:
            dep_names = [graph.concepts[d].name for d in concept.dependents if d in graph.concepts]
            print(f"  Leads to: {', '.join(dep_names)}")
        else:
            print("  Leads to: None")
        
        # Show related concepts
        if concept.related:
            related_names = [graph.concepts[r].name for r in concept.related if r in graph.concepts]
            print(f"  Related to: {', '.join(related_names)}")


def main():
    """Main function to display the algebra knowledge graph."""
    print("Building K-12 Algebra Knowledge Graph...")
    graph = build_full_algebra_graph()
    
    # Print summary
    print(f"\nK-12 Algebra Knowledge Graph: {len(graph.concepts)} concepts")
    
    # Elementary School (K-5)
    elementary_concepts = [c for c in graph.concepts.values() if c.difficulty <= 2]
    print(f"Elementary School Algebra (K-5): {len(elementary_concepts)} concepts")
    
    # Middle School (6-8)
    middle_concepts = [c for c in graph.concepts.values() if c.difficulty == 3]
    print(f"Middle School Algebra (6-8): {len(middle_concepts)} concepts")
    
    # High School (9-12)
    high_concepts = [c for c in graph.concepts.values() if c.difficulty >= 4]
    print(f"High School Algebra (9-12): {len(high_concepts)} concepts")
    
    # Display detailed information
    display_concepts_by_difficulty(graph)
    display_concept_relationships(graph)


if __name__ == "__main__":
    main() 