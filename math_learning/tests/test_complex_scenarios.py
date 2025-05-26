"""
Test complex scenarios for math learning system.

This module tests more complex scenarios to verify gap identification,
exercise bank functionality, and recommendation accuracy.
"""

import unittest
import os
import sys
import random
from typing import Dict, List, Set

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender


class TestComplexScenarios(unittest.TestCase):
    """Tests for complex learning scenarios."""

    def setUp(self):
        """Set up test fixtures for all tests."""
        # Create a more complex knowledge graph with multiple paths
        self.knowledge_graph, self.concept_ids = self._create_complex_knowledge_graph()
        self.exercise_bank = self._create_diverse_exercises(self.concept_ids)
        self.gap_analyzer = GapAnalyzer(self.knowledge_graph)
        self.recommender = Recommender(self.knowledge_graph, self.exercise_bank)

    def _create_complex_knowledge_graph(self):
        """Create a more complex knowledge graph with multiple learning paths."""
        graph = KnowledgeGraph(name="Advanced Math")
        
        # Core concepts (level 1)
        numbers = Concept(
            name="Numbers",
            description="Basic number concepts and operations.",
            difficulty=1,
            time_to_master=20,
            category="Arithmetic",
            concept_id="numbers-id"
        )
        
        variables = Concept(
            name="Variables",
            description="Using letters to represent unknown quantities.",
            difficulty=1,
            time_to_master=25,
            category="Algebra",
            concept_id="variables-id"
        )
        
        points = Concept(
            name="Points",
            description="Locations in coordinate space.",
            difficulty=1,
            time_to_master=15,
            category="Geometry",
            concept_id="points-id"
        )
        
        # Intermediate concepts (level 2)
        equations = Concept(
            name="Equations",
            description="Mathematical statements asserting equality.",
            difficulty=2,
            time_to_master=40,
            category="Algebra",
            concept_id="equations-id"
        )
        
        functions = Concept(
            name="Functions",
            description="Relations mapping inputs to outputs.",
            difficulty=2,
            time_to_master=45,
            category="Algebra",
            concept_id="functions-id"
        )
        
        lines = Concept(
            name="Lines",
            description="Straight paths extending infinitely.",
            difficulty=2,
            time_to_master=30,
            category="Geometry",
            concept_id="lines-id"
        )
        
        shapes = Concept(
            name="Shapes",
            description="Basic geometric shapes and properties.",
            difficulty=2,
            time_to_master=35,
            category="Geometry",
            concept_id="shapes-id"
        )
        
        # Advanced concepts (level 3)
        linear_equations = Concept(
            name="Linear Equations",
            description="Equations forming straight lines when graphed.",
            difficulty=3,
            time_to_master=50,
            category="Algebra",
            concept_id="linear-equations-id"
        )
        
        graphing = Concept(
            name="Graphing",
            description="Visual representation of mathematical relations.",
            difficulty=3,
            time_to_master=55,
            category="Visualization",
            concept_id="graphing-id"
        )
        
        polygons = Concept(
            name="Polygons",
            description="Closed shapes with straight sides.",
            difficulty=3,
            time_to_master=40,
            category="Geometry",
            concept_id="polygons-id"
        )
        
        # Expert concepts (level 4)
        systems = Concept(
            name="Systems of Equations",
            description="Multiple equations solved simultaneously.",
            difficulty=4,
            time_to_master=70,
            category="Algebra",
            concept_id="systems-id"
        )
        
        transformations = Concept(
            name="Transformations",
            description="Operations that change position, size, or shape.",
            difficulty=4,
            time_to_master=60,
            category="Geometry",
            concept_id="transformations-id"
        )
        
        # Add all concepts to the graph
        for concept in [numbers, variables, points, equations, functions, 
                      lines, shapes, linear_equations, graphing, 
                      polygons, systems, transformations]:
            graph.add_concept(concept)
        
        # Add prerequisite relationships - multiple paths and dependencies
        
        # Core to intermediate
        graph.add_prerequisite(numbers.id, equations.id)
        graph.add_prerequisite(variables.id, equations.id)
        graph.add_prerequisite(variables.id, functions.id)
        graph.add_prerequisite(points.id, lines.id)
        graph.add_prerequisite(points.id, shapes.id)
        
        # Intermediate to advanced
        graph.add_prerequisite(equations.id, linear_equations.id)
        graph.add_prerequisite(functions.id, linear_equations.id)
        graph.add_prerequisite(functions.id, graphing.id)
        graph.add_prerequisite(lines.id, graphing.id)
        graph.add_prerequisite(shapes.id, polygons.id)
        
        # Advanced to expert
        graph.add_prerequisite(linear_equations.id, systems.id)
        graph.add_prerequisite(graphing.id, systems.id)
        graph.add_prerequisite(polygons.id, transformations.id)
        graph.add_prerequisite(lines.id, transformations.id)
        
        # Cross-domain prerequisites
        graph.add_prerequisite(linear_equations.id, transformations.id)  # Algebra -> Geometry
        
        return graph, {
            "Numbers": numbers.id,
            "Variables": variables.id,
            "Points": points.id,
            "Equations": equations.id,
            "Functions": functions.id,
            "Lines": lines.id,
            "Shapes": shapes.id,
            "Linear Equations": linear_equations.id,
            "Graphing": graphing.id,
            "Polygons": polygons.id,
            "Systems of Equations": systems.id,
            "Transformations": transformations.id,
        }
        
    def _create_diverse_exercises(self, concept_ids):
        """Create diverse exercises with varying difficulties and relationships."""
        bank = ExerciseBank(name="Advanced Exercises")
        
        # Helper to create exercises with proper IDs
        def create_exercise(title, problem, solution, difficulty, primary_concept, related_concepts=None):
            if related_concepts is None:
                related_concepts = []
                
            # Create a unique ID based on the title
            exercise_id = f"{title.lower().replace(' ', '-')}-id"
            
            exercise = Exercise(
                title=title,
                problem=problem,
                solution=solution,
                difficulty=difficulty,
                exercise_id=exercise_id
            )
            
            # Add primary concept relationship
            exercise.add_concept_relationship(
                concept_id=concept_ids[primary_concept],
                weight=0.9,
                relationship_type="primary"
            )
            
            # Add relationships to related concepts
            for concept_name, weight in related_concepts:
                exercise.add_concept_relationship(
                    concept_id=concept_ids[concept_name],
                    weight=weight,
                    relationship_type="foundational"
                )
                
            bank.add_exercise(exercise)
            return exercise
        
        # Create exercises for each concept with various relationships
        
        # Numbers exercises
        create_exercise(
            "Basic Arithmetic",
            "Calculate: 24 + 36 × 2 - 15 ÷ 3",
            "91",
            1,
            "Numbers"
        )
        
        create_exercise(
            "Number Properties",
            "Identify all prime numbers between 20 and 40.",
            "23, 29, 31, 37",
            2,
            "Numbers"
        )
        
        # Variables exercises
        create_exercise(
            "Variable Substitution",
            "If x = 5 and y = 3, calculate 2x + 3y.",
            "19",
            1,
            "Variables",
            [("Numbers", 0.5)]
        )
        
        create_exercise(
            "Variable Expressions",
            "Simplify: 3(x + 2) - 2(x - 1)",
            "x + 8",
            2,
            "Variables",
            [("Numbers", 0.4)]
        )
        
        # Points exercises
        create_exercise(
            "Coordinate Points",
            "Plot the points A(3,4), B(-2,5), C(0,-3) on a coordinate plane.",
            "Points plotted correctly.",
            1,
            "Points"
        )
        
        create_exercise(
            "Distance Between Points",
            "Find the distance between points (2,3) and (5,7).",
            "5",
            2,
            "Points",
            [("Numbers", 0.3)]
        )
        
        # Equations exercises
        create_exercise(
            "Solving Basic Equations",
            "Solve for x: 3x + 7 = 22",
            "x = 5",
            2,
            "Equations",
            [("Variables", 0.6), ("Numbers", 0.3)]
        )
        
        create_exercise(
            "Equation Word Problems",
            "A number plus twice the number equals 36. Find the number.",
            "12",
            3,
            "Equations",
            [("Variables", 0.5), ("Numbers", 0.4)]
        )
        
        # Functions exercises
        create_exercise(
            "Function Evaluation",
            "If f(x) = 2x² + 3x - 1, find f(2).",
            "11",
            2,
            "Functions",
            [("Variables", 0.6), ("Numbers", 0.3)]
        )
        
        create_exercise(
            "Function Composition",
            "If f(x) = 2x + 1 and g(x) = x² - 3, find (f ∘ g)(2).",
            "f(g(2)) = f(1) = 3",
            3,
            "Functions",
            [("Variables", 0.5), ("Equations", 0.4)]
        )
        
        # Lines exercises
        create_exercise(
            "Line Drawing",
            "Draw a line passing through points (1,3) and (4,9).",
            "Line drawn correctly.",
            2,
            "Lines",
            [("Points", 0.7)]
        )
        
        create_exercise(
            "Line Equation",
            "Find the equation of a line passing through (2,5) and (4,9).",
            "y = 2x + 1",
            3,
            "Lines",
            [("Points", 0.5), ("Equations", 0.6)]
        )
        
        # Shapes exercises
        create_exercise(
            "Identifying Shapes",
            "Identify the following shapes: △, □, ○, ◇.",
            "Triangle, square, circle, diamond.",
            1,
            "Shapes"
        )
        
        create_exercise(
            "Shape Properties",
            "Calculate the area and perimeter of a rectangle with length 8 and width 5.",
            "Area = 40, Perimeter = 26",
            2,
            "Shapes",
            [("Numbers", 0.4)]
        )
        
        # Linear Equations exercises
        create_exercise(
            "Slope-Intercept Form",
            "Convert the equation 3x - 2y = 6 to slope-intercept form.",
            "y = (3/2)x - 3",
            3,
            "Linear Equations",
            [("Equations", 0.7), ("Lines", 0.5)]
        )
        
        create_exercise(
            "Parallel and Perpendicular Lines",
            "Find the equation of a line passing through (3,4) that is parallel to y = 2x + 1.",
            "y = 2x - 2",
            4,
            "Linear Equations",
            [("Lines", 0.6), ("Equations", 0.5)]
        )
        
        # Graphing exercises
        create_exercise(
            "Function Graphing",
            "Graph the function f(x) = x² - 4.",
            "Parabola with vertex at (0,-4)",
            3,
            "Graphing",
            [("Functions", 0.7), ("Points", 0.4)]
        )
        
        create_exercise(
            "Graphing Linear Equations",
            "Graph the line 2x + 3y = 6.",
            "Line with x-intercept at (3,0) and y-intercept at (0,2)",
            3,
            "Graphing",
            [("Linear Equations", 0.7), ("Lines", 0.5)]
        )
        
        # Polygons exercises
        create_exercise(
            "Triangle Classification",
            "Classify the triangle with sides 3, 4, and 5.",
            "Right triangle",
            3,
            "Polygons",
            [("Shapes", 0.6)]
        )
        
        create_exercise(
            "Polygon Interior Angles",
            "Find the sum of interior angles in a hexagon.",
            "720 degrees",
            3,
            "Polygons",
            [("Shapes", 0.5), ("Numbers", 0.3)]
        )
        
        # Systems of Equations exercises
        create_exercise(
            "Solving Linear Systems",
            "Solve the system: 2x + y = 7, 3x - 2y = 8",
            "x = 3, y = 1",
            4,
            "Systems of Equations",
            [("Linear Equations", 0.8), ("Equations", 0.5)]
        )
        
        create_exercise(
            "Systems Word Problems",
            "A theater sold 200 tickets for a total of $1,750. Adult tickets cost $10 and child tickets cost $5. How many of each type were sold?",
            "150 adult tickets and 50 child tickets",
            4,
            "Systems of Equations",
            [("Linear Equations", 0.7), ("Equations", 0.5)]
        )
        
        # Transformations exercises
        create_exercise(
            "Reflecting Points",
            "Reflect the point (3,4) across the y-axis.",
            "(-3,4)",
            3,
            "Transformations",
            [("Points", 0.6), ("Lines", 0.4)]
        )
        
        create_exercise(
            "Rotation of Shapes",
            "Rotate a triangle with vertices at (0,0), (2,0), and (1,2) by 90° counterclockwise around the origin.",
            "New vertices: (0,0), (0,2), and (-2,1)",
            4,
            "Transformations",
            [("Polygons", 0.7), ("Points", 0.5)]
        )
        
        return bank

    def test_multi_path_learning_boundary(self):
        """Test learning boundary identification with multiple possible learning paths."""
        # Create a user with knowledge of core concepts
        learning_graph = LearningGraph(user_id="multi_path_user")
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.8)
        learning_graph.set_mastery(self.concept_ids["Points"], 0.8)
        
        # Find learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        
        # Boundary should contain Equations, Functions, Lines, and Shapes
        # These are all concepts that have prerequisites satisfied
        self.assertEqual(len(boundary), 4)
        boundary_names = [self.knowledge_graph.get_concept(cid).name for cid in boundary]
        self.assertIn("Equations", boundary_names)
        self.assertIn("Functions", boundary_names)
        self.assertIn("Lines", boundary_names)
        self.assertIn("Shapes", boundary_names)
        
        # Test recommendations
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=3)
        self.assertEqual(len(recommendations), 3)
        
        # All recommended exercises should be for concepts at the boundary
        recommended_concepts = [rec.concept_id for rec in recommendations]
        self.assertTrue(all(concept_id in boundary for concept_id in recommended_concepts))

    def test_partial_prerequisites_gap(self):
        """Test identification of concepts with partially met prerequisites."""
        # Create a user with partial knowledge of prerequisites for linear equations
        learning_graph = LearningGraph(user_id="partial_prereq_user")
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Equations"], 0.8)
        # Missing Functions which is also a prerequisite for Linear Equations
        
        # Detect gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        
        # Get names of gap concepts for easier testing
        gap_names = [self.knowledge_graph.get_concept(gap.concept_id).name for gap in gaps]
        
        # Functions should be a high priority gap as it blocks Linear Equations
        self.assertIn("Functions", gap_names)
        
        # Test learning boundary - should not include Linear Equations yet
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        boundary_names = [self.knowledge_graph.get_concept(cid).name for cid in boundary]
        self.assertIn("Functions", boundary_names)
        self.assertNotIn("Linear Equations", boundary_names)
        
        # Now master Functions
        learning_graph.set_mastery(self.concept_ids["Functions"], 0.8)
        
        # Linear Equations should now be in the boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        boundary_names = [self.knowledge_graph.get_concept(cid).name for cid in boundary]
        self.assertIn("Linear Equations", boundary_names)

    def test_cross_domain_prerequisites(self):
        """Test handling of prerequisites across different knowledge domains."""
        # Create user with knowledge spanning multiple domains
        learning_graph = LearningGraph(user_id="cross_domain_user")
        
        # Algebra track
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Equations"], 0.8)
        learning_graph.set_mastery(self.concept_ids["Functions"], 0.8)
        learning_graph.set_mastery(self.concept_ids["Linear Equations"], 0.8)
        
        # Geometry track - partial
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)
        # Missing Shapes and Polygons
        
        # Verify transformations requires both tracks
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        boundary_names = [self.knowledge_graph.get_concept(cid).name for cid in boundary]
        
        # Should have Graphing from algebra track
        self.assertIn("Graphing", boundary_names)
        # Should have Shapes from geometry track
        self.assertIn("Shapes", boundary_names)
        # Should NOT have Transformations yet (missing Polygons)
        self.assertNotIn("Transformations", boundary_names)
        
        # Complete missing geometry prerequisites
        learning_graph.set_mastery(self.concept_ids["Shapes"], 0.8)
        learning_graph.set_mastery(self.concept_ids["Polygons"], 0.8)
        
        # Now check if Transformations is in boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        boundary_names = [self.knowledge_graph.get_concept(cid).name for cid in boundary]
        self.assertIn("Transformations", boundary_names)

    def test_exercise_recommendation_relevance(self):
        """Test that recommended exercises are relevant to the user's knowledge state."""
        # Create user with specific knowledge
        learning_graph = LearningGraph(user_id="exercise_relevance_user")
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Points"], 0.5)  # Partial mastery
        
        # Get recommendations
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=5)
        
        # Get concept names for each recommendation
        rec_concepts = {
            self.knowledge_graph.get_concept(rec.concept_id).name: rec.exercise.title 
            for rec in recommendations
        }
        
        # Should recommend Points exercises for partial mastery
        self.assertIn("Points", rec_concepts)
        
        # Should recommend Lines, Shapes, Equations, or Functions for next steps
        boundary_concepts = ["Lines", "Shapes", "Equations", "Functions"]
        boundary_recs = [c for c in rec_concepts.keys() if c in boundary_concepts]
        self.assertGreater(len(boundary_recs), 0)
        
        # Should NOT recommend advanced concepts
        advanced_concepts = ["Linear Equations", "Graphing", "Polygons", "Systems of Equations", "Transformations"]
        for concept in advanced_concepts:
            self.assertNotIn(concept, rec_concepts)

    def test_non_linear_learning_path(self):
        """Test non-linear learning path with multiple options at each step."""
        # Create new user
        learning_graph = LearningGraph(user_id="nonlinear_path_user")
        
        # Initial state - no knowledge
        path = self.recommender.get_learning_path(learning_graph)
        initial_concepts = [step["name"] for step in path]
        
        # Should recommend foundation concepts
        for concept in ["Numbers", "Variables", "Points"]:
            self.assertIn(concept, initial_concepts)
            
        # Simulate learning Numbers first
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.8)
        
        # Get new learning path
        path = self.recommender.get_learning_path(learning_graph)
        concepts_after_numbers = [step["name"] for step in path]
        
        # Variables should still be recommended
        self.assertIn("Variables", concepts_after_numbers)
        
        # Simulate learning Variables instead of Points
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.8)
        
        # Get updated learning path
        path = self.recommender.get_learning_path(learning_graph)
        concepts_after_vars = [step["name"] for step in path]
        
        # Should include Equations and Functions
        self.assertTrue("Equations" in concepts_after_vars or "Functions" in concepts_after_vars)
        
        # Also should still include Points as an option
        self.assertIn("Points", concepts_after_vars)

    def test_learning_boundary_prioritization(self):
        """Test that learning boundary concepts are prioritized correctly."""
        # Create user with varied mastery levels
        learning_graph = LearningGraph(user_id="boundary_priority_user")
        learning_graph.set_mastery(self.concept_ids["Numbers"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Variables"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)
        learning_graph.set_mastery(self.concept_ids["Equations"], 0.4)  # Low mastery
        learning_graph.set_mastery(self.concept_ids["Functions"], 0.2)  # Very low mastery
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.7)      # Higher mastery
        learning_graph.set_mastery(self.concept_ids["Shapes"], 0.8)     # Higher mastery
        
        # Recommendations should prioritize concepts with lower mastery levels
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=4)
        rec_concepts = [self.knowledge_graph.get_concept(rec.concept_id).name for rec in recommendations]
        
        # Functions should be recommended before Shapes and Lines (lower mastery)
        functions_index = rec_concepts.index("Functions") if "Functions" in rec_concepts else -1
        shapes_index = rec_concepts.index("Shapes") if "Shapes" in rec_concepts else -1
        lines_index = rec_concepts.index("Lines") if "Lines" in rec_concepts else -1
        
        # Verify Functions is in recommendations
        self.assertNotEqual(functions_index, -1, "Functions should be in recommendations")
        
        # If Shapes is in recommendations, Functions should come first
        if shapes_index != -1:
            self.assertLess(functions_index, shapes_index, 
                          "Functions (20% mastery) should be recommended before Shapes (80% mastery)")
            
        # If Lines is in recommendations, Functions should come first
        if lines_index != -1:
            self.assertLess(functions_index, lines_index,
                          "Functions (20% mastery) should be recommended before Lines (70% mastery)")
            
        # Equations should be recommended before Shapes (if both present)
        equations_index = rec_concepts.index("Equations") if "Equations" in rec_concepts else -1
        if equations_index != -1 and shapes_index != -1:
            self.assertLess(equations_index, shapes_index,
                          "Equations (40% mastery) should be recommended before Shapes (80% mastery)")

    def test_random_user_simulation(self):
        """Test system with a simulated random user knowledge state."""
        # Create user with random mastery of concepts
        learning_graph = LearningGraph(user_id="random_user")
        
        # Set random mastery for some concepts
        all_concepts = list(self.concept_ids.items())
        random.shuffle(all_concepts)
        
        # Master some random concepts
        for i, (name, concept_id) in enumerate(all_concepts):
            if i < len(all_concepts) // 3:  # Master about 1/3 of concepts
                learning_graph.set_mastery(concept_id, random.uniform(0.7, 0.9))
            elif i < len(all_concepts) // 2:  # Partial mastery for some
                learning_graph.set_mastery(concept_id, random.uniform(0.3, 0.6))
        
        # Analyze gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        
        # Find learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        
        # Get recommendations
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=3)
        
        # Verify that recommended concepts are at the learning boundary
        rec_concepts = [rec.concept_id for rec in recommendations]
        self.assertTrue(any(concept_id in boundary for concept_id in rec_concepts))
        
        # Verify that recommendations include appropriate exercises
        for rec in recommendations:
            exercise = rec.exercise
            concept_id = rec.concept_id
            self.assertIn(concept_id, exercise.get_all_concept_ids())


if __name__ == "__main__":
    unittest.main() 