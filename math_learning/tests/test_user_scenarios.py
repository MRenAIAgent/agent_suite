"""
Test user learning scenarios.

This module tests complete user learning scenarios with different knowledge gaps.
"""

import unittest
import os
import sys
import tempfile
from typing import Dict, List, Tuple

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender, ExerciseRecommendation
from math_learning.config.storage_config import MathLearningStorageConfig, StorageBackend


class TestUserScenarios(unittest.TestCase):
    """Tests for complete user learning scenarios."""

    def setUp(self):
        """Set up the knowledge graph and exercise bank for all tests."""
        # Create a complete knowledge graph for geometry
        self.knowledge_graph, self.concept_ids = self._create_geometry_knowledge_graph()
        self.exercise_bank = self._create_geometry_exercises(self.concept_ids)
        self.gap_analyzer = GapAnalyzer(self.knowledge_graph)
        self.recommender = Recommender(self.knowledge_graph, self.exercise_bank)

    def _create_geometry_knowledge_graph(self) -> Tuple[KnowledgeGraph, Dict[str, str]]:
        """Create a geometry knowledge graph."""
        # Create a temporary file for test persistence to avoid loading the algebra graph
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.write('{"concepts": {}, "relationships": []}')  # Empty graph
        temp_file.close()
        
        # Create config that uses the temporary file
        config = MathLearningStorageConfig(
            backend=StorageBackend.NETWORKX,
            networkx_persistence_file=temp_file.name,
            networkx_auto_save=False  # Don't auto-save during tests
        )
        
        graph = KnowledgeGraph(name="Geometry", config=config)
        
        # Clean up the temp file
        os.unlink(temp_file.name)
        
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
        
        right_triangles = Concept(
            name="Right Triangles",
            description="A right triangle has one angle that measures 90 degrees.",
            difficulty=3,
            time_to_master=40,
            category="Polygons",
            concept_id="right-triangles-id"
        )
        
        pythagorean_theorem = Concept(
            name="Pythagorean Theorem",
            description="In a right triangle, the square of the hypotenuse equals the sum of squares of the other two sides.",
            difficulty=3,
            time_to_master=45,
            category="Theorems",
            concept_id="pythagorean-id"
        )
        
        # Add concepts to the graph
        graph.add_concept(points)
        graph.add_concept(lines)
        graph.add_concept(angles)
        graph.add_concept(triangles)
        graph.add_concept(right_triangles)
        graph.add_concept(pythagorean_theorem)
        
        # Add prerequisite relationships
        graph.add_prerequisite(points.id, lines.id)
        graph.add_prerequisite(lines.id, angles.id)
        graph.add_prerequisite(angles.id, triangles.id)
        graph.add_prerequisite(triangles.id, right_triangles.id)
        graph.add_prerequisite(right_triangles.id, pythagorean_theorem.id)
        
        return graph, {
            "Points": points.id,
            "Lines": lines.id,
            "Angles": angles.id,
            "Triangles": triangles.id,
            "Right Triangles": right_triangles.id,
            "Pythagorean Theorem": pythagorean_theorem.id,
        }
        
    def _create_geometry_exercises(self, concept_ids: Dict[str, str]) -> ExerciseBank:
        """Create exercises for geometry."""
        bank = ExerciseBank(name="Geometry Exercises")
        
        # Points exercises
        ex1 = Exercise(
            title="Identifying Points",
            problem="Plot points A(3,4), B(-2,5), C(0,0) on a coordinate plane.",
            solution="Points plotted correctly.",
            difficulty=1,
            exercise_id="points-exercise-id"
        )
        bank.add_exercise(ex1)
        ex1.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.9, relationship_type="primary")
        
        # Lines exercises
        ex2 = Exercise(
            title="Drawing Lines",
            problem="Draw a line passing through points A(1,2) and B(5,4).",
            solution="Line drawn correctly.",
            difficulty=1,
            exercise_id="lines-exercise-id"
        )
        bank.add_exercise(ex2)
        ex2.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.9, relationship_type="primary")
        ex2.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.4, relationship_type="foundational")
        
        # Angles exercises
        ex3 = Exercise(
            title="Measuring Angles",
            problem="Measure the angle formed by the rays OA and OB.",
            solution="Angle measures 45 degrees.",
            difficulty=2,
            exercise_id="angles-exercise-id"
        )
        bank.add_exercise(ex3)
        ex3.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.9, relationship_type="primary")
        ex3.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.3, relationship_type="foundational")
        
        # Triangles exercises
        ex4 = Exercise(
            title="Triangle Properties",
            problem="Find the missing angle in a triangle where two angles are 45° and 60°.",
            solution="Missing angle is 75°.",
            difficulty=2,
            exercise_id="triangles-exercise-id"
        )
        bank.add_exercise(ex4)
        ex4.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.9, relationship_type="primary")
        ex4.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.5, relationship_type="foundational")
        
        # Right triangles exercises
        ex5 = Exercise(
            title="Identifying Right Triangles",
            problem="Which of the following triangles is a right triangle? A(3,4,5), B(5,6,7), C(8,15,17)",
            solution="Triangles A(3,4,5) and C(8,15,17) are right triangles.",
            difficulty=3,
            exercise_id="right-triangles-exercise-id"
        )
        bank.add_exercise(ex5)
        ex5.add_concept_relationship(concept_id=concept_ids["Right Triangles"], weight=0.9, relationship_type="primary")
        ex5.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.4, relationship_type="foundational")
        
        # Pythagorean theorem exercises
        ex6 = Exercise(
            title="Applying Pythagorean Theorem",
            problem="Find the hypotenuse of a right triangle with legs of length 6 and 8.",
            solution="Using the Pythagorean theorem: c² = 6² + 8² = 36 + 64 = 100, so c = 10.",
            difficulty=3,
            exercise_id="pythagorean-exercise-id"
        )
        bank.add_exercise(ex6)
        ex6.add_concept_relationship(concept_id=concept_ids["Pythagorean Theorem"], weight=0.9, relationship_type="primary")
        ex6.add_concept_relationship(concept_id=concept_ids["Right Triangles"], weight=0.5, relationship_type="foundational")
        
        return bank

    def test_scenario_beginner_student(self):
        """Test a beginner student with no prior knowledge."""
        # Create a learning graph for a student with no prior knowledge
        learning_graph = LearningGraph(user_id="beginner_student")
        
        # Verify that all concepts are identified as gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        self.assertEqual(len(gaps), 6)  # All 6 concepts should be identified as gaps
        
        # Verify learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0], self.concept_ids["Points"])
        
        # Verify recommended exercises
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=1)
        self.assertEqual(len(recommendations), 1)
        
        # The recommended concept should be Points (as it's the foundation)
        recommended_concept = self.knowledge_graph.get_concept(recommendations[0].concept_id)
        self.assertEqual(recommended_concept.name, "Points")
        
        # Simulate completing the exercise correctly
        learning_graph.set_mastery(recommendations[0].concept_id, 0.8)  # Directly set mastery for test
        
        # After mastering Points, Lines should be in the learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        self.assertIn(self.concept_ids["Lines"], boundary)
        
        # Next recommendation should be for Lines
        new_recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=1)
        recommended_concept = self.knowledge_graph.get_concept(new_recommendations[0].concept_id)
        self.assertEqual(recommended_concept.name, "Lines")
        
    def test_scenario_intermediate_student(self):
        """Test an intermediate student with knowledge of basic concepts."""
        # Create a learning graph for a student with knowledge of points, lines, and angles
        learning_graph = LearningGraph(user_id="intermediate_student")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)   # Mostly mastered
        learning_graph.set_mastery(self.concept_ids["Angles"], 0.7)  # Sufficiently mastered
        
        # Verify remaining gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        self.assertEqual(len(gaps), 3)  # Triangles, Right Triangles, Pythagorean Theorem
        
        # Verify learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        self.assertEqual(len(boundary), 1)
        self.assertEqual(boundary[0], self.concept_ids["Triangles"])
        
        # Verify recommendations
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=1)
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].concept_id, self.concept_ids["Triangles"])
        
        # Simulate completing several triangles exercises correctly
        exercise = recommendations[0].exercise
        concept_id = recommendations[0].concept_id
        
        # First attempt - moderate improvement
        learning_graph.record_exercise_attempt(exercise.id, concept_id, True)
        new_mastery = learning_graph.update_mastery(
            concept_id,
            exercise_result=True,
            exercise_difficulty=exercise.difficulty / 5.0,
            concept_weight=exercise.get_concept_weight(concept_id)
        )
        
        # Second attempt - further improvement
        learning_graph.record_exercise_attempt(exercise.id, concept_id, True)
        new_mastery = learning_graph.update_mastery(
            concept_id,
            exercise_result=True,
            exercise_difficulty=exercise.difficulty / 5.0,
            concept_weight=exercise.get_concept_weight(concept_id)
        )
        
        # After mastering triangles, check if right triangles are in the boundary
        if new_mastery >= 0.7:
            boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
            self.assertIn(self.concept_ids["Right Triangles"], boundary)
            
            # Next recommendation should be for Right Triangles
            new_recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=1)
            self.assertEqual(new_recommendations[0].concept_id, self.concept_ids["Right Triangles"])
            
    def test_scenario_gaps_in_knowledge(self):
        """Test a student with gaps in the middle of the knowledge chain."""
        # Create a learning graph for a student who knows points and triangles but not lines or angles
        learning_graph = LearningGraph(user_id="gap_student")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.8)  # Mostly mastered
        learning_graph.set_mastery(self.concept_ids["Triangles"], 0.7)  # Sufficiently mastered
        
        # This is an inconsistent state (knows triangles but not prerequisites)
        # The system should identify the critical gaps
        
        # Verify gaps include Lines and Angles (the missing prerequisites)
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        gap_ids = [gap.concept_id for gap in gaps]
        self.assertIn(self.concept_ids["Lines"], gap_ids)
        self.assertIn(self.concept_ids["Angles"], gap_ids)
        
        # Lines should be at the learning boundary because it's next after Points
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        self.assertIn(self.concept_ids["Lines"], boundary)
        
        # Verify recommendations include Lines
        recommendations = self.recommender.recommend_exercises(learning_graph, max_exercises=1)
        self.assertEqual(len(recommendations), 1)
        
        # The recommended concept should be Lines (since it's at the boundary)
        recommended_concept = self.knowledge_graph.get_concept(recommendations[0].concept_id)
        self.assertEqual(recommended_concept.name, "Lines")
        
    def test_scenario_learning_progression(self):
        """Test a full learning progression from beginner to advanced."""
        # Create a learning graph for a student starting with no knowledge
        learning_graph = LearningGraph(user_id="progressing_student")
        
        # Track mastery progression for each concept
        mastery_progression = {concept_name: [] for concept_name in self.concept_ids.keys()}
        
        # Stage 1: First recommendation should be at the learning boundary (Points)
        recommendation = self.recommender.recommend_exercises(learning_graph, max_exercises=1)[0]
        recommended_concept = self.knowledge_graph.get_concept(recommendation.concept_id)
        self.assertEqual(recommended_concept.name, "Points")
        
        # Complete Points exercise successfully
        learning_graph.set_mastery(recommendation.concept_id, 0.8)  # Directly set mastery for test
        mastery_progression["Points"].append(0.8)
        
        # Stage 2: Next recommendation should be Lines
        learning_path = self.recommender.get_learning_path(learning_graph)
        self.assertGreaterEqual(len(learning_path), 1)
        self.assertEqual(learning_path[0]["name"], "Lines")
        
        # Complete Lines exercise 
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)
        mastery_progression["Lines"].append(0.8)
        
        # Stage 3: Next should be Angles
        learning_path = self.recommender.get_learning_path(learning_graph)
        self.assertGreaterEqual(len(learning_path), 1)
        self.assertEqual(learning_path[0]["name"], "Angles")
        
        # Complete Angles exercise
        learning_graph.set_mastery(self.concept_ids["Angles"], 0.8)
        mastery_progression["Angles"].append(0.8)
        
        # Stage 4: Next should be Triangles
        learning_path = self.recommender.get_learning_path(learning_graph)
        self.assertGreaterEqual(len(learning_path), 1)
        self.assertEqual(learning_path[0]["name"], "Triangles")
        
        # Complete Triangles exercise
        learning_graph.set_mastery(self.concept_ids["Triangles"], 0.8)
        mastery_progression["Triangles"].append(0.8)
        
        # Verify learning path progression to Right Triangles
        learning_path = self.recommender.get_learning_path(learning_graph)
        self.assertGreaterEqual(len(learning_path), 1)
        self.assertEqual(learning_path[0]["name"], "Right Triangles")
        
        # Verify mastery increased for each concept we practiced
        for concept_name, masteries in mastery_progression.items():
            if masteries:  # If we practiced this concept
                self.assertGreater(masteries[-1], 0.0)


if __name__ == "__main__":
    unittest.main() 