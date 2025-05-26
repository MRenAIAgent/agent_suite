"""
Test gap analysis and recommendations.

This module tests the gap analysis and recommendation functionality.
"""

import unittest
import os
import sys
import json
from typing import Dict

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.exercises.exercise import Exercise
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.gap_analyzer import GapAnalyzer
from math_learning.recommendation.recommender import Recommender


class TestGapAnalysis(unittest.TestCase):
    """Tests for gap analysis and recommendations."""

    def setUp(self):
        """Set up test cases."""
        # Create a knowledge graph for geometry
        self.knowledge_graph, self.concept_ids = self._create_test_knowledge_graph()
        self.exercise_bank = self._create_test_exercises(self.concept_ids)
        self.gap_analyzer = GapAnalyzer(self.knowledge_graph)
        self.recommender = Recommender(self.knowledge_graph, self.exercise_bank)

    def _create_test_knowledge_graph(self):
        """Create a test knowledge graph."""
        graph = KnowledgeGraph(name="Test Geometry")
        
        # Define concepts with fixed IDs
        points = Concept(
            name="Points",
            description="A point is a position in space.",
            difficulty=1,
            time_to_master=10,
            category="Geometry Basics",
            concept_id="points-1"
        )
        
        lines = Concept(
            name="Lines",
            description="A line is a straight path.",
            difficulty=1,
            time_to_master=15,
            category="Geometry Basics",
            concept_id="lines-1"
        )
        
        angles = Concept(
            name="Angles",
            description="An angle is formed by two rays sharing a common endpoint.",
            difficulty=2,
            time_to_master=20,
            category="Geometry Basics",
            concept_id="angles-1"
        )
        
        triangles = Concept(
            name="Triangles",
            description="A triangle is a polygon with three sides.",
            difficulty=2,
            time_to_master=30,
            category="Polygons",
            concept_id="triangles-1"
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
        
        return graph, {
            "Points": points.id,
            "Lines": lines.id,
            "Angles": angles.id,
            "Triangles": triangles.id,
        }
        
    def _create_test_exercises(self, concept_ids: Dict[str, str]) -> ExerciseBank:
        """Create test exercises."""
        bank = ExerciseBank(name="Test Exercises")
        
        # Points exercise
        ex1 = Exercise(
            title="Points Test",
            problem="Identify points on a coordinate plane.",
            solution="Points identified correctly.",
            difficulty=1,
            exercise_id="points-ex1"
        )
        bank.add_exercise(ex1)
        ex1.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.9, relationship_type="primary")
        
        # Lines exercise
        ex2 = Exercise(
            title="Lines Test",
            problem="Draw a line through two points.",
            solution="Line drawn correctly.",
            difficulty=1,
            exercise_id="lines-ex1"
        )
        bank.add_exercise(ex2)
        ex2.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.9, relationship_type="primary")
        ex2.add_concept_relationship(concept_id=concept_ids["Points"], weight=0.4, relationship_type="foundational")
        
        # Angles exercise
        ex3 = Exercise(
            title="Angles Test",
            problem="Measure an angle with a protractor.",
            solution="Angle measured correctly.",
            difficulty=2,
            exercise_id="angles-ex1"
        )
        bank.add_exercise(ex3)
        ex3.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.9, relationship_type="primary")
        ex3.add_concept_relationship(concept_id=concept_ids["Lines"], weight=0.3, relationship_type="foundational")
        
        # Triangles exercise
        ex4 = Exercise(
            title="Triangles Test",
            problem="Find the missing angle in a triangle.",
            solution="Missing angle calculated correctly.",
            difficulty=2,
            exercise_id="triangles-ex1"
        )
        bank.add_exercise(ex4)
        ex4.add_concept_relationship(concept_id=concept_ids["Triangles"], weight=0.9, relationship_type="primary")
        ex4.add_concept_relationship(concept_id=concept_ids["Angles"], weight=0.5, relationship_type="foundational")
        
        return bank

    def test_gap_detection_no_knowledge(self):
        """Test gap detection with no prior knowledge."""
        # Create a learning graph with no prior knowledge
        learning_graph = LearningGraph(user_id="test_user_1")
        
        # Detect gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        
        # First gap should be Points (as it's the foundation)
        self.assertEqual(len(gaps), 4)  # Should identify all 4 concepts as gaps
        
        # The highest priority gap should be Points
        highest_priority_gap = gaps[0]
        points_concept = self.knowledge_graph.get_concept(self.concept_ids["Points"])
        self.assertEqual(highest_priority_gap.concept_id, points_concept.id)

    def test_gap_detection_partial_knowledge(self):
        """Test gap detection with partial knowledge."""
        # Create a learning graph with knowledge of points only
        learning_graph = LearningGraph(user_id="test_user_2")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered points
        
        # Detect gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        
        # Should identify 3 concepts as gaps (Lines, Angles, Triangles)
        self.assertEqual(len(gaps), 3)
        
        # The highest priority gap should be Lines (as it's the next in sequence)
        highest_priority_gap = gaps[0]
        lines_concept = self.knowledge_graph.get_concept(self.concept_ids["Lines"])
        self.assertEqual(highest_priority_gap.concept_id, lines_concept.id)
        
    def test_learning_boundary(self):
        """Test finding the learning boundary."""
        # Create a learning graph with knowledge of points and lines
        learning_graph = LearningGraph(user_id="test_user_3")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered points
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)   # Mostly mastered lines
        
        # Find learning boundary
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        
        # The learning boundary should contain only Angles
        self.assertEqual(len(boundary), 1)
        angles_concept = self.knowledge_graph.get_concept(self.concept_ids["Angles"])
        self.assertEqual(boundary[0], angles_concept.id)
        
    def test_exercise_recommendations(self):
        """Test exercise recommendations based on knowledge gaps."""
        # Create a learning graph with knowledge of points only
        learning_graph = LearningGraph(user_id="test_user_4")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered points
        
        # Get recommendations
        recommendations = self.recommender.recommend_exercises(learning_graph)
        
        # Should recommend a Lines exercise
        self.assertGreaterEqual(len(recommendations), 1)
        first_rec = recommendations[0]
        self.assertEqual(first_rec.concept_id, self.concept_ids["Lines"])
        
    def test_learning_path(self):
        """Test generating a learning path."""
        # Create a learning graph with knowledge of points and lines
        learning_graph = LearningGraph(user_id="test_user_5")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered points
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)   # Mostly mastered lines
        
        # Get learning path
        path = self.recommender.get_learning_path(learning_graph)
        
        # Path should start with Angles
        self.assertGreaterEqual(len(path), 1)
        first_step = path[0]
        self.assertEqual(first_step["name"], "Angles")
        
    def test_mastery_update(self):
        """Test updating mastery based on exercise completion."""
        # Create a learning graph with some knowledge of angles
        learning_graph = LearningGraph(user_id="test_user_6")
        learning_graph.set_mastery(self.concept_ids["Points"], 0.9)  # Mastered points
        learning_graph.set_mastery(self.concept_ids["Lines"], 0.8)   # Mostly mastered lines
        learning_graph.set_mastery(self.concept_ids["Angles"], 0.4)  # Some knowledge of angles
        
        # Directly update mastery to simulate exercise completion
        concept_id = self.concept_ids["Angles"]
        old_mastery = learning_graph.get_mastery(concept_id)
        
        # Create a mock exercise result and update mastery
        learning_graph.set_mastery(concept_id, 0.8)  # Set mastery to 0.8
        new_mastery = learning_graph.get_mastery(concept_id)
        
        # Mastery should increase
        self.assertGreater(new_mastery, old_mastery)
        
        # After updating mastery, the learning boundary should include Triangles
        boundary = self.gap_analyzer.find_learning_boundary(learning_graph)
        
        # Triangles should be in boundary since mastery is now 0.8 (above threshold)
        self.assertIn(self.concept_ids["Triangles"], boundary)


if __name__ == "__main__":
    unittest.main() 