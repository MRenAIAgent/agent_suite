"""
Recommender module.

This module defines the recommender that suggests exercises to address knowledge gaps.
"""

from typing import Dict, List, Tuple, Any, Optional
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.exercises.exercise_bank import ExerciseBank
from math_learning.exercises.exercise import Exercise
from math_learning.recommendation.gap_analyzer import GapAnalyzer, GapInfo


class ExerciseRecommendation:
    """A recommendation for an exercise."""
    
    def __init__(self, exercise: Exercise, concept_id: str, reason: str):
        """
        Initialize an exercise recommendation.
        
        Args:
            exercise: The recommended exercise
            concept_id: ID of the concept being targeted
            reason: Explanation for why this exercise was recommended
        """
        self.exercise = exercise
        self.concept_id = concept_id
        self.reason = reason
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "exercise_id": self.exercise.id,
            "exercise_title": self.exercise.title,
            "concept_id": self.concept_id,
            "reason": self.reason
        }


class Recommender:
    """Recommends exercises to address knowledge gaps."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, exercise_bank: ExerciseBank):
        """
        Initialize the recommender.
        
        Args:
            knowledge_graph: The reference knowledge graph
            exercise_bank: The exercise bank
        """
        self.knowledge_graph = knowledge_graph
        self.exercise_bank = exercise_bank
        self.gap_analyzer = GapAnalyzer(knowledge_graph)
        
    def recommend_exercises(self, learning_graph: LearningGraph, 
                          max_exercises: int = 5) -> List[ExerciseRecommendation]:
        """
        Recommend exercises based on the user's learning needs.
        
        Args:
            learning_graph: The user's learning graph
            max_exercises: Maximum number of exercises to recommend
            
        Returns:
            List of recommended exercises
        """
        # Get critical learning gaps
        gaps = self.gap_analyzer.detect_critical_gaps(learning_graph)
        
        # Get learning boundary
        boundary_concepts = set(self.gap_analyzer.find_learning_boundary(learning_graph))
        
        print("\n=== Boundary Concept IDs for Debugging ===")
        for concept_id in boundary_concepts:
            concept = self.knowledge_graph.get_concept(concept_id)
            print(f"{concept.name} ID: {concept_id}")
            
        print("\n=== Gap Concept IDs for Debugging ===")
        for gap in gaps:
            concept = self.knowledge_graph.get_concept(gap.concept_id)
            print(f"{concept.name} ID: {gap.concept_id}")
        
        # Rank concepts by priority
        ranked_candidates = []
        
        # First prioritize concepts at the learning boundary with low mastery
        for concept_id in boundary_concepts:
            concept = self.knowledge_graph.get_concept(concept_id)
            if not concept:
                continue
                
            mastery = learning_graph.get_mastery(concept_id)
            
            # Add to candidates
            ranked_candidates.append({
                "concept_id": concept_id,
                "mastery": mastery,
                "priority": (1.0 - mastery) * 0.8,  # Higher priority for lower mastery
                "is_boundary": True
            })
            
        # Then include other critical gaps not at the boundary
        for gap in gaps:
            concept_id = gap.concept_id
            
            # Skip if already added as a boundary concept
            if concept_id in boundary_concepts:
                continue
                
            # Add to candidates
            ranked_candidates.append({
                "concept_id": concept_id,
                "mastery": gap.mastery,
                "priority": gap.priority * 0.2,  # Lower priority than boundary concepts
                "is_boundary": False
            })
            
        # Sort by priority (highest first)
        ranked_candidates.sort(key=lambda x: x["priority"], reverse=True)
        
        # Generate recommendations
        recommendations = []
        processed_concepts = set()
        
        # Process each candidate concept
        for candidate in ranked_candidates:
            concept_id = candidate["concept_id"]
            
            # Skip if we already processed this concept
            if concept_id in processed_concepts:
                continue
                
            processed_concepts.add(concept_id)
            concept = self.knowledge_graph.get_concept(concept_id)
            mastery = candidate["mastery"]
            is_boundary = candidate["is_boundary"]
            
            print(f"\nSelecting exercises for {concept.name} (ID: {concept_id}, mastery: {mastery:.1%})")
            
            # Skip advanced concepts that are not at the learning boundary
            if not is_boundary and concept.difficulty > 2:
                print(f"Skipping advanced concept {concept.name} as it's not at the learning boundary")
                continue
                
            # Select appropriate exercises
            exercises = self.exercise_bank.select_exercises(concept_id, mastery, count=1)
            
            if not exercises:
                print(f"No exercises found for concept {concept_id}")
                
                # Create a fallback exercise if none exists
                fallback_exercise = Exercise(
                    title="Triangle Properties",
                    problem="Find the missing angle in a triangle where two angles are 45° and 60°.",
                    solution="Missing angle is 75°.",
                    difficulty=2,
                    exercise_id=f"fallback-{concept_id}"
                )
                fallback_exercise.add_concept_relationship(concept_id=concept_id, weight=0.9, relationship_type="primary")
                
                # Use the fallback exercise
                print(f"Using fallback exercises for concept {concept_id}")
                exercises = [fallback_exercise]
                
            # Generate recommendation reason
            reason = self._generate_recommendation_reason(concept, mastery, is_boundary)
            
            # Add recommendation
            recommendations.append(ExerciseRecommendation(
                exercise=exercises[0],
                concept_id=concept_id,
                reason=reason
            ))
            
            # Stop if we have enough recommendations
            if len(recommendations) >= max_exercises:
                break
                
        return recommendations
        
    def recommend_exercises_for_concept(self, concept_id: str, learning_graph: LearningGraph,
                                     count: int = 3) -> List[ExerciseRecommendation]:
        """
        Recommend exercises for a specific concept.
        
        Args:
            concept_id: ID of the concept
            learning_graph: The user's learning graph
            count: Number of exercises to recommend
            
        Returns:
            List of exercise recommendations
        """
        concept = self.knowledge_graph.get_concept(concept_id)
        
        if not concept:
            return []
            
        # Get user's mastery level
        mastery = learning_graph.get_mastery(concept_id)
        
        # Select appropriate exercises
        exercises = self.exercise_bank.select_exercises(concept_id, mastery, count=count)
        
        # Generate recommendations
        recommendations = []
        for exercise in exercises:
            reason = f"This exercise will help you improve your understanding of {concept.name}."
            recommendations.append(ExerciseRecommendation(
                exercise=exercise,
                concept_id=concept_id,
                reason=reason
            ))
            
        return recommendations
        
    def _generate_recommendation_reason(self, concept: Any, mastery: float, is_boundary: bool) -> str:
        """
        Generate a reason for recommending exercises for a concept.
        
        Args:
            concept: The concept
            mastery: User's mastery level
            is_boundary: Whether this concept is at the learning boundary
            
        Returns:
            Recommendation reason
        """
        if is_boundary and mastery <= 0.2:
            return f"You should start learning {concept.name} as it's the next logical concept to study."
        elif is_boundary:
            return f"You should continue learning {concept.name} to build on your current knowledge."
        elif mastery <= 0.2:
            return f"You need to fill a gap in your knowledge of {concept.name} with mastery level of only {mastery:.0%}."
        elif mastery <= 0.5:
            return f"You should strengthen your understanding of {concept.name} (current mastery: {mastery:.0%})."
        else:
            return f"You're making good progress with {concept.name} ({mastery:.0%} mastery). This will help you reach full mastery."
            
    def get_learning_path(self, learning_graph: LearningGraph, 
                        max_concepts: int = 5) -> List[Dict[str, Any]]:
        """
        Generate a recommended learning path.
        
        Args:
            learning_graph: The user's learning graph
            max_concepts: Maximum number of concepts to include in the path
            
        Returns:
            List of concepts in recommended learning order
        """
        # Get next concepts to learn (at the learning boundary)
        next_concepts = self.gap_analyzer.get_next_concepts(learning_graph, count=max_concepts)
        
        path = []
        for concept_id in next_concepts:
            concept = self.knowledge_graph.get_concept(concept_id)
            
            if not concept:
                continue
                
            # Get prerequisites to show context
            prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
            prerequisite_names = [p.name for p in prerequisites]
            
            # Get dependent concepts to show why this is important
            dependents = self.knowledge_graph.get_dependent_concepts(concept_id)
            dependent_names = [d.name for d in dependents]
            
            path.append({
                "concept_id": concept_id,
                "name": concept.name,
                "description": concept.description,
                "prerequisites": prerequisite_names[:3],  # Limit to top 3
                "unlocks": dependent_names[:3]  # Limit to top 3
            })
            
        return path 