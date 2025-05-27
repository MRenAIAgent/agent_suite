"""
Algebra Learning Module

This module builds a comprehensive K-12 algebra learning system with:
1. A knowledge graph of algebra concepts
2. A connected bank of exercises 
3. Utilities for finding appropriate exercises for any concept
"""

import os
import json
from typing import Dict, List, Optional, Tuple

from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.concept import Concept
from exercises.exercise_bank import ExerciseBank
from exercises.exercise import Exercise

# Import our algebra knowledge graph and exercise modules
from knowledge_graph.algebra_graph import build_algebra_knowledge_graph
from exercises.algebra_exercises import create_elementary_exercises
from exercises.middle_algebra_exercises import create_middle_school_exercises
from exercises.high_algebra_exercises import create_high_school_exercises


class AlgebraLearningSystem:
    """A complete algebra learning system with knowledge graph and exercises."""
    
    def __init__(self):
        """Initialize the algebra learning system."""
        self.knowledge_graph = build_algebra_knowledge_graph()
        self.exercise_bank = self._build_complete_exercise_bank()
        
    def _build_complete_exercise_bank(self) -> ExerciseBank:
        """
        Build a complete exercise bank by combining exercises from all levels.
        
        Returns:
            ExerciseBank: The combined exercise bank
        """
        # Create a new combined bank
        combined_bank = ExerciseBank(name="K-12 Algebra Exercises")
        
        # Add exercises from all levels
        elementary_bank = create_elementary_exercises()
        middle_school_bank = create_middle_school_exercises()
        high_school_bank = create_high_school_exercises()
        
        # Combine all exercises
        for bank in [elementary_bank, middle_school_bank, high_school_bank]:
            for exercise in bank.get_all_exercises():
                combined_bank.add_exercise(exercise)
                
        return combined_bank
    
    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """
        Find a concept by name (case-insensitive partial match).
        
        Args:
            name: The name to search for
            
        Returns:
            Concept or None if not found
        """
        name_lower = name.lower()
        for concept in self.knowledge_graph.get_all_concepts():
            if name_lower in concept.name.lower():
                return concept
        return None
    
    def get_learning_path(self, target_concept_id: str) -> List[Concept]:
        """
        Generate a learning path to reach a target concept.
        
        Args:
            target_concept_id: The ID of the target concept
            
        Returns:
            List of concepts in order that should be learned
        """
        if target_concept_id not in self.knowledge_graph.concepts:
            return []
            
        # Get all prerequisites recursively
        prerequisites = set()
        visited = set()
        
        def collect_prerequisites(concept_id):
            if concept_id in visited:
                return
            visited.add(concept_id)
            
            prereqs = self.knowledge_graph.get_prerequisites(concept_id)
            for prereq in prereqs:
                prerequisites.add(prereq.id)
                collect_prerequisites(prereq.id)
        
        collect_prerequisites(target_concept_id)
        
        # Sort prerequisites by difficulty
        sorted_prereqs = [self.knowledge_graph.get_concept(prereq_id) for prereq_id in prerequisites]
        sorted_prereqs.sort(key=lambda x: x.difficulty)
        
        # Add the target concept at the end
        target_concept = self.knowledge_graph.get_concept(target_concept_id)
        learning_path = sorted_prereqs + [target_concept]
        
        return learning_path
    
    def get_exercises_for_learning_path(self, learning_path: List[Concept], 
                                      exercises_per_concept: int = 3) -> Dict[str, List[Exercise]]:
        """
        Get appropriate exercises for each concept in a learning path.
        
        Args:
            learning_path: List of concepts in learning order
            exercises_per_concept: Number of exercises to include per concept
            
        Returns:
            Dictionary mapping concept IDs to lists of exercises
        """
        result = {}
        
        for concept in learning_path:
            # Get exercises for this concept
            exercises = self.exercise_bank.get_exercises_for_concept(concept.id)
            
            # Sort by difficulty
            exercises.sort(key=lambda x: x.difficulty)
            
            # Take the requested number (or all if fewer are available)
            result[concept.id] = exercises[:exercises_per_concept]
            
        return result
    
    def get_concept_details(self, concept_id: str) -> Dict:
        """
        Get detailed information about a concept and related concepts.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            Dictionary with concept details, prerequisites, and related concepts
        """
        concept = self.knowledge_graph.get_concept(concept_id)
        if not concept:
            return {}
            
        prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
        dependents = self.knowledge_graph.get_dependent_concepts(concept_id)
        
        # Get exercises for this concept
        exercises = self.exercise_bank.get_exercises_for_concept(concept_id)
        
        return {
            "concept": concept.to_dict(),
            "prerequisites": [prereq.to_dict() for prereq in prerequisites],
            "dependents": [dep.to_dict() for dep in dependents],
            "exercises_count": len(exercises),
            "difficulty_distribution": self._get_difficulty_distribution(exercises)
        }
    
    def _get_difficulty_distribution(self, exercises: List[Exercise]) -> Dict[int, int]:
        """
        Get the distribution of exercises by difficulty level.
        
        Args:
            exercises: List of exercises
            
        Returns:
            Dictionary mapping difficulty levels to counts
        """
        distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for exercise in exercises:
            distribution[exercise.difficulty] += 1
            
        return distribution
    
    def recommend_next_concepts(self, learned_concept_ids: List[str], 
                              max_recommendations: int = 3) -> List[Tuple[Concept, float]]:
        """
        Recommend next concepts to learn based on already learned concepts.
        
        Args:
            learned_concept_ids: List of concept IDs already learned
            max_recommendations: Maximum number of recommendations to return
            
        Returns:
            List of (concept, relevance_score) tuples
        """
        if not learned_concept_ids:
            # If no concepts are learned, recommend the most basic ones
            basic_concepts = [c for c in self.knowledge_graph.get_all_concepts() 
                             if c.difficulty <= 2 and not c.prerequisites]
            return [(concept, 1.0) for concept in basic_concepts[:max_recommendations]]
        
        # Find all potential next concepts (any concept that has prerequisites
        # that are a subset of learned concepts)
        candidates = []
        
        for concept in self.knowledge_graph.get_all_concepts():
            # Skip already learned concepts
            if concept.id in learned_concept_ids:
                continue
                
            # Check if all prerequisites are satisfied
            prereq_ids = {prereq_id for prereq_id in concept.prerequisites}
            unsatisfied_prereqs = prereq_ids - set(learned_concept_ids)
            
            # If at least half of prerequisites are satisfied, consider it
            if len(unsatisfied_prereqs) <= len(prereq_ids) / 2:
                # Score based on % of prerequisites met and inverse of difficulty
                prereqs_met_ratio = (len(prereq_ids) - len(unsatisfied_prereqs)) / max(1, len(prereq_ids))
                difficulty_factor = 1.0 - (concept.difficulty - 1) / 4  # 1 -> 1.0, 5 -> 0.0
                
                # Calculate score (higher is better)
                score = prereqs_met_ratio * 0.7 + difficulty_factor * 0.3
                
                candidates.append((concept, score))
        
        # Sort by score (highest first) and return top recommendations
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_recommendations]
    
    def save_to_files(self, output_dir: str) -> None:
        """
        Save the complete algebra learning system to files.
        
        Args:
            output_dir: Directory to save files to
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save knowledge graph
        self.knowledge_graph.save_to_file(os.path.join(output_dir, "algebra_knowledge_graph.json"))
        
        # Save exercise bank
        self.exercise_bank.save_to_file(os.path.join(output_dir, "algebra_exercises.json"))
        
        # Save concept statistics
        concept_stats = []
        for concept in self.knowledge_graph.get_all_concepts():
            exercises = self.exercise_bank.get_exercises_for_concept(concept.id)
            concept_stats.append({
                "id": concept.id,
                "name": concept.name,
                "category": concept.category,
                "difficulty": concept.difficulty,
                "prerequisites_count": len(concept.prerequisites),
                "dependents_count": len(concept.dependents),
                "exercises_count": len(exercises),
                "centrality": self.knowledge_graph.calculate_centrality(concept.id)
            })
        
        with open(os.path.join(output_dir, "concept_statistics.json"), 'w') as f:
            json.dump(concept_stats, f, indent=2)


if __name__ == "__main__":
    # Create the algebra learning system
    system = AlgebraLearningSystem()
    
    # Print some statistics
    print(f"Algebra Knowledge Graph: {len(system.knowledge_graph.concepts)} concepts")
    print(f"Algebra Exercise Bank: {len(system.exercise_bank.get_all_exercises())} exercises")
    
    # Identify the most central concepts
    central_concepts = system.knowledge_graph.get_central_concepts(10)
    print("\nMost central concepts:")
    for concept in central_concepts:
        print(f"- {concept.name} (Centrality: {system.knowledge_graph.calculate_centrality(concept.id):.2f})")
    
    # Create a learning path for logarithmic functions
    log_concept = system.get_concept_by_name("Logarithmic")
    if log_concept:
        print(f"\nLearning path for {log_concept.name}:")
        learning_path = system.get_learning_path(log_concept.id)
        for i, concept in enumerate(learning_path, 1):
            print(f"{i}. {concept.name} (Difficulty: {concept.difficulty})")
            exercises = system.exercise_bank.get_exercises_for_concept(concept.id)
            print(f"   - {len(exercises)} exercises available")
    
    # Save the system to files
    system.save_to_files("algebra_learning_system")
    print("\nSaved algebra learning system to algebra_learning_system/ directory") 