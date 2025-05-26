"""
User Model module.

This module defines the learning graph that represents a user's knowledge state.
"""

import json
import datetime
from typing import Dict, List, Optional, Set, Any, Tuple


class LearningGraph:
    """A learning graph representing a user's mastery of concepts."""

    def __init__(self, user_id: str = "", name: str = ""):
        """
        Initialize a learning graph for a user.

        Args:
            user_id: Identifier for the user
            name: Optional name for the learning graph
        """
        self.user_id = user_id
        self.name = name
        self.concept_mastery: Dict[str, Dict[str, Any]] = {}
        self.exercise_history: Dict[str, List[Dict[str, Any]]] = {}
        
    def set_mastery(self, concept_id: str, mastery_level: float, confidence: float = 0.8) -> None:
        """
        Set the mastery level for a concept.

        Args:
            concept_id: The ID of the concept
            mastery_level: The mastery level (0.0-1.0)
            confidence: Confidence in the mastery assessment (0.0-1.0)
        """
        mastery_level = max(0.0, min(1.0, mastery_level))  # Ensure 0.0-1.0 range
        confidence = max(0.0, min(1.0, confidence))  # Ensure 0.0-1.0 range
        
        if concept_id not in self.concept_mastery:
            self.concept_mastery[concept_id] = {
                "mastery": mastery_level,
                "confidence": confidence,
                "last_practiced": datetime.datetime.now().isoformat(),
                "history": []
            }
        else:
            # Add current mastery to history
            self.concept_mastery[concept_id]["history"].append({
                "mastery": self.concept_mastery[concept_id]["mastery"],
                "confidence": self.concept_mastery[concept_id]["confidence"],
                "timestamp": self.concept_mastery[concept_id]["last_practiced"]
            })
            
            # Update current mastery
            self.concept_mastery[concept_id]["mastery"] = mastery_level
            self.concept_mastery[concept_id]["confidence"] = confidence
            self.concept_mastery[concept_id]["last_practiced"] = datetime.datetime.now().isoformat()
        
    def update_mastery(self, concept_id: str, exercise_result: bool, 
                     exercise_difficulty: float, concept_weight: float) -> float:
        """
        Update the mastery level for a concept based on exercise performance.

        Args:
            concept_id: The ID of the concept
            exercise_result: Whether the exercise was completed correctly
            exercise_difficulty: The difficulty of the exercise (0.0-1.0)
            concept_weight: How strongly the exercise tests the concept (0.0-1.0)

        Returns:
            The updated mastery level
        """
        # Get current mastery or default to 0.0
        current_mastery = self.get_mastery(concept_id)
        current_confidence = self.get_confidence(concept_id)
        
        # Adjust likelihood based on difficulty
        likelihood_correct = 0.5 + (1 - exercise_difficulty) * 0.5
        
        if exercise_result:
            # Correct answer: increase mastery more for difficult exercises
            posterior_mastery = current_mastery + (1 - current_mastery) * likelihood_correct * concept_weight
            # Increase confidence
            posterior_confidence = min(1.0, current_confidence + 0.1 * concept_weight)
        else:
            # Incorrect answer: decrease mastery more for easier exercises
            posterior_mastery = current_mastery * (1 - likelihood_correct) * concept_weight
            # Decrease confidence slightly
            posterior_confidence = max(0.1, current_confidence - 0.2 * concept_weight)
            
        # Update mastery
        self.set_mastery(concept_id, posterior_mastery, posterior_confidence)
        
        return posterior_mastery
        
    def record_exercise_attempt(self, exercise_id: str, concept_id: str, result: bool) -> None:
        """
        Record an attempt at an exercise.

        Args:
            exercise_id: The ID of the exercise
            concept_id: The ID of the concept being tested
            result: Whether the exercise was completed correctly
        """
        timestamp = datetime.datetime.now().isoformat()
        
        if exercise_id not in self.exercise_history:
            self.exercise_history[exercise_id] = []
            
        self.exercise_history[exercise_id].append({
            "concept_id": concept_id,
            "result": result,
            "timestamp": timestamp
        })
        
    def has_seen_exercise(self, exercise_id: str) -> bool:
        """
        Check if the user has seen an exercise before.

        Args:
            exercise_id: The ID of the exercise

        Returns:
            True if the user has seen the exercise, False otherwise
        """
        return exercise_id in self.exercise_history
        
    def get_mastery(self, concept_id: str, default: float = 0.0) -> float:
        """
        Get the mastery level for a concept.

        Args:
            concept_id: The ID of the concept
            default: Default value if concept mastery not found

        Returns:
            The mastery level (0.0-1.0)
        """
        if concept_id not in self.concept_mastery:
            return default
        return self.concept_mastery[concept_id]["mastery"]
        
    def get_confidence(self, concept_id: str, default: float = 0.5) -> float:
        """
        Get the confidence in mastery assessment for a concept.

        Args:
            concept_id: The ID of the concept
            default: Default value if concept mastery not found

        Returns:
            The confidence level (0.0-1.0)
        """
        if concept_id not in self.concept_mastery:
            return default
        return self.concept_mastery[concept_id]["confidence"]
        
    def get_mastered_concepts(self, threshold: float = 0.7) -> List[str]:
        """
        Get concepts that have been mastered.

        Args:
            threshold: Mastery threshold for considering a concept mastered

        Returns:
            List of mastered concept IDs
        """
        return [concept_id for concept_id, data in self.concept_mastery.items()
                if data["mastery"] >= threshold]
                
    def get_struggling_concepts(self, threshold: float = 0.3) -> List[str]:
        """
        Get concepts that the user is struggling with.

        Args:
            threshold: Mastery threshold below which a concept is considered challenging

        Returns:
            List of challenging concept IDs
        """
        return [concept_id for concept_id, data in self.concept_mastery.items()
                if data["mastery"] < threshold]
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert the learning graph to a dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "concept_mastery": self.concept_mastery,
            "exercise_history": self.exercise_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningGraph":
        """Create a learning graph from a dictionary."""
        graph = cls(
            user_id=data.get("user_id", ""),
            name=data.get("name", "")
        )
        
        graph.concept_mastery = data.get("concept_mastery", {})
        graph.exercise_history = data.get("exercise_history", {})
        
        return graph
        
    def save_to_file(self, file_path: str) -> None:
        """
        Save the learning graph to a JSON file.

        Args:
            file_path: Path to save the file
        """
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load_from_file(cls, file_path: str) -> "LearningGraph":
        """
        Load a learning graph from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded learning graph
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return cls.from_dict(data) 