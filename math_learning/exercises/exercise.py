"""
Exercise module.

This module defines the exercise model which represents problems
that test understanding of concepts.
"""

import uuid
from typing import Dict, List, Optional, Any, Tuple


class Exercise:
    """An exercise that tests understanding of concepts."""

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
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an exercise.

        Args:
            title: Title of the exercise
            problem: The problem statement/question
            solution: The solution or answer
            difficulty: A value from 1-5 indicating the difficulty level
            estimated_time: Estimated time in minutes to complete
            format_type: Type of exercise (e.g., "multiple-choice", "free-response")
            exercise_id: Unique identifier (auto-generated if not provided)
            hints: List of progressive hints
            metadata: Additional metadata about the exercise
        """
        self.id = exercise_id or str(uuid.uuid4())
        self.title = title
        self.problem = problem
        self.solution = solution
        self.difficulty = max(1, min(5, difficulty))  # Ensure 1-5 range
        self.estimated_time = estimated_time
        self.format_type = format_type
        self.hints = hints or []
        self.metadata = metadata or {}
        
        # Concept relationships
        self.concept_relationships: List[Dict[str, Any]] = []
        
    def add_concept_relationship(
        self,
        concept_id: str,
        weight: float = 0.8,
        relationship_type: str = "primary"
    ) -> None:
        """
        Add a relationship to a concept.

        Args:
            concept_id: ID of the concept
            weight: Strength of the relationship (0.0-1.0)
            relationship_type: Type of relationship ("primary", "secondary", "foundational")
        """
        self.concept_relationships.append({
            "concept_id": concept_id,
            "weight": max(0.0, min(1.0, weight)),  # Ensure 0.0-1.0 range
            "type": relationship_type
        })
        
    def get_primary_concepts(self) -> List[str]:
        """
        Get the primary concepts tested by this exercise.

        Returns:
            List of primary concept IDs
        """
        return [rel["concept_id"] for rel in self.concept_relationships 
                if rel["type"] == "primary"]
                
    def get_all_concept_ids(self) -> List[str]:
        """
        Get all concept IDs related to this exercise.

        Returns:
            List of all concept IDs
        """
        return [rel["concept_id"] for rel in self.concept_relationships]
        
    def get_concept_weight(self, concept_id: str) -> float:
        """
        Get the weight of the relationship to a specific concept.

        Args:
            concept_id: ID of the concept

        Returns:
            Weight of the relationship (0.0-1.0) or 0.0 if not related
        """
        for rel in self.concept_relationships:
            if rel["concept_id"] == concept_id:
                return rel["weight"]
        return 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the exercise to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "problem": self.problem,
            "solution": self.solution,
            "difficulty": self.difficulty,
            "estimated_time": self.estimated_time,
            "format_type": self.format_type,
            "hints": self.hints,
            "metadata": self.metadata,
            "concept_relationships": self.concept_relationships
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Exercise":
        """Create an exercise from a dictionary."""
        exercise = cls(
            title=data["title"],
            problem=data["problem"],
            solution=data["solution"],
            difficulty=data.get("difficulty", 3),
            estimated_time=data.get("estimated_time", 5),
            format_type=data.get("format_type", "free-response"),
            exercise_id=data.get("id"),
            hints=data.get("hints", []),
            metadata=data.get("metadata", {})
        )
        
        for rel in data.get("concept_relationships", []):
            exercise.add_concept_relationship(
                concept_id=rel["concept_id"],
                weight=rel.get("weight", 0.8),
                relationship_type=rel.get("type", "primary")
            )
            
        return exercise 