"""
Concept module for the knowledge graph.

This module defines the concept class which represents nodes in the knowledge graph.
"""

import uuid
from typing import Dict, List, Optional, Set, Any


class Concept:
    """A concept node in the knowledge graph."""

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
        """
        Initialize a concept node.

        Args:
            name: The name of the concept
            description: A detailed description of the concept
            difficulty: A value from 1-5 indicating the difficulty level
            time_to_master: Estimated time in minutes to master the concept
            category: The category this concept belongs to (e.g., "Algebra")
            concept_id: Unique identifier (auto-generated if not provided)
            examples: List of example descriptions or problems
        """
        self.id = concept_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.difficulty = max(1, min(5, difficulty))  # Ensure 1-5 range
        self.time_to_master = time_to_master
        self.category = category
        self.examples = examples or []
        
        # Relationships (to be populated by knowledge graph)
        self.prerequisites: Set[str] = set()
        self.dependents: Set[str] = set()
        self.related: Set[str] = set()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the concept to a dictionary for serialization."""
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Concept":
        """Create a concept from a dictionary."""
        concept = cls(
            name=data["name"],
            description=data["description"],
            difficulty=data.get("difficulty", 3),
            time_to_master=data.get("time_to_master", 60),
            category=data.get("category", ""),
            concept_id=data.get("id"),
            examples=data.get("examples", [])
        )
        
        concept.prerequisites = set(data.get("prerequisites", []))
        concept.dependents = set(data.get("dependents", []))
        concept.related = set(data.get("related", []))
        
        return concept 