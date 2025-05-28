"""
Abstract storage interface for the math learning knowledge graph.

This module defines the interface that storage backends must implement,
allowing the math learning system to work with different storage backends
(NetworkX, RAG, etc.) through a unified API.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, Tuple
from .concept import Concept


class KnowledgeGraphStorageInterface(ABC):
    """Abstract interface for knowledge graph storage backends."""
    
    @abstractmethod
    def add_concept(self, concept: Concept) -> None:
        """
        Add a concept to the knowledge graph.
        
        Args:
            concept: The concept to add
        """
        pass
    
    @abstractmethod
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        """
        Add a prerequisite relationship between concepts.
        
        Args:
            prerequisite_id: The ID of the prerequisite concept
            concept_id: The ID of the dependent concept
            strength: The strength of the relationship (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def add_related(self, concept_id1: str, concept_id2: str, strength: float = 0.5) -> None:
        """
        Add a related relationship between concepts.
        
        Args:
            concept_id1: The ID of the first concept
            concept_id2: The ID of the second concept
            strength: The strength of the relationship (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """
        Get a concept by ID.
        
        Args:
            concept_id: The ID of the concept to retrieve
            
        Returns:
            The concept or None if not found
        """
        pass
    
    @abstractmethod
    def get_all_concepts(self) -> List[Concept]:
        """
        Get all concepts in the knowledge graph.
        
        Returns:
            List of all concepts
        """
        pass
    
    @abstractmethod
    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """
        Get all prerequisite concepts for a given concept.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            List of prerequisite concepts
        """
        pass
    
    @abstractmethod
    def get_dependent_concepts(self, concept_id: str) -> List[Concept]:
        """
        Get all concepts that depend on the given concept.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            List of dependent concepts
        """
        pass
    
    @abstractmethod
    def get_central_concepts(self, limit: int = 5) -> List[Concept]:
        """
        Get the most central concepts in the knowledge graph.
        
        Args:
            limit: Maximum number of concepts to return
            
        Returns:
            List of central concepts
        """
        pass
    
    @abstractmethod
    def calculate_centrality(self, concept_id: str) -> float:
        """
        Calculate the centrality of a concept in the knowledge graph.
        
        Args:
            concept_id: The ID of the concept
            
        Returns:
            Centrality score (0.0-1.0)
        """
        pass
    
    @abstractmethod
    def find_learning_path(self, start_concept: str, end_concept: str) -> List[Concept]:
        """
        Find a learning path from start concept to end concept.
        
        Args:
            start_concept: The ID of the starting concept
            end_concept: The ID of the target concept
            
        Returns:
            List of concepts representing the learning path
        """
        pass
    
    @abstractmethod
    def save_to_file(self, file_path: str) -> None:
        """
        Save the knowledge graph to a file.
        
        Args:
            file_path: Path to save the file
        """
        pass
    
    @abstractmethod
    def load_from_file(self, file_path: str) -> None:
        """
        Load the knowledge graph from a file.
        
        Args:
            file_path: Path to the file to load
        """
        pass
    
    @abstractmethod
    def get_concept_count(self) -> int:
        """
        Get the total number of concepts in the graph.
        
        Returns:
            Number of concepts
        """
        pass
    
    @abstractmethod
    def get_relationship_count(self) -> int:
        """
        Get the total number of relationships in the graph.
        
        Returns:
            Number of relationships
        """
        pass
    
    @abstractmethod
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """
        Search for concepts by name or description.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        pass
    
    @abstractmethod
    def get_concepts_by_category(self, category: str) -> List[Concept]:
        """
        Get all concepts in a specific category.
        
        Args:
            category: The category name
            
        Returns:
            List of concepts in the category
        """
        pass
    
    @abstractmethod
    def get_concepts_by_difficulty(self, min_difficulty: int, max_difficulty: int) -> List[Concept]:
        """
        Get concepts within a difficulty range.
        
        Args:
            min_difficulty: Minimum difficulty level
            max_difficulty: Maximum difficulty level
            
        Returns:
            List of concepts within the difficulty range
        """
        pass 