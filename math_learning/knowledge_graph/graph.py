"""
Knowledge Graph module.

This module defines the knowledge graph structure that represents domain concepts and their relationships.
Now supports configurable storage backends (NetworkX or RAG).
"""

import json
import os
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any

from .concept import Concept
from .storage_interface import KnowledgeGraphStorageInterface
from .storage_factory import create_storage, create_storage_from_env
from math_learning.config.storage_config import MathLearningStorageConfig


class KnowledgeGraph:
    """A knowledge graph representing concepts and their relationships with configurable storage."""

    def __init__(self, name: str = "", config: Optional[MathLearningStorageConfig] = None):
        """
        Initialize a knowledge graph with configurable storage.

        Args:
            name: Optional name for the knowledge graph
            config: Storage configuration. If None, uses default configuration.
        """
        self.name = name
        self._storage: KnowledgeGraphStorageInterface = create_storage(config, name)
        
    @classmethod
    def from_env(cls, name: str = "") -> "KnowledgeGraph":
        """
        Create a knowledge graph using configuration from environment variables.
        
        Args:
            name: Optional name for the knowledge graph
            
        Returns:
            A new KnowledgeGraph instance
        """
        instance = cls.__new__(cls)
        instance.name = name
        instance._storage = create_storage_from_env(name)
        return instance
        
    def add_concept(self, concept: Concept) -> None:
        """
        Add a concept to the knowledge graph.

        Args:
            concept: The concept to add
        """
        self._storage.add_concept(concept)
        
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        """
        Add a prerequisite relationship between concepts.

        Args:
            prerequisite_id: The ID of the prerequisite concept
            concept_id: The ID of the dependent concept
            strength: The strength of the relationship (0.0-1.0)
        """
        self._storage.add_prerequisite(prerequisite_id, concept_id, strength)
        
    def add_related(self, concept_id1: str, concept_id2: str, strength: float = 0.5) -> None:
        """
        Add a related relationship between concepts.

        Args:
            concept_id1: The ID of the first concept
            concept_id2: The ID of the second concept
            strength: The strength of the relationship (0.0-1.0)
        """
        self._storage.add_related(concept_id1, concept_id2, strength)
        
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """
        Get a concept by ID.

        Args:
            concept_id: The ID of the concept to retrieve

        Returns:
            The concept or None if not found
        """
        return self._storage.get_concept(concept_id)
        
    def get_all_concepts(self) -> List[Concept]:
        """
        Get all concepts in the knowledge graph.

        Returns:
            List of all concepts
        """
        return self._storage.get_all_concepts()
        
    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """
        Get all prerequisite concepts for a given concept.

        Args:
            concept_id: The ID of the concept

        Returns:
            List of prerequisite concepts
        """
        return self._storage.get_prerequisites(concept_id)
                
    def get_dependent_concepts(self, concept_id: str) -> List[Concept]:
        """
        Get all concepts that depend on the given concept.

        Args:
            concept_id: The ID of the concept

        Returns:
            List of dependent concepts
        """
        return self._storage.get_dependent_concepts(concept_id)
                
    def get_central_concepts(self, limit: int = 5) -> List[Concept]:
        """
        Get the most central concepts in the knowledge graph.

        Args:
            limit: Maximum number of concepts to return

        Returns:
            List of central concepts
        """
        return self._storage.get_central_concepts(limit)
                
    def calculate_centrality(self, concept_id: str) -> float:
        """
        Calculate the centrality of a concept in the knowledge graph.

        Args:
            concept_id: The ID of the concept

        Returns:
            Centrality score (0.0-1.0)
        """
        return self._storage.calculate_centrality(concept_id)
        
    def find_learning_path(self, start_concept: str, end_concept: str) -> List[Concept]:
        """
        Find a learning path from start concept to end concept.
        
        Args:
            start_concept: The ID of the starting concept
            end_concept: The ID of the target concept
            
        Returns:
            List of concepts representing the learning path
        """
        return self._storage.find_learning_path(start_concept, end_concept)
        
    def save_to_file(self, file_path: str) -> None:
        """
        Save the knowledge graph to a JSON file.

        Args:
            file_path: Path to save the file
        """
        self._storage.save_to_file(file_path)
            
    @classmethod
    def load_from_file(cls, file_path: str, config: Optional[MathLearningStorageConfig] = None) -> "KnowledgeGraph":
        """
        Load a knowledge graph from a JSON file.

        Args:
            file_path: Path to the JSON file
            config: Storage configuration. If None, uses default configuration.

        Returns:
            Loaded knowledge graph
        """
        # Create a new instance
        graph = cls(config=config)
        
        # Load data into the storage backend
        graph._storage.load_from_file(file_path)
        
        return graph
    
    # Additional convenience methods
    def get_concept_count(self) -> int:
        """Get the total number of concepts in the graph."""
        return self._storage.get_concept_count()
    
    def get_relationship_count(self) -> int:
        """Get the total number of relationships in the graph."""
        return self._storage.get_relationship_count()
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search for concepts by name or description."""
        return self._storage.search_concepts(query, limit)
    
    def get_concepts_by_category(self, category: str) -> List[Concept]:
        """Get all concepts in a specific category."""
        return self._storage.get_concepts_by_category(category)
    
    def get_concepts_by_difficulty(self, min_difficulty: int, max_difficulty: int) -> List[Concept]:
        """Get concepts within a difficulty range."""
        return self._storage.get_concepts_by_difficulty(min_difficulty, max_difficulty)
    
    # Backward compatibility properties
    @property
    def concepts(self) -> Dict[str, Concept]:
        """
        Get concepts dictionary for backward compatibility.
        
        Note: This creates a new dictionary each time it's accessed.
        For better performance, use get_all_concepts() or get_concept().
        """
        return {concept.id: concept for concept in self._storage.get_all_concepts()}
    
    @property 
    def graph(self) -> nx.DiGraph:
        """
        Get NetworkX graph for backward compatibility.
        
        Note: This only works with NetworkX storage backend.
        For RAG storage, this will raise an AttributeError.
        """
        if hasattr(self._storage, 'graph'):
            return self._storage.graph
        else:
            raise AttributeError("Graph property is only available with NetworkX storage backend")
    
    def __repr__(self) -> str:
        """String representation of the knowledge graph."""
        concept_count = self.get_concept_count()
        relationship_count = self.get_relationship_count()
        storage_type = type(self._storage).__name__
        
        return (f"KnowledgeGraph(name='{self.name}', "
                f"concepts={concept_count}, "
                f"relationships={relationship_count}, "
                f"storage={storage_type})") 