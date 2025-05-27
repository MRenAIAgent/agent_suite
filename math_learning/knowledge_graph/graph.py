"""
Knowledge Graph module.

This module defines the knowledge graph structure that represents domain concepts and their relationships.
"""

import json
import os
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple, Any
from .concept import Concept


class KnowledgeGraph:
    """A knowledge graph representing concepts and their relationships."""

    def __init__(self, name: str = ""):
        """
        Initialize a knowledge graph.

        Args:
            name: Optional name for the knowledge graph
        """
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        self.graph = nx.DiGraph()  # Directed graph
        
    def add_concept(self, concept: Concept) -> None:
        """
        Add a concept to the knowledge graph.

        Args:
            concept: The concept to add
        """
        self.concepts[concept.id] = concept
        self.graph.add_node(concept.id, data=concept.to_dict())
        
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        """
        Add a prerequisite relationship between concepts.

        Args:
            prerequisite_id: The ID of the prerequisite concept
            concept_id: The ID of the dependent concept
            strength: The strength of the relationship (0.0-1.0)
        """
        if prerequisite_id not in self.concepts or concept_id not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
            
        self.graph.add_edge(prerequisite_id, concept_id, 
                            type="prerequisite", 
                            strength=strength)
        
        # Update concept relationship sets
        self.concepts[prerequisite_id].dependents.add(concept_id)
        self.concepts[concept_id].prerequisites.add(prerequisite_id)
        
    def add_related(self, concept_id1: str, concept_id2: str, strength: float = 0.5) -> None:
        """
        Add a related relationship between concepts.

        Args:
            concept_id1: The ID of the first concept
            concept_id2: The ID of the second concept
            strength: The strength of the relationship (0.0-1.0)
        """
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
            
        # Add as undirected by adding both directions
        self.graph.add_edge(concept_id1, concept_id2, 
                           type="related", 
                           strength=strength)
        self.graph.add_edge(concept_id2, concept_id1, 
                           type="related", 
                           strength=strength)
        
        # Update concept relationship sets
        self.concepts[concept_id1].related.add(concept_id2)
        self.concepts[concept_id2].related.add(concept_id1)
        
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """
        Get a concept by ID.

        Args:
            concept_id: The ID of the concept to retrieve

        Returns:
            The concept or None if not found
        """
        return self.concepts.get(concept_id)
        
    def get_all_concepts(self) -> List[Concept]:
        """
        Get all concepts in the knowledge graph.

        Returns:
            List of all concepts
        """
        return list(self.concepts.values())
        
    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """
        Get all prerequisite concepts for a given concept.

        Args:
            concept_id: The ID of the concept

        Returns:
            List of prerequisite concepts
        """
        if concept_id not in self.concepts:
            return []
            
        return [self.concepts[pre_id] for pre_id in self.concepts[concept_id].prerequisites
                if pre_id in self.concepts]
                
    def get_dependent_concepts(self, concept_id: str) -> List[Concept]:
        """
        Get all concepts that depend on the given concept.

        Args:
            concept_id: The ID of the concept

        Returns:
            List of dependent concepts
        """
        if concept_id not in self.concepts:
            return []
            
        return [self.concepts[dep_id] for dep_id in self.concepts[concept_id].dependents
                if dep_id in self.concepts]
                
    def get_central_concepts(self, limit: int = 5) -> List[Concept]:
        """
        Get the most central concepts in the knowledge graph based on degree centrality.

        Args:
            limit: Maximum number of concepts to return

        Returns:
            List of central concepts
        """
        if not self.graph.nodes:
            return []
            
        centrality = nx.degree_centrality(self.graph)
        sorted_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return [self.concepts[concept_id] for concept_id, _ in sorted_concepts[:limit]
                if concept_id in self.concepts]
                
    def calculate_centrality(self, concept_id: str) -> float:
        """
        Calculate the centrality of a concept in the knowledge graph.

        Args:
            concept_id: The ID of the concept

        Returns:
            Centrality score (0.0-1.0)
        """
        if not self.graph.nodes or concept_id not in self.graph:
            return 0.0
            
        centrality = nx.degree_centrality(self.graph)
        return centrality.get(concept_id, 0.0)
        
    def save_to_file(self, file_path: str) -> None:
        """
        Save the knowledge graph to a JSON file.

        Args:
            file_path: Path to save the file
        """
        data = {
            "name": self.name,
            "concepts": [concept.to_dict() for concept in self.concepts.values()],
            "relationships": [
                {
                    "source": u,
                    "target": v,
                    "type": data["type"],
                    "strength": data["strength"]
                }
                for u, v, data in self.graph.edges(data=True)
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load_from_file(cls, file_path: str) -> "KnowledgeGraph":
        """
        Load a knowledge graph from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Loaded knowledge graph
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        graph = cls(name=data.get("name", ""))
        
        # Add concepts
        for concept_data in data.get("concepts", []):
            concept = Concept.from_dict(concept_data)
            graph.add_concept(concept)
            
        # Add relationships
        for rel in data.get("relationships", []):
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            strength = rel.get("strength", 0.5)
            
            if rel_type == "prerequisite":
                graph.add_prerequisite(source, target, strength)
            elif rel_type == "related":
                # Only add once since add_related adds both directions
                if not graph.graph.has_edge(target, source):
                    graph.add_related(source, target, strength)
                    
        return graph 