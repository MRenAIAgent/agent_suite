"""
NetworkX storage implementation for the math learning knowledge graph.

This module provides a NetworkX-based storage backend that implements
the KnowledgeGraphStorageInterface, maintaining backward compatibility
with the existing NetworkX-based implementation.
"""

import json
import os
import networkx as nx
from typing import Dict, List, Optional, Set, Any, Tuple

from .storage_interface import KnowledgeGraphStorageInterface
from .concept import Concept
from math_learning.config.storage_config import MathLearningStorageConfig


class NetworkXStorage(KnowledgeGraphStorageInterface):
    """NetworkX-based storage implementation for knowledge graphs."""
    
    def __init__(self, config: MathLearningStorageConfig, name: str = ""):
        """
        Initialize NetworkX storage.
        
        Args:
            config: Storage configuration
            name: Optional name for the knowledge graph
        """
        self.config = config
        self.name = name
        self.concepts: Dict[str, Concept] = {}
        self.graph = nx.DiGraph()  # Directed graph
        
        # Load existing data if file exists
        if (self.config.backend.value == "networkx" and 
            os.path.exists(self.config.networkx_persistence_file)):
            try:
                self.load_from_file(self.config.networkx_persistence_file)
            except Exception as e:
                print(f"Warning: Could not load existing graph file: {e}")
    
    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the knowledge graph."""
        self.concepts[concept.id] = concept
        self.graph.add_node(concept.id, data=concept.to_dict())
        
        if self.config.networkx_auto_save:
            self._auto_save()
    
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        """Add a prerequisite relationship between concepts."""
        if prerequisite_id not in self.concepts or concept_id not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
            
        self.graph.add_edge(prerequisite_id, concept_id, 
                            type="prerequisite", 
                            strength=strength)
        
        # Update concept relationship sets
        self.concepts[prerequisite_id].dependents.add(concept_id)
        self.concepts[concept_id].prerequisites.add(prerequisite_id)
        
        if self.config.networkx_auto_save:
            self._auto_save()
    
    def add_related(self, concept_id1: str, concept_id2: str, strength: float = 0.5) -> None:
        """Add a related relationship between concepts."""
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
        
        if self.config.networkx_auto_save:
            self._auto_save()
    
    def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)
    
    def get_all_concepts(self) -> List[Concept]:
        """Get all concepts in the knowledge graph."""
        return list(self.concepts.values())
    
    def get_prerequisites(self, concept_id: str) -> List[Concept]:
        """Get all prerequisite concepts for a given concept."""
        if concept_id not in self.concepts:
            return []
            
        return [self.concepts[pre_id] for pre_id in self.concepts[concept_id].prerequisites
                if pre_id in self.concepts]
    
    def get_dependent_concepts(self, concept_id: str) -> List[Concept]:
        """Get all concepts that depend on the given concept."""
        if concept_id not in self.concepts:
            return []
            
        return [self.concepts[dep_id] for dep_id in self.concepts[concept_id].dependents
                if dep_id in self.concepts]
    
    def get_central_concepts(self, limit: int = 5) -> List[Concept]:
        """Get the most central concepts in the knowledge graph based on degree centrality."""
        if not self.graph.nodes:
            return []
            
        centrality = nx.degree_centrality(self.graph)
        sorted_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return [self.concepts[concept_id] for concept_id, _ in sorted_concepts[:limit]
                if concept_id in self.concepts]
    
    def calculate_centrality(self, concept_id: str) -> float:
        """Calculate the centrality of a concept in the knowledge graph."""
        if not self.graph.nodes or concept_id not in self.graph:
            return 0.0
            
        centrality = nx.degree_centrality(self.graph)
        return centrality.get(concept_id, 0.0)
    
    def find_learning_path(self, start_concept: str, end_concept: str) -> List[Concept]:
        """Find a learning path from start concept to end concept."""
        if start_concept not in self.concepts or end_concept not in self.concepts:
            return []
            
        try:
            # Use NetworkX to find shortest path
            path_ids = nx.shortest_path(self.graph, start_concept, end_concept)
            return [self.concepts[concept_id] for concept_id in path_ids]
        except nx.NetworkXNoPath:
            # No path exists
            return []
        except nx.NodeNotFound:
            # One of the nodes doesn't exist
            return []
    
    def save_to_file(self, file_path: str) -> None:
        """Save the knowledge graph to a JSON file."""
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
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load the knowledge graph from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        self.name = data.get("name", "")
        self.concepts.clear()
        self.graph.clear()
        
        # Add concepts
        for concept_data in data.get("concepts", []):
            concept = Concept.from_dict(concept_data)
            self.concepts[concept.id] = concept
            self.graph.add_node(concept.id, data=concept.to_dict())
            
        # Add relationships
        for rel in data.get("relationships", []):
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            strength = rel.get("strength", 0.5)
            
            self.graph.add_edge(source, target, type=rel_type, strength=strength)
            
            # Update concept relationship sets
            if rel_type == "prerequisite":
                if source in self.concepts and target in self.concepts:
                    self.concepts[source].dependents.add(target)
                    self.concepts[target].prerequisites.add(source)
            elif rel_type == "related":
                if source in self.concepts and target in self.concepts:
                    self.concepts[source].related.add(target)
                    self.concepts[target].related.add(source)
    
    def get_concept_count(self) -> int:
        """Get the total number of concepts in the graph."""
        return len(self.concepts)
    
    def get_relationship_count(self) -> int:
        """Get the total number of relationships in the graph."""
        return self.graph.number_of_edges()
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search for concepts by name or description."""
        query_lower = query.lower()
        matches = []
        
        for concept in self.concepts.values():
            name_match = query_lower in concept.name.lower()
            desc_match = query_lower in concept.description.lower()
            
            if name_match or desc_match:
                matches.append(concept)
                
            if len(matches) >= limit:
                break
        
        return matches
    
    def get_concepts_by_category(self, category: str) -> List[Concept]:
        """Get all concepts in a specific category."""
        return [concept for concept in self.concepts.values() 
                if concept.category == category]
    
    def get_concepts_by_difficulty(self, min_difficulty: int, max_difficulty: int) -> List[Concept]:
        """Get concepts within a difficulty range."""
        return [concept for concept in self.concepts.values()
                if min_difficulty <= concept.difficulty <= max_difficulty]
    
    def _auto_save(self) -> None:
        """Automatically save the graph if auto-save is enabled."""
        if self.config.networkx_auto_save:
            try:
                self.save_to_file(self.config.networkx_persistence_file)
            except Exception as e:
                print(f"Warning: Auto-save failed: {e}") 