"""
RAG storage implementation for the math learning knowledge graph.

This module provides a RAG-based storage backend that implements
the KnowledgeGraphStorageInterface, using the repository's RAG system
for persistent graph storage.
"""

import json
import os
import asyncio
from typing import Dict, List, Optional, Set, Any, Tuple

from .storage_interface import KnowledgeGraphStorageInterface
from .concept import Concept
from math_learning.config.storage_config import MathLearningStorageConfig

# Import RAG components
from agents.rag.api.rag_service import RagService
from agents.rag.storage.graph.base import GraphStorageAdaptor
from agents.rag.storage.graph.memory_storage import MemoryGraphStorageAdaptor
from agents.rag.storage.graph.neo4j_storage import Neo4jGraphStorageAdaptor
from agents.rag.middleware.storage_router import StorageType
from agents.rag.models.knowledge import Entity, Relationship


class RagStorage(KnowledgeGraphStorageInterface):
    """RAG-based storage implementation for knowledge graphs."""
    
    def __init__(self, config: MathLearningStorageConfig, name: str = ""):
        """
        Initialize RAG storage.
        
        Args:
            config: Storage configuration
            name: Optional name for the knowledge graph
        """
        self.config = config
        self.name = name or config.rag_graph_name
        self.concepts: Dict[str, Concept] = {}  # Local cache
        
        # Initialize RAG service
        if config.rag_config is None:
            raise ValueError("RAG configuration is required for RAG storage backend")
            
        self.rag_service = RagService()
        
        # Register the appropriate graph storage adaptor
        if config.rag_backend == "memory":
            graph_adaptor = MemoryGraphStorageAdaptor()
        elif config.rag_backend == "neo4j":
            # For now, use memory as fallback - Neo4j would need connection config
            graph_adaptor = MemoryGraphStorageAdaptor()
        else:
            graph_adaptor = MemoryGraphStorageAdaptor()
            
        self.rag_service.register_storage_adaptor(StorageType.GRAPH, graph_adaptor)
        self.graph_storage = graph_adaptor
        
        # Load existing concepts from RAG storage
        self._load_concepts_from_rag()
    
    def _run_async(self, coro):
        """Helper method to run async operations synchronously."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
    
    def _load_concepts_from_rag(self) -> None:
        """Load existing concepts from RAG storage into local cache."""
        try:
            # Query for math concept entities
            query_params = {
                "type": "math_concept",
                "properties": {"graph_name": self.name},
                "limit": 1000
            }
            
            entities_data = self._run_async(
                self.graph_storage.query_graph("get_entities", query_params)
            )
            
            for entity_data in entities_data:
                try:
                    # The entity data should contain the concept information
                    if entity_data.get("type") == "math_concept":
                        properties = entity_data.get("properties", {})
                        if properties.get("graph_name") == self.name:
                            # Try to reconstruct concept from properties
                            concept_data = {
                                "id": entity_data["id"].replace("math_concept_", ""),
                                "name": properties.get("name", ""),
                                "description": properties.get("description", ""),
                                "category": properties.get("category", ""),
                                "difficulty": properties.get("difficulty", 1),
                                "grade_level": properties.get("grade_level", 1),
                                "prerequisites": set(),
                                "dependents": set(),
                                "related": set()
                            }
                            concept = Concept.from_dict(concept_data)
                            self.concepts[concept.id] = concept
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse concept from entity {entity_data.get('id', 'unknown')}: {e}")
                        
        except Exception as e:
            print(f"Warning: Could not load concepts from RAG storage: {e}")
    
    def _save_concept_to_rag(self, concept: Concept) -> None:
        """Save a concept to RAG storage."""
        try:
            # Create entity for the concept
            entity_id = f"math_concept_{concept.id}"
            
            # Create Entity object
            entity = Entity(
                id=entity_id,
                type="math_concept",
                properties={
                    "name": concept.name,
                    "description": concept.description,
                    "category": concept.category,
                    "difficulty": concept.difficulty,
                    "grade_level": concept.grade_level,
                    "graph_name": self.name
                }
            )
            
            self._run_async(self.graph_storage.store_entity(entity))
            
        except Exception as e:
            print(f"Warning: Could not save concept {concept.id} to RAG storage: {e}")
    
    def _save_relationship_to_rag(self, source_id: str, target_id: str, 
                                  relationship_type: str, strength: float) -> None:
        """Save a relationship to RAG storage."""
        try:
            source_entity_id = f"math_concept_{source_id}"
            target_entity_id = f"math_concept_{target_id}"
            
            # Create Relationship object
            relationship = Relationship(
                source_id=source_entity_id,
                target_id=target_entity_id,
                type=relationship_type,
                properties={
                    "strength": strength,
                    "graph_name": self.name
                }
            )
            
            self._run_async(self.graph_storage.store_relationship(relationship))
            
        except Exception as e:
            print(f"Warning: Could not save relationship {source_id}->{target_id} to RAG storage: {e}")
    
    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the knowledge graph."""
        self.concepts[concept.id] = concept
        self._save_concept_to_rag(concept)
    
    def add_prerequisite(self, prerequisite_id: str, concept_id: str, strength: float = 0.8) -> None:
        """Add a prerequisite relationship between concepts."""
        if prerequisite_id not in self.concepts or concept_id not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
        
        # Update local concept relationships
        self.concepts[prerequisite_id].dependents.add(concept_id)
        self.concepts[concept_id].prerequisites.add(prerequisite_id)
        
        # Save to RAG storage
        self._save_relationship_to_rag(prerequisite_id, concept_id, "prerequisite", strength)
    
    def add_related(self, concept_id1: str, concept_id2: str, strength: float = 0.5) -> None:
        """Add a related relationship between concepts."""
        if concept_id1 not in self.concepts or concept_id2 not in self.concepts:
            raise ValueError("Both concepts must exist in the knowledge graph")
        
        # Update local concept relationships (bidirectional)
        self.concepts[concept_id1].related.add(concept_id2)
        self.concepts[concept_id2].related.add(concept_id1)
        
        # Save to RAG storage (bidirectional)
        self._save_relationship_to_rag(concept_id1, concept_id2, "related", strength)
        self._save_relationship_to_rag(concept_id2, concept_id1, "related", strength)
    
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
        """Get the most central concepts based on relationship count."""
        # Calculate centrality based on total relationships
        concept_centrality = []
        
        for concept in self.concepts.values():
            total_relationships = (len(concept.prerequisites) + 
                                 len(concept.dependents) + 
                                 len(concept.related))
            concept_centrality.append((concept, total_relationships))
        
        # Sort by centrality and return top concepts
        concept_centrality.sort(key=lambda x: x[1], reverse=True)
        return [concept for concept, _ in concept_centrality[:limit]]
    
    def calculate_centrality(self, concept_id: str) -> float:
        """Calculate the centrality of a concept based on relationship count."""
        if concept_id not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_id]
        total_relationships = (len(concept.prerequisites) + 
                             len(concept.dependents) + 
                             len(concept.related))
        
        # Normalize by total possible relationships
        max_possible = len(self.concepts) - 1  # Can relate to all other concepts
        if max_possible == 0:
            return 0.0
            
        return min(total_relationships / max_possible, 1.0)
    
    def find_learning_path(self, start_concept: str, end_concept: str) -> List[Concept]:
        """Find a learning path from start concept to end concept using BFS."""
        if start_concept not in self.concepts or end_concept not in self.concepts:
            return []
        
        if start_concept == end_concept:
            return [self.concepts[start_concept]]
        
        # Use BFS to find shortest path through prerequisites
        from collections import deque
        
        queue = deque([(start_concept, [start_concept])])
        visited = {start_concept}
        
        while queue:
            current_id, path = queue.popleft()
            current_concept = self.concepts[current_id]
            
            # Check dependents (concepts that depend on current)
            for dependent_id in current_concept.dependents:
                if dependent_id == end_concept:
                    # Found the target
                    final_path = path + [dependent_id]
                    return [self.concepts[concept_id] for concept_id in final_path]
                
                if dependent_id not in visited and dependent_id in self.concepts:
                    visited.add(dependent_id)
                    queue.append((dependent_id, path + [dependent_id]))
        
        return []  # No path found
    
    def save_to_file(self, file_path: str) -> None:
        """Save the knowledge graph to a JSON file (for backup/export)."""
        data = {
            "name": self.name,
            "concepts": [concept.to_dict() for concept in self.concepts.values()],
            "relationships": []
        }
        
        # Extract relationships from concepts
        for concept in self.concepts.values():
            for prereq_id in concept.prerequisites:
                data["relationships"].append({
                    "source": prereq_id,
                    "target": concept.id,
                    "type": "prerequisite",
                    "strength": 0.8
                })
            
            for related_id in concept.related:
                # Only add one direction to avoid duplicates
                if concept.id < related_id:
                    data["relationships"].append({
                        "source": concept.id,
                        "target": related_id,
                        "type": "related",
                        "strength": 0.5
                    })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str) -> None:
        """Load the knowledge graph from a JSON file (for import)."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.name = data.get("name", self.name)
        
        # Clear existing data
        self.concepts.clear()
        
        # Add concepts
        for concept_data in data.get("concepts", []):
            concept = Concept.from_dict(concept_data)
            self.add_concept(concept)
        
        # Add relationships
        for rel in data.get("relationships", []):
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            strength = rel.get("strength", 0.5)
            
            if rel_type == "prerequisite":
                self.add_prerequisite(source, target, strength)
            elif rel_type == "related":
                self.add_related(source, target, strength)
    
    def get_concept_count(self) -> int:
        """Get the total number of concepts in the graph."""
        return len(self.concepts)
    
    def get_relationship_count(self) -> int:
        """Get the total number of relationships in the graph."""
        total = 0
        for concept in self.concepts.values():
            total += len(concept.prerequisites)
            total += len(concept.related)  # Count each direction
        return total
    
    def search_concepts(self, query: str, limit: int = 10) -> List[Concept]:
        """Search for concepts by name or description."""
        if self.config.rag_enable_vector_search:
            try:
                # Use RAG's vector search if available
                results = self.rag_service.search(query, limit=limit)
                concepts = []
                
                for result in results:
                    # Extract concept from search result
                    if hasattr(result, 'metadata') and result.metadata.get("graph_name") == self.name:
                        concept_data = json.loads(result.metadata.get("concept_data", "{}"))
                        concept = Concept.from_dict(concept_data)
                        concepts.append(concept)
                        
                        if len(concepts) >= limit:
                            break
                
                return concepts
                
            except Exception as e:
                print(f"Warning: Vector search failed, falling back to local search: {e}")
        
        # Fallback to local search
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