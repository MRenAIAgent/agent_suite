import uuid
import json
import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple
import graphiti
import networkx as nx

from agent_suite.agents.storage.long_term.base import LongTermMemoryBase


class GraphitiKnowledgeGraphMemory(LongTermMemoryBase):
    """
    Knowledge graph memory implementation using Graphiti.
    
    Graphiti provides a simple but powerful API for managing knowledge graphs,
    with built-in support for graph-based queries and persistence.
    """
    
    def __init__(self, graph_name: str = "agent_memory", 
                persist: bool = True,
                persistence_path: Optional[str] = None):
        """
        Initialize the knowledge graph memory.
        
        Args:
            graph_name: Name of the graph for this agent's memory
            persist: Whether to persist the graph to disk
            persistence_path: Optional path to persistence directory
        """
        # Initialize the graph
        self.graph_name = graph_name
        self.persist = persist
        self.persistence_path = persistence_path
        
        # Setup the graph
        self.graph = self._setup_graph()
        self.entity_types = set()  # Track entity types
        
        # Vector memory is None for this implementation
        self.vector_memory = None
        
    def _setup_graph(self) -> nx.MultiDiGraph:
        """Initialize the graph backend."""
        try:
            # Try to load existing graph
            if self.persist and self.persistence_path:
                graph = graphiti.Graph.load(f"{self.persistence_path}/{self.graph_name}.graphml")
            else:
                graph = graphiti.Graph(name=self.graph_name)
        except (FileNotFoundError, ValueError):
            # Create new graph
            graph = graphiti.Graph(name=self.graph_name)
            
        return graph
    
    def _save_graph(self) -> None:
        """Save the graph to disk if persistence is enabled."""
        if self.persist and self.persistence_path:
            self.graph.save(f"{self.persistence_path}/{self.graph_name}.graphml")
    
    def store_fact(self, subject: str, predicate: str, object_value: Any, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single fact in the knowledge graph.
        
        Args:
            subject: The subject entity of the fact
            predicate: The relationship type
            object_value: The object value or entity
            metadata: Optional metadata about this fact (confidence, source, etc.)
            
        Returns:
            fact_id: A unique identifier for the stored fact
        """
        # Generate a unique ID for the fact
        fact_id = str(uuid.uuid4())
        
        # Process metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add fact ID to metadata
        metadata["fact_id"] = fact_id
        
        # Handle primitive data types vs entity references
        if isinstance(object_value, (str, int, float, bool)):
            # This is a property relationship (entity -> value)
            if not self.graph.has_node(subject):
                self.graph.add_node(subject)
                
            # Store as node property
            self.graph.set_node_property(subject, predicate, object_value)
            
            # Store metadata as additional properties
            for key, value in metadata.items():
                meta_prop = f"{predicate}__{key}"
                self.graph.set_node_property(subject, meta_prop, value)
        else:
            # This is an entity-to-entity relationship
            if not self.graph.has_node(subject):
                self.graph.add_node(subject)
                
            if not self.graph.has_node(object_value):
                self.graph.add_node(object_value)
                
            # Create relationship
            edge_props = {**metadata, "fact_id": fact_id}
            self.graph.add_edge(subject, object_value, predicate, properties=edge_props)
        
        # Persist changes if enabled
        self._save_graph()
        
        return fact_id
    
    def store_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store unstructured text.
        
        Note: This implementation does not handle vector embeddings directly.
        Text is stored as a node in the graph with properties.
        
        Args:
            text: The text content to store
            metadata: Optional metadata about this text
            
        Returns:
            text_id: A unique identifier for the stored text
        """
        # Generate a unique ID for the text
        text_id = str(uuid.uuid4())
        
        # Process metadata
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Create a text node in the graph
        self.graph.add_node(text_id)
        self.graph.set_node_property(text_id, "content", text)
        self.graph.set_node_property(text_id, "type", "text")
        
        # Add metadata as properties
        for key, value in metadata.items():
            self.graph.set_node_property(text_id, key, value)
        
        # Persist changes if enabled
        self._save_graph()
        
        return text_id
    
    def query_facts(self, query: Union[str, Dict[str, Any]], 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for facts.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results to return
            
        Returns:
            List of matching facts with their metadata
        """
        results = []
        
        if isinstance(query, str):
            # Natural language query - use keyword matching
            keywords = query.lower().split()
            
            # Search node properties
            for node_id in self.graph.node_ids():
                node_props = self.graph.get_node_properties(node_id)
                for prop, value in node_props.items():
                    if any(keyword in str(value).lower() for keyword in keywords):
                        results.append({
                            "subject": node_id,
                            "predicate": prop,
                            "object": value,
                            "confidence": 0.7,  # Basic matching confidence
                            "source": "node_property"
                        })
                        
            # Search relationships
            for edge in self.graph.edges():
                if any(keyword in edge["label"].lower() for keyword in keywords):
                    results.append({
                        "subject": edge["source"],
                        "predicate": edge["label"],
                        "object": edge["target"],
                        "confidence": 0.8,  # Relationship matches are higher confidence
                        "source": "edge"
                    })
                    
                # Check edge properties
                edge_props = edge.get("properties", {})
                for prop, value in edge_props.items():
                    if any(keyword in str(value).lower() for keyword in keywords):
                        results.append({
                            "subject": edge["source"],
                            "predicate": edge["label"],
                            "object": edge["target"],
                            "matched_property": prop,
                            "matched_value": value,
                            "confidence": 0.6,
                            "source": "edge_property"
                        })
        else:
            # Structured query
            subject = query.get("subject")
            predicate = query.get("predicate")
            object_value = query.get("object")
            
            # Filter edges based on query
            for edge in self.graph.edges():
                match = True
                
                if subject and edge["source"] != subject:
                    match = False
                if predicate and edge["label"] != predicate:
                    match = False
                if object_value and edge["target"] != object_value:
                    match = False
                
                if match:
                    edge_props = edge.get("properties", {})
                    results.append({
                        "subject": edge["source"],
                        "predicate": edge["label"],
                        "object": edge["target"],
                        "metadata": edge_props,
                        "source": "edge"
                    })
            
            # If looking for properties of a specific subject
            if subject and not object_value:
                node_props = self.graph.get_node_properties(subject) if self.graph.has_node(subject) else {}
                for prop, value in node_props.items():
                    if not predicate or prop == predicate:
                        results.append({
                            "subject": subject,
                            "predicate": prop,
                            "object": value,
                            "confidence": 1.0,  # Direct property match
                            "source": "node_property"
                        })
        
        # Sort by confidence if available
        results = sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return results[:limit]
    
    def query_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query for text nodes using keyword matching.
        
        Args:
            query: The query text
            limit: Maximum number of results to return
            
        Returns:
            List of matching text nodes with relevance scores
        """
        results = []
        keywords = query.lower().split()
        
        # Find all text nodes
        for node_id in self.graph.node_ids():
            node_props = self.graph.get_node_properties(node_id)
            
            # Check if it's a text node
            if node_props.get("type") == "text":
                content = node_props.get("content", "")
                
                # Simple keyword matching
                keyword_matches = sum(1 for keyword in keywords if keyword in content.lower())
                if keyword_matches > 0:
                    score = keyword_matches / len(keywords)
                    
                    # Add metadata
                    metadata = {k: v for k, v in node_props.items() if k != "content" and k != "type"}
                    
                    results.append({
                        "id": node_id,
                        "text": content,
                        "score": score,
                        "metadata": metadata
                    })
        
        # Sort by score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        return results[:limit]
    
    def get_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all entities or entities of a specific type.
        
        Args:
            entity_type: Optional type to filter entities
            
        Returns:
            List of entity identifiers
        """
        entities = []
        
        for node_id in self.graph.node_ids():
            node_props = self.graph.get_node_properties(node_id)
            
            # Check if node has a type property
            node_type = node_props.get("type")
            
            # Filter by type if specified
            if entity_type is None or node_type == entity_type:
                entities.append(node_id)
        
        return entities
    
    def get_entity_facts(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to a specific entity.
        
        Args:
            entity: Entity identifier
            
        Returns:
            List of facts where the entity is either subject or object
        """
        facts = []
        
        # Get properties of the entity
        if self.graph.has_node(entity):
            props = self.graph.get_node_properties(entity)
            for predicate, value in props.items():
                # Skip metadata properties
                if "__" not in predicate:
                    # Get metadata for this property
                    metadata = {}
                    for meta_key, meta_value in props.items():
                        if meta_key.startswith(f"{predicate}__"):
                            meta_name = meta_key.split("__", 1)[1]
                            metadata[meta_name] = meta_value
                    
                    facts.append({
                        "subject": entity,
                        "predicate": predicate,
                        "object": value,
                        "metadata": metadata,
                        "source": "property"
                    })
        
        # Get relationships where entity is subject
        for edge in self.graph.out_edges(entity):
            facts.append({
                "subject": entity,
                "predicate": edge["label"],
                "object": edge["target"],
                "metadata": edge.get("properties", {}),
                "source": "outgoing_edge"
            })
        
        # Get relationships where entity is object
        for edge in self.graph.in_edges(entity):
            facts.append({
                "subject": edge["source"],
                "predicate": edge["label"],
                "object": entity,
                "metadata": edge.get("properties", {}),
                "source": "incoming_edge"
            })
        
        return facts
    
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a specific fact.
        
        Args:
            fact_id: The unique identifier of the fact
            
        Returns:
            True if deleted successfully, False otherwise
        """
        deleted = False
        
        # Search for the fact in node properties
        for node_id in self.graph.node_ids():
            node_props = self.graph.get_node_properties(node_id)
            
            # Find properties with fact_id in metadata
            for prop_name, value in list(node_props.items()):
                if prop_name.endswith("__fact_id") and value == fact_id:
                    base_prop = prop_name.split("__", 1)[0]
                    
                    # Delete the property
                    self.graph.remove_node_property(node_id, base_prop)
                    
                    # Delete all metadata properties
                    for meta_prop in list(node_props.keys()):
                        if meta_prop.startswith(f"{base_prop}__"):
                            self.graph.remove_node_property(node_id, meta_prop)
                    
                    deleted = True
        
        # Search for the fact in edge properties
        for edge in self.graph.edges():
            edge_props = edge.get("properties", {})
            if edge_props.get("fact_id") == fact_id:
                # Delete the edge
                self.graph.remove_edge(edge["source"], edge["target"], edge["label"])
                deleted = True
        
        # Persist changes if enabled
        if deleted:
            self._save_graph()
        
        return deleted
    
    def delete_entity(self, entity: str) -> int:
        """
        Delete an entity and all its related facts.
        
        Args:
            entity: Entity identifier
            
        Returns:
            Number of facts deleted
        """
        if not self.graph.has_node(entity):
            return 0
            
        # Count facts to be deleted
        fact_count = 0
        
        # Count node properties
        node_props = self.graph.get_node_properties(entity)
        unique_props = set()
        for prop_name in node_props.keys():
            if "__" not in prop_name:
                unique_props.add(prop_name)
        fact_count += len(unique_props)
        
        # Count relationships
        fact_count += len(list(self.graph.out_edges(entity)))
        fact_count += len(list(self.graph.in_edges(entity)))
        
        # Delete the node
        self.graph.remove_node(entity)
        
        # Persist changes if enabled
        self._save_graph()
        
        return fact_count
    
    def delete_text(self, text_id: str) -> bool:
        """
        Delete a specific text passage.
        
        Args:
            text_id: The unique identifier of the text
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.graph.has_node(text_id):
            return False
            
        # Check if it's a text node
        props = self.graph.get_node_properties(text_id)
        if props.get("type") != "text":
            return False
            
        # Delete the node
        self.graph.remove_node(text_id)
        
        # Persist changes if enabled
        self._save_graph()
        
        return True
    
    def clear(self) -> None:
        """
        Clear all stored data.
        """
        # Create a new empty graph
        self.graph = graphiti.Graph(name=self.graph_name)
        
        # Persist changes if enabled
        self._save_graph()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the stored data.
        
        Returns:
            Dictionary with statistics like entity count, fact count, etc.
        """
        # Count nodes by type
        node_types = {}
        total_properties = 0
        
        for node_id in self.graph.node_ids():
            props = self.graph.get_node_properties(node_id)
            node_type = props.get("type", "unknown")
            
            if node_type not in node_types:
                node_types[node_type] = 0
            node_types[node_type] += 1
            
            # Count properties
            for prop in props.keys():
                if "__" not in prop:
                    total_properties += 1
        
        # Count relationships by type
        edge_types = {}
        for edge in self.graph.edges():
            edge_type = edge["label"]
            
            if edge_type not in edge_types:
                edge_types[edge_type] = 0
            edge_types[edge_type] += 1
        
        return {
            "total_nodes": self.graph.num_nodes(),
            "total_edges": self.graph.num_edges(),
            "total_properties": total_properties,
            "node_types": node_types,
            "edge_types": edge_types
        } 