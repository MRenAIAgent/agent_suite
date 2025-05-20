import uuid
import datetime
from typing import Any, Dict, List, Optional, Union, Set
import os

try:
    import graphiti
except ImportError:
    raise ImportError("Graphiti library not found. Install it with 'pip install graphiti'")

from knowledge.base import KnowledgeBase


class GraphitiKnowledgeGraph(KnowledgeBase):
    """
    Knowledge Graph implementation using Graphiti.
    
    This implementation stores entities and relations in a graph structure,
    allowing for sophisticated querying and traversal.
    """
    
    def __init__(self, 
                graph_name: str = "knowledge_graph", 
                persist: bool = True,
                persistence_path: Optional[str] = None):
        """
        Initialize the knowledge graph.
        
        Args:
            graph_name: Name of the graph
            persist: Whether to persist the graph to disk
            persistence_path: Optional path for persistence
        """
        self.graph_name = graph_name
        self.persist = persist
        
        # Set default persistence path if not provided
        if persist and persistence_path is None:
            self.persistence_path = os.path.join(os.getcwd(), "knowledge_data")
            os.makedirs(self.persistence_path, exist_ok=True)
        else:
            self.persistence_path = persistence_path
            
        # Initialize the graph
        self.graph = self._setup_graph()
        
        # Track entity types for easier querying
        self.entity_types = {}  # entity_type -> set of entity_ids
        
        # Track relations for faster lookup
        self.relation_ids = {}  # relation_id -> (source_id, relation_type, target_id)
        
    def _setup_graph(self):
        """Initialize or load the graph."""
        try:
            # Try to load existing graph
            if self.persist and self.persistence_path:
                graph_path = os.path.join(self.persistence_path, f"{self.graph_name}.graphml")
                if os.path.exists(graph_path):
                    return graphiti.Graph.load(graph_path)
            
            # Create new graph if loading fails or not persisting
            return graphiti.Graph(name=self.graph_name)
        except Exception as e:
            print(f"Error loading graph: {e}. Creating new graph.")
            return graphiti.Graph(name=self.graph_name)
    
    def _save_graph(self):
        """Persist the graph to disk if enabled."""
        if self.persist and self.persistence_path:
            graph_path = os.path.join(self.persistence_path, f"{self.graph_name}.graphml")
            self.graph.save(graph_path)
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Dictionary of properties for the entity
            
        Returns:
            The entity ID
        """
        # Add the node if it doesn't exist
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id)
        
        # Set the entity type
        self.graph.set_node_property(entity_id, "type", entity_type)
        
        # Track entity type
        if entity_type not in self.entity_types:
            self.entity_types[entity_type] = set()
        self.entity_types[entity_type].add(entity_id)
        
        # Set properties
        for key, value in properties.items():
            self.graph.set_node_property(entity_id, key, value)
        
        # Add creation timestamp if not present
        if "created_at" not in properties:
            self.graph.set_node_property(entity_id, "created_at", datetime.datetime.now().isoformat())
        
        # Persist changes
        self._save_graph()
        
        return entity_id
    
    def add_relation(self, source_id: str, relation_type: str, target_id: str, 
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two entities.
        
        Args:
            source_id: ID of the source entity
            relation_type: Type of the relation
            target_id: ID of the target entity
            properties: Optional properties for the relation
            
        Returns:
            Unique identifier for the relation
        """
        # Ensure both entities exist
        if not self.graph.has_node(source_id):
            raise ValueError(f"Source entity {source_id} does not exist")
        
        if not self.graph.has_node(target_id):
            raise ValueError(f"Target entity {target_id} does not exist")
        
        # Generate a unique ID for the relation
        relation_id = str(uuid.uuid4())
        
        # Process properties
        if properties is None:
            properties = {}
            
        # Add relation ID and timestamp to properties
        properties["relation_id"] = relation_id
        if "created_at" not in properties:
            properties["created_at"] = datetime.datetime.now().isoformat()
        
        # Create the edge
        self.graph.add_edge(source_id, target_id, relation_type, properties=properties)
        
        # Track relation ID
        self.relation_ids[relation_id] = (source_id, relation_type, target_id)
        
        # Persist changes
        self._save_graph()
        
        return relation_id
    
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to the knowledge graph.
        
        Args:
            text: The text content
            properties: Optional metadata for the text
            
        Returns:
            Unique identifier for the text
        """
        # Generate a unique ID for the text
        text_id = str(uuid.uuid4())
        
        # Process properties
        if properties is None:
            properties = {}
            
        # Add content and type
        properties["content"] = text
        
        # Create an entity for the text
        return self.add_entity(text_id, "text", properties)
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      properties: Optional[Dict[str, Any]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities in the knowledge graph.
        
        Args:
            entity_type: Optional type to filter entities
            properties: Optional properties to filter entities
            limit: Maximum number of results
            
        Returns:
            List of matching entities with their properties
        """
        results = []
        
        # If entity_type is specified, use the tracked set for efficiency
        if entity_type and entity_type in self.entity_types:
            entity_ids_to_check = self.entity_types[entity_type]
        else:
            entity_ids_to_check = self.graph.node_ids()
        
        # Process each entity
        for entity_id in entity_ids_to_check:
            node_props = self.graph.get_node_properties(entity_id)
            
            # Skip if entity_type is specified and doesn't match
            if entity_type and node_props.get("type") != entity_type:
                continue
                
            # Check if properties match
            if properties:
                match = True
                for key, value in properties.items():
                    if key not in node_props or node_props[key] != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            # Add to results
            results.append({
                "id": entity_id,
                "type": node_props.get("type", "unknown"),
                "properties": node_props
            })
            
            # Check limit
            if len(results) >= limit:
                break
                
        return results
    
    def query_relations(self, 
                       source_id: Optional[str] = None,
                       relation_type: Optional[str] = None,
                       target_id: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relations in the knowledge graph.
        
        Args:
            source_id: Optional source entity ID
            relation_type: Optional relation type
            target_id: Optional target entity ID
            limit: Maximum number of results
            
        Returns:
            List of matching relations with their properties
        """
        results = []
        
        # Function to check if an edge matches the criteria
        def matches_criteria(edge):
            if source_id and edge["source"] != source_id:
                return False
            if relation_type and edge["label"] != relation_type:
                return False
            if target_id and edge["target"] != target_id:
                return False
            return True
        
        # Get all edges and filter
        for edge in self.graph.edges():
            if matches_criteria(edge):
                results.append({
                    "source_id": edge["source"],
                    "relation_type": edge["label"],
                    "target_id": edge["target"],
                    "relation_id": edge.get("properties", {}).get("relation_id", "unknown"),
                    "properties": edge.get("properties", {})
                })
                
                # Check limit
                if len(results) >= limit:
                    break
        
        return results
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform simple keyword-based search on the knowledge graph.
        
        Note: This is a basic implementation. For true semantic search,
        use the vector database implementation.
        
        Args:
            query: Keywords to search for
            limit: Maximum number of results
            
        Returns:
            List of relevant entities and text with relevance scores
        """
        results = []
        keywords = query.lower().split()
        
        # Search text entities
        text_entities = self.query_entities(entity_type="text")
        
        for entity in text_entities:
            content = entity["properties"].get("content", "").lower()
            
            # Count keyword matches
            match_count = sum(1 for keyword in keywords if keyword in content)
            
            if match_count > 0:
                score = match_count / len(keywords)
                results.append({
                    "id": entity["id"],
                    "type": "text",
                    "content": entity["properties"].get("content", ""),
                    "score": score,
                    "properties": entity["properties"]
                })
        
        # Search all other entities
        for entity_id in self.graph.node_ids():
            props = self.graph.get_node_properties(entity_id)
            entity_type = props.get("type")
            
            # Skip text entities (already processed)
            if entity_type == "text":
                continue
                
            # Convert properties to string for searching
            entity_text = str(props).lower()
            
            # Count keyword matches
            match_count = sum(1 for keyword in keywords if keyword in entity_text)
            
            if match_count > 0:
                score = match_count / len(keywords) * 0.7  # Lower weight than text matches
                results.append({
                    "id": entity_id,
                    "type": entity_type,
                    "score": score,
                    "properties": props
                })
                
        # Sort by score and apply limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data if found, None otherwise
        """
        if not self.graph.has_node(entity_id):
            return None
            
        props = self.graph.get_node_properties(entity_id)
        
        return {
            "id": entity_id,
            "type": props.get("type", "unknown"),
            "properties": props
        }
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties.
        
        Args:
            entity_id: The entity ID
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph.has_node(entity_id):
            return False
            
        # Update properties
        for key, value in properties.items():
            self.graph.set_node_property(entity_id, key, value)
            
        # Update modification timestamp
        self.graph.set_node_property(entity_id, "updated_at", datetime.datetime.now().isoformat())
        
        # Persist changes
        self._save_graph()
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph.has_node(entity_id):
            return False
            
        # Get entity type for tracking
        props = self.graph.get_node_properties(entity_id)
        entity_type = props.get("type")
        
        # Remove from entity type tracking
        if entity_type and entity_type in self.entity_types:
            self.entity_types[entity_type].discard(entity_id)
            
        # Remove from relation tracking
        relation_ids_to_remove = []
        for relation_id, (source, _, target) in self.relation_ids.items():
            if source == entity_id or target == entity_id:
                relation_ids_to_remove.append(relation_id)
                
        for relation_id in relation_ids_to_remove:
            del self.relation_ids[relation_id]
            
        # Remove the node from the graph (this also removes connected edges)
        self.graph.remove_node(entity_id)
        
        # Persist changes
        self._save_graph()
        
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from the knowledge graph.
        
        Args:
            relation_id: The relation ID
            
        Returns:
            True if successful, False otherwise
        """
        if relation_id not in self.relation_ids:
            return False
            
        # Get relation details
        source_id, relation_type, target_id = self.relation_ids[relation_id]
        
        # Find and remove the edge
        for edge in self.graph.edges():
            edge_props = edge.get("properties", {})
            if (edge["source"] == source_id and 
                edge["target"] == target_id and 
                edge["label"] == relation_type and
                edge_props.get("relation_id") == relation_id):
                
                self.graph.remove_edge(source_id, target_id, relation_type)
                del self.relation_ids[relation_id]
                
                # Persist changes
                self._save_graph()
                
                return True
                
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with statistics
        """
        # Count entities by type
        entity_counts = {entity_type: len(entities) for entity_type, entities in self.entity_types.items()}
        
        # Count relations by type
        relation_counts = {}
        for edge in self.graph.edges():
            relation_type = edge["label"]
            if relation_type not in relation_counts:
                relation_counts[relation_type] = 0
            relation_counts[relation_type] += 1
            
        return {
            "total_entities": self.graph.num_nodes(),
            "total_relations": self.graph.num_edges(),
            "entity_counts": entity_counts,
            "relation_counts": relation_counts
        } 