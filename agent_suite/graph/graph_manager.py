#!/usr/bin/env python
"""
GraphManager - A centralized graph database manager for the Agent Suite.

This module provides a unified interface for managing graph data across
different components of the Agent Suite. It supports in-memory graphs
and persistence via GraphML files.
"""

import os
import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional, Set, Tuple

import networkx as nx

# Set up logging
logger = logging.getLogger(__name__)

class GraphManager:
    """
    Centralized graph manager for the Agent Suite.
    
    This class provides unified access to graph operations and
    persistence across the Agent Suite.
    """
    
    def __init__(
        self,
        graph_name: str = "default_graph",
        persist: bool = True,
        persistence_path: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the GraphManager.
        
        Args:
            graph_name: Name of the graph
            persist: Whether to persist the graph to disk
            persistence_path: Path to store persistent graphs
            api_key: Optional API key for cloud graph storage (if supported)
        """
        self.graph_name = graph_name
        self.persist = persist
        self.persistence_path = persistence_path
        self.api_key = api_key
        
        # Create directories if needed
        if self.persist and self.persistence_path:
            os.makedirs(self.persistence_path, exist_ok=True)
        
        # Initialize the graph
        self.graph = self._setup_graph()
        
        # Tracking dictionaries for efficient querying
        self.entity_types: Dict[str, Set[str]] = {}  # entity_type -> set of entity_ids
        self.relation_ids: Dict[str, Tuple[str, str, str]] = {}  # relation_id -> (source_id, relation_type, target_id)
        
        # Load entity types and relation IDs
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load entity types and relation IDs from the graph."""
        # Rebuild entity types dictionary
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if 'type' in node_data:
                entity_type = node_data['type']
                if entity_type not in self.entity_types:
                    self.entity_types[entity_type] = set()
                self.entity_types[entity_type].add(node_id)
        
        # Rebuild relation IDs dictionary
        for source, target, edge_data in self.graph.edges(data=True):
            if 'relation_id' in edge_data:
                relation_id = edge_data['relation_id']
                relation_type = edge_data.get('relation_type', 'unknown')
                self.relation_ids[relation_id] = (source, relation_type, target)
    
    def _setup_graph(self):
        """Initialize or load a graph."""
        try:
            if self.persist and self.persistence_path:
                graph_path = os.path.join(self.persistence_path, f"{self.graph_name}.graphml")
                if os.path.exists(graph_path):
                    logger.info(f"Loading existing graph from {graph_path}")
                    try:
                        return nx.read_graphml(graph_path)
                    except Exception as e:
                        logger.error(f"Error reading GraphML: {e}")
            
            # Create new graph if loading fails or not persisting
            logger.info(f"Creating new graph: {self.graph_name}")
            return nx.MultiDiGraph(name=self.graph_name)
        except Exception as e:
            logger.error(f"Error loading graph: {e}. Creating new graph.")
            return nx.MultiDiGraph(name=self.graph_name)
    
    def _save_graph(self):
        """Persist the graph to disk if enabled."""
        if self.persist and self.persistence_path:
            graph_path = os.path.join(self.persistence_path, f"{self.graph_name}.graphml")
            try:
                # Create a copy of the graph for saving
                # GraphML doesn't support nested dictionaries or None values
                save_graph = nx.MultiDiGraph()
                
                # Copy nodes with string-converted properties
                for node, node_data in self.graph.nodes(data=True):
                    node_attrs = {}
                    for key, value in node_data.items():
                        if isinstance(value, dict):
                            # Convert dictionaries to strings
                            node_attrs[key] = str(value)
                        elif value is None:
                            # Convert None to empty string
                            node_attrs[key] = ""
                        else:
                            node_attrs[key] = value
                    save_graph.add_node(node, **node_attrs)
                
                # Copy edges with string-converted properties
                for u, v, key, edge_data in self.graph.edges(data=True, keys=True):
                    edge_attrs = {}
                    for key_attr, value in edge_data.items():
                        if isinstance(value, dict):
                            # Convert dictionaries to strings
                            edge_attrs[key_attr] = str(value)
                        elif value is None:
                            # Convert None to empty string
                            edge_attrs[key_attr] = ""
                        else:
                            edge_attrs[key_attr] = value
                    save_graph.add_edge(u, v, key=key, **edge_attrs)
                
                # Save the graph
                nx.write_graphml(save_graph, graph_path)
                logger.debug(f"Graph saved to {graph_path}")
            except Exception as e:
                logger.error(f"Error saving graph: {e}")
    
    def add_entity(self, entity_id: Optional[str] = None, entity_type: str = "entity", 
                 properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an entity to the graph.
        
        Args:
            entity_id: Optional unique identifier (generated if None)
            entity_type: Type of the entity
            properties: Dictionary of properties for the entity
            
        Returns:
            The entity ID
        """
        # Generate ID if not provided
        if entity_id is None:
            entity_id = str(uuid.uuid4())
            
        # Initialize properties if None
        if properties is None:
            properties = {}
        
        # Add the node with properties
        all_properties = properties.copy()
        all_properties["type"] = entity_type
        
        # Add creation timestamp if not present
        if "created_at" not in all_properties:
            all_properties["created_at"] = datetime.datetime.now().isoformat()
        
        # Add node to graph
        self.graph.add_node(entity_id, **all_properties)
        
        # Track entity type
        if entity_type not in self.entity_types:
            self.entity_types[entity_type] = set()
        self.entity_types[entity_type].add(entity_id)
        
        # Persist changes
        self._save_graph()
        
        return entity_id
    
    def add_relation(self, source_id: str, relation_type: str, target_id: str, 
                    relation_id: Optional[str] = None,
                    properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relation between two entities.
        
        Args:
            source_id: ID of the source entity
            relation_type: Type of the relation
            target_id: ID of the target entity
            relation_id: Optional unique identifier (generated if None)
            properties: Optional properties for the relation
            
        Returns:
            Unique identifier for the relation
        """
        # Ensure both entities exist
        if not self.graph.has_node(source_id):
            raise ValueError(f"Source entity {source_id} does not exist")
        
        if not self.graph.has_node(target_id):
            raise ValueError(f"Target entity {target_id} does not exist")
        
        # Generate a unique ID for the relation if not provided
        if relation_id is None:
            relation_id = str(uuid.uuid4())
        
        # Process properties
        if properties is None:
            properties = {}
            
        # Create edge properties 
        edge_props = properties.copy()
        edge_props["relation_id"] = relation_id
        edge_props["relation_type"] = relation_type
        
        if "created_at" not in edge_props:
            edge_props["created_at"] = datetime.datetime.now().isoformat()
        
        # Create the edge
        self.graph.add_edge(source_id, target_id, **edge_props)
        
        # Track relation ID
        self.relation_ids[relation_id] = (source_id, relation_type, target_id)
        
        # Persist changes
        self._save_graph()
        
        return relation_id
    
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to the graph.
        
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
            
        # Add content to properties
        properties["content"] = text
        
        # Create an entity for the text
        return self.add_entity(entity_id=text_id, entity_type="text", properties=properties)
    
    def query_entities(self, entity_type: Optional[str] = None, 
                      properties: Optional[Dict[str, Any]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities in the graph.
        
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
            entity_ids_to_check = self.graph.nodes()
        
        # Process each entity
        for entity_id in entity_ids_to_check:
            node_props = self.graph.nodes[entity_id]
            
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
        Query relations in the graph.
        
        Args:
            source_id: Optional ID of the source entity
            relation_type: Optional type of the relation
            target_id: Optional ID of the target entity
            limit: Maximum number of results
            
        Returns:
            List of matching relations with their properties
        """
        results = []
        
        # Define a function to check if an edge matches the criteria
        def matches_criteria(source, target, edge_data):
            if source_id is not None and source != source_id:
                return False
            
            if target_id is not None and target != target_id:
                return False
            
            if relation_type is not None and edge_data.get("relation_type") != relation_type:
                return False
            
            return True
        
        # Iterate through edges
        for source, target, edge_key in self.graph.edges(keys=True):
            edge_data = self.graph.edges[source, target, edge_key]
            
            if matches_criteria(source, target, edge_data):
                # Get source and target node types
                source_type = self.graph.nodes[source].get("type", "unknown")
                target_type = self.graph.nodes[target].get("type", "unknown")
                
                # Create a result entry
                results.append({
                    "id": edge_data.get("relation_id", str(uuid.uuid4())),
                    "source_id": source,
                    "source_type": source_type,
                    "relation_type": edge_data.get("relation_type", "unknown"),
                    "target_id": target,
                    "target_type": target_type,
                    "properties": edge_data
                })
                
                # Check limit
                if len(results) >= limit:
                    break
        
        return results
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search entities and relations for a keyword.
        
        Args:
            query: Search term
            limit: Maximum number of results
            
        Returns:
            List of matching entities and relations
        """
        results = []
        query = query.lower()
        
        # Search entities
        for node_id, node_data in self.graph.nodes(data=True):
            # Check for match in node properties
            found = False
            for key, value in node_data.items():
                if isinstance(value, str) and query in value.lower():
                    found = True
                    break
            
            if found:
                results.append({
                    "id": node_id,
                    "type": "entity",
                    "entity_type": node_data.get("type", "unknown"),
                    "properties": node_data,
                    "match_type": "entity"
                })
                
                if len(results) >= limit:
                    return results
        
        # Search relations
        for source, target, edge_key in self.graph.edges(keys=True):
            edge_data = self.graph.edges[source, target, edge_key]
            
            # Check for match in edge properties
            found = False
            relation_type = edge_data.get("relation_type", "unknown")
            
            # Check relation type first
            if query in relation_type.lower():
                found = True
            else:
                # Check other properties
                for key, value in edge_data.items():
                    if isinstance(value, str) and query in value.lower():
                        found = True
                        break
            
            if found:
                results.append({
                    "id": edge_data.get("relation_id", str(uuid.uuid4())),
                    "source_id": source,
                    "relation_type": relation_type,
                    "target_id": target,
                    "properties": edge_data,
                    "match_type": "relation"
                })
                
                if len(results) >= limit:
                    return results
        
        return results
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Entity data or None if not found
        """
        if not self.graph.has_node(entity_id):
            return None
        
        node_props = self.graph.nodes[entity_id]
        
        return {
            "id": entity_id,
            "type": node_props.get("type", "unknown"),
            "properties": node_props
        }
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties.
        
        Args:
            entity_id: ID of the entity
            properties: New or updated properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph.has_node(entity_id):
            return False
        
        # Update each property
        for key, value in properties.items():
            self.graph.nodes[entity_id][key] = value
        
        # Add updated timestamp
        self.graph.nodes[entity_id]["updated_at"] = datetime.datetime.now().isoformat()
        
        # Update entity type tracking if type changed
        if "type" in properties:
            new_type = properties["type"]
            old_type = None
            
            # Find and remove from old type
            for entity_type, entity_ids in self.entity_types.items():
                if entity_id in entity_ids and entity_type != new_type:
                    old_type = entity_type
                    entity_ids.remove(entity_id)
                    break
            
            # Add to new type
            if new_type not in self.entity_types:
                self.entity_types[new_type] = set()
            self.entity_types[new_type].add(entity_id)
        
        # Persist changes
        self._save_graph()
        
        return True
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and its relations.
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph.has_node(entity_id):
            return False
        
        # Get entity type for tracking
        entity_type = self.graph.nodes[entity_id].get("type", "unknown")
        
        # Remove from entity type tracking
        if entity_type in self.entity_types and entity_id in self.entity_types[entity_type]:
            self.entity_types[entity_type].remove(entity_id)
        
        # Identify relations to be removed
        relations_to_remove = []
        for relation_id, (source_id, _, target_id) in self.relation_ids.items():
            if source_id == entity_id or target_id == entity_id:
                relations_to_remove.append(relation_id)
        
        # Remove from relation tracking
        for relation_id in relations_to_remove:
            if relation_id in self.relation_ids:
                del self.relation_ids[relation_id]
        
        # Remove the node (this also removes connected edges)
        self.graph.remove_node(entity_id)
        
        # Persist changes
        self._save_graph()
        
        return True
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation by ID.
        
        Args:
            relation_id: ID of the relation
            
        Returns:
            True if successful, False otherwise
        """
        if relation_id not in self.relation_ids:
            return False
        
        source_id, relation_type, target_id = self.relation_ids[relation_id]
        
        # Find the edge with the matching relation_id
        edge_keys_to_remove = []
        for edge_key in self.graph.get_edge_data(source_id, target_id):
            edge_data = self.graph.edges[source_id, target_id, edge_key]
            if edge_data.get("relation_id") == relation_id:
                edge_keys_to_remove.append(edge_key)
        
        # Remove the edges
        for edge_key in edge_keys_to_remove:
            self.graph.remove_edge(source_id, target_id, edge_key)
        
        # Remove from tracking
        del self.relation_ids[relation_id]
        
        # Persist changes
        self._save_graph()
        
        return True
    
    def get_neighbors(self, entity_id: str, relation_type: Optional[str] = None, 
                     direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get neighbors of an entity.
        
        Args:
            entity_id: ID of the entity
            relation_type: Optional type of relation to filter by
            direction: Direction of relations ('in', 'out', or 'both')
            
        Returns:
            List of neighboring entities with relation info
        """
        if not self.graph.has_node(entity_id):
            return []
        
        results = []
        
        # Get outgoing connections if needed
        if direction == "out" or direction == "both":
            for _, neighbor_id, edge_key in self.graph.out_edges(entity_id, keys=True):
                edge_data = self.graph.edges[entity_id, neighbor_id, edge_key]
                edge_relation_type = edge_data.get("relation_type", "unknown")
                
                # Skip if relation_type specified and doesn't match
                if relation_type is not None and edge_relation_type != relation_type:
                    continue
                
                # Get neighbor information
                neighbor_props = self.graph.nodes[neighbor_id]
                
                results.append({
                    "entity_id": neighbor_id,
                    "entity_type": neighbor_props.get("type", "unknown"),
                    "relation_id": edge_data.get("relation_id", str(uuid.uuid4())),
                    "relation_type": edge_relation_type,
                    "direction": "out",
                    "properties": neighbor_props
                })
        
        # Get incoming connections if needed
        if direction == "in" or direction == "both":
            for neighbor_id, _, edge_key in self.graph.in_edges(entity_id, keys=True):
                edge_data = self.graph.edges[neighbor_id, entity_id, edge_key]
                edge_relation_type = edge_data.get("relation_type", "unknown")
                
                # Skip if relation_type specified and doesn't match
                if relation_type is not None and edge_relation_type != relation_type:
                    continue
                
                # Get neighbor information
                neighbor_props = self.graph.nodes[neighbor_id]
                
                results.append({
                    "entity_id": neighbor_id,
                    "entity_type": neighbor_props.get("type", "unknown"),
                    "relation_id": edge_data.get("relation_id", str(uuid.uuid4())),
                    "relation_type": edge_relation_type,
                    "direction": "in",
                    "properties": neighbor_props
                })
        
        return results
    
    def graph_traversal(self, start_id: str, max_depth: int = 2, 
                       relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Traverse the graph from a starting node.
        
        Args:
            start_id: ID of the starting entity
            max_depth: Maximum traversal depth
            relation_types: Optional list of relation types to follow
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        if not self.graph.has_node(start_id):
            return {"nodes": [], "edges": []}
        
        nodes = {}  # id -> node data
        edges = {}  # (source_id, target_id, key) -> edge data
        
        # Add the starting node
        nodes[start_id] = self.graph.nodes[start_id]
        
        def traverse(node_id, current_depth):
            if current_depth >= max_depth:
                return
            
            # Explore outgoing edges
            for source, target, edge_key in self.graph.out_edges(node_id, keys=True):
                edge_data = self.graph.edges[source, target, edge_key]
                relation_type = edge_data.get("relation_type", "unknown")
                
                # Skip if we're filtering by relation type
                if relation_types is not None and relation_type not in relation_types:
                    continue
                
                # Add target node if not seen
                if target not in nodes:
                    nodes[target] = self.graph.nodes[target]
                    
                    # Continue traversal from this node
                    traverse(target, current_depth + 1)
                
                # Add the edge
                edges[(source, target, edge_key)] = edge_data
        
        # Start traversal
        traverse(start_id, 0)
        
        # Format results
        formatted_nodes = []
        for node_id, node_data in nodes.items():
            formatted_nodes.append({
                "id": node_id,
                "type": node_data.get("type", "unknown"),
                "properties": node_data
            })
        
        formatted_edges = []
        for (source, target, edge_key), edge_data in edges.items():
            formatted_edges.append({
                "id": edge_data.get("relation_id", str(uuid.uuid4())),
                "source": source,
                "target": target,
                "type": edge_data.get("relation_type", "unknown"),
                "properties": edge_data
            })
        
        return {
            "nodes": formatted_nodes,
            "edges": formatted_edges
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "name": self.graph_name,
            "total_entities": self.graph.number_of_nodes(),
            "total_relations": self.graph.number_of_edges(),
            "entity_counts": {},
            "relation_counts": {}
        }
        
        # Count entities by type
        for entity_type, entity_ids in self.entity_types.items():
            stats["entity_counts"][entity_type] = len(entity_ids)
        
        # Count relations by type
        relation_type_counts = {}
        for _, _, edge_data in self.graph.edges(data=True):
            relation_type = edge_data.get("relation_type", "unknown")
            relation_type_counts[relation_type] = relation_type_counts.get(relation_type, 0) + 1
        
        stats["relation_counts"] = relation_type_counts
        
        return stats 