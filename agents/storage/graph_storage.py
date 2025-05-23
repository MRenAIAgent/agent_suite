"""Graph storage interface.

This module defines specialized interfaces for graph storage operations,
extending the base storage interface with graph-specific functionality.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Set

from agents.storage.base_storage import BaseStorage


class GraphStorage(BaseStorage[Dict[str, Any]]):
    """Interface for graph storage operations.
    
    Extends the base storage interface with graph-specific operations
    like node and relationship management.
    """
    
    async def create_node(
        self, 
        node_type: str, 
        properties: Dict[str, Any],
        **kwargs
    ) -> str:
        """Create a node in the graph.
        
        Args:
            node_type: Type of node to create
            properties: Properties for the node
            **kwargs: Additional node creation parameters
            
        Returns:
            Identifier for the created node
        """
        data = {
            "type": "node",
            "node_type": node_type,
            "properties": properties
        }
        return await self.store(f"node:{properties.get('id', None)}", data, **kwargs)
    
    async def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Create a relationship between nodes.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: Properties for the relationship
            **kwargs: Additional relationship creation parameters
            
        Returns:
            Identifier for the created relationship
        """
        data = {
            "type": "relationship",
            "source_id": source_id,
            "target_id": target_id,
            "relationship_type": relationship_type,
            "properties": properties or {}
        }
        key = f"rel:{source_id}:{relationship_type}:{target_id}"
        return await self.store(key, data, **kwargs)
    
    async def get_nodes(
        self, 
        node_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get nodes matching criteria.
        
        Args:
            node_type: Type of nodes to retrieve
            properties: Filter by properties
            limit: Maximum number of nodes to return
            **kwargs: Additional query parameters
            
        Returns:
            List of matching nodes
        """
        raise NotImplementedError("Node retrieval must be implemented by subclasses")
    
    async def get_relationships(
        self, 
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get relationships matching criteria.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: Filter by properties
            limit: Maximum number of relationships to return
            **kwargs: Additional query parameters
            
        Returns:
            List of matching relationships
        """
        raise NotImplementedError("Relationship retrieval must be implemented by subclasses")
    
    async def query(
        self, 
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a graph query.
        
        Args:
            query: The query to execute (e.g., Cypher for Neo4j)
            parameters: Query parameters
            **kwargs: Additional query execution parameters
            
        Returns:
            Query results
        """
        raise NotImplementedError("Graph query must be implemented by subclasses")
    
    async def traverse(
        self, 
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Traverse the graph from a starting node.
        
        Args:
            start_node_id: Starting node identifier
            relationship_types: Types of relationships to traverse
            direction: Direction of traversal ("outgoing", "incoming", "both")
            max_depth: Maximum traversal depth
            **kwargs: Additional traversal parameters
            
        Returns:
            Traversal results
        """
        raise NotImplementedError("Graph traversal must be implemented by subclasses") 