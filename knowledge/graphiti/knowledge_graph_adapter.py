"""
Adapter for using the centralized GraphManager with the knowledge module.

This adapter connects the knowledge module's interface with the centralized
graph infrastructure, promoting code reuse and consistency.
"""

from typing import Any, Dict, List, Optional, Union

from knowledge.base import KnowledgeBase
from agent_suite.graph import GraphManager


class GraphitiKnowledgeGraphAdapter(KnowledgeBase):
    """
    Knowledge Graph implementation using the centralized GraphManager.
    
    This adapter allows the knowledge module to use the shared graph infrastructure
    while maintaining its existing interface.
    """
    
    def __init__(self, 
                graph_name: str = "knowledge_graph", 
                persist: bool = True,
                persistence_path: Optional[str] = None):
        """
        Initialize the knowledge graph adapter.
        
        Args:
            graph_name: Name of the graph
            persist: Whether to persist the graph to disk
            persistence_path: Optional path for persistence
        """
        # Initialize the centralized graph manager
        self.graph_manager = GraphManager(
            graph_name=graph_name,
            persist=persist,
            persistence_path=persistence_path
        )
        
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
        return self.graph_manager.add_entity(
            entity_id=entity_id, 
            entity_type=entity_type, 
            properties=properties
        )
    
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
        return self.graph_manager.add_relation(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
            properties=properties
        )
    
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to the knowledge graph.
        
        Args:
            text: The text content
            properties: Optional metadata for the text
            
        Returns:
            Unique identifier for the text
        """
        return self.graph_manager.add_text(text=text, properties=properties)
    
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
        return self.graph_manager.query_entities(
            entity_type=entity_type,
            properties=properties,
            limit=limit
        )
    
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
        return self.graph_manager.query_relations(
            source_id=source_id,
            relation_type=relation_type,
            target_id=target_id,
            limit=limit
        )
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform simple keyword-based search on the knowledge graph.
        
        Note: This is a basic implementation using keyword matching.
        For true semantic search, use the vector database implementation.
        
        Args:
            query: Keywords to search for
            limit: Maximum number of results
            
        Returns:
            List of relevant entities and text with relevance scores
        """
        return self.graph_manager.keyword_search(query=query, limit=limit)
    
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data if found, None otherwise
        """
        return self.graph_manager.get_entity(entity_id)
    
    def update_entity(self, entity_id: str, properties: Dict[str, Any]) -> bool:
        """
        Update an entity's properties.
        
        Args:
            entity_id: The entity ID
            properties: Properties to update
            
        Returns:
            True if successful, False otherwise
        """
        return self.graph_manager.update_entity(entity_id, properties)
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity from the knowledge graph.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.graph_manager.delete_entity(entity_id)
    
    def delete_relation(self, relation_id: str) -> bool:
        """
        Delete a relation from the knowledge graph.
        
        Args:
            relation_id: The relation ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.graph_manager.delete_relation(relation_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with statistics
        """
        return self.graph_manager.get_stats() 