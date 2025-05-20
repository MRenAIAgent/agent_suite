"""
Base classes for the knowledge module.

This module defines the abstract interfaces that all knowledge implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class KnowledgeBase(ABC):
    """
    Abstract base class for all knowledge storage backends.
    
    This defines the interface that all knowledge implementations must implement.
    """
    
    @abstractmethod
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity to the knowledge base.
        
        Args:
            entity_id: Unique identifier for the entity
            entity_type: Type of the entity
            properties: Dictionary of properties for the entity
            
        Returns:
            The entity ID
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def add_text(self, text: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add unstructured text to the knowledge base.
        
        Args:
            text: The text content
            properties: Optional metadata for the text
            
        Returns:
            Unique identifier for the text
        """
        pass
    
    @abstractmethod
    def query_entities(self, entity_type: Optional[str] = None, 
                      properties: Optional[Dict[str, Any]] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query entities in the knowledge base.
        
        Args:
            entity_type: Optional type to filter entities
            properties: Optional properties to filter entities
            limit: Maximum number of results
            
        Returns:
            List of matching entities with their properties
        """
        pass
    
    @abstractmethod
    def query_relations(self, 
                       source_id: Optional[str] = None,
                       relation_type: Optional[str] = None,
                       target_id: Optional[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relations in the knowledge base.
        
        Args:
            source_id: Optional source entity ID
            relation_type: Optional relation type
            target_id: Optional target entity ID
            limit: Maximum number of results
            
        Returns:
            List of matching relations with their properties
        """
        pass
    
    @abstractmethod
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search on the knowledge base.
        
        Args:
            query: The search query
            limit: Maximum number of results
            
        Returns:
            List of relevant items with relevance scores
        """
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            Entity data if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        pass 