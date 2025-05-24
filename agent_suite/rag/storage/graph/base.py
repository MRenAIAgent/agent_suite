"""Graph storage adaptor interface for the RAG system.

This module provides the abstract base class for graph storage adaptors in the RAG system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

from agent_suite.rag.storage.base import StorageAdaptor
from agent_suite.rag.models.knowledge import Entity, Relationship, KnowledgeGraph


class GraphStorageAdaptor(StorageAdaptor, ABC):
    """Base interface for graph storage adaptors."""
    
    @abstractmethod
    async def store_entity(self, entity: Entity) -> str:
        """
        Store an entity in the graph.
        
        Args:
            entity: The entity to store
            
        Returns:
            The ID of the stored entity
        """
        pass
    
    @abstractmethod
    async def store_relationship(self, relationship: Relationship) -> str:
        """
        Store a relationship in the graph.
        
        Args:
            relationship: The relationship to store
            
        Returns:
            The ID of the stored relationship
        """
        pass
    
    @abstractmethod
    async def store_knowledge_graph(self, graph: KnowledgeGraph) -> str:
        """
        Store a complete knowledge graph.
        
        Args:
            graph: The knowledge graph to store
            
        Returns:
            The ID of the stored graph
        """
        pass
    
    @abstractmethod
    async def query_graph(self, query_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a graph query.
        
        Args:
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query results
        """
        pass
    
    @abstractmethod
    async def find_paths(self, start_entity_id: str, end_entity_id: str, 
                       max_depth: int = 3) -> List[List[Union[Entity, Relationship]]]:
        """
        Find paths between entities.
        
        Args:
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each a list of alternating entities and relationships)
        """
        pass 