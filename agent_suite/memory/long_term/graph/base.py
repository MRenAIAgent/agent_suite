"""
Base class for graph-based memory implementations.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from agent_suite.memory.long_term.base import LongTermMemory


class GraphMemory(LongTermMemory):
    """
    Base class for graph-based memory implementations.
    
    This class extends the LongTermMemory base class with graph-specific
    methods and properties for storing and querying relationships.
    """
    
    @abstractmethod
    def store(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store structured data in the graph.
        
        Args:
            data: Dictionary containing entity data and relationships
            metadata: Optional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        pass
    
    @abstractmethod
    def add_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity node to the graph.
        
        Args:
            entity_type: Type of entity
            properties: Entity properties
            
        Returns:
            entity_id: Unique identifier for the stored entity
        """
        pass
    
    @abstractmethod
    def add_relationship(self, from_entity: str, relationship_type: str, to_entity: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between entities.
        
        Args:
            from_entity: Source entity ID
            relationship_type: Type of relationship
            to_entity: Target entity ID
            properties: Optional relationship properties
            
        Returns:
            relationship_id: Unique identifier for the relationship
        """
        pass
    
    @abstractmethod
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results
            
        Returns:
            List of matching subgraphs or paths
        """
        pass 