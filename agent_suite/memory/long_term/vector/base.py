"""
Base class for vector-based memory implementations.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

from agent_suite.memory.long_term.base import LongTermMemory


class VectorMemory(LongTermMemory):
    """
    Base class for vector-based memory implementations.
    
    This class extends the LongTermMemory base class with vector-specific
    methods and properties for semantic search.
    """
    
    @abstractmethod
    def store(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store text data with vector embedding.
        
        Args:
            data: Text to store
            metadata: Optional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        pass
    
    @abstractmethod
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query for similar vectors.
        
        Args:
            query: Text query or dict with query parameters
            limit: Maximum number of results
            
        Returns:
            List of matching memories with similarity scores
        """
        pass 