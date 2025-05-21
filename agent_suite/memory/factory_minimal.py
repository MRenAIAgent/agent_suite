"""
Minimal factory module for creating memory components.

This module provides a simplified factory to create basic memory components
without requiring external dependencies like Redis.
"""

from typing import Dict, Any, Optional, Union

from agent_suite.memory.memory_manager import MemoryManager
from agent_suite.memory.context.base import ContextMemory
from agent_suite.memory.context.in_memory import InMemoryContextMemory


class MemoryFactoryMinimal:
    """
    Minimal factory for creating memory components and managers.
    
    This class provides static methods to create memory components
    with minimal dependencies.
    """
    
    @staticmethod
    def create_context_memory() -> ContextMemory:
        """
        Create a basic in-memory context memory implementation.
                
        Returns:
            Implementation of ContextMemory
        """
        return InMemoryContextMemory()
    
    @staticmethod
    def create_memory_manager(user_id: Optional[str] = None) -> MemoryManager:
        """
        Create a minimal memory manager with basic in-memory components.
        
        Args:
            user_id: Optional user ID for user-specific memory
            
        Returns:
            Configured MemoryManager instance
        """
        # Create memory components
        context_memory = MemoryFactoryMinimal.create_context_memory()
        
        # Create and return memory manager without long-term memory
        return MemoryManager(
            context_memory=context_memory,
            long_term_memory=None,  # No long-term memory
            user_id=user_id
        ) 