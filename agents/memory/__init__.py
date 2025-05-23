"""Memory management module for agent systems.

This module provides components for managing agent memory, including
short-term context memory and long-term persistent memory.
"""

from agents.memory.memory_manager import MemoryManager
from agents.memory.context_memory import ContextMemory
from agents.memory.longterm_memory import LongtermMemory
from agents.memory.base_memory_manager import MemoryManagerBase
from agents.memory.factory import (
    create_memory_manager,
    create_redis_memory_manager,
    create_graphiti_memory_manager,
    create_context_memory
)

# Import GraphitiStore if available
try:
    from agents.memory.stores.graphiti_store import GraphitiStore
    has_graphiti = True
except ImportError:
    has_graphiti = False

__all__ = [
    'MemoryManager',
    'ContextMemory',
    'LongtermMemory',
    'MemoryManagerBase',
    'create_memory_manager',
    'create_redis_memory_manager',
    'create_graphiti_memory_manager',
    'create_context_memory',
    'has_graphiti'
]

# Add GraphitiStore to __all__ if available
if has_graphiti:
    __all__.append('GraphitiStore') 