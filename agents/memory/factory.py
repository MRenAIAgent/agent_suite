"""Factory functions for creating memory managers with various store configurations.

This module provides convenient factory functions for creating memory managers
with different storage backends, including Redis and in-memory storage.
"""

from typing import Optional, Dict, Any
from agents.memory.memory_manager import MemoryManager
from agents.memory.context_memory import ContextMemory
from agents.memory.longterm_memory import LongtermMemory
from agents.memory.stores.in_memory_store import InMemoryStore
from agents.memory.stores.redis_store import RedisStore

try:
    from agents.memory.stores.graphiti_store import GraphitiStore, GRAPHITI_AVAILABLE
except ImportError:
    GRAPHITI_AVAILABLE = False


def create_memory_manager(
    max_history: int = 20,
    use_longterm: bool = False,
    storage_dir: str = "./memory_storage",
    session_id: Optional[str] = None
) -> MemoryManager:
    """Create a memory manager with default in-memory storage.
    
    Args:
        max_history: Maximum number of history items to keep in context
        use_longterm: Whether to use longterm storage
        storage_dir: Directory for longterm storage
        session_id: Optional session ID for organizing memories
        
    Returns:
        Configured MemoryManager instance
    """
    return MemoryManager(
        max_history=max_history,
        use_longterm=use_longterm,
        storage_dir=storage_dir,
        session_id=session_id
    )


def create_redis_memory_manager(
    max_history: int = 20,
    use_longterm: bool = False,
    storage_dir: str = "./memory_storage",
    session_id: Optional[str] = None,
    redis_config: Optional[Dict[str, Any]] = None
) -> MemoryManager:
    """Create a memory manager with Redis-backed context storage.
    
    Args:
        max_history: Maximum number of history items to keep in context
        use_longterm: Whether to use longterm storage
        storage_dir: Directory for longterm storage
        session_id: Optional session ID for organizing memories
        redis_config: Redis configuration parameters (host, port, etc.)
        
    Returns:
        Configured MemoryManager instance with Redis for context memory
    """
    # Use default Redis config if none provided
    if redis_config is None:
        redis_config = {}
    
    # Create Redis store
    redis_store = RedisStore(**redis_config)
    
    # Create memory manager with Redis store for context
    return MemoryManager(
        max_history=max_history,
        use_longterm=use_longterm,
        storage_dir=storage_dir,
        session_id=session_id,
        context_store=redis_store
    )


def create_graphiti_memory_manager(
    max_history: int = 20,
    session_id: Optional[str] = None,
    db_path: str = "memory.graph",
    namespace: str = "agent",
    context_store_type: str = "in_memory"
) -> MemoryManager:
    """Create a memory manager with Graphiti-based longterm memory.
    
    This factory creates a memory manager that uses a graph database for
    long-term memory storage, allowing for more sophisticated semantic
    relationship queries and memory organization.
    
    Args:
        max_history: Maximum number of history items to keep in context
        session_id: Optional session ID for organizing memories (used as namespace if provided)
        db_path: Path to the graph database file
        namespace: Namespace for grouping memory nodes
        context_store_type: Type of store to use for context memory ("in_memory" or "redis")
        
    Returns:
        Configured MemoryManager instance with Graphiti for longterm memory
        
    Raises:
        ImportError: If Graphiti is not installed
    """
    if not GRAPHITI_AVAILABLE:
        raise ImportError(
            "Graphiti is required for GraphitiStore. "
            "Install it with: pip install graphiti"
        )
    
    # Use session_id as namespace if provided
    ns = session_id or namespace
    
    # Create context store
    if context_store_type == "redis":
        context_store = RedisStore(prefix=f"{ns}:")
    else:
        context_store = InMemoryStore()
    
    # Create a manager with custom longterm memory
    manager = MemoryManager(
        max_history=max_history,
        use_longterm=True,  # Always use longterm with Graphiti
        session_id=session_id,
        context_store=context_store
    )
    
    # Replace the file-based longterm memory with Graphiti-based memory
    graphiti_store = GraphitiStore(
        db_path=db_path,
        namespace=ns
    )
    
    # Create a new LongtermMemory with GraphitiStore
    manager.longterm_memory = LongtermMemory(
        max_history=max_history * 5,
        storage_dir="",  # Not used with GraphitiStore
        session_id=session_id,
        store=graphiti_store
    )
    
    return manager


def create_context_memory(
    max_history: int = 20,
    use_redis: bool = False,
    redis_config: Optional[Dict[str, Any]] = None
) -> ContextMemory:
    """Create a context memory manager with specified storage.
    
    Args:
        max_history: Maximum number of history items to keep
        use_redis: Whether to use Redis for storage
        redis_config: Redis configuration parameters (if use_redis is True)
        
    Returns:
        Configured ContextMemory instance
    """
    if use_redis:
        # Use default Redis config if none provided
        if redis_config is None:
            redis_config = {}
        
        # Create Redis store
        store = RedisStore(**redis_config)
    else:
        # Use in-memory store
        store = InMemoryStore()
    
    return ContextMemory(max_history=max_history, store=store) 