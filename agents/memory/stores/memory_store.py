# Compatibility layer for the renamed MemoryStore
# This will be deprecated - use agents.memory.stores.in_memory_store instead

from agents.memory.stores.in_memory_store import InMemoryStore

# Alias for backward compatibility
MemoryStore = InMemoryStore

__all__ = ["MemoryStore"] 