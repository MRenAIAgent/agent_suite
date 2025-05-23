# Compatibility layer for the renamed BaseStore
# This will be deprecated - use agents.memory.stores.store instead

from agents.memory.stores.store import Store

# Alias for backward compatibility
BaseStore = Store

__all__ = ["BaseStore"] 