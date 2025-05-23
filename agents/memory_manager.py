# Compatibility layer for older imports
# This will be deprecated - use agents.memory.memory_manager instead

from agents.memory.memory_manager import MemoryManager

# Re-export the class to maintain backward compatibility
__all__ = ["MemoryManager"] 