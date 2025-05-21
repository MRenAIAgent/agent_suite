"""
In-memory implementation of context memory.
"""

import json
from typing import Dict, List, Any, Optional

from agent_suite.memory.context.base import ContextMemory


class InMemoryContextMemory(ContextMemory):
    """
    Simple in-memory implementation of context memory.
    
    This implementation stores all context memory in local Python dictionaries
    and lists, making it fast but non-persistent.
    """
    
    def __init__(self):
        """
        Initialize empty memory storage.
        """
        self.memory = {}
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new memory item.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item
        """
        if key not in self.memory:
            self.memory[key] = []
        
        # Store as a dict with value and metadata
        self.memory[key].append({
            "value": value,
            "metadata": metadata or {}
        })
    
    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve memory items with optional filtering.
        
        Args:
            key: Optional key to filter items (if None, return all items)
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        if key is None:
            # Return all memory
            return self.memory
        
        # Return memory for specific key
        return self.memory.get(key, [])
    
    def get_formatted(self, format_type: str = "default") -> str:
        """
        Get formatted memory for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use ('default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of context memory
        """
        if format_type == "default":
            return self._format_default()
        elif format_type == "compact":
            return self._format_compact()
        elif format_type == "detailed":
            return self._format_detailed()
        else:
            return self._format_default()
    
    def clear(self) -> None:
        """
        Clear all memory items.
        """
        self.memory = {}
    
    def _format_default(self) -> str:
        """
        Default format for memory presentation.
        """
        result = []
        for key, items in self.memory.items():
            result.append(f"==== {key.upper()} ====")
            for item in items:
                result.append(f"- {item['value']}")
            result.append("")  # Empty line
        return "\n".join(result)
    
    def _format_compact(self) -> str:
        """
        Compact format for memory presentation.
        """
        result = []
        for key, items in self.memory.items():
            values = [str(item["value"]) for item in items]
            result.append(f"{key}: {', '.join(values)}")
        return "\n".join(result)
    
    def _format_detailed(self) -> str:
        """
        Detailed format for memory presentation including metadata.
        """
        result = []
        for key, items in self.memory.items():
            result.append(f"==== {key.upper()} ====")
            for idx, item in enumerate(items):
                result.append(f"Item {idx+1}:")
                result.append(f"  Value: {item['value']}")
                result.append(f"  Metadata: {json.dumps(item['metadata'], indent=2)}")
            result.append("")  # Empty line
        return "\n".join(result) 