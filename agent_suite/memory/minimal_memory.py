"""
Minimal memory module for compatibility testing.

This module includes simplified versions of the memory components
needed for testing the adapter without external dependencies.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union


# --- Base class for context memory ---

class ContextMemory(ABC):
    """
    Base class for context memory implementations.
    
    This defines the interface for storing, retrieving, and formatting
    context memory during agent execution.
    """
    
    @abstractmethod
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new memory item.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item (e.g. timestamps, user ID)
        """
        pass
    
    @abstractmethod
    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve memory items with optional filtering.
        
        Args:
            key: Optional key to filter items (if None, return all items)
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        pass
    
    @abstractmethod
    def get_formatted(self, format_type: str = "default") -> str:
        """
        Get formatted memory for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use (e.g. 'default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of context memory
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        pass


# --- In-memory implementation ---

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


# --- Adapter classes ---

class IntermediateStorageBase(ABC):
    """
    Simplified abstract base class for IntermediateStorage.
    
    This is a minimal version just for the compatibility test.
    """
    
    @abstractmethod
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    @abstractmethod
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        pass
    
    @abstractmethod
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def get_formatted_context(self, format_type: str = "default") -> str:
        pass
    
    @abstractmethod
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def clear(self) -> None:
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass


class ContextToIntermediateAdapter(IntermediateStorageBase):
    """
    Adapter that exposes a ContextMemory as an IntermediateStorage.
    
    This allows the new ContextMemory implementations to be
    used with the existing agent code.
    """
    
    def __init__(self, context_memory: ContextMemory):
        """
        Initialize with a ContextMemory instance.
        
        Args:
            context_memory: The memory implementation to adapt
        """
        self.memory = context_memory
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an intermediate step to storage.
        
        Args:
            step_type: Type of step (e.g., 'thought', 'action', 'tool_result')
            content: The content of the step
            metadata: Optional metadata about the step (e.g., timestamps)
        """
        self.memory.add(step_type, content, metadata)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a tool call result to storage.
        
        Args:
            tool_name: Name of the tool called
            tool_input: Input provided to the tool
            tool_output: Output returned by the tool
            metadata: Optional metadata about the tool call (e.g., latency)
        """
        tool_data = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output
        }
        self.memory.add("tool_call", tool_data, metadata)
    
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored intermediate steps.
        
        Args:
            step_types: Optional filter for specific step types
            limit: Optional limit on number of steps to return
            
        Returns:
            List of stored steps matching criteria
        """
        result = []
        
        if step_types and len(step_types) == 1:
            # Get specific step type
            steps = self.memory.get(step_types[0])
            if steps:
                result = steps if isinstance(steps, list) else [steps]
        else:
            # Get all steps
            all_data = self.memory.get()
            
            # Flatten the data
            for key, items in all_data.items():
                if step_types is None or key in step_types:
                    for item in items if isinstance(items, list) else [items]:
                        step = {
                            "type": key,
                            "content": item.get("value"),
                            "metadata": item.get("metadata", {})
                        }
                        result.append(step)
        
        # Apply limit if needed
        if limit is not None and limit > 0:
            result = result[-limit:]
            
        return result
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """
        Get formatted context string for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use (e.g., 'default', 'compact', 'detailed')
            
        Returns:
            Formatted string representation of relevant context
        """
        return self.memory.get_formatted(format_type)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get stored steps as message objects for LLM context.
        
        Args:
            max_tokens: Optional maximum token limit to respect
            
        Returns:
            List of message objects for LLM context
        """
        # This is a simplified implementation
        formatted_context = self.memory.get_formatted("default")
        
        # Create a simple message containing the context
        return [{"role": "system", "content": formatted_context}]
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored steps.
        
        Returns:
            Dictionary with statistics (e.g., count by type, token estimates)
        """
        all_data = self.memory.get()
        
        stats = {
            "total_steps": 0,
            "steps_by_type": {}
        }
        
        # Count steps by type
        for key, items in all_data.items():
            item_count = len(items) if isinstance(items, list) else 1
            stats["total_steps"] += item_count
            stats["steps_by_type"][key] = item_count
        
        return stats


class ContextMemoryAdapter:
    """
    Adapter utility class for memory compatibility.
    
    This class provides static methods to create adapters between
    IntermediateStorage and ContextMemory.
    """
    
    @staticmethod
    def adapt_context_to_intermediate(context_memory: ContextMemory) -> IntermediateStorageBase:
        """
        Adapt a ContextMemory to the IntermediateStorage interface.
        
        Args:
            context_memory: An instance of ContextMemory
            
        Returns:
            An IntermediateStorage implementation that delegates to the context memory
        """
        return ContextToIntermediateAdapter(context_memory) 