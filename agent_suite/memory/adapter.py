"""
Adapter module for connecting existing IntermediateStorage to the new context memory system.

This module provides adapter classes that allow the existing agent implementation 
to work with the new memory system without requiring changes to the agent code.
"""

from typing import Dict, List, Any, Optional

from agent_suite.agents.storage.intermediate_storage import IntermediateStorageBase
from agent_suite.memory.context.base import ContextMemory


class ContextMemoryAdapter:
    """
    Adapter to connect existing IntermediateStorage to new ContextMemory.
    
    This adapter allows an IntermediateStorage implementation to be
    used with the new memory system or vice versa.
    """
    
    @staticmethod
    def adapt_intermediate_to_context(intermediate_storage: IntermediateStorageBase) -> ContextMemory:
        """
        Adapt an existing IntermediateStorage to the new ContextMemory interface.
        
        Args:
            intermediate_storage: An instance of IntermediateStorageBase
            
        Returns:
            A ContextMemory implementation that delegates to the intermediate storage
        """
        return IntermediateToContextAdapter(intermediate_storage)
    
    @staticmethod
    def adapt_context_to_intermediate(context_memory: ContextMemory) -> IntermediateStorageBase:
        """
        Adapt a new ContextMemory to the existing IntermediateStorage interface.
        
        Args:
            context_memory: An instance of ContextMemory
            
        Returns:
            An IntermediateStorage implementation that delegates to the context memory
        """
        return ContextToIntermediateAdapter(context_memory)


class IntermediateToContextAdapter(ContextMemory):
    """
    Adapter that exposes an IntermediateStorage as a ContextMemory.
    
    This allows existing IntermediateStorage implementations to be
    used with the new memory system.
    """
    
    def __init__(self, intermediate_storage: IntermediateStorageBase):
        """
        Initialize with an IntermediateStorage instance.
        
        Args:
            intermediate_storage: The storage implementation to adapt
        """
        self.storage = intermediate_storage
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new memory item.
        
        Args:
            key: The key/type of the memory item
            value: The value to store
            metadata: Optional metadata about this item
        """
        # Map to appropriate add method based on key
        if key == "tool_call" and isinstance(value, dict) and "tool_name" in value:
            self.storage.add_tool_call(
                value["tool_name"],
                value.get("input", {}),
                value.get("output", {}),
                metadata
            )
        else:
            self.storage.add_step(key, value, metadata)
    
    def get(self, key: Optional[str] = None) -> Any:
        """
        Retrieve memory items with optional filtering.
        
        Args:
            key: Optional key to filter items (if None, return all items)
            
        Returns:
            Memory items matching the key, or all items if key is None
        """
        # Get steps with optional type filter
        return self.storage.get_steps([key] if key else None)
    
    def get_formatted(self, format_type: str = "default") -> str:
        """
        Get formatted memory for inclusion in LLM prompts.
        
        Args:
            format_type: Type of formatting to use
            
        Returns:
            Formatted string representation of context memory
        """
        return self.storage.get_formatted_context(format_type)
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.storage.clear()


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