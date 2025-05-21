#!/usr/bin/env python3
"""
Completely self-contained test of memory adapter functionality.

This script contains all necessary code in a single file to demonstrate
the adapter pattern for connecting React agents with the new memory system.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


# ====================== New Memory System ======================

class ContextMemory(ABC):
    """Base class for context memory implementations."""
    
    @abstractmethod
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory item."""
        pass
    
    @abstractmethod
    def get(self, key: Optional[str] = None) -> Any:
        """Retrieve memory items with optional filtering."""
        pass
    
    @abstractmethod
    def get_formatted(self, format_type: str = "default") -> str:
        """Get formatted memory for inclusion in LLM prompts."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memory items."""
        pass


class InMemoryContextMemory(ContextMemory):
    """Simple in-memory implementation of context memory."""
    
    def __init__(self):
        """Initialize empty memory storage."""
        self.memory = {}
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory item."""
        if key not in self.memory:
            self.memory[key] = []
        
        # Store as a dict with value and metadata
        self.memory[key].append({
            "value": value,
            "metadata": metadata or {}
        })
    
    def get(self, key: Optional[str] = None) -> Any:
        """Retrieve memory items with optional filtering."""
        if key is None:
            # Return all memory
            return self.memory
        
        # Return memory for specific key
        return self.memory.get(key, [])
    
    def get_formatted(self, format_type: str = "default") -> str:
        """Get formatted memory for inclusion in LLM prompts."""
        if format_type == "default":
            return self._format_default()
        elif format_type == "compact":
            return self._format_compact()
        elif format_type == "detailed":
            return self._format_detailed()
        else:
            return self._format_default()
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.memory = {}
    
    def _format_default(self) -> str:
        """Default format for memory presentation."""
        result = []
        for key, items in self.memory.items():
            result.append(f"==== {key.upper()} ====")
            for item in items:
                result.append(f"- {item['value']}")
            result.append("")  # Empty line
        return "\n".join(result)
    
    def _format_compact(self) -> str:
        """Compact format for memory presentation."""
        result = []
        for key, items in self.memory.items():
            values = [str(item["value"]) for item in items]
            result.append(f"{key}: {', '.join(values)}")
        return "\n".join(result)
    
    def _format_detailed(self) -> str:
        """Detailed format for memory presentation including metadata."""
        result = []
        for key, items in self.memory.items():
            result.append(f"==== {key.upper()} ====")
            for idx, item in enumerate(items):
                result.append(f"Item {idx+1}:")
                result.append(f"  Value: {item['value']}")
                result.append(f"  Metadata: {json.dumps(item['metadata'], indent=2)}")
            result.append("")  # Empty line
        return "\n".join(result)


# ====================== Old Storage System ======================

class IntermediateStorageBase(ABC):
    """Abstract base class for storing intermediate reasoning steps."""
    
    @abstractmethod
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        pass
    
    @abstractmethod
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        pass
    
    @abstractmethod
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps."""
        pass
    
    @abstractmethod
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        pass
    
    @abstractmethod
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps."""
        pass


class InMemoryIntermediateStorage(IntermediateStorageBase):
    """Simple in-memory implementation of intermediate step storage."""
    
    def __init__(self):
        """Initialize empty storage."""
        self.steps = []
        self.tool_calls = []
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        step = {
            "type": step_type,
            "content": content,
            "metadata": metadata or {}
        }
        self.steps.append(step)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        tool_call = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output,
            "metadata": metadata or {}
        }
        self.tool_calls.append(tool_call)
        
        # Also add as a step for unified access
        self.add_step(
            "tool_call", 
            {"tool_name": tool_name, "input": tool_input, "output": tool_output},
            metadata
        )
    
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps."""
        filtered_steps = self.steps
        
        if step_types:
            filtered_steps = [step for step in filtered_steps if step["type"] in step_types]
        
        if limit is not None:
            filtered_steps = filtered_steps[-limit:]
            
        return filtered_steps
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        formatted_steps = []
        for step in self.steps:
            if step["type"] == "thought":
                formatted_steps.append(f"Thought: {step['content']}")
            elif step["type"] == "action":
                formatted_steps.append(f"Action: {step['content']}")
            elif step["type"] == "action_input":
                formatted_steps.append(f"Action Input: {step['content']}")
            elif step["type"] == "observation" or step["type"] == "tool_call":
                if step["type"] == "tool_call":
                    tool_info = step["content"]
                    formatted_steps.append(f"Observation: {json.dumps(tool_info['output'], ensure_ascii=False)}")
                else:
                    formatted_steps.append(f"Observation: {step['content']}")
        return "\n".join(formatted_steps)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
        formatted_context = self.get_formatted_context("default")
        return [{"role": "system", "content": formatted_context}]
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.steps = []
        self.tool_calls = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps."""
        stats = {
            "total_steps": len(self.steps),
            "total_tool_calls": len(self.tool_calls),
            "steps_by_type": {}
        }
        
        # Count steps by type
        for step in self.steps:
            step_type = step["type"]
            if step_type not in stats["steps_by_type"]:
                stats["steps_by_type"][step_type] = 0
            stats["steps_by_type"][step_type] += 1
        
        return stats


# ====================== Adapter Classes ======================

class ContextToIntermediateAdapter(IntermediateStorageBase):
    """Adapter that exposes a ContextMemory as an IntermediateStorage."""
    
    def __init__(self, context_memory: ContextMemory):
        """Initialize with a ContextMemory instance."""
        self.memory = context_memory
    
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        self.memory.add(step_type, content, metadata)
    
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        tool_data = {
            "tool_name": tool_name,
            "input": tool_input,
            "output": tool_output
        }
        self.memory.add("tool_call", tool_data, metadata)
    
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps."""
        result = []
        
        if step_types and len(step_types) == 1:
            # Get specific step type
            steps = self.memory.get(step_types[0])
            if steps:
                for item in steps:
                    step = {
                        "type": step_types[0],
                        "content": item["value"],
                        "metadata": item["metadata"]
                    }
                    result.append(step)
        else:
            # Get all steps
            all_data = self.memory.get()
            
            # Flatten the data
            for key, items in all_data.items():
                if step_types is None or key in step_types:
                    for item in items:
                        step = {
                            "type": key,
                            "content": item["value"],
                            "metadata": item["metadata"]
                        }
                        result.append(step)
        
        # Apply limit if needed
        if limit is not None and limit > 0:
            result = result[-limit:]
            
        return result
    
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        return self.memory.get_formatted(format_type)
    
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
        formatted_context = self.memory.get_formatted("default")
        return [{"role": "system", "content": formatted_context}]
    
    def clear(self) -> None:
        """Clear all stored intermediate steps."""
        self.memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored steps."""
        all_data = self.memory.get()
        
        stats = {
            "total_steps": 0,
            "steps_by_type": {}
        }
        
        # Count steps by type
        for key, items in all_data.items():
            item_count = len(items)
            stats["total_steps"] += item_count
            stats["steps_by_type"][key] = item_count
        
        return stats


class IntermediateToContextAdapter(ContextMemory):
    """Adapter that exposes an IntermediateStorage as a ContextMemory."""
    
    def __init__(self, intermediate_storage: IntermediateStorageBase):
        """Initialize with an IntermediateStorage instance."""
        self.storage = intermediate_storage
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new memory item."""
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
        """Retrieve memory items with optional filtering."""
        # Get steps with optional type filter
        steps = self.storage.get_steps([key] if key else None)
        
        # Organize steps into the ContextMemory format
        result = {}
        for step in steps:
            step_type = step["type"]
            if step_type not in result:
                result[step_type] = []
            
            result[step_type].append({
                "value": step["content"],
                "metadata": step["metadata"]
            })
        
        if key:
            return result.get(key, [])
        return result
    
    def get_formatted(self, format_type: str = "default") -> str:
        """Get formatted memory for inclusion in LLM prompts."""
        return self.storage.get_formatted_context(format_type)
    
    def clear(self) -> None:
        """Clear all memory items."""
        self.storage.clear()


class ContextMemoryAdapter:
    """Adapter utility class for memory compatibility."""
    
    @staticmethod
    def adapt_intermediate_to_context(intermediate_storage: IntermediateStorageBase) -> ContextMemory:
        """Adapt an existing IntermediateStorage to the new ContextMemory interface."""
        return IntermediateToContextAdapter(intermediate_storage)
    
    @staticmethod
    def adapt_context_to_intermediate(context_memory: ContextMemory) -> IntermediateStorageBase:
        """Adapt a new ContextMemory to the existing IntermediateStorage interface."""
        return ContextToIntermediateAdapter(context_memory)


# ====================== Memory Manager Mocks ======================

class ReactAgentMemoryManager:
    """Simplified mock of the React agent's memory manager."""
    
    def __init__(self, store):
        """Initialize with a store."""
        self.store = store  # This will be the adapted storage
    
    def add(self, message):
        """Add a message to history."""
        self.store.add_step(message["role"], message["content"])
    
    def get_history(self):
        """Get conversation history."""
        steps = self.store.get_steps()
        
        # Convert steps to messages
        messages = []
        for step in steps:
            if step["type"] in ["user", "assistant", "system"]:
                messages.append({
                    "role": step["type"],
                    "content": step["content"]
                })
        
        return messages
    
    def clear(self):
        """Clear conversation history."""
        self.store.clear()


# ====================== Test Code ======================

def run_test():
    """Run the memory adapter test."""
    print("=== Memory Adapter Compatibility Test ===\n")
    
    # --- Test adapting new memory to old interface ---
    print("Testing adaptation of new memory to old interface:")
    
    # Create new memory system
    new_memory = InMemoryContextMemory()
    print("  - Created new InMemoryContextMemory instance")
    
    # Add some test data to the memory
    new_memory.add("system_message", 
                 "You are a helpful assistant using the new memory system.", 
                 {"timestamp": datetime.now().isoformat()})
    new_memory.add("user_query", 
                 "How can I learn Python?", 
                 {"timestamp": datetime.now().isoformat()})
    print("  - Added initial messages to new memory")
    
    # Display the formatted memory content
    print("\nNew Memory Content:")
    print(new_memory.get_formatted())
    
    # Create an adapter to convert the new memory to the old IntermediateStorage format
    adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(new_memory)
    print("\nCreated adapter from new memory to old IntermediateStorage")
    
    # Test intermediate storage operations through adapter
    adapted_storage.add_step("thought", "Python is a great language for beginners")
    adapted_storage.add_tool_call(
        "search", 
        {"query": "Python tutorials"}, 
        {"result": "Found several great tutorials for Python beginners."}
    )
    print("Added steps and tool call through adapter")
    
    # Display memory through both interfaces
    print("\nAccessing through old interface:")
    print(adapted_storage.get_formatted_context())
    
    print("\nAccessing through new interface (should include the added steps):")
    print(new_memory.get_formatted())
    
    # Test using with React agent memory manager
    agent_memory = ReactAgentMemoryManager(adapted_storage)
    print("\nCreated agent memory manager with adapted storage")
    
    # Add a message through the agent's memory manager
    agent_memory.add({
        "role": "assistant", 
        "content": "I recommend starting with the official Python tutorial at python.org."
    })
    print("Added message through agent memory manager")
    
    # Check that everything flowed through correctly
    print("\nFinal memory content through new interface:")
    print(new_memory.get_formatted())
    
    # --- Test adapting old storage to new interface ---
    print("\n\nTesting adaptation of old storage to new interface:")
    
    # Create old storage
    old_storage = InMemoryIntermediateStorage()
    print("  - Created old InMemoryIntermediateStorage instance")
    
    # Add steps to old storage
    old_storage.add_step("system", "You are a helpful assistant.")
    old_storage.add_step("user", "How do I install Python?")
    old_storage.add_step("thought", "I should provide installation instructions.")
    old_storage.add_tool_call(
        "knowledge_base", 
        {"query": "Python installation"}, 
        {"result": "Python can be downloaded from python.org."}
    )
    print("  - Added steps and tool call to old storage")
    
    # Display original storage content
    print("\nOld Storage Content:")
    print(old_storage.get_formatted_context())
    
    # Create adapter to use with new interface
    adapted_memory = ContextMemoryAdapter.adapt_intermediate_to_context(old_storage)
    print("\nCreated adapter from old storage to new ContextMemory")
    
    # Add something through the new interface
    adapted_memory.add("assistant_response", 
                      "To install Python, go to python.org and download the latest version.", 
                      {"timestamp": datetime.now().isoformat()})
    print("Added response through adapted memory")
    
    # Display memory through both interfaces
    print("\nAccessing through new interface:")
    print(adapted_memory.get_formatted())
    
    print("\nAccessing through old interface (should include the added response):")
    print(old_storage.get_formatted_context())
    
    print("\nMemory adapter test completed successfully.")


if __name__ == "__main__":
    run_test() 