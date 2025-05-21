# React Agent Compatibility with New Memory System

This document explains how to use the new memory system with existing React agents through the adapter pattern.

## Overview

The new memory system provides a more flexible and powerful way to manage both context and long-term memory. To maintain backward compatibility with existing React agents, the adapter pattern is used to connect the new memory components with the existing `IntermediateStorage` interface.

## Usage with React Agents

### Basic Usage

```python
from agents.react_agent import ReActAgent
from llm.litellm.litellm import LiteLLM
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter
from agents.memory_manager import MemoryManager

# Create new memory system
context_memory = InMemoryContextMemory()

# Create an adapter to convert the new memory to the old IntermediateStorage
adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(context_memory)

# Create a custom memory manager that uses our adapted storage
class CustomMemoryManager(MemoryManager):
    def __init__(self, storage):
        super().__init__()
        self.store = storage  # Override the default store with our adapted storage

memory_manager = CustomMemoryManager(adapted_storage)

# Initialize the React agent with the adapted memory
agent = ReActAgent(
    llm=LiteLLM.create_llm(),
    role="You are a helpful assistant.",
    task="Answer user questions concisely.",
    guide="Provide brief, helpful responses to user queries.",
    tools=[...],
    memory_manager=memory_manager,  # Using the adapted memory
    max_iterations=5
)

# Run the agent - all memory operations will flow through to the new memory system
response = agent.run("What is the capital of France?", "model_name")
```

### Redis-Based Context Memory (Persistent)

For more persistent context memory, you can use the Redis-based implementation:

```python
from agent_suite.memory.context.redis import RedisContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter

# Create Redis-based context memory
context_memory = RedisContextMemory(
    redis_url="redis://localhost:6379",
    namespace="agent_context:",
    ttl=3600  # 1-hour TTL
)

# Then adapt and use as above
adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(context_memory)
```

### Complete Example with Long-Term Memory

For a complete setup with both context and long-term memory:

```python
from agent_suite.memory.factory import MemoryFactory
from agent_suite.memory.adapter import ContextMemoryAdapter
from agents.memory_manager import MemoryManager

# Create a memory manager with default configuration
memory_manager = MemoryFactory.create_memory_manager()

# Get the context memory component
context_memory = memory_manager.context_memory

# Create adapter for the React agent
adapted_storage = ContextMemoryAdapter.adapt_context_to_intermediate(context_memory)

# Create custom memory manager
custom_memory_manager = CustomMemoryManager(adapted_storage)

# Initialize React agent with this memory manager
# ...

# When you want to search long-term memory from the React agent
results = memory_manager.query("What information do I have about Python?")

# Add results to context through the memory manager
for result in results:
    memory_manager.add_to_context("retrieved_info", result["text"])
```

## Standalone Testing

You can verify the adapter is working correctly using the standalone test:

```python
python standalone_memory_test.py
```

This script contains a complete, self-contained implementation of the memory adapter pattern and demonstrates bidirectional compatibility between new and old memory systems.

## Best Practices

1. **Memory Initialization**: Initialize memory with any system messages or context before starting the agent.

2. **Dual Access**: You can access memory through both the new interface and the adapted old interface - they will stay in sync.

3. **Long-Term Memory**: Use the `memory_manager.query()` method to search long-term memory and add results to context using `memory_manager.add_to_context()`.

4. **User-Specific Memory**: For multi-user systems, provide a `user_id` when creating memory components to isolate memory between users.

5. **Performance**: The adapter pattern adds a small overhead - for performance-critical applications, consider migrating to use the new memory API directly.

## Troubleshooting

If you encounter issues:

- Verify you're passing the CustomMemoryManager instance to the ReActAgent constructor
- Check Redis connection details if using RedisContextMemory
- Ensure the adapter is correctly converting between formats by checking memory contents

## Migration Path

The adapter pattern provides a migration path for existing agent implementations:

1. **Phase 1**: Use adapters to integrate new memory components without changing agent code
2. **Phase 2**: Gradually migrate agent implementations to use new memory system directly
3. **Phase 3**: Eventually remove the adapter and old IntermediateStorage when no longer needed 