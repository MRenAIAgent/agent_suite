# Memory System Adapter

The memory adapter module provides a bridge between the existing `IntermediateStorage` system and the new structured `ContextMemory` system, ensuring backward compatibility while allowing a gradual transition to the new memory architecture.

## Overview

The adapter pattern enables:

1. Using existing agent code with the new memory system
2. Using new memory implementations with the existing agent code
3. Gradual migration from the old storage system to the new memory system

## Usage

### Adapting Existing IntermediateStorage to New ContextMemory Interface

```python
from agent_suite.agents.storage.intermediate_storage import InMemoryIntermediateStorage
from agent_suite.memory.adapter import ContextMemoryAdapter

# Create an instance of the existing storage
existing_storage = InMemoryIntermediateStorage()

# Add data using existing interface
existing_storage.add_step("thought", "I need to search for information.")
existing_storage.add_tool_call("search", {"query": "Python"}, {"result": "..."})

# Adapt to new ContextMemory interface
context_memory = ContextMemoryAdapter.adapt_intermediate_to_context(existing_storage)

# Now use with new memory system components
memory_manager = MemoryManager(
    context_memory=context_memory,  # Use adapted storage here
    long_term_memory=long_term_memory
)
```

### Adapting New ContextMemory to Existing IntermediateStorage Interface

```python
from agent_suite.memory.context.in_memory import InMemoryContextMemory
from agent_suite.memory.adapter import ContextMemoryAdapter

# Create an instance of the new memory system
new_memory = InMemoryContextMemory()

# Add data using new interface
new_memory.add("message", "Hello, world!", {"role": "system"})
new_memory.add("query", "How can I help?", {"role": "user"})

# Adapt to existing IntermediateStorage interface
intermediate_storage = ContextMemoryAdapter.adapt_context_to_intermediate(new_memory)

# Now use with existing agent code
agent = Agent(
    storage=intermediate_storage  # Use adapted memory here
)
```

## Key Components

### ContextMemoryAdapter

Static utility class that provides two adapter methods:

- `adapt_intermediate_to_context()`: Converts an IntermediateStorage to a ContextMemory
- `adapt_context_to_intermediate()`: Converts a ContextMemory to an IntermediateStorage

### IntermediateToContextAdapter

Adapter class that exposes an existing IntermediateStorage through the ContextMemory interface.

### ContextToIntermediateAdapter

Adapter class that exposes a new ContextMemory through the existing IntermediateStorage interface.

## Migration Strategy

1. **Phase 1**: Use adapters to integrate new memory components without changing agent code
2. **Phase 2**: Gradually migrate agent implementations to use new memory system directly
3. **Phase 3**: Eventually remove the adapter and old IntermediateStorage when no longer needed

## Compatibility Considerations

- Some advanced features of either system might not be perfectly mapped in the adapter
- Performance may be slightly impacted when using adapters compared to native implementations
- Both memory systems share similar concepts but with different structures, so the mapping is generally straightforward 