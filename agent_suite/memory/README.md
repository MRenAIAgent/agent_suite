# Agent Suite Memory System

The Agent Suite Memory System provides a flexible and unified interface for managing both context (short-term) and long-term memory for AI agents, with support for different storage backends.

## Architecture

The memory system is designed with a layered architecture:

```
MemoryManager
    |
    |-- Context Memory (short-term)
    |     |-- InMemoryContextMemory
    |     |-- RedisContextMemory
    |
    |-- Long-Term Memory
          |-- Vector Memory
          |     |-- RedisVectorMemory
          |
          |-- Graph Memory
                |-- GraphitiMemory
```

### Key Components

1. **MemoryManager**: Provides a unified interface for memory operations, handling both context and long-term memory, as well as user-specific and global memory.

2. **Context Memory**: Stores short-term, transient information needed during agent execution.
   - `InMemoryContextMemory`: Simple in-memory implementation for short-lived sessions
   - `RedisContextMemory`: Redis-backed implementation for persistent, shared context

3. **Long-Term Memory**: Stores information persistently for recall across multiple sessions.
   - **Vector Memory**: Semantic search-capable memory using vector embeddings
     - `RedisVectorMemory`: Uses Redis with vector search capabilities
   - **Graph Memory**: Graph-based memory for storing entity relationships
     - `GraphitiMemory`: Uses Graphiti knowledge graph service

4. **MemoryFactory**: Factory class for easily creating memory components with different configurations.

5. **Adapter**: Bridge between existing IntermediateStorage and the new memory system for backward compatibility.
   - `ContextMemoryAdapter`: Provides utility methods to adapt between the two interfaces
   - `IntermediateToContextAdapter`: Uses existing storage with new interface
   - `ContextToIntermediateAdapter`: Uses new memory with existing interface

## Usage

### Basic Usage

```python
from agent_suite.memory.factory import MemoryFactory

# Create a memory manager with default configuration
memory_manager = MemoryFactory.create_memory_manager()

# Add to context memory
memory_manager.add_to_context("message", "Hello, world!")

# Store in long-term memory
memory_id = memory_manager.store("This is important information to remember.")

# Query long-term memory
results = memory_manager.query("What information do I have?")

# Convenience method to query long-term memory and add to context
memory_manager.remember("What information do I have?", context_key="retrieved_info")
```

### Customized Memory Manager

```python
# Custom configuration for memory components
context_config = {
    "type": "redis",
    "redis_url": "redis://localhost:6379",
    "namespace": "agent_context:",
    "ttl": 3600  # 1 hour TTL
}

long_term_config = {
    "type": "redis_vector",
    "redis_url": "redis://localhost:6379",
    "index_name": "agent_memory",
    "vector_dimensions": 384,
    "distance_metric": "COSINE"
}

# Create a user-specific memory manager
user_id = "user123"
memory_manager = MemoryFactory.create_memory_manager(
    context_config=context_config,
    long_term_config=long_term_config,
    user_id=user_id
)
```

### Using the Adapter for Backward Compatibility

```python
from agent_suite.agents.storage.intermediate_storage import InMemoryIntermediateStorage
from agent_suite.memory.adapter import ContextMemoryAdapter

# Create and use existing storage
existing_storage = InMemoryIntermediateStorage()
existing_storage.add_step("thought", "I need to find information.")

# Adapt to new memory system
context_memory = ContextMemoryAdapter.adapt_intermediate_to_context(existing_storage)

# Use in memory manager
memory_manager = MemoryFactory.create_memory_manager(
    context_memory=context_memory,  # Use the adapted storage
    long_term_config={"type": "redis_vector"}
)
```

See `adapter.md` for more detailed information on using the adapter.

### User-Specific and Global Memory

The memory system supports both:

- **User-specific memory**: When a `user_id` is provided, queries and storage operations are scoped to that user
- **Global memory**: Without a `user_id`, memory is shared across all users

## Memory Type Selection Guide

### Context Memory

- **In-Memory**: Best for stateless agent sessions, fast but not persistent.
- **Redis**: Best for persistent context, shared across instances, or when durability is needed.

### Long-Term Memory

- **Redis Vector Memory**: Best for semantic similarity search when text is the primary data type.
- **Graphiti Memory**: Best for storing complex relationships between entities, enabling graph traversal and relationship queries.

## Requirements

- For Redis-based implementations:
  - Redis server with RediSearch module
  - `redis-py` Python package
- For vector implementations:
  - Embedding service for converting text to vectors
- For Graphiti implementations:
  - Graphiti API access (endpoint and API key)

## Examples

- See `examples/memory_example.py` for working examples of using the memory system
- See `examples/compatibility_example.py` for examples of using the memory adapter 