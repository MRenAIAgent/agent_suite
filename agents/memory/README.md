# Memory Management Module

This module provides memory capabilities for AI agents, including conversation history management, context memory, and long-term persistent storage.

## Overview

The memory module consists of:

1. **MemoryManagerBase**: Abstract base class defining the interface for all memory managers
2. **MemoryManager**: Unified manager that combines context and longterm memory features
3. **ContextMemory**: Short-term memory for conversation history and working context
4. **LongtermMemory**: Persistent storage for longer-term memory across sessions

## Storage Backends

The memory system uses different storage backends:

1. **InMemoryStore**: Fast in-memory storage (default)
2. **RedisStore**: Redis-backed persistent storage
3. Custom stores can be implemented by extending the **Store** base class

## Usage Examples

### Basic Usage with Default MemoryManager

```python
from agents.memory import MemoryManager

# Create a memory manager with default settings
memory_manager = MemoryManager()

# Add a new message to history
memory_manager.add({
    "role": "user",
    "content": "Hello, AI!"
})

# Add assistant's response
memory_manager.add({
    "role": "assistant",
    "content": "Hello, human!"
})

# Get the current conversation history
history = memory_manager.get_history()
print(history)

# Clear conversation history
memory_manager.clear()
```

### Using Context Memory Features

```python
from agents.memory import ContextMemory

# Create a context memory manager
context_memory = ContextMemory(max_history=20)

# Add conversation messages
context_memory.add({
    "role": "user",
    "content": "What's the capital of France?"
})

context_memory.add({
    "role": "assistant",
    "content": "The capital of France is Paris."
})

# Store additional context data
context_memory.set_context("user_location", "New York")
context_memory.set_context("current_topic", "Geography")

# Get context data
location = context_memory.get_context("user_location")
print(f"User location: {location}")

# Get all context data
all_context = context_memory.get_context()
print(all_context)

# Get formatted history for prompts
formatted_history = context_memory.get_formatted_history()
print(formatted_history)

# Search memory
results = context_memory.search("France", limit=5)
print(results)
```

### Using Longterm Memory

```python
from agents.memory import LongtermMemory

# Create a longterm memory manager
longterm_memory = LongtermMemory(
    max_history=100,
    storage_dir="./my_memories",
    session_id="user_123"
)

# Add messages
longterm_memory.add({
    "role": "user",
    "content": "My favorite color is blue.",
    "timestamp": "2023-05-01T12:00:00"
})

# Later, load memory from persistent storage
loaded_memory = longterm_memory.load_memory()
print(loaded_memory)

# Search for relevant information
color_info = longterm_memory.search("favorite color")
print(color_info)

# Delete persistent storage
longterm_memory.purge()
```

### Using the Unified MemoryManager

```python
from agents.memory import MemoryManager

# Create a memory manager with longterm storage enabled
memory_manager = MemoryManager(
    max_history=20,
    use_longterm=True,
    storage_dir="./agent_memories",
    session_id="user_456"
)

# Add messages (stored in both context and longterm)
memory_manager.add({
    "role": "user",
    "content": "Remember that my birthday is June 15."
})

memory_manager.add({
    "role": "assistant",
    "content": "I'll remember that your birthday is June 15."
})

# Add working context (only in context memory)
memory_manager.set_context("user_preferences", {
    "color": "green",
    "language": "English"
})

# Get recent history
history = memory_manager.get_history(limit=10)

# Search across all memory
birthday_info = memory_manager.search("birthday")
```

### Using Redis-Backed Storage

```python
from agents.memory.stores import RedisStore
from agents.memory import ContextMemory

# Create Redis store with custom configuration
redis_store = RedisStore(
    host="localhost",
    port=6379,
    db=0,
    prefix="agent:",
    password=None,
    expire_seconds=3600  # Expire after 1 hour
)

# Create a context memory manager with Redis storage
context_memory = ContextMemory(max_history=20)
context_memory.store = redis_store

# Now use context_memory normally, with Redis persistence
context_memory.add({
    "role": "user", 
    "content": "Remember this is stored in Redis"
})

# Data persists even after restarting application
```

## Extending

To create a new memory manager implementation:

1. Subclass `MemoryManagerBase` from `agents.memory.base_memory_manager`
2. Implement all the required methods with your custom logic
3. Use your new memory manager in your agent

To create a new storage backend:

1. Subclass `Store` from `agents.memory.stores.store`
2. Implement all the required methods for your storage technology
3. Use your custom store with existing memory managers 