# Memory Store Implementations

This directory contains different storage implementations for the agent memory system.

## Available Store Implementations

### InMemoryStore

A simple in-memory store that keeps history and cache in memory. 
This store is not persistent across program restarts unless explicitly 
saved and loaded with the file-based functionality in `LongtermMemory`.

```python
from agents.memory.stores.in_memory_store import InMemoryStore

store = InMemoryStore()
```

### RedisStore

Redis-backed implementation that stores history and cache in Redis,
providing cross-instance memory persistence and sharing capabilities.

```python
from agents.memory.stores.redis_store import RedisStore

store = RedisStore(
    host="localhost",  # Redis server host
    port=6379,         # Redis server port
    db=0,              # Redis database number
    prefix="agent:"    # Key prefix for Redis keys
)
```

### Neo4jStore

Neo4j-backed implementation that stores conversation history and cache in a Neo4j graph database.
This is ideal for shared memory across instances with the power of a graph database.

```python
from agents.memory.stores.neo4j_store import Neo4jStore

store = Neo4jStore(
    uri="neo4j+s://xxx.databases.neo4j.io",  # Neo4j connection URI
    username="neo4j",                         # Neo4j username
    password="your-password",                 # Neo4j password
    database="neo4j",                         # Database name (default: neo4j)
    prefix="agent:"                           # Key prefix for Neo4j nodes
)
```

#### Neo4j AuraDB Free

The Neo4jStore works great with Neo4j AuraDB Free, which provides:
- Free hosted Neo4j database with no credit card required
- 50MB storage (more than enough for agent memory)
- No multi-database support (uses the default `neo4j` database)
- 1 million node limit

Sign up at [Neo4j AuraDB](https://console.neo4j.io/) to create a free instance.

## Using Stores with Memory Managers

All store implementations can be used with the memory managers:

```python
from agents.memory.longterm_memory import LongtermMemory
from agents.memory.stores.neo4j_store import Neo4jStore

# Create a Neo4j-backed store
store = Neo4jStore(
    uri="neo4j+s://xxx.databases.neo4j.io",
    username="neo4j", 
    password="your-password"
)

# Use it with LongtermMemory
memory = LongtermMemory(
    max_history=100,
    session_id="user-123",
    store=store  # Pass your custom store
)

# Memory operations now use Neo4j for storage
memory.add({"role": "user", "content": "Hello"})
history = memory.get_history()
```

## Creating Custom Store Implementations

Custom stores should implement the `Store` base class:

```python
from agents.memory.stores.store import Store

class MyCustomStore(Store):
    def __init__(self, **kwargs):
        super().__init__()  # Initialize empty in-memory history and cache
        # Initialize your custom storage

    # Implement required methods
    def add_history(self, message):
        # Add to in-memory and custom storage
        pass
        
    # ... (implement other required methods)
``` 