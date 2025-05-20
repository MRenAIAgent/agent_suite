# Centralized Graph Infrastructure

This module provides a centralized graph database infrastructure for Agent Suite, supporting both the knowledge module and long-term storage features.

## Overview

The graph infrastructure consists of:

1. **GraphManager**: Core class providing a unified interface for graph operations
2. **Knowledge Module Adapter**: Connects the knowledge module to the centralized graph
3. **Long-term Storage Adapter**: Connects the agent's memory to the centralized graph

## Features

- **Shared Infrastructure**: Common codebase for graph operations across modules
- **Multiple Storage Options**: 
  - In-memory storage (non-persistent)
  - File-based storage (persistent using GraphML format)
- **Rich Graph Operations**:
  - Add/update/delete entities
  - Add/update/delete relations
  - Query by entity type or properties
  - Query by relation type or endpoints
  - Keyword search across graph
  - Graph traversal
  - Entity neighborhood exploration

## Usage Examples

### Basic Usage

```python
from agent_suite.graph import GraphManager

# Create a graph manager with file-based storage
graph = GraphManager(
    graph_name="my_graph",
    persist=True,
    persistence_path="./data/graphs"
)

# Add entities
person_id = graph.add_entity(
    entity_type="person",
    properties={
        "name": "John Doe",
        "age": 30
    }
)

# Add relations
graph.add_relation(
    source_id=person_id,
    relation_type="WORKS_AT",
    target_id=company_id
)

# Query entities
people = graph.query_entities(entity_type="person")

# Query relations
employment = graph.query_relations(relation_type="WORKS_AT")

# Search
results = graph.keyword_search("John")
```

### With Knowledge Module

```python
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter

# Create a knowledge graph
kg = GraphitiKnowledgeGraphAdapter(
    graph_name="knowledge_graph",
    persist=True
)

# Add text
text_id = kg.add_text(
    text="Python is a programming language.",
    properties={"title": "Python Info"}
)

# Search
results = kg.semantic_search("programming language")
```

### With Long-term Storage

```python
from agent_suite.agents.storage.long_term.graphiti import GraphitiKnowledgeGraphMemoryAdapter

# Create memory
memory = GraphitiKnowledgeGraphMemoryAdapter(
    graph_name="agent_memory",
    persist=True
)

# Store facts
fact_id = memory.store_fact(
    subject="user:john",
    predicate="likes",
    object_value="python",
    metadata={"confidence": 0.9}
)

# Query facts
facts = memory.query_facts("john likes")
```

## Implementation Details

The graph infrastructure is built on top of Graphiti, a Python library for graph operations. It provides:

- Entity and relation management
- Property storage on both entities and relations
- Traversal capabilities
- Persistence via GraphML format

## Storage Format

Graph data is stored in GraphML format, which preserves:
- Node IDs and properties
- Edge directions, types, and properties
- Graph structure

## Dependencies

- Graphiti: Core graph library
- Python 3.7+
- dotenv: For environment variable loading 