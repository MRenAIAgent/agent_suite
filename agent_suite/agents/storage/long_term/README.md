# Long-Term Memory for Agents

This package provides long-term memory capabilities for agents using knowledge graphs and vector databases.

## Features

- **Knowledge Graph Memory**: Store and query structured information using relationship-based graphs
- **Vector Database Memory**: Store and retrieve unstructured text using semantic similarity
- **Combined Memory**: Leverage both knowledge graphs and vector databases in a unified interface

## Overview

Long-term memory allows agents to remember and recall information across sessions, improving their ability to:

1. **Build Knowledge**: Accumulate facts, relationships, and context over time
2. **Reason Over Relationships**: Understand connections between entities and concepts
3. **Find Relevant Context**: Retrieve information semantically relevant to the current task
4. **Improve Over Time**: Learn from past interactions and experiences

## Implementations

### Knowledge Graph Memory (GraphitiKnowledgeGraphMemory)

Stores information as entities and relationships in a knowledge graph using the Graphiti library.

```python
from agent_suite.agents.storage.long_term.factory import LongTermMemoryFactory

# Create a knowledge graph memory
kg_memory = LongTermMemoryFactory.create_memory("kg", {
    "graph_name": "my_agent_memory",
    "persist": True,
    "persistence_path": "./memory"
})

# Store facts about entities
fact_id = kg_memory.store_fact(
    subject="apple",
    predicate="is_a",
    object_value="fruit",
    metadata={"confidence": 0.95, "source": "user"}
)

# Query for facts
facts = kg_memory.query_facts("What is an apple?")
```

### Vector Database Memory (VectorDatabaseMemory)

Stores unstructured text with vector embeddings for semantic similarity search using Qdrant.

```python
from agent_suite.agents.storage.long_term.factory import LongTermMemoryFactory

# Create a vector database memory
vector_memory = LongTermMemoryFactory.create_memory("vector", {
    "collection_name": "my_agent_memory",
    "embedding_model": "all-MiniLM-L6-v2"
})

# Store text content
text_id = vector_memory.store_text(
    text="Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    metadata={"source": "wikipedia", "topic": "AI"}
)

# Query for similar text
results = vector_memory.query_text("What is machine learning?")
```

### Combined Memory (CombinedLongTermMemory)

Uses both knowledge graph and vector database to provide a unified memory system.

```python
from agent_suite.agents.storage.long_term.factory import LongTermMemoryFactory

# Create a combined memory
memory = LongTermMemoryFactory.create_memory("combined", {
    "kg_graph_name": "my_kg_memory",
    "vector_collection_name": "my_vector_memory", 
    "persistence_path": "./memory"
})

# Store both facts and text
fact_id = memory.store_fact("python", "is_a", "programming_language")
text_id = memory.store_text("Python is a high-level, interpreted programming language.")

# Query both systems
facts = memory.query_facts("programming language")
texts = memory.query_text("What is Python?")
```

## Dependencies

- **Graphiti**: Knowledge graph library
- **Qdrant**: Vector database for similarity search
- **SentenceTransformers**: Models for generating text embeddings

## Usage with Agents

To integrate long-term memory with an agent:

```python
from agent_suite.agents.storage.long_term.factory import LongTermMemoryFactory
from agent_suite.agents.storage_aware_react_agent import StorageAwareReActAgent

# Create the memory
memory = LongTermMemoryFactory.create_memory("combined", {
    "persistence_path": "./agent_memory"  
})

# Create an agent with long-term memory
agent = StorageAwareReActAgent(
    long_term_memory=memory
)

# Store information learned during interactions
memory.store_fact("user", "name", "John")
memory.store_text("The user is interested in machine learning and artificial intelligence.")

# Retrieve relevant context for new interactions
relevant_facts = memory.query_facts("user preferences")
relevant_context = memory.query_text("What is the user interested in?")
```

## Best Practices

1. **Structured vs. Unstructured**: Use facts for structured, discrete information and text storage for detailed explanations
2. **Metadata**: Include source, confidence, and timestamp information in metadata
3. **Context Retrieval**: Query both fact and text stores for comprehensive context
4. **Persistence**: Configure persistence for long-running agents
5. **Entity Linking**: Use consistent entity identifiers across fact storage 