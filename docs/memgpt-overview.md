# MemGPT: Virtual Context Management for LLMs

## Overview

MemGPT (Memory-GPT) is an approach to context management that draws inspiration from operating system virtual memory. It enables LLMs to work with unbounded context by intelligently moving data between fast (in-context) and slow (external storage) memory tiers.

**Key Insight**: Just as operating systems create the illusion of unlimited RAM through virtual memory and paging, MemGPT creates the illusion of unlimited context for LLMs.

> MemGPT is now part of **Letta**, an open-source framework for building stateful agents.
> - GitHub: https://github.com/letta-ai/letta
> - Docs: https://docs.letta.com

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMGPT MEMORY HIERARCHY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  MAIN CONTEXT (In-Context Memory)           ≈ RAM         │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │  System Prompt (immutable)                          │  │  │
│  │  │  ┌─────────────────────────────────────────────┐    │  │  │
│  │  │  │  Core Memory (LLM-editable)                 │    │  │  │
│  │  │  │  ├── Persona Block: Agent's identity        │    │  │  │
│  │  │  │  └── Human Block: User information          │    │  │  │
│  │  │  └─────────────────────────────────────────────┘    │  │  │
│  │  │  Working Context (recent messages)                  │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                    LLM Memory Tools                              │
│                              │                                   │
│         ┌────────────────────┴────────────────────┐             │
│         ▼                                         ▼             │
│  ┌─────────────────────┐            ┌─────────────────────┐     │
│  │  RECALL MEMORY      │            │  ARCHIVAL MEMORY    │     │
│  │  (Conversation DB)  │            │  (Vector Database)  │     │
│  │  ≈ Disk             │            │  ≈ Disk             │     │
│  │                     │            │                     │     │
│  │  • Full message log │            │  • Long-term facts  │     │
│  │  • Text search      │            │  • External data    │     │
│  │  • Date search      │            │  • Semantic search  │     │
│  └─────────────────────┘            └─────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Tiers

### 1. Core Memory (In-Context, Editable)

The core memory is always present in the LLM's context window and can be directly edited by the agent via tool calls.

| Block | Purpose | Example |
|-------|---------|---------|
| **Persona** | Agent's identity and behavior | "I am a helpful coding assistant who writes clean Python" |
| **Human** | Information about the user | "Name: Alice. Prefers concise answers. Expert in React." |

```python
# Agent can edit core memory via tools
core_memory_append("human", "Prefers dark mode in IDE")
core_memory_replace("persona", "I am a Python expert...")
```

### 2. Recall Memory (Conversation History)

Complete log of all messages, stored externally but searchable.

```python
# Search past conversations
conversation_search("what did we discuss about APIs")
conversation_search_date("2024-01-15", "2024-01-20")
```

### 3. Archival Memory (Long-Term Storage)

Vector database for semantic search over facts and external data.

```python
# Store important information
archival_memory_insert("User's project uses PostgreSQL with pgvector")

# Retrieve relevant information
archival_memory_search("database setup")
```

---

## Self-Directed Memory Management

The key innovation of MemGPT is that the **LLM itself decides** what to remember:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SELF-DIRECTED MEMORY FLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User: "By the way, I'm allergic to peanuts"                    │
│                              │                                   │
│                              ▼                                   │
│  LLM thinks: "This is important safety information.             │
│               I should store this in core memory."              │
│                              │                                   │
│                              ▼                                   │
│  LLM calls: core_memory_append("human", "ALLERGY: peanuts")     │
│                              │                                   │
│                              ▼                                   │
│  Core memory updated. Information persists across sessions.     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

This is different from traditional RAG where an external system decides what to retrieve.

---

## Available Memory Tools

| Tool | Purpose |
|------|---------|
| `send_message` | Send response to user |
| `core_memory_append` | Add text to a core memory block |
| `core_memory_replace` | Replace text in a core memory block |
| `conversation_search` | Search recall memory by text |
| `conversation_search_date` | Search recall memory by date range |
| `archival_memory_insert` | Store information in archival memory |
| `archival_memory_search` | Semantic search over archival memory |

---

## Code Example (Letta)

```python
from letta import create_client, LLMConfig
from letta.memory import ChatMemory

# Create client
client = create_client()

# Create agent with memory
agent = client.create_agent(
    name="assistant",
    memory=ChatMemory(
        human="User information will be stored here.",
        persona="I am a helpful assistant that remembers everything."
    ),
    llm_config=LLMConfig(model="gpt-4"),
)

# Send message - agent manages its own memory
response = agent.send_message("Hi, I'm Alice and I work at Anthropic")
# Agent might call: core_memory_append("human", "Name: Alice, Works at: Anthropic")

# Later conversation - agent recalls from memory
response = agent.send_message("Where do I work?")
# Agent checks core memory and responds: "You work at Anthropic"

# Search past conversations
response = agent.send_message("What did we talk about last week?")
# Agent calls: conversation_search_date(...) to find relevant history
```

---

## OS Analogy

| Operating System | MemGPT |
|------------------|--------|
| RAM | Main context (what LLM sees) |
| Disk | Recall + Archival memory |
| Page fault | Information not in context, need to search |
| Paging | Moving data between context and external storage |
| Process | Single conversation/agent |

---

## Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Memory Manager** | LLM itself (via tool calls) |
| **Persistence** | All data persisted to databases |
| **Retrieval** | On-demand via explicit search tools |
| **Flexibility** | Agent decides what's important |
| **Unbounded Context** | Virtual memory creates illusion of infinite context |

---

## Limitations

1. **Token Overhead**: LLM spends tokens on memory management decisions
2. **Reliability**: LLM might forget to save important information
3. **No Compression**: Relies on search rather than summarization
4. **Single Agent**: Limited multi-agent memory sharing
5. **No Cache Optimization**: Doesn't optimize for KV-cache hit rates

---

## References

- **Paper**: [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- **Research Site**: https://research.memgpt.ai/
- **Letta (Implementation)**: https://github.com/letta-ai/letta
- **Documentation**: https://docs.letta.com/concepts/memgpt/
