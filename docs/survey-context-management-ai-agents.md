# A Survey on Context Management in AI Agent Frameworks: Architecture, Implementation, and Comparative Analysis

**Authors:** Research Survey
**Date:** January 2026
**Version:** 1.0

---

## Abstract

Context management is a fundamental challenge in building effective AI agents and multi-turn conversational systems. As large language models (LLMs) become central to autonomous agent architectures, the efficient handling of conversation history, memory persistence, and state management has emerged as a critical differentiator between frameworks. This survey presents a comprehensive analysis of context management approaches across five leading AI agent frameworks: Claude SDK (Anthropic), Google ADK, LangChain, Manus, and OpenAI Agents SDK. We examine the architectural design patterns, implementation details, and technical trade-offs of each framework. Our analysis reveals distinct philosophical approaches—from stateless explicit management to hierarchical session-based systems to KV-cache-optimized designs. We provide detailed architecture diagrams, implementation examples, and a systematic comparison to guide practitioners in selecting appropriate frameworks for their use cases.

**Keywords:** Context Management, AI Agents, Large Language Models, Multi-Agent Systems, Memory Systems, Conversation State

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Preliminaries](#2-background-and-preliminaries)
3. [Claude SDK (Anthropic)](#3-claude-sdk-anthropic)
4. [Google Agent Development Kit (ADK)](#4-google-agent-development-kit-adk)
5. [LangChain Framework](#5-langchain-framework)
6. [Manus AI](#6-manus-ai)
7. [OpenAI Agents SDK](#7-openai-agents-sdk)
8. [Comparative Analysis](#8-comparative-analysis)
9. [Discussion and Future Directions](#9-discussion-and-future-directions)
10. [Conclusion](#10-conclusion)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Motivation

The emergence of large language models (LLMs) has catalyzed a paradigm shift in artificial intelligence, enabling the development of autonomous agents capable of complex reasoning, tool use, and multi-step task execution. Central to the effectiveness of these agents is **context management**—the mechanisms by which conversation history, intermediate state, and long-term memory are maintained and utilized across interactions.

Context management presents several fundamental challenges:

1. **Token Limitations**: LLMs have finite context windows (ranging from 8K to 1M+ tokens), requiring strategies for prioritization and compression
2. **State Persistence**: Conversations may span multiple sessions requiring durable storage
3. **Multi-Agent Coordination**: Complex workflows involve multiple agents sharing or isolating context
4. **Cost Optimization**: Token usage directly impacts operational costs, with cached tokens offering 10x savings
5. **Latency Constraints**: Context retrieval and injection must not impede real-time interactions

### 1.2 Scope and Contributions

This survey provides:

- **Architectural analysis** of five major AI agent frameworks
- **Implementation details** with code examples and design patterns
- **Comparative evaluation** across multiple dimensions
- **Practical guidance** for framework selection and migration

### 1.3 Framework Selection Criteria

We selected frameworks based on:
- Industry adoption and community support
- Distinct architectural approaches
- Production readiness
- Documentation availability

| Framework | Organization | Release | Primary Language |
|-----------|--------------|---------|------------------|
| Claude SDK | Anthropic | 2023 | Python, TypeScript |
| Google ADK | Google | 2024 | Python |
| LangChain | LangChain Inc. | 2022 | Python, TypeScript |
| Manus | Monica.im | 2025 | Python |
| OpenAI Agents SDK | OpenAI | 2025 | Python, TypeScript |

---

## 2. Background and Preliminaries

### 2.1 Context in Large Language Models

LLMs process input as sequences of tokens within a fixed **context window**. The context typically includes:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Context Window                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐    │
│  │  System Prompt                                       │    │
│  │  - Agent persona and instructions                   │    │
│  │  - Tool definitions and schemas                     │    │
│  │  - Behavioral constraints                           │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Conversation History                               │    │
│  │  - User messages                                    │    │
│  │  - Assistant responses                              │    │
│  │  - Tool calls and results                           │    │
│  └─────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Current Input                                      │    │
│  │  - Latest user message                              │    │
│  │  - Retrieved context (RAG)                          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Context Window Sizes (2026)

| Model | Context Window | Extended |
|-------|----------------|----------|
| GPT-4o | 128K tokens | - |
| Claude Sonnet 4.5 | 200K tokens | 1M tokens |
| Claude Opus 4.5 | 200K tokens | - |
| Gemini 2.0 | 1M tokens | 2M tokens |
| Llama 3.3 | 128K tokens | - |

### 2.3 Key Terminology

| Term | Definition |
|------|------------|
| **Context Window** | Maximum tokens processable in single inference |
| **KV-Cache** | Key-value cache storing attention computations |
| **Session** | Logical grouping of related interactions |
| **State** | Mutable data persisted across turns |
| **Handoff** | Transfer of control between agents |
| **Memory** | Persistent storage of information across sessions |

### 2.4 Architectural Patterns

We identify four primary context management patterns:

1. **Stateless Explicit**: Developer manages complete history (Claude SDK)
2. **Stateful Session**: Framework maintains session state (Google ADK, OpenAI)
3. **Pluggable Memory**: Interchangeable memory backends (LangChain)
4. **Cache-Optimized**: KV-cache hit rate as primary metric (Manus)

---

## 3. Claude SDK (Anthropic)

### 3.1 Design Philosophy

The Claude SDK follows a **stateless API architecture** where the API does not maintain conversation state server-side. Developers explicitly pass the complete message history with each request, providing maximum control and transparency.

### 3.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Claude SDK Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    Application Layer                        │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │     │
│  │  │ClaudeSDK     │  │  Tool        │  │  Message         │  │     │
│  │  │Client        │  │  Runner      │  │  Stream          │  │     │
│  │  │(sessions)    │  │  (agent loop)│  │  (accumulation)  │  │     │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │     │
│  │         │                 │                    │            │     │
│  │         └─────────────────┼────────────────────┘            │     │
│  └───────────────────────────┼─────────────────────────────────┘     │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              Anthropic Client (sync/async)                  │     │
│  │  ┌────────────────────────────────────────────────────┐    │     │
│  │  │  messages.create()  │  messages.stream()           │    │     │
│  │  │  messages.count_tokens()  │  beta.messages.*       │    │     │
│  │  └────────────────────────────────────────────────────┘    │     │
│  └───────────────────────────┬─────────────────────────────────┘     │
│                              │                                       │
│  ┌───────────────────────────┼─────────────────────────────────┐     │
│  │              Context Management Layer                        │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │     │
│  │  │ Token        │  │ Context      │  │ Memory Tool      │   │     │
│  │  │ Counting     │  │ Editing      │  │ (file-based)     │   │     │
│  │  │ API          │  │ (beta)       │  │                  │   │     │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                              │                                       │
│  ┌───────────────────────────┼─────────────────────────────────┐     │
│  │              Subagent Isolation Layer                        │     │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │     │
│  │  │Subagent 1│  │Subagent 2│  │Subagent 3│  │Subagent N│    │     │
│  │  │(isolated)│  │(isolated)│  │(isolated)│  │(isolated)│    │     │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Message Protocol

#### 3.3.1 Message Structure

Messages follow a strict alternating pattern between `user` and `assistant` roles:

```python
from anthropic import Anthropic

client = Anthropic()

messages = [
    {"role": "user", "content": "Hello, Claude"},
    {"role": "assistant", "content": "Hello! How can I help you today?"},
    {"role": "user", "content": "Explain context management in LLMs"}
]

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system="You are a helpful AI assistant.",
    messages=messages
)
```

#### 3.3.2 Content Block Types

| Block Type | Direction | Token Cost | Description |
|------------|-----------|------------|-------------|
| `text` | Input/Output | Variable | Plain text content |
| `image` | Input | ~1,600 tokens | Base64 or URL image |
| `tool_use` | Output | Variable | Tool invocation request |
| `tool_result` | Input | Variable | Tool execution result |
| `thinking` | Output | Variable | Extended thinking (Claude 4+) |
| `document` | Input | 1,500-3,000/page | PDF documents |

```python
# Multi-modal message with image
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_encoded_image
            }
        }
    ]
}
```

### 3.4 Token Counting API

The Token Counting API enables precise context window management:

```python
# Count tokens before sending
count = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    system="You are a scientist specializing in AI.",
    messages=[
        {"role": "user", "content": "Explain transformer architecture"}
    ],
    tools=[
        {
            "name": "search_papers",
            "description": "Search academic papers",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]
)

print(f"Input tokens: {count.input_tokens}")
# Includes: system prompt + messages + tool definitions
```

### 3.5 Context Editing (Beta)

Context editing provides declarative rules for automatic context management:

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    betas=["context-management-2025-06-27"],
    messages=conversation_history,
    tools=tool_definitions,
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {
                    "type": "input_tokens",
                    "value": 30000  # Trigger at 30K tokens
                },
                "keep": {
                    "type": "tool_uses",
                    "value": 3  # Keep 3 most recent
                },
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 5000  # Free at least 5K tokens
                },
                "exclude_tools": ["web_search"]  # Never clear these
            },
            {
                "type": "clear_thinking_20251015",
                "trigger": {"type": "input_tokens", "value": 50000},
                "keep": {"type": "thinking_blocks", "value": 2}
            }
        ]
    }
)
```

#### Context Editing Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                 Context Edit Processing Pipeline                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 1: Calculate Current Token Count                     │   │
│  │   input_tokens = count(system + messages + tools)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 2: Evaluate Triggers                                 │   │
│  │   FOR each edit_rule in context_management.edits:         │   │
│  │     IF input_tokens >= trigger.value:                     │   │
│  │       activate_rule(edit_rule)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 3: Identify Clearable Content                        │   │
│  │   - Sort tool_uses by timestamp (oldest first)            │   │
│  │   - Exclude tools in exclude_tools list                   │   │
│  │   - Preserve keep.value most recent items                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 4: Clear Until Threshold Met                         │   │
│  │   cleared_tokens = 0                                      │   │
│  │   WHILE cleared_tokens < clear_at_least.value:            │   │
│  │     item = next_clearable_item()                          │   │
│  │     replace_with_placeholder(item)                        │   │
│  │     cleared_tokens += item.token_count                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Step 5: Invalidate Prompt Cache                           │   │
│  │   - Modified context invalidates cached prefixes          │   │
│  │   - New cache created from modified context               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.6 Memory Tool Implementation

The Memory Tool provides file-based persistent memory:

```python
from anthropic.types.beta import BetaAbstractMemoryTool
from pathlib import Path
import json

class LocalFilesystemMemoryTool(BetaAbstractMemoryTool):
    """Custom memory tool implementation using local filesystem."""

    def __init__(self, base_path: str = "./memory"):
        super().__init__()
        self.memory_root = Path(base_path) / "memories"
        self.memory_root.mkdir(parents=True, exist_ok=True)

    def view(self, path: str, view_range: list[int] | None = None) -> dict:
        """View directory contents or file sections."""
        full_path = self.memory_root / path.lstrip("/")

        if full_path.is_dir():
            entries = []
            for item in sorted(full_path.iterdir()):
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            return {
                "type": "directory",
                "path": path,
                "entries": entries
            }

        elif full_path.is_file():
            content = full_path.read_text()
            lines = content.splitlines()

            if view_range:
                start, end = view_range[0] - 1, view_range[1]
                lines = lines[start:end]

            return {
                "type": "file",
                "path": path,
                "content": "\n".join(lines),
                "total_lines": len(content.splitlines())
            }

        return {"error": f"Path not found: {path}"}

    def create(self, path: str, content: str) -> dict:
        """Create a new memory file."""
        full_path = self.memory_root / path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return {"status": "created", "path": path}

    def str_replace(self, path: str, old_str: str, new_str: str) -> dict:
        """Replace string in memory file."""
        full_path = self.memory_root / path.lstrip("/")
        if not full_path.is_file():
            return {"error": f"File not found: {path}"}

        content = full_path.read_text()
        if old_str not in content:
            return {"error": f"String not found in file"}

        new_content = content.replace(old_str, new_str, 1)
        full_path.write_text(new_content)
        return {"status": "replaced", "path": path}

    def insert(self, path: str, line_number: int, content: str) -> dict:
        """Insert content at specific line."""
        full_path = self.memory_root / path.lstrip("/")
        if not full_path.is_file():
            return {"error": f"File not found: {path}"}

        lines = full_path.read_text().splitlines()
        lines.insert(line_number - 1, content)
        full_path.write_text("\n".join(lines))
        return {"status": "inserted", "path": path, "line": line_number}

    def delete(self, path: str) -> dict:
        """Delete file or directory."""
        full_path = self.memory_root / path.lstrip("/")
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
        else:
            return {"error": f"Path not found: {path}"}
        return {"status": "deleted", "path": path}

    def rename(self, old_path: str, new_path: str) -> dict:
        """Rename/move file or directory."""
        old_full = self.memory_root / old_path.lstrip("/")
        new_full = self.memory_root / new_path.lstrip("/")

        if not old_full.exists():
            return {"error": f"Path not found: {old_path}"}

        new_full.parent.mkdir(parents=True, exist_ok=True)
        old_full.rename(new_full)
        return {"status": "renamed", "from": old_path, "to": new_path}


# Usage with API
memory_tool = LocalFilesystemMemoryTool("./agent_memory")

response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=2048,
    betas=["context-management-2025-06-27"],
    system="""You have access to a persistent memory system.
    Use it to store important information about the user and conversation.
    Organize memories in a logical directory structure.""",
    messages=[{"role": "user", "content": "Remember that my name is Alice and I prefer Python."}],
    tools=[{"type": "memory_20250818", "name": "memory"}]
)
```

### 3.7 Streaming Protocol

Claude uses Server-Sent Events (SSE) for real-time streaming:

```python
# Streaming implementation
async def stream_response(client, messages):
    accumulated_text = ""
    accumulated_thinking = ""
    tool_calls = []

    async with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 10000}
    ) as stream:
        async for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    print("[Text Block Started]")
                elif event.content_block.type == "thinking":
                    print("[Thinking Block Started]")
                elif event.content_block.type == "tool_use":
                    tool_calls.append({
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input": ""
                    })

            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    accumulated_text += event.delta.text
                    print(event.delta.text, end="", flush=True)
                elif event.delta.type == "thinking_delta":
                    accumulated_thinking += event.delta.thinking
                elif event.delta.type == "input_json_delta":
                    tool_calls[-1]["input"] += event.delta.partial_json

            elif event.type == "message_stop":
                print("\n[Message Complete]")

    return {
        "text": accumulated_text,
        "thinking": accumulated_thinking,
        "tool_calls": tool_calls
    }
```

#### SSE Event Types

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}

event: message_stop
data: {"type":"message_stop"}
```

### 3.8 Tool Use Protocol

```python
# Tool definition schema
tool_definition = {
    "name": "search_database",
    "description": "Search the company database for relevant information",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_range": {"type": "string"},
                    "category": {"type": "string"}
                }
            },
            "max_results": {
                "type": "integer",
                "default": 10
            }
        },
        "required": ["query"]
    }
}

# Tool execution loop
def execute_tool_loop(client, messages, tools):
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            messages=messages,
            tools=tools
        )

        # Check if tool use is requested
        tool_use_blocks = [
            block for block in response.content
            if block.type == "tool_use"
        ]

        if not tool_use_blocks:
            # No tool calls, return final response
            return response

        # Execute tools and add results
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_use_blocks:
            result = execute_tool(tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            })

        messages.append({"role": "user", "content": tool_results})
```

### 3.9 Subagent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                            │
│                (Full context, global planning)                   │
├─────────────────────────────────────────────────────────────────┤
│  Context: [System Prompt + Full History + All Tool Results]     │
│  Role: Decompose tasks, coordinate subagents, synthesize        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│  Code Reviewer   │ │  Test Writer     │ │  Doc Generator   │
│    Subagent      │ │    Subagent      │ │    Subagent      │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ Context:         │ │ Context:         │ │ Context:         │
│ - Task-specific  │ │ - Task-specific  │ │ - Task-specific  │
│ - Isolated       │ │ - Isolated       │ │ - Isolated       │
│ - Limited tools  │ │ - Limited tools  │ │ - Limited tools  │
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ Returns:         │ │ Returns:         │ │ Returns:         │
│ Summary only     │ │ Summary only     │ │ Summary only     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
```

```python
# Subagent configuration (filesystem-based)
# .claude/agents/code_reviewer.md
"""
---
model: claude-sonnet-4-5
system_prompt: |
  You are an expert code reviewer.
  Focus on: security, performance, readability.
  Output structured review comments.
allowed_tools:
  - read_file
  - grep
  - glob
max_turns: 5
---
"""

# Subagent configuration (programmatic)
from claude_agent_sdk import ClaudeAgentOptions

reviewer_config = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    system_prompt="You are an expert code reviewer...",
    allowed_tools=["read_file", "grep", "glob"],
    max_turns=5
)
```

### 3.10 Performance Metrics

| Feature | Performance Impact |
|---------|-------------------|
| Context editing alone | 29% improvement |
| Memory tool + context editing | 39% improvement |
| Token reduction (100-turn) | 84% reduction |
| Prompt caching (cache hit) | 90% cost reduction |

---

## 4. Google Agent Development Kit (ADK)

### 4.1 Design Philosophy

Google ADK provides a **hierarchical context system** with built-in session management, state scoping, and multi-agent coordination. The framework emphasizes structured state management with explicit scoping prefixes.

### 4.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Google ADK Architecture                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         Runner                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │                   Event Processor                         │  │ │
│  │  │  - Process events from agents                             │  │ │
│  │  │  - Apply state deltas                                     │  │ │
│  │  │  - Handle agent transfers                                 │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │                   Session Manager                         │  │ │
│  │  │  - Create/retrieve sessions                               │  │ │
│  │  │  - Persist state changes                                  │  │ │
│  │  │  - Manage event history                                   │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │                  Agent Orchestrator                       │  │ │
│  │  │  - Route to appropriate agent                             │  │ │
│  │  │  - Handle handoffs                                        │  │ │
│  │  │  - Manage turn lifecycle                                  │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                    Agent Layer                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │   LlmAgent   │  │ Sequential   │  │     Parallel         │ │  │
│  │  │              │  │   Agent      │  │      Agent           │ │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │  │
│  │         │                 │                      │             │  │
│  │         └─────────────────┼──────────────────────┘             │  │
│  └───────────────────────────┼────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                  Context Hierarchy                             │  │
│  │  ┌────────────────────────────────────────────────────────┐   │  │
│  │  │  InvocationContext (Full Access)                        │   │  │
│  │  │    - session, state, events, services                   │   │  │
│  │  │    ┌────────────────────────────────────────────────┐   │   │  │
│  │  │    │  CallbackContext (Callbacks)                    │   │   │  │
│  │  │    │    - read/write state, agent info               │   │   │  │
│  │  │    │    ┌────────────────────────────────────────┐   │   │   │  │
│  │  │    │    │  ToolContext (Tools)                   │   │   │   │  │
│  │  │    │    │    - state access, transfers           │   │   │   │  │
│  │  │    │    │    ┌──────────────────────────────┐    │   │   │   │  │
│  │  │    │    │    │  ReadonlyContext (Dynamic)   │    │   │   │   │  │
│  │  │    │    │    │    - read-only state view    │    │   │   │   │  │
│  │  │    │    │    └──────────────────────────────┘    │   │   │   │  │
│  │  │    │    └────────────────────────────────────────┘   │   │   │  │
│  │  │    └────────────────────────────────────────────────┘   │   │  │
│  │  └────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                   Services Layer                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │   Session    │  │   Memory     │  │     Artifact         │ │  │
│  │  │   Service    │  │   Service    │  │     Service          │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Context Type Hierarchy

```python
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator

# InvocationContext - Full access in agent implementation
class CustomAgent(BaseAgent):
    async def _run_async_impl(
        self,
        ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # Full access to all context
        agent_name = ctx.agent.name
        session = ctx.session
        session_id = session.id
        user_content = ctx.user_content

        # Access services
        artifact_service = ctx.artifact_service
        memory_service = ctx.memory_service
        session_service = ctx.session_service

        # Read/write state
        user_name = ctx.session.state.get("user:name", "Unknown")
        ctx.session.state["last_action"] = "processed"

        # Control flow
        if ctx.session.state.get("should_stop"):
            ctx.end_invocation = True

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=Content(parts=[Part(text=f"Hello, {user_name}!")])
        )


# ToolContext - Access in tool functions
def search_database(
    query: str,
    tool_context: ToolContext
) -> dict:
    """Tool with context access."""
    # Read state
    user_id = tool_context.state.get("user:id")

    # Write state
    tool_context.state["temp:last_query"] = query

    # Trigger agent transfer
    if "escalate" in query.lower():
        tool_context.actions.transfer_to_agent = "support_agent"

    # Access services
    results = tool_context.memory_service.search_memory(query)

    return {"results": results, "user_id": user_id}


# CallbackContext - Access in callbacks
def before_model_callback(
    ctx: CallbackContext,
    request: LlmRequest
) -> Optional[LlmResponse]:
    """Callback with state access."""
    # Read state
    call_count = ctx.state.get("temp:call_count", 0)
    ctx.state["temp:call_count"] = call_count + 1

    # Can modify or intercept request
    if call_count > 10:
        return LlmResponse(
            content=Content(parts=[Part(text="Rate limit exceeded")])
        )
    return None


# ReadonlyContext - Access in dynamic instructions
def get_dynamic_instructions(
    ctx: ReadonlyContext
) -> str:
    """Dynamic instructions with read-only access."""
    user_name = ctx.state.get("user:name", "User")
    language = ctx.state.get("user:language", "English")

    return f"""You are helping {user_name}.
    Respond in {language}.
    Be concise and helpful."""
```

### 4.4 State Management System

#### 4.4.1 State Prefix Scoping

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

# Create session with initial state
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
    state={
        # Session-scoped (default) - this session only
        "current_topic": "context management",
        "turn_count": 0,

        # User-scoped - persists across all user's sessions
        "user:name": "Alice",
        "user:preferences": {"language": "English", "tone": "professional"},
        "user:history_summary": "Interested in AI development",

        # App-scoped - shared across all users
        "app:version": "2.1.0",
        "app:feature_flags": {"new_ui": True},

        # Temp-scoped - current invocation only, never persisted
        "temp:request_id": "req_789",
        "temp:start_time": "2026-01-20T10:00:00Z"
    }
)
```

#### 4.4.2 State Scope Behavior

| Prefix | Scope | Persistence | Use Case |
|--------|-------|-------------|----------|
| (none) | Session | Session lifetime | Conversation state |
| `user:` | User | Across sessions | Preferences, profile |
| `app:` | Application | Global | Configuration, flags |
| `temp:` | Invocation | Never | Request-specific data |

#### 4.4.3 State Delta System

```
┌─────────────────────────────────────────────────────────────────────┐
│                     State Delta Flow                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Step 1: Code Modifies State via Context                      │   │
│  │                                                                │   │
│  │    callback_context.state["user:preference"] = "dark_mode"    │   │
│  │                                                                │   │
│  │    Note: This does NOT immediately modify session.state       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Framework Records Delta                              │   │
│  │                                                                │   │
│  │    StateDeltaTracker intercepts modification                  │   │
│  │    pending_delta = {"user:preference": "dark_mode"}           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Step 3: Event Carries Delta                                  │   │
│  │                                                                │   │
│  │    event = Event(                                             │   │
│  │        author="agent_name",                                   │   │
│  │        actions=EventActions(                                  │   │
│  │            state_delta={"user:preference": "dark_mode"}       │   │
│  │        )                                                      │   │
│  │    )                                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Step 4: SessionService Applies Delta Atomically              │   │
│  │                                                                │   │
│  │    session_service.append_event(session, event)               │   │
│  │    - Merges state_delta into session.state                    │   │
│  │    - Persists event to storage                                │   │
│  │    - Updates appropriate scope (session/user/app)             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
# Manual state delta in custom agent
class StatefulAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        # Create event with explicit state delta
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=Content(parts=[Part(text="Processing...")]),
            actions=EventActions(
                state_delta={
                    "user:last_seen": datetime.now().isoformat(),
                    "interaction_count": ctx.session.state.get("interaction_count", 0) + 1
                }
            )
        )
```

### 4.5 Template Injection

```python
from google.adk.agents import LlmAgent

# State values are automatically injected into instructions
agent = LlmAgent(
    name="PersonalizedAgent",
    model="gemini-2.0-flash",
    instruction="""You are a helpful assistant for {user:name}.

    User Preferences:
    - Language: {user:language}
    - Tone: {user:tone}

    Current Session:
    - Topic: {current_topic}
    - Turn: {turn_count}

    Optional context: {additional_context?}
    """,  # ? makes the variable optional
    output_key="response"  # Auto-save output to state["response"]
)
```

### 4.6 Session Services

```python
from google.adk.sessions import (
    InMemorySessionService,
    DatabaseSessionService,
    VertexAiSessionService
)

# Development: In-memory (no persistence)
dev_service = InMemorySessionService()

# Production: Database-backed
db_service = DatabaseSessionService(
    db_url="postgresql+asyncpg://user:pass@localhost:5432/adk_db"
)

# Google Cloud: Vertex AI managed
vertex_service = VertexAiSessionService(
    project_id="my-project",
    location="us-central1"
)
```

#### Database Schema (DatabaseSessionService)

```sql
-- Sessions table
CREATE TABLE sessions (
    id VARCHAR PRIMARY KEY,
    app_name VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Events table (append-only)
CREATE TABLE raw_events (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR REFERENCES sessions(id),
    invocation_id VARCHAR NOT NULL,
    author VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    content JSONB,
    actions JSONB,
    timestamp FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User state (cross-session)
CREATE TABLE user_state (
    user_id VARCHAR NOT NULL,
    app_name VARCHAR NOT NULL,
    key VARCHAR NOT NULL,
    value JSONB,
    PRIMARY KEY (user_id, app_name, key)
);

-- App state (global)
CREATE TABLE app_state (
    app_name VARCHAR NOT NULL,
    key VARCHAR NOT NULL,
    value JSONB,
    PRIMARY KEY (app_name, key)
);

CREATE INDEX idx_events_session ON raw_events(session_id);
CREATE INDEX idx_events_timestamp ON raw_events(timestamp);
```

### 4.7 Memory Service (Long-term)

```python
from google.adk.memory import InMemoryMemoryService, VertexAiRagMemoryService
from google.adk.tools import load_memory

# Simple in-memory (keyword-based)
memory_service = InMemoryMemoryService()

# Production: Vertex AI RAG
rag_memory = VertexAiRagMemoryService(
    project_id="my-project",
    location="us-central1",
    rag_corpus_id="my-corpus"
)

# Agent with memory access
agent = LlmAgent(
    name="MemoryAgent",
    model="gemini-2.0-flash",
    instruction="""You can recall information from past conversations.
    Use the load_memory tool to retrieve relevant context.""",
    tools=[load_memory]
)

# Runner with memory service
runner = Runner(
    agent=agent,
    app_name="memory_app",
    session_service=session_service,
    memory_service=memory_service
)

# Archive completed session to long-term memory
async def archive_session(session):
    await memory_service.add_session_to_memory(session)
```

### 4.8 Multi-Agent Patterns

#### 4.8.1 Sequential Agent (Shared Context)

```python
from google.adk.agents import LlmAgent, SequentialAgent

# Research agent writes to state
researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash",
    instruction="Research the given topic thoroughly.",
    output_key="research_findings"  # Saved to state
)

# Writer reads from state
writer = LlmAgent(
    name="Writer",
    model="gemini-2.0-flash",
    instruction="""Write a report based on the research.

    Research findings: {research_findings}"""
)

# Reviewer reads both
reviewer = LlmAgent(
    name="Reviewer",
    model="gemini-2.0-flash",
    instruction="""Review the report for accuracy.

    Original research: {research_findings}
    Draft report: {report_draft}""",
    output_key="final_report"
)

# Pipeline shares context
pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, writer, reviewer]
)
```

#### 4.8.2 AgentTool (Isolated Context)

```python
from google.adk.tools import AgentTool

# Calculator runs in isolation
calculator = LlmAgent(
    name="Calculator",
    model="gemini-2.0-flash",
    instruction="Perform mathematical calculations accurately."
)

# Wrap as tool - isolated execution context
calculator_tool = AgentTool(agent=calculator)

# Main agent uses calculator without sharing context
main_agent = LlmAgent(
    name="Assistant",
    model="gemini-2.0-flash",
    instruction="Help users with various tasks.",
    tools=[calculator_tool, search_tool, ...]
)
```

#### 4.8.3 Agent Transfer Protocol

```python
# Transfer via tool
def escalate_to_human(
    reason: str,
    tool_context: ToolContext
) -> dict:
    """Escalate complex issues to human support."""
    tool_context.actions.transfer_to_agent = "human_support"
    tool_context.state["escalation_reason"] = reason
    return {"status": "transferring", "reason": reason}

# Transfer via agent
class TriageAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        # Analyze and route
        category = await self._categorize(ctx.user_content)

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=Content(parts=[Part(text=f"Routing to {category}")]),
            actions=EventActions(
                transfer_to_agent=f"{category}_specialist",
                state_delta={"routing_category": category}
            )
        )
```

### 4.9 Callbacks System

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from typing import Optional

# All callback types
def before_agent_callback(ctx: CallbackContext) -> Optional[Content]:
    """Called before agent starts processing."""
    print(f"Agent starting: {ctx.agent.name}")
    # Return Content to skip agent execution
    return None

def after_agent_callback(ctx: CallbackContext, output: Content) -> Optional[Content]:
    """Called after agent completes."""
    print(f"Agent completed with {len(output.parts)} parts")
    # Return modified Content or None
    return output

def before_model_callback(
    ctx: CallbackContext,
    request: LlmRequest
) -> Optional[LlmResponse]:
    """Called before LLM API call."""
    # Implement caching
    cache_key = hash(str(request))
    if cached := get_cache(cache_key):
        return cached

    # Implement guardrails
    if contains_pii(request):
        return LlmResponse(content=Content(
            parts=[Part(text="Cannot process PII")]
        ))

    return None

def after_model_callback(
    ctx: CallbackContext,
    response: LlmResponse
) -> Optional[LlmResponse]:
    """Called after LLM response."""
    # Log usage
    log_tokens(response.usage_metadata)

    # Post-process response
    return response

# Agent with all callbacks
agent = LlmAgent(
    name="CallbackAgent",
    model="gemini-2.0-flash",
    instruction="...",
    before_agent_callback=before_agent_callback,
    after_agent_callback=after_agent_callback,
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback
)
```

---

## 5. LangChain Framework

### 5.1 Design Philosophy

LangChain provides **pluggable memory abstractions** with extensive integration options. The framework has evolved significantly, transitioning from legacy memory classes to modern patterns based on LangGraph and RunnableWithMessageHistory.

### 5.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LangChain Architecture                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Application Layer                            │ │
│  │  ┌────────────────────────────────────────────────────────┐    │ │
│  │  │                    LCEL Chains                          │    │ │
│  │  │  prompt | llm | output_parser                           │    │ │
│  │  └────────────────────────────────────────────────────────┘    │ │
│  │  ┌────────────────────────────────────────────────────────┐    │ │
│  │  │              RunnableWithMessageHistory                 │    │ │
│  │  │  - Wraps chains with automatic history management       │    │ │
│  │  │  - Session-based history retrieval                      │    │ │
│  │  └────────────────────────────────────────────────────────┘    │ │
│  │  ┌────────────────────────────────────────────────────────┐    │ │
│  │  │                    LangGraph                            │    │ │
│  │  │  - Stateful graph execution                             │    │ │
│  │  │  - Checkpointers for persistence                        │    │ │
│  │  │  - Cross-thread memory via Store                        │    │ │
│  │  └────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                    Memory Layer                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Modern (Recommended)                                    │  │  │
│  │  │  ┌─────────────────┐  ┌─────────────────────────────┐   │  │  │
│  │  │  │ BaseChatMessage │  │ LangGraph Checkpointers     │   │  │  │
│  │  │  │ History         │  │ - MemorySaver               │   │  │  │
│  │  │  │ - InMemory      │  │ - PostgresSaver             │   │  │  │
│  │  │  │ - Redis         │  │ - SqliteSaver               │   │  │  │
│  │  │  │ - PostgreSQL    │  │                             │   │  │  │
│  │  │  │ - MongoDB       │  │ LangGraph Store             │   │  │  │
│  │  │  └─────────────────┘  │ - Cross-thread memory       │   │  │  │
│  │  │                       └─────────────────────────────┘   │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Legacy (Deprecated v0.3.1+)                             │  │  │
│  │  │  - ConversationBufferMemory                              │  │  │
│  │  │  - ConversationSummaryMemory                             │  │  │
│  │  │  - ConversationBufferWindowMemory                        │  │  │
│  │  │  - VectorStoreRetrieverMemory                            │  │  │
│  │  │  - ConversationEntityMemory                              │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                  Message System                                │  │
│  │  BaseMessage                                                   │  │
│  │  ├── HumanMessage    (type: "human")                          │  │
│  │  ├── AIMessage       (type: "ai", tool_calls, usage)          │  │
│  │  ├── SystemMessage   (type: "system")                         │  │
│  │  ├── ToolMessage     (type: "tool", tool_call_id)             │  │
│  │  └── ChatMessage     (type: "chat", dynamic role)             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                  Callback System                               │  │
│  │  CallbackManager                                               │  │
│  │  ├── on_llm_start / on_llm_end                                │  │
│  │  ├── on_chain_start / on_chain_end                            │  │
│  │  ├── on_tool_start / on_tool_end                              │  │
│  │  └── on_retriever_start / on_retriever_end                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Message Type System

```python
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    ChatMessage
)
from pydantic import Field
from typing import Union, List, Dict, Optional, Literal

# BaseMessage structure
class BaseMessage:
    content: Union[str, List[Union[str, Dict]]]
    additional_kwargs: dict = Field(default_factory=dict)
    response_metadata: dict = Field(default_factory=dict)
    type: str
    name: Optional[str] = None
    id: Optional[str] = None

# AIMessage with tool calls
class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"
    tool_calls: list[ToolCall] = Field(default_factory=list)
    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    usage_metadata: Optional[UsageMetadata] = None

# ToolCall structure
class ToolCall(TypedDict):
    name: str
    args: Dict[str, Any]
    id: Optional[str]
    type: Literal["tool_call"] = "tool_call"

# ToolMessage for responses
class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    tool_call_id: str  # Must match AIMessage.tool_calls[].id
    artifact: Optional[Any] = None
    status: Literal["success", "error"] = "success"

# Example usage
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's the weather?"),
    AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather",
            "args": {"city": "London"},
            "id": "call_123",
            "type": "tool_call"
        }]
    ),
    ToolMessage(
        content='{"temp": 15, "condition": "cloudy"}',
        tool_call_id="call_123"
    ),
    AIMessage(content="The weather in London is 15°C and cloudy.")
]
```

### 5.4 Modern Approach: RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from typing import Callable

# Session store
session_store: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Factory function for session history."""
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

# Build chain with history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | ChatOpenAI(model="gpt-4o")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Invoke with session
response = chain_with_history.invoke(
    {"input": "Hi, I'm Alice"},
    config={"configurable": {"session_id": "user_123"}}
)

# Follow-up (history is maintained)
response = chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user_123"}}
)
# Response: "Your name is Alice."
```

### 5.5 Chat History Implementations

```python
# Redis implementation
from langchain_redis import RedisChatMessageHistory

class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379",
        key_prefix: str = "chat_history:",
        ttl: Optional[int] = None
    ):
        self.session_id = session_id
        self.key = f"{key_prefix}{session_id}"
        self.ttl = ttl
        self._client = Redis.from_url(url)

    @property
    def messages(self) -> List[BaseMessage]:
        data = self._client.json().get(self.key)
        if data is None:
            return []
        return [_message_from_dict(m) for m in data["messages"]]

    def add_message(self, message: BaseMessage) -> None:
        if not self._client.exists(self.key):
            self._client.json().set(self.key, "$", {"messages": []})
        self._client.json().arrappend(self.key, "$.messages", message.dict())
        if self.ttl:
            self._client.expire(self.key, self.ttl)

    def clear(self) -> None:
        self._client.delete(self.key)


# PostgreSQL implementation
from langchain_postgres import PostgresChatMessageHistory

history = PostgresChatMessageHistory(
    session_id="user_123",
    connection_string="postgresql://user:pass@localhost/db",
    table_name="chat_history"
)


# MongoDB implementation
from langchain_mongodb import MongoDBChatMessageHistory

history = MongoDBChatMessageHistory(
    connection_string="mongodb://localhost:27017",
    session_id="user_123",
    database_name="chat_db",
    collection_name="histories",
    history_size=100  # Keep last 100 messages
)
```

### 5.6 LangGraph State Management

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore

# State schema with reducer
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    context: dict

# add_messages reducer implementation
def add_messages(
    left: list[BaseMessage],
    right: Union[BaseMessage, list[BaseMessage]]
) -> list[BaseMessage]:
    """Append messages with ID-based deduplication."""
    if isinstance(right, BaseMessage):
        right = [right]

    left_by_id = {m.id: m for m in left if m.id}
    result = list(left)

    for msg in right:
        if msg.id and msg.id in left_by_id:
            # Update existing message
            idx = next(i for i, m in enumerate(result) if m.id == msg.id)
            result[idx] = msg
        else:
            result.append(msg)

    return result

# Build graph
graph_builder = StateGraph(State)

def chatbot(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

# Compile with persistence
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/db"
)

store = InMemoryStore()  # Cross-thread memory

graph = graph_builder.compile(
    checkpointer=checkpointer,
    store=store
)

# Invoke with thread_id
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "conv_123"}}
)
```

### 5.7 Token Management

```python
from langchain_core.messages.utils import (
    trim_messages,
    filter_messages,
    merge_message_runs,
    count_tokens_approximately
)

# Trim by token count
trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",  # Keep most recent
    token_counter=count_tokens_approximately,
    start_on="human",  # Ensure starts with human
    include_system=True,  # Always keep system
    allow_partial=False
)

# Trim by message count
trimmed = trim_messages(
    messages,
    max_tokens=10,
    token_counter=len  # Counts messages
)

# Filter by type
filtered = filter_messages(
    messages,
    include_types=[HumanMessage, AIMessage],
    exclude_ids=["msg_to_remove"]
)

# Merge consecutive same-type
merged = merge_message_runs(messages)

# Exact counting with tiktoken
import tiktoken

def count_tokens_exact(messages: list, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # Message overhead
        if isinstance(message.content, str):
            num_tokens += len(encoding.encode(message.content))
    num_tokens += 3  # Reply priming
    return num_tokens
```

---

## 6. Manus AI

### 6.1 Design Philosophy

Manus AI represents a paradigm shift in context management, prioritizing **KV-cache optimization** as the primary design principle. The framework treats the file system as extended memory and uses append-only context modification to maximize cache hit rates.

### 6.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Manus Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    Agent Orchestration                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │ │
│  │  │   Planner    │  │   Executor   │  │     Verifier         │  │ │
│  │  │    Agent     │──▶│    Agent     │──▶│      Agent          │  │ │
│  │  │  (planning)  │  │  (actions)   │  │   (validation)       │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                    Event Stream                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  [message] → [action] → [observation] → [plan] → ...    │  │  │
│  │  │                                                          │  │  │
│  │  │  Append-only for KV-cache optimization                   │  │  │
│  │  │  Never modify previous events                            │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                  Context Management                            │  │
│  │  ┌──────────────────┐  ┌──────────────────────────────────┐   │  │
│  │  │  KV-Cache        │  │  Recoverable Compression          │   │  │
│  │  │  Optimization    │  │  - Drop content, keep references  │   │  │
│  │  │  - Stable prefix │  │  - Summarize only when needed     │   │  │
│  │  │  - Append-only   │  │  - Preserve recent turns raw      │   │  │
│  │  │  - Logit masking │  │                                   │   │  │
│  │  └──────────────────┘  └──────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │              Three-Tier Memory Architecture                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │ Tier 1:      │  │ Tier 2:      │  │ Tier 3:              │ │  │
│  │  │ Active       │  │ Session      │  │ Long-term            │ │  │
│  │  │ Context      │  │ Files        │  │ Memory               │ │  │
│  │  │ (in LLM)     │  │ (filesystem) │  │ (vector DB)          │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                    Execution Environment                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │   Browser    │  │    Shell     │  │    File System       │ │  │
│  │  │ (Playwright) │  │  (Execute)   │  │  (Extended Memory)   │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  │                              │                                 │  │
│  │                    ┌─────────▼─────────┐                      │  │
│  │                    │   Ubuntu Sandbox   │                      │  │
│  │                    │     (Docker)       │                      │  │
│  │                    └───────────────────┘                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.3 KV-Cache Optimization Principles

```python
class KVCacheOptimizer:
    """
    KV-cache hit rate is the single most important metric
    for production AI agents.

    Cost difference: 10x ($0.30/MTok cached vs $3.00/MTok uncached)
    Input-to-output ratio: ~100:1
    """

    # Principle 1: Stable Prefixes
    # BAD - timestamp invalidates entire cache
    def bad_system_prompt(self):
        return f"Current time: {datetime.now()}\n{self.instructions}"

    # GOOD - static prefix, dynamic data passed separately
    def good_system_prompt(self):
        return self.instructions  # Timestamp in user message

    # Principle 2: Append-Only Context
    def add_to_context(self, event):
        # NEVER modify previous events
        self.context.append(event)  # Always append

    # Principle 3: Explicit Cache Breakpoints
    cache_breakpoints = [
        "system_prompt_end",
        "tool_definitions_end",
        "conversation_history_start"
    ]
```

### 6.4 Tool Management via Logit Masking

```python
class ToolManager:
    """
    Instead of dynamically adding/removing tools (which breaks KV-cache),
    use logit masking to control tool availability.
    """

    def __init__(self, all_tools: list):
        # Keep ALL tools in context for cache stability
        self.all_tools = all_tools
        self.tool_groups = {
            'browser': ['browser_navigate', 'browser_click', 'browser_input',
                       'browser_scroll', 'browser_screenshot'],
            'shell': ['shell_exec', 'shell_view', 'shell_wait'],
            'file': ['file_read', 'file_write', 'file_str_replace',
                    'file_find_replace', 'file_insert']
        }

    def get_logit_mask(self, allowed_groups: list[str]) -> dict:
        """
        Returns mask that prevents selection of disallowed tools
        during decoding, WITHOUT modifying tool definitions.
        """
        mask = {}
        for group, tools in self.tool_groups.items():
            if group not in allowed_groups:
                for tool in tools:
                    mask[tool] = float('-inf')  # Impossible to select
        return mask

    def get_tools(self) -> list:
        """Always return all tools for cache stability."""
        return self.all_tools  # Never modify this list
```

### 6.5 File System as Extended Memory

```python
class FileSystemMemory:
    """
    Treat the file system as unlimited, persistent memory.
    Keep only references in LLM context.
    """

    def __init__(self, workspace: str = "/home/ubuntu"):
        self.workspace = workspace
        self.todo_file = f"{workspace}/todo.md"
        self.notes_dir = f"{workspace}/notes"
        self.data_dir = f"{workspace}/data"

    def save_intermediate_result(self, filename: str, content: str) -> str:
        """Save large content to file, return only path."""
        path = f"{self.notes_dir}/{filename}"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        # Return path for context, NOT full content
        return f"[Content saved to: {path}]"

    def update_todo(self, tasks: list[dict]) -> None:
        """
        The todo.md pattern: constantly rewrite to push
        objectives into recent attention span.
        """
        with open(self.todo_file, 'w') as f:
            f.write(f"# Current Task\n\n")
            for i, task in enumerate(tasks, 1):
                status = "x" if task.get('done') else " "
                current = " <-- Current" if task.get('current') else ""
                f.write(f"- [{status}] {task['description']}{current}\n")

    def read_with_caching(self, path: str) -> str:
        """Read file, potentially from cache."""
        cache_key = f"{path}:{os.path.getmtime(path)}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        content = Path(path).read_text()
        self._cache[cache_key] = content
        return content
```

### 6.6 Recoverable Compression Strategy

```python
class ContextCompressor:
    """
    Compression strategies (in order of preference):
    1. Raw (no compression) - best quality
    2. Compaction (drop recoverable content) - references retained
    3. Summarization (only when necessary) - lossy
    """

    def __init__(self, max_tokens: int = 128000):
        self.max_tokens = max_tokens

    def compress(self, context: list) -> list:
        current_tokens = self.count_tokens(context)

        if current_tokens < self.max_tokens:
            return context  # No compression needed

        # Phase 1: Compaction - drop recoverable content
        for item in context:
            if item['type'] == 'webpage':
                # Replace content with reference
                item['content'] = f"[Content available at: {item['url']}]"
            elif item['type'] == 'file':
                item['content'] = f"[File contents at: {item['path']}]"
            elif item['type'] == 'tool_result' and len(item['content']) > 1000:
                # Truncate large tool results
                item['content'] = item['content'][:500] + "\n...[truncated]..."

        if self.count_tokens(context) < self.max_tokens:
            return context

        # Phase 2: Summarization - preserve recent turns raw
        recent_turns = context[-3:]  # Keep last 3 interactions raw
        older_turns = context[:-3]

        summary = self.llm_summarize(older_turns)
        return [{'type': 'summary', 'content': summary}] + recent_turns

    def llm_summarize(self, turns: list) -> str:
        """Use LLM to create summary of older turns."""
        prompt = f"""Summarize the following conversation history,
        preserving key facts, decisions, and outcomes:

        {json.dumps(turns, indent=2)}"""

        return self.llm.invoke(prompt)
```

### 6.7 Event Stream Architecture

```python
class EventStream:
    """
    Chronological log of all interactions.
    Backbone of context management.
    """

    EVENT_TYPES = [
        'message',      # User/assistant messages
        'action',       # Tool invocations
        'observation',  # Tool results
        'plan',         # Planning outputs
        'knowledge',    # Retrieved information
        'datasource'    # External data references
    ]

    def __init__(self):
        self.events = []

    def add_event(
        self,
        event_type: str,
        content: Any,
        metadata: dict = None
    ) -> dict:
        """Append event (never modify existing)."""
        event = {
            'id': str(uuid4()),
            'type': event_type,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.events.append(event)  # Append-only for KV-cache
        return event

    def get_context_window(self, max_tokens: int) -> list:
        """Build context prioritizing recent events."""
        recent = self.events[-10:]
        older = self.events[:-10]

        # Summarize older if too large
        if self.count_tokens(older) > max_tokens * 0.3:
            older = self.summarize_events(older)

        return older + recent

    def handle_error(self, action: dict, error: Exception) -> dict:
        """
        Error Preservation Pattern:
        Leave wrong turns visible for model to learn.
        """
        observation = {
            'type': 'observation',
            'action': action,
            'success': False,
            'error': str(error),
            'stacktrace': traceback.format_exc()
        }
        return self.add_event('observation', observation)
```

### 6.8 Wide Research Pattern (100+ Parallel Agents)

```python
class WideResearch:
    """Execute large-scale research with massive parallelization."""

    async def execute(self, query: str, num_agents: int = 100) -> dict:
        # Decompose query into independent research tracks
        tracks = self.decompose_query(query, num_agents)

        # Create isolated agents
        agents = [
            ResearchAgent(
                context=self.create_isolated_context(track),
                sandbox=self.create_sandbox()
            )
            for track in tracks
        ]

        # Execute in parallel
        results = await asyncio.gather(
            *[agent.research() for agent in agents],
            return_exceptions=True
        )

        # Filter successes
        successes = [r for r in results if not isinstance(r, Exception)]

        # Synthesize with consensus mechanism
        return self.synthesize_with_consensus(successes)

    def create_isolated_context(self, track: dict) -> dict:
        """Each agent gets isolated context."""
        return {
            'task': track['task'],
            'constraints': track['constraints'],
            'output_format': track['format'],
            # No shared state - isolation by design
        }

    def synthesize_with_consensus(self, results: list) -> dict:
        """Aggregate results with conflict resolution."""
        # Group by topic
        grouped = defaultdict(list)
        for result in results:
            grouped[result['topic']].append(result)

        # Resolve conflicts via voting/confidence
        synthesized = {}
        for topic, findings in grouped.items():
            synthesized[topic] = self.resolve_conflicts(findings)

        return synthesized
```

### 6.9 GAIA Benchmark Performance

| Level | Manus | OpenAI Operator | Previous SOTA |
|-------|-------|-----------------|---------------|
| Level 1 | **86.5%** | 74.3% | 67.9% |
| Level 2 | **70.1%** | 69.1% | 67.4% |
| Level 3 | **57.7%** | 47.6% | 42.3% |

---

## 7. OpenAI Agents SDK

### 7.1 Design Philosophy

The OpenAI Agents SDK is a lightweight, Python-first framework for building multi-agent workflows. It emphasizes **dependency injection** for context management and provides built-in session persistence with multiple storage backends.

### 7.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OpenAI Agents SDK Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                         Runner                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐  │ │
│  │  │                    Execution Loop                         │  │ │
│  │  │  1. Invoke LLM with current context                       │  │ │
│  │  │  2. Process tool calls or handoffs                        │  │ │
│  │  │  3. Switch agent if handoff occurs                        │  │ │
│  │  │  4. Update context variables                              │  │ │
│  │  │  5. Continue until final output                           │  │ │
│  │  └──────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                    Agent Layer                                 │  │
│  │  ┌────────────────────────────────────────────────────────┐   │  │
│  │  │  Agent                                                  │   │  │
│  │  │  ├── name: str                                         │   │  │
│  │  │  ├── instructions: str | Callable                      │   │  │
│  │  │  ├── model: str                                        │   │  │
│  │  │  ├── tools: list[Tool]                                 │   │  │
│  │  │  ├── handoffs: list[Handoff]                           │   │  │
│  │  │  ├── output_type: Type (Pydantic)                      │   │  │
│  │  │  ├── input_guardrails: list[Guardrail]                 │   │  │
│  │  │  ├── output_guardrails: list[Guardrail]                │   │  │
│  │  │  └── hooks: AgentHooks                                 │   │  │
│  │  └────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                  Context System                                │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  RunContextWrapper[T]                                    │  │  │
│  │  │  - Wraps user-defined context object                     │  │  │
│  │  │  - Shared across agents, tools, hooks                    │  │  │
│  │  │  - NOT sent to LLM (local only)                          │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Sessions (Memory)                                       │  │  │
│  │  │  ├── SQLiteSession                                       │  │  │
│  │  │  ├── SQLAlchemySession                                   │  │  │
│  │  │  ├── RedisSession                                        │  │  │
│  │  │  ├── DaprSession                                         │  │  │
│  │  │  └── EncryptedSession                                    │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                Multi-Agent Patterns                            │  │
│  │  ┌──────────────────────┐  ┌──────────────────────────────┐   │  │
│  │  │  Handoffs            │  │  Agent as Tool               │   │  │
│  │  │  (Decentralized)     │  │  (Centralized)               │   │  │
│  │  │  - Agent transfers   │  │  - Manager invokes           │   │  │
│  │  │    control to peer   │  │    sub-agents as tools       │   │  │
│  │  │  - Full history      │  │  - Manager retains           │   │  │
│  │  │    passes (default)  │  │    control                   │   │  │
│  │  └──────────────────────┘  └──────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────┼───────────────────────────────────┐  │
│  │                 Safety & Observability                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │  │
│  │  │  Guardrails  │  │   Tracing    │  │      Hooks           │ │  │
│  │  │  - Input     │  │  - Built-in  │  │  - on_start/end      │ │  │
│  │  │  - Output    │  │  - Langfuse  │  │  - on_tool_*         │ │  │
│  │  │  - Tool      │  │  - LangSmith │  │  - on_llm_*          │ │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Context Management via Dependency Injection

```python
from dataclasses import dataclass
from agents import Agent, Runner, RunContextWrapper

# Define context type
@dataclass
class AppContext:
    user_id: str
    user_name: str
    preferences: dict
    session_data: dict = None

    def __post_init__(self):
        self.session_data = self.session_data or {}

# Context is NOT sent to LLM - purely local
# Must be injected into instructions to be visible

def dynamic_instructions(
    ctx: RunContextWrapper[AppContext],
    agent: Agent[AppContext]
) -> str:
    """Dynamic instructions that inject context."""
    return f"""You are helping {ctx.context.user_name}.

    User preferences:
    - Language: {ctx.context.preferences.get('language', 'English')}
    - Tone: {ctx.context.preferences.get('tone', 'professional')}

    Be helpful and concise."""

# Tool with context access
from agents import function_tool

@function_tool
def get_user_history(
    ctx: RunContextWrapper[AppContext],
    query: str
) -> str:
    """Search user's history."""
    user_id = ctx.context.user_id
    # Access context for user-specific data
    return f"History for {user_id}: ..."

# Agent with typed context
agent = Agent[AppContext](
    name="Assistant",
    instructions=dynamic_instructions,
    tools=[get_user_history]
)

# Run with context
context = AppContext(
    user_id="user_123",
    user_name="Alice",
    preferences={"language": "English", "tone": "friendly"}
)

result = await Runner.run(agent, "Hello!", context=context)
```

### 7.4 Session Management

```python
from agents import SQLiteSession, Agent, Runner

# SQLite session (persistent)
session = SQLiteSession(
    session_id="user_123",
    db_path="conversations.db"
)

# Run with automatic history management
result = await session.run(agent, "Hello, I'm Alice")
result = await session.run(agent, "What's my name?")
# Response: "Your name is Alice."

# Advanced session with features
from agents import AdvancedSQLiteSession

advanced_session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="conversations.db"
)

# Conversation branching
branch_id = await advanced_session.create_branch("experiment_1")

# Usage analytics
stats = await advanced_session.get_usage_stats()
print(f"Total tokens: {stats['total_tokens']}")

# Query history
history = await advanced_session.query_history(
    filter={"role": "assistant"},
    limit=10
)
```

### 7.5 Handoff System

```python
from agents import Agent, handoff, HandoffInputData
from agents.extensions import handoff_filters

# Define specialized agents
sales_agent = Agent(
    name="Sales",
    instructions="Handle sales inquiries. Be persuasive but honest."
)

support_agent = Agent(
    name="Support",
    instructions="Handle technical support. Be patient and thorough."
)

# Triage agent with handoffs
triage_agent = Agent(
    name="Triage",
    instructions="Route customers to the appropriate specialist.",
    handoffs=[
        handoff(
            agent=sales_agent,
            tool_description="Transfer to sales for pricing/purchasing"
        ),
        handoff(
            agent=support_agent,
            tool_description="Transfer to support for technical issues"
        )
    ]
)

# Custom input filter for handoffs
def filter_sensitive(data: HandoffInputData) -> HandoffInputData:
    """Remove sensitive info before handoff."""
    filtered_history = [
        msg for msg in data.input_history
        if "password" not in str(msg).lower()
    ]
    return HandoffInputData(
        input_history=filtered_history,
        pre_handoff_items=data.pre_handoff_items,
        new_items=data.new_items
    )

# Handoff with filtering
secure_handoff = handoff(
    agent=support_agent,
    input_filter=filter_sensitive,
    nest_handoff_history=True  # Summarize instead of full history
)

# Conditional handoff
def should_enable(ctx: RunContextWrapper) -> bool:
    return ctx.context.preferences.get("premium", False)

premium_handoff = handoff(
    agent=premium_agent,
    is_enabled=should_enable
)
```

### 7.6 Agent as Tool Pattern

```python
from agents import Agent

# Specialist agents
research_agent = Agent(
    name="Researcher",
    instructions="Conduct thorough research on given topics."
)

calculator_agent = Agent(
    name="Calculator",
    instructions="Perform mathematical calculations accurately."
)

# Manager uses specialists as tools (retains control)
manager_agent = Agent(
    name="Manager",
    instructions="Coordinate specialists to complete complex tasks.",
    tools=[
        research_agent.as_tool(
            tool_name="research",
            tool_description="Conduct research on a topic"
        ),
        calculator_agent.as_tool(
            tool_name="calculate",
            tool_description="Perform calculations"
        )
    ]
)

# Manager invokes specialists but retains control
result = await Runner.run(
    manager_agent,
    "Research AI trends and calculate market growth"
)
```

### 7.7 Guardrails System

```python
from agents import (
    Agent, InputGuardrail, OutputGuardrail,
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered
)

# Input guardrail
async def check_input_safety(ctx, agent, input) -> GuardrailFunctionOutput:
    """Check input for harmful content."""
    is_safe = not contains_harmful_content(input)
    return GuardrailFunctionOutput(
        output_info={"safe": is_safe},
        tripwire_triggered=not is_safe
    )

# Output guardrail
async def check_output_pii(ctx, agent, output) -> GuardrailFunctionOutput:
    """Check output for PII leakage."""
    has_pii = detect_pii(output)
    return GuardrailFunctionOutput(
        output_info={"has_pii": has_pii},
        tripwire_triggered=has_pii
    )

# Agent with guardrails
agent = Agent(
    name="SafeAgent",
    instructions="...",
    input_guardrails=[InputGuardrail(func=check_input_safety)],
    output_guardrails=[OutputGuardrail(func=check_output_pii)]
)

# Handle tripwire
try:
    result = await Runner.run(agent, user_input)
except InputGuardrailTripwireTriggered as e:
    print(f"Input blocked: {e.guardrail_result}")
```

### 7.8 Lifecycle Hooks

```python
from agents import Agent, AgentHooks

class MyHooks(AgentHooks):
    async def on_start(self, context, agent):
        print(f"[{agent.name}] Starting")
        context.context.session_data["start_time"] = time.time()

    async def on_end(self, context, agent, output):
        duration = time.time() - context.context.session_data["start_time"]
        print(f"[{agent.name}] Completed in {duration:.2f}s")

    async def on_handoff(self, context, agent, source):
        print(f"[{agent.name}] Received handoff from {source.name}")

    async def on_tool_start(self, context, agent, tool):
        print(f"[{agent.name}] Calling tool: {tool.name}")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"[{agent.name}] Tool {tool.name} returned")

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        print(f"[{agent.name}] LLM call starting")

    async def on_llm_end(self, context, agent, response):
        print(f"[{agent.name}] LLM responded")

agent = Agent(
    name="HookedAgent",
    instructions="...",
    hooks=MyHooks()
)
```

### 7.9 Streaming Support

```python
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent

agent = Agent(name="Streamer", instructions="Tell stories.")

async def stream_response():
    result = Runner.run_streamed(agent, "Tell me a story")

    async for event in result.stream_events():
        # Raw LLM tokens
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

        # High-level events
        elif event.type == "run_item_stream_event":
            if event.name == "message_output_created":
                print("\n[Message complete]")
            elif event.name == "tool_called":
                print(f"\n[Tool called]")
            elif event.name == "handoff_occurred":
                print(f"\n[Handoff]")

        # Agent changed
        elif event.type == "agent_updated_stream_event":
            print(f"\n[Now: {event.new_agent.name}]")

    return await result.get_final_output()
```

### 7.10 Tracing Integration

```python
from agents.tracing import add_trace_processor, custom_span

# Built-in tracing (enabled by default)
# Disable globally
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

# Or per-run
from agents import RunConfig
config = RunConfig(tracing_disabled=True)

# Custom spans
async with custom_span("my_operation") as span:
    span.set_attribute("user_id", "123")
    # ... your code ...

# Third-party integration (Langfuse)
from langfuse.openai_agents import LangfuseTraceProcessor
add_trace_processor(LangfuseTraceProcessor())
```

---

## 8. Comparative Analysis

### 8.1 Design Philosophy Comparison

| Aspect | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|--------|------------|------------|-----------|-------|---------------|
| **State Model** | Stateless (explicit) | Stateful (sessions) | Flexible (patterns) | Append-only (KV) | Stateless + Sessions |
| **Philosophy** | Simplicity, control | Hierarchical | Modularity | Cache-first | Python-first |
| **Primary Focus** | Token efficiency | Multi-agent | Ecosystem | Production cost | Ergonomics |

### 8.2 Feature Comparison Matrix

| Feature | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|---------|------------|------------|-----------|-------|---------------|
| **Built-in Sessions** | Agent SDK | Yes | Via integrations | File-based | Yes (SQLite+) |
| **State Scoping** | Manual | Prefix-based | Store selection | 3-tier | Context object |
| **Token Counting** | Native API | Via model | Utilities | Implicit | Via model |
| **Context Editing** | Native (beta) | Manual | trim_messages | Recoverable | Truncation param |
| **Memory Persistence** | File tool | Services | Multiple backends | File + Vector | Multiple backends |
| **Multi-Agent** | Subagents | Sub-agents/AgentTool | LangGraph | Wide Research | Handoffs/as_tool |
| **Cache Optimization** | Prompt caching | Manual | None | Primary goal | None |
| **Guardrails** | Manual | Callbacks | Callbacks | Manual | Built-in |
| **Tracing** | Manual | Manual | LangSmith | Manual | Built-in |

### 8.3 Architecture Pattern Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Context Flow Patterns                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Claude SDK (Stateless Explicit):                                           │
│  ┌────────┐                                                                  │
│  │ App    │──[messages]──▶ API ──[response]──▶ App ──[append]──▶ messages   │
│  └────────┘     ▲                                       │                    │
│                 └───────────────────────────────────────┘                    │
│                                                                              │
│  Google ADK (Hierarchical Sessions):                                         │
│  ┌────────┐    ┌─────────┐    ┌─────────┐                                   │
│  │ Runner │───▶│ Session │───▶│ Agent   │                                   │
│  └────────┘    │ Service │    │ Context │                                   │
│                └─────────┘    └─────────┘                                   │
│                     │              │                                         │
│                     └──[state delta]──▶ Persist                             │
│                                                                              │
│  LangChain (Pluggable Memory):                                              │
│  ┌────────┐    ┌──────────┐    ┌─────────┐                                  │
│  │ Chain  │───▶│ Memory   │───▶│ Backend │                                  │
│  └────────┘    │ Interface│    │ (Redis/ │                                  │
│                └──────────┘    │ Postgres)│                                  │
│                                └─────────┘                                   │
│                                                                              │
│  Manus (KV-Cache Optimized):                                                │
│  ┌────────┐    ┌───────────┐    ┌────────────┐                              │
│  │ Agent  │───▶│ Event     │───▶│ File       │                              │
│  └────────┘    │ Stream    │    │ System     │                              │
│                │ (append)  │    │ (extended) │                              │
│                └───────────┘    └────────────┘                              │
│                     │                                                        │
│                     └──[stable prefix]──▶ KV-Cache                          │
│                                                                              │
│  OpenAI Agents (Dependency Injection):                                      │
│  ┌────────┐    ┌───────────┐    ┌─────────┐                                 │
│  │ Runner │───▶│ Context   │───▶│ Session │                                 │
│  └────────┘    │ Wrapper   │    │ (SQLite)│                                 │
│                └───────────┘    └─────────┘                                 │
│                     │                                                        │
│                     └──[inject into instructions]──▶ LLM                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.4 Multi-Agent Pattern Comparison

| Pattern | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|---------|------------|------------|-----------|-------|---------------|
| **Orchestrator** | Manual | SequentialAgent | LangGraph | Planner/Executor | Handoffs |
| **Sub-agents** | Isolated | Shared/Isolated | Graph nodes | Isolated | as_tool |
| **Parallel Exec** | Manual async | ParallelAgent | LangGraph | Wide Research (100+) | Manual async |
| **Context Sharing** | Summary return | State output_key | Checkpointer | File references | Full history |
| **Handoff Filter** | N/A | include_contents | N/A | N/A | input_filter |

### 8.5 Persistence Backend Comparison

| Backend | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|---------|------------|------------|-----------|-------|---------------|
| In-Memory | Manual | InMemorySession | InMemoryChatHistory | Active context | Manual |
| SQLite | Manual | DatabaseSession | SqliteSaver | N/A | SQLiteSession |
| PostgreSQL | Manual | DatabaseSession | PostgresSaver | N/A | SQLAlchemySession |
| Redis | Manual | Custom | RedisChatHistory | N/A | RedisSession |
| MongoDB | Manual | Custom | MongoDBChatHistory | N/A | Custom |
| File System | Memory Tool | Custom | Custom | Primary | Custom |
| Vector DB | Custom | VertexAI RAG | VectorStore | Tier 3 | Custom |

### 8.6 Token Management Comparison

| Approach | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|----------|------------|------------|-----------|-------|---------------|
| **Counting** | Native API | Model-based | tiktoken/approx | Implicit | Model-based |
| **Truncation** | Context editing | Manual | trim_messages | Recoverable compress | truncation param |
| **Strategy** | Declarative rules | Manual logic | Multiple options | Append + compress | Auto/disabled |
| **Cache** | Prompt caching | None | None | KV-cache first | None |

### 8.7 Code Complexity Comparison

**Simple Chat with Memory:**

```python
# Claude SDK (~10 lines)
messages = []
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})
    response = client.messages.create(model="claude-sonnet-4-5", messages=messages)
    messages.append({"role": "assistant", "content": response.content[0].text})
    print(response.content[0].text)

# Google ADK (~15 lines)
runner = Runner(agent=agent, session_service=InMemorySessionService())
session = await runner.session_service.create_session("app", "user", "session")
while True:
    user_input = input("> ")
    async for event in runner.run_async("user", "session", Content(parts=[Part(text=user_input)])):
        if event.is_final_response():
            print(event.content.parts[0].text)

# LangChain (~12 lines)
chain_with_history = RunnableWithMessageHistory(chain, get_session_history, ...)
while True:
    user_input = input("> ")
    response = chain_with_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "user123"}}
    )
    print(response.content)

# Manus (~8 lines + file system)
context = []
while True:
    user_input = input("> ")
    context.append({"type": "message", "role": "user", "content": user_input})
    response = llm.invoke(context)
    context.append({"type": "message", "role": "assistant", "content": response})
    save_to_file("/workspace/context.json", context)

# OpenAI Agents (~8 lines)
session = SQLiteSession("user123", "chat.db")
while True:
    user_input = input("> ")
    result = await session.run(agent, user_input)
    print(result.final_output)
```

### 8.8 Performance Characteristics

| Metric | Claude SDK | Google ADK | LangChain | Manus | OpenAI Agents |
|--------|------------|------------|-----------|-------|---------------|
| **Latency Overhead** | Minimal | Medium | Medium | Minimal | Minimal |
| **Memory Efficiency** | High | Medium | Variable | High | Medium |
| **Scalability** | Manual | Good | Good | Excellent | Good |
| **Cost Optimization** | 90% (cache) | Manual | Manual | 10x (KV-cache) | Manual |

### 8.9 Use Case Recommendations

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Claude-specific apps | **Claude SDK** | Native features, context editing |
| Google Cloud deploy | **Google ADK** | Vertex AI integration |
| Multi-LLM apps | **LangChain** | Provider abstraction |
| Production at scale | **Manus patterns** | 10x cost reduction |
| Rapid prototyping | **OpenAI Agents** | Simple API, good defaults |
| Complex multi-agent | **Google ADK** | Rich orchestration patterns |
| Token-constrained | **Claude SDK / Manus** | Best optimization tools |

---

## 9. Discussion and Future Directions

### 9.1 Key Insights

1. **No One-Size-Fits-All**: Each framework optimizes for different priorities
2. **KV-Cache Awareness**: Emerging as critical for production systems
3. **File System as Memory**: Manus demonstrates unlimited context via files
4. **Session Abstraction**: Built-in sessions becoming standard (OpenAI, Google)
5. **Handoff Patterns**: Multi-agent coordination converging on similar patterns

### 9.2 Emerging Trends

1. **Automatic Context Management**: Declarative rules replacing manual logic
2. **Hybrid Memory**: Combining short-term context with long-term retrieval
3. **Cost-Aware Design**: Cache optimization becoming first-class concern
4. **Standardization**: MCP (Model Context Protocol) for interoperability

### 9.3 Open Challenges

1. **Cross-Framework Portability**: No standard context format
2. **Long-Context Utilization**: Efficiently using 1M+ token windows
3. **Memory Coherence**: Maintaining consistency in distributed agents
4. **Evaluation Metrics**: Standardizing context management quality

### 9.4 Future Research Directions

1. **Learned Context Compression**: Neural summarization for context
2. **Adaptive Caching**: Dynamic cache strategies based on usage
3. **Federated Memory**: Privacy-preserving cross-session learning
4. **Context-Aware Routing**: Intelligent agent selection based on context

---

## 10. Conclusion

This survey has presented a comprehensive analysis of context management across five leading AI agent frameworks. Our key findings include:

1. **Architectural Diversity**: Frameworks range from stateless explicit (Claude SDK) to hierarchical sessions (Google ADK) to cache-optimized append-only (Manus)

2. **Trade-offs**: Each approach balances control, complexity, and performance differently:
   - Claude SDK: Maximum control, manual management
   - Google ADK: Structured hierarchy, Google ecosystem
   - LangChain: Maximum flexibility, integration breadth
   - Manus: Production efficiency, cache optimization
   - OpenAI Agents: Developer ergonomics, good defaults

3. **Convergent Patterns**: Despite different approaches, common patterns emerge:
   - Session-based persistence
   - Handoff mechanisms for multi-agent
   - Declarative context management
   - Built-in observability

4. **Manus Insight**: "Context engineering is not about adding more context—it is about finding the minimal effective context required for the next step."

The choice of framework should be guided by specific requirements: deployment environment, scale, cost constraints, and team expertise. All frameworks continue to evolve rapidly, with context management remaining a key area of innovation in AI agent development.

---

## 11. References

### Framework Documentation

1. Anthropic. "Claude SDK Documentation." https://docs.anthropic.com/
2. Anthropic. "Context Editing." https://docs.claude.com/en/docs/build-with-claude/context-editing
3. Anthropic. "Memory Tool." https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
4. Google. "Agent Development Kit Documentation." https://google.github.io/adk-docs/
5. LangChain. "LangChain Documentation." https://docs.langchain.com/
6. LangChain. "LangGraph Persistence." https://docs.langchain.com/oss/python/langgraph/persistence
7. OpenAI. "Agents SDK Documentation." https://openai.github.io/openai-agents-python/

### GitHub Repositories

8. Anthropic. "anthropic-sdk-python." https://github.com/anthropics/anthropic-sdk-python
9. Google. "adk-python." https://github.com/google/adk-python
10. LangChain. "langchain." https://github.com/langchain-ai/langchain
11. LangChain. "langgraph." https://github.com/langchain-ai/langgraph
12. OpenAI. "openai-agents-python." https://github.com/openai/openai-agents-python
13. FoundationAgents. "OpenManus." https://github.com/FoundationAgents/OpenManus

### Technical Articles

14. Manus Team. "Context Engineering for AI Agents: Lessons from Building Manus." https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus
15. Ren, S. "In-depth Technical Investigation into Manus AI." https://gist.github.com/renschni/4fbc70b31bad8dd57f3370239dccd58f
16. Wang, X., et al. "CodeAct: Executable Code Actions Elicit Better LLM Agents." arXiv:2402.01030, 2024.

### Related Work

17. Shinn, N., et al. "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
18. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
19. Wu, Q., et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv:2308.08155, 2023.

---

*End of Survey*
