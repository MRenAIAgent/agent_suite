# Context Management in AI Agent SDKs: A Comprehensive Research Report

**Research Date:** January 2026
**Scope:** Claude SDK (Anthropic), Google ADK, LangChain, and Manus

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Claude SDK Context Management](#2-claude-sdk-context-management)
3. [Google ADK Context Management](#3-google-adk-context-management)
4. [LangChain Context Management](#4-langchain-context-management)
5. [Manus Context Management](#5-manus-context-management)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Recommendations](#7-recommendations)
8. [Conclusion](#8-conclusion)
9. [Appendices](#appendices)

---

## 1. Executive Summary

Context management is fundamental to building effective AI agents and conversational applications. This report analyzes four leading frameworks for managing conversation context, memory, and state:

| Framework | Philosophy | Key Strength |
|-----------|------------|--------------|
| **Claude SDK** | Stateless API with explicit context passing | Token-aware context editing, memory tools |
| **Google ADK** | Hierarchical context with built-in session management | Rich state scoping, multi-agent patterns |
| **LangChain** | Pluggable memory abstractions with extensive integrations | Flexibility, large ecosystem |
| **Manus** | KV-cache optimized with file system as extended memory | Production efficiency, context engineering |

---

## 2. Claude SDK Context Management

### 2.1 Architecture Overview

The Anthropic Claude SDK follows a **stateless API architecture**. This means the API does not maintain conversation state server-side—developers must explicitly pass the complete message history with each request.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ ClaudeSDKClient │  │  Tool Runner    │  │  Message Stream │  │
│  │ (sessions)      │  │  (agent loop)   │  │  (accumulation) │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           └────────────────────┼────────────────────┘            │
│                                ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Anthropic Client (sync/async)                │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Context Management                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Token       │  │ Context     │  │ Memory Tool             │  │
│  │ Counting    │  │ Editing     │  │ (file-based persistence)│  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Concepts

#### Stateless Message Passing

```python
from anthropic import Anthropic

client = Anthropic()

# Multi-turn conversation - developer manages the history
messages = [
    {"role": "user", "content": "Hello, Claude"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Can you describe LLMs to me?"}
]

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    messages=messages
)
```

**Key Constraints:**
- Messages must strictly alternate between `user` and `assistant` roles
- Conversations always begin with a `user` message
- Assistant message prefilling is supported for response guidance

#### Claude Agent SDK (Higher-Level)

The Agent SDK provides session management abstractions:

```python
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock

async def multi_turn_conversation():
    async with ClaudeSDKClient() as client:
        # First turn
        await client.query("What's the capital of France?")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # Second turn - context is maintained
        await client.query("What's the population of that city?")
        async for message in client.receive_response():
            print_response(message)
```

### 2.3 Content Block Types

Claude supports multiple content block types for complex multi-modal interactions:

| Block Type | Direction | Description |
|------------|-----------|-------------|
| `text` | Input/Output | Plain text content |
| `image` | Input | Base64 or URL image data |
| `tool_use` | Output | Tool invocation by Claude |
| `tool_result` | Input | Results returned to Claude |
| `thinking` | Output | Extended thinking blocks (Claude 4+) |
| `document` | Input | PDF documents (base64) |

#### Tool Use Content Block
```json
{
    "type": "tool_use",
    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "name": "get_stock_price",
    "input": {"ticker": "^GSPC"}
}
```

#### Tool Result Content Block
```json
{
    "type": "tool_result",
    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "content": "259.75 USD",
    "is_error": false
}
```

### 2.4 Key Features

#### Token Counting API

Free API for pre-calculating token usage:

```python
# Count tokens before sending
count = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    system="You are a scientist",
    messages=[{"role": "user", "content": "Hello, Claude"}],
    tools=[...]  # Tools also counted
)
print(f"Input tokens: {count.input_tokens}")
```

#### Context Editing (Beta)

Automatic context management to stay within token limits:

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    betas=["context-management-2025-06-27"],
    messages=[...],
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 30000},
                "keep": {"type": "tool_uses", "value": 3},
                "clear_at_least": {"type": "input_tokens", "value": 5000},
                "exclude_tools": ["web_search"]
            }
        ]
    }
)
```

**Available Strategies:**
- `clear_tool_uses_20250919` - Clears old tool results chronologically
- `clear_thinking_20251015` - Clears extended thinking blocks

**Processing Logic:**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Edit Processing                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Calculate current input_tokens                              │
│  2. FOR each edit rule:                                         │
│     a. Check if trigger.value < current_tokens                  │
│     b. If triggered:                                            │
│        - Identify clearable content (chronologically oldest)    │
│        - Exclude tools in exclude_tools list                    │
│        - Calculate tokens to clear (>= clear_at_least.value)    │
│        - Preserve keep.value most recent items                  │
│        - Replace cleared content with placeholder text          │
│  3. Invalidate any cached prompt prefixes                       │
│  4. Send modified context to model                              │
└─────────────────────────────────────────────────────────────────┘
```

#### Memory Tool (Beta)

Persistent file-based memory system:

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=2048,
    betas=["context-management-2025-06-27"],
    messages=[{"role": "user", "content": "Remember that I prefer Python"}],
    tools=[{"type": "memory_20250818", "name": "memory"}]
)
```

**Memory Operations:**
| Operation | Purpose |
|-----------|---------|
| `view` | Read directory contents or file sections |
| `create` | Create new memory files |
| `str_replace` | Update existing memory content |
| `insert` | Add text at specific line numbers |
| `delete` | Remove files or directories |
| `rename` | Reorganize memory structure |

Custom implementation via `BetaAbstractMemoryTool`:

```python
from anthropic.types.beta import BetaAbstractMemoryTool
from pathlib import Path

class LocalFilesystemMemoryTool(BetaAbstractMemoryTool):
    def __init__(self, base_path: str = "./memory"):
        super().__init__()
        self.memory_root = Path(base_path) / "memories"
        self.memory_root.mkdir(parents=True, exist_ok=True)

    def view(self, path: str, view_range: list[int] | None = None) -> dict:
        full_path = self.memory_root / path.lstrip("/")
        if full_path.is_dir():
            entries = [{"name": item.name, "type": "directory" if item.is_dir() else "file"}
                      for item in full_path.iterdir()]
            return {"type": "directory", "path": path, "entries": entries}
        elif full_path.is_file():
            lines = full_path.read_text().splitlines()
            if view_range:
                lines = lines[view_range[0]-1:view_range[1]]
            return {"type": "file", "path": path, "content": "\n".join(lines)}

    # Implement: create(), str_replace(), insert(), delete(), rename()
```

### 2.5 Streaming Implementation

Claude uses Server-Sent Events (SSE) for streaming:

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_stop
data: {"type":"message_stop"}
```

**Delta Accumulation Algorithm:**
```python
def accumulate_delta(acc: dict, delta: dict) -> dict:
    """Recursively merge delta into accumulated state."""
    for key, delta_value in delta.items():
        if key not in acc:
            acc[key] = delta_value
        elif key in ("index", "type"):
            acc[key] = delta_value  # Replace, don't accumulate
        elif isinstance(acc[key], str) and isinstance(delta_value, str):
            acc[key] += delta_value  # String concatenation
        elif isinstance(acc[key], dict) and isinstance(delta_value, dict):
            acc[key] = accumulate_delta(acc[key], delta_value)  # Recursive
    return acc
```

### 2.6 Tool Use Protocol

**Tool Schema Format (JSON Schema):**
```python
tool_definition = {
    "name": "get_weather",  # a-z, A-Z, 0-9, _, - (max 64 chars)
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
}
```

**Tool Choice Parameters:**
| Tool Choice | Behavior |
|------------|----------|
| `{"type": "auto"}` | Claude decides whether to use tools (default) |
| `{"type": "any"}` | Must use one of the provided tools |
| `{"type": "tool", "name": "X"}` | Must use specific tool X |
| `{"type": "none"}` | Cannot use any tools |

**Parallel Tool Calls:**
```python
# Disable parallel tool use
response = client.messages.create(
    tools=tools,
    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    messages=messages
)
```

### 2.7 Extended Thinking

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=messages
)
```

| Budget Range | Recommendation |
|--------------|----------------|
| 1,024 - 4,096 | Simple reasoning tasks |
| 4,096 - 16,384 | Moderate complexity |
| 16,384 - 32,768 | Complex analysis |
| > 32,768 | Use batch processing |

**Streaming with Thinking (required when max_tokens > 21,333):**
```python
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=25000,
    thinking={"type": "enabled", "budget_tokens": 15000},
    messages=messages
) as stream:
    for event in stream:
        if event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="")
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="")
```

### 2.8 Multi-modal Context

#### Image Token Cost
| Image Size | Approximate Tokens |
|------------|-------------------|
| Up to 1568px (long edge) | ~1,600 tokens |
| Larger images | Auto-scaled down |

#### PDF Constraints
| Constraint | Limit |
|------------|-------|
| Max file size | 32 MB |
| Max pages | 100 per request |
| Token cost | 1,500 - 3,000 per page |

### 2.9 Subagents for Context Isolation

```
┌─────────────────────────────────────────────────┐
│              Main Orchestrator                   │
│         (Claude Opus 4 - global plan)           │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Subagent │ │ Subagent │ │ Subagent │
   │ (tests)  │ │ (review) │ │ (docs)   │
   └──────────┘ └──────────┘ └──────────┘
   (isolated)   (isolated)   (isolated)
```

### 2.10 Context Window Specifications

| Model | Standard Context | Extended Context |
|-------|------------------|------------------|
| Claude Sonnet 4 | 200K tokens | 1M tokens |
| Claude Sonnet 4.5 | 200K tokens | 1M tokens |
| Claude Opus 4.5 | 200K tokens | - |

### 2.11 Performance Impact

| Feature Combination | Improvement |
|---------------------|-------------|
| Context editing alone | 29% |
| Memory tool + context editing | 39% |
| Token reduction (100-turn workflows) | 84% |

---

## 3. Google ADK Context Management

### 3.1 Architecture Overview

Google's Agent Development Kit (ADK) provides a comprehensive, **hierarchical context system** with built-in session management, state scoping, and multi-agent coordination patterns.

```
┌──────────────────────────────────────────────────────────────┐
│                        Runner                                 │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  InvocationContext (Full Access)                        │  │
│  │    - session, state, events, services                   │  │
│  │    ┌────────────────────────────────────────────────┐   │  │
│  │    │  CallbackContext (Callbacks)                    │   │  │
│  │    │    - read/write state, agent control            │   │  │
│  │    │    ┌────────────────────────────────────────┐   │   │  │
│  │    │    │  ToolContext (Tools)                   │   │   │  │
│  │    │    │    - state access, agent transfer      │   │   │  │
│  │    │    │    ┌──────────────────────────────┐    │   │   │  │
│  │    │    │    │  ReadonlyContext (Instrs)    │    │   │   │  │
│  │    │    │    │    - read-only state view    │    │   │   │  │
│  │    │    │    └──────────────────────────────┘    │   │   │  │
│  │    │    └────────────────────────────────────────┘   │   │  │
│  │    └────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Context Type Hierarchy

| Context Type | Usage Location | Access Level |
|--------------|----------------|--------------|
| `InvocationContext` | Agent's core methods | Full access |
| `CallbackContext` | Agent/model callbacks | Read/write state |
| `ToolContext` | Tool functions | State access, agent transfer |
| `ReadonlyContext` | Instruction providers | Read-only |

#### InvocationContext Example

```python
from google.adk.agents import BaseAgent, InvocationContext
from google.adk.events import Event
from typing import AsyncGenerator

class MyAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        # Full context access
        agent_name = ctx.agent.name
        session_id = ctx.session.id
        user_content = ctx.user_content

        # Service access
        artifact_service = ctx.artifact_service
        memory_service = ctx.memory_service
        session_service = ctx.session_service

        if ctx.session.state.get("critical_error_flag"):
            ctx.end_invocation = True
            yield Event(author=self.name, content="Stopping due to error.")
```

#### ToolContext Example

```python
from google.adk.tools.tool_context import ToolContext

def remember_favorite_city(city: str, tool_context: ToolContext) -> dict:
    """Tool with state access and agent transfer capability."""
    tool_context.state["user:favorite_city"] = city
    tool_context.state["temp:last_operation"] = "city_saved"

    if tool_context.state.get("needs_escalation"):
        tool_context.actions.transfer_to_agent = "human_support_agent"

    return {"status": "success", "city": city}
```

### 3.3 State Management

#### State Prefixes for Scoping

| Prefix | Scope | Persistence |
|--------|-------|-------------|
| (none) | Current session only | Session lifetime |
| `user:` | Across all user's sessions | Persistent per user |
| `app:` | Across all users | Persistent globally |
| `temp:` | Current invocation only | Not persisted |

```python
# Session-scoped (default)
session.state["last_message"] = "hello"

# User-scoped (persists across sessions)
session.state["user:preferred_language"] = "Spanish"
session.state["user:name"] = "Sarah"

# App-scoped (shared across all users)
session.state["app:version"] = "2.1.0"

# Temporary (current invocation only)
session.state["temp:api_response"] = response_data
```

#### Template Injection in Instructions

```python
agent = LlmAgent(
    name="PersonalizedAgent",
    model="gemini-2.0-flash",
    instruction="""You are helping {user:name}.
    Their preferred language is {user:preferred_language}.
    Current topic: {conversation_topic?}"""  # ? = optional
)
```

#### Automatic State Storage with output_key

```python
greeting_agent = LlmAgent(
    name="Greeter",
    model="gemini-2.0-flash",
    instruction="Generate a short, friendly greeting.",
    output_key="last_greeting"  # Auto-saved to state["last_greeting"]
)
```

### 3.4 State Delta System

State modifications in ADK are **event-driven**, not direct mutations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     State Delta Flow                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Code modifies context.state:                                        │
│     callback_context.state["key"] = "value"                             │
│                                                                          │
│  2. Framework intercepts modification:                                   │
│     - Change is NOT immediately applied to session.state                │
│     - Change is recorded in pending_delta dictionary                    │
│                                                                          │
│  3. When event is yielded:                                              │
│     event.actions.state_delta = {"key": "value"}                        │
│                                                                          │
│  4. SessionService.append_event(event):                                 │
│     - Merges state_delta into session.state atomically                  │
│     - Persists event with delta to storage                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.5 Session Services

| Service | Persistence | Best For |
|---------|-------------|----------|
| `InMemorySessionService` | None | Development, testing |
| `DatabaseSessionService` | PostgreSQL, MySQL, SQLite | Production with custom DB |
| `VertexAiSessionService` | Vertex AI Agent Engine | Google Cloud deployments |

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService

# Development
session_service = InMemorySessionService()

# Production
db_url = "postgresql+asyncpg://user:pass@host/db"
session_service = DatabaseSessionService(db_url=db_url)

# Create session with initial state
session = await session_service.create_session(
    app_name="my_app",
    user_id="user_123",
    session_id="session_456",
    state={"user:name": "Alice"}
)
```

### 3.6 Memory Systems (Long-Term)

| Concept | Analogy | Scope |
|---------|---------|-------|
| Session State | Short-term memory | Single conversation |
| MemoryService | Long-term archive | Across all sessions |

```python
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory

memory_service = InMemoryMemoryService()

agent = LlmAgent(
    name="MemoryAgent",
    model="gemini-2.0-flash",
    instruction="You can recall past conversations using load_memory.",
    tools=[load_memory]
)

runner = Runner(
    agent=agent,
    app_name="memory_app",
    session_service=session_service,
    memory_service=memory_service
)

# After session ends, archive to long-term memory
await memory_service.add_session_to_memory(completed_session)
```

### 3.7 Multi-Agent Context Patterns

#### Sub-Agents (Shared Context)

```python
from google.adk.agents import LlmAgent, SequentialAgent

researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash",
    instruction="Research the topic.",
    output_key="research_findings"
)

writer = LlmAgent(
    name="Writer",
    model="gemini-2.0-flash",
    instruction="Write based on: {research_findings}"
)

pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, writer]
)
```

#### AgentTool (Isolated Context)

```python
from google.adk.tools import AgentTool

calculator_agent = LlmAgent(
    name="Calculator",
    model="gemini-2.0-flash",
    instruction="Perform calculations."
)

calculator_tool = AgentTool(agent=calculator_agent)

main_agent = LlmAgent(
    name="MainAgent",
    tools=[calculator_tool]
)
```

### 3.8 Events System

```python
from google.adk.events import Event, EventActions

event = Event(
    id="unique_event_id",
    invocation_id="inv_123",
    author="agent_name",
    content=Content(...),
    actions=EventActions(
        state_delta={"key": "value"},
        transfer_to_agent="other_agent",
        escalate=False
    ),
    timestamp=1234567890.0
)
```

### 3.9 Agent Transfer Protocol

```python
# Transfer via tool
def escalate_to_human(reason: str, tool_context: ToolContext) -> dict:
    tool_context.actions.transfer_to_agent = "human_support_agent"
    return {"status": "transferring", "reason": reason}

# Transfer via agent logic
class TriageAgent(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext):
        category = await self._categorize(ctx.user_content)
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=Content(parts=[Part(text=f"Routing to {category}")]),
            actions=EventActions(transfer_to_agent=f"{category}_specialist")
        )
```

### 3.10 Callbacks System

```python
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from typing import Optional

def before_model_callback(ctx: CallbackContext, request: LlmRequest) -> Optional[LlmResponse]:
    """Called before LLM API call - implement guardrails, caching."""
    if contains_forbidden_content(request):
        return LlmResponse(content=Content(parts=[Part(text="Cannot process.")]))
    return None

def after_model_callback(ctx: CallbackContext, response: LlmResponse) -> Optional[LlmResponse]:
    """Called after LLM response - implement logging, modification."""
    print(f"LLM responded with {len(response.content.parts)} parts")
    return None

agent = LlmAgent(
    name="CallbackDemo",
    model="gemini-2.0-flash",
    before_model_callback=before_model_callback,
    after_model_callback=after_model_callback
)
```

---

## 4. LangChain Context Management

### 4.1 Architecture Overview

LangChain provides **pluggable memory abstractions** with extensive integration options. The framework has evolved significantly, with current recommendations favoring LangGraph for production:

```
┌──────────────────────────────────────────────────────────────────┐
│                    LangChain Memory Architecture                  │
├──────────────────────────────────────────────────────────────────┤
│  Modern (Recommended)          │  Legacy (Deprecated v0.3.1+)    │
│  ┌─────────────────────────┐   │  ┌─────────────────────────┐    │
│  │ LangGraph Persistence   │   │  │ ConversationBufferMemory│    │
│  │  - Checkpointers        │   │  │ ConversationSummaryMem  │    │
│  │  - Store Interface      │   │  │ VectorStoreRetrieverMem │    │
│  └─────────────────────────┘   │  │ ConversationEntityMem   │    │
│  ┌─────────────────────────┐   │  └─────────────────────────┘    │
│  │ RunnableWithMsgHistory  │   │                                  │
│  │  - LCEL Integration     │   │                                  │
│  └─────────────────────────┘   │                                  │
├──────────────────────────────────────────────────────────────────┤
│  Storage Backends: Redis, PostgreSQL, MongoDB, SQLite, FAISS     │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Message Type System

LangChain's message system is built on a class hierarchy:

```
BaseMessage (abstract)
├── HumanMessage      (type: "human")
├── AIMessage         (type: "ai")
├── SystemMessage     (type: "system")
├── ToolMessage       (type: "tool")
├── FunctionMessage   (type: "function") [deprecated]
└── ChatMessage       (type: "chat", dynamic role)
```

#### BaseMessage Core Structure
```python
class BaseMessage(Serializable):
    content: Union[str, List[Union[str, Dict]]]
    additional_kwargs: dict = Field(default_factory=dict)
    response_metadata: dict = Field(default_factory=dict)
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
```

#### AIMessage with Tool Calls
```python
AIMessage(
    content='',
    tool_calls=[{
        'name': 'add',
        'args': {'x': 10, 'y': 10},
        'id': 'call_abc123',
        'type': 'tool_call'
    }],
    response_metadata={
        'model': 'gpt-4',
        'finish_reason': 'tool_calls',
        'usage': {'prompt_tokens': 50, 'completion_tokens': 20}
    }
)
```

#### ToolMessage
```python
class ToolMessage(BaseMessage):
    type: Literal["tool"] = "tool"
    tool_call_id: str  # Essential for parallel tool calls
    artifact: Optional[Any] = None
    status: Literal["success", "error"] = "success"
```

### 4.3 Memory Types

#### ConversationBufferMemory
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=ChatOpenAI(model="gpt-4"), memory=memory)

conversation.predict(input="Hi, I'm Alice")
conversation.predict(input="What's my name?")  # Remembers "Alice"
```

#### ConversationBufferWindowMemory
```python
from langchain.memory import ConversationBufferWindowMemory
memory = ConversationBufferWindowMemory(k=3)  # Last 3 exchanges
```

#### ConversationSummaryMemory
```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context(
    {"input": "Hi, I'm working on a project about AI"},
    {"output": "That sounds interesting! Tell me more."}
)
memory.load_memory_variables({})  # Returns summarized history
```

#### VectorStoreRetrieverMemory
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts([""], embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

memory = VectorStoreRetrieverMemory(retriever=retriever)
memory.save_context(
    {"input": "My favorite food is pizza"},
    {"output": "That's great!"}
)
memory.load_memory_variables({"input": "What food do I like?"})
```

### 4.4 Memory Selection Guide

| Memory Type | Best For | Token Usage | Context Retention |
|-------------|----------|-------------|-------------------|
| Buffer | Short conversations | High (grows) | Complete |
| BufferWindow | Recent context only | Fixed | Last K messages |
| Summary | Long conversations | Low (constant) | Summarized |
| SummaryBuffer | Balanced needs | Medium | Recent + Summary |
| VectorStoreRetriever | Semantic recall | Variable | Query-based |
| Entity | Entity tracking | Medium | Entity-focused |

### 4.5 Modern Approach: RunnableWithMessageHistory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | ChatOpenAI(model="gpt-4")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

response = chain_with_history.invoke(
    {"input": "Hi, I'm Bob"},
    config={"configurable": {"session_id": "user123"}}
)
```

### 4.6 Token Management

```python
from langchain_core.messages.utils import trim_messages, count_tokens_approximately

trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",
    token_counter=count_tokens_approximately,
    start_on="human",
    include_system=True,
    allow_partial=False
)
```

### 4.7 LangGraph State Management

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages, StateGraph

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str

graph = StateGraph(State)
```

**Reducer Functions:**
```python
def add_messages(left: list, right: list) -> list:
    """Append messages with ID-based deduplication."""
    left_by_id = {m.id: m for m in left if m.id}
    result = list(left)
    for msg in right:
        if msg.id and msg.id in left_by_id:
            idx = next(i for i, m in enumerate(result) if m.id == msg.id)
            result[idx] = msg
        else:
            result.append(msg)
    return result
```

### 4.8 LangGraph Persistence (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/db"
)

store = InMemoryStore()

graph = workflow.compile(checkpointer=checkpointer, store=store)

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "conversation_123"}}
)
```

### 4.9 Persistence Options

| Backend | Implementation | Best For |
|---------|----------------|----------|
| Redis | `RedisChatMessageHistory` | High-performance, TTL support |
| PostgreSQL | `PostgresChatMessageHistory` | Relational, ACID compliance |
| MongoDB | `MongoDBChatMessageHistory` | Document storage, flexibility |
| SQLite | `SqliteSaver` | Local development |

### 4.10 Callback System

```python
from langchain_core.callbacks import BaseCallbackHandler

class MemoryTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        self.token_counts = []

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, 'generation_info'):
                    usage = gen.generation_info.get('token_usage', {})
                    self.token_counts.append({
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0)
                    })
```

---

## 5. Manus Context Management

### 5.1 Architecture Overview

Manus AI is an autonomous agent system developed by Monica.im (launched March 2025) that represents a significant advancement in **context engineering**. The team has rebuilt their agent framework four times, with each iteration focused on context optimization.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Manus Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Planner    │    │   Executor   │    │   Verifier   │      │
│  │    Agent     │───▶│    Agent     │───▶│    Agent     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         └───────────────────┼───────────────────┘               │
│                             ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Event Stream                          │   │
│  │   (Messages, Actions, Observations, Plans, Knowledge)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐              │
│         ▼                   ▼                   ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   Browser    │    │    Shell     │    │    File      │     │
│  │  (Playwright)│    │   Execute    │    │   System     │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│                             │                                   │
│                    ┌────────▼────────┐                         │
│                    │  Ubuntu Sandbox  │                         │
│                    │    (Docker)      │                         │
│                    └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Core Design Philosophy: CodeAct Paradigm

Manus operates on a **"CodeAct" paradigm** (from arXiv:2402.01030) where executable Python code serves as the universal action format rather than rigid JSON function calls:

```python
# Traditional: JSON function calling
{"function": "search", "parameters": {"query": "AI news"}}

# CodeAct: Executable Python
search_results = web_search("AI news")
for result in search_results[:5]:
    print(f"- {result.title}: {result.url}")
```

### 5.3 KV-Cache Optimization

Manus considers **KV-cache hit rate** as "the single most important metric for a production-stage AI agent":

- **Input-to-output token ratio**: ~100:1
- **Cost difference**: Cached tokens ($0.30/MTok) vs uncached ($3.00/MTok) = **10x savings**

**Three Core Principles:**

```python
# Principle 1: Stable Prefixes
# BAD - timestamp invalidates cache
system_prompt = f"Current time: {datetime.now()}\n{instructions}"

# GOOD - static prefix
system_prompt = f"{instructions}"  # Timestamp passed separately

# Principle 2: Append-Only Context
# NEVER modify previous actions/observations
context.append(new_observation)  # Always append, never mutate

# Principle 3: Explicit Cache Breakpoints
cache_breakpoints = [
    system_prompt_end,
    tool_definitions_end,
    conversation_history_start
]
```

### 5.4 Tool Management via Logit Masking

Instead of dynamically removing tools (which breaks KV-cache), Manus uses **logit masking**:

```python
class ToolManager:
    def __init__(self, all_tools):
        self.all_tools = all_tools  # Keep stable for KV-cache
        self.tool_prefixes = {
            'browser': ['browser_navigate', 'browser_click', 'browser_input'],
            'shell': ['shell_exec', 'shell_view', 'shell_wait'],
            'file': ['file_read', 'file_write', 'file_str_replace']
        }

    def get_logit_mask(self, allowed_groups):
        """Returns mask that prevents selection of disallowed tools
        during decoding, without modifying the tool definitions."""
        mask = {}
        for group, tools in self.tool_prefixes.items():
            if group not in allowed_groups:
                for tool in tools:
                    mask[tool] = float('-inf')
        return mask
```

### 5.5 File System as Extended Context

Manus treats the **file system as unlimited, persistent memory**:

```python
class FileSystemMemory:
    def __init__(self, workspace="/home/ubuntu"):
        self.workspace = workspace
        self.todo_file = f"{workspace}/todo.md"
        self.notes_dir = f"{workspace}/notes"

    def save_intermediate_result(self, filename, content):
        """Save large content to file, keep only path in context"""
        path = f"{self.notes_dir}/{filename}"
        with open(path, 'w') as f:
            f.write(content)
        return path  # Return path for context, not full content

    def update_todo(self, tasks):
        """Constantly rewrite todo.md to maintain focus"""
        with open(self.todo_file, 'w') as f:
            for i, task in enumerate(tasks):
                status = "x" if task['done'] else " "
                f.write(f"- [{status}] {task['description']}\n")
```

**The `todo.md` Pattern:**
```markdown
# Task: Research Manus AI architecture

- [x] Search for official documentation
- [x] Analyze system prompts
- [ ] Document context management strategies  <-- Current
- [ ] Compile technical examples
- [ ] Write final report
```

By constantly rewriting `todo.md`, Manus **recites its objectives into the end of context**, pushing the global plan into the model's recent attention span.

### 5.6 Recoverable Compression Strategy

```python
class ContextCompressor:
    def __init__(self, max_tokens=128000):
        self.max_tokens = max_tokens

    def compress(self, context):
        """
        Compression strategies (in order of preference):
        1. Raw (no compression) - best quality
        2. Compaction (drop recoverable content)
        3. Summarization (only when necessary)
        """
        if self.count_tokens(context) < self.max_tokens:
            return context

        # Phase 1: Compaction - drop recoverable content
        for item in context:
            if item.type == 'webpage':
                item.content = f"[Content available at: {item.url}]"
            elif item.type == 'file':
                item.content = f"[File contents at: {item.path}]"

        if self.count_tokens(context) < self.max_tokens:
            return context

        # Phase 2: Summarization - preserve recent turns raw
        recent_turns = context[-3:]
        older_turns = context[:-3]
        summary = self.llm_summarize(older_turns)
        return [{'type': 'summary', 'content': summary}] + recent_turns
```

### 5.7 Three-Tier Memory Architecture

```python
class ManusMemory:
    def __init__(self):
        # Tier 1: Active Context (in LLM context window)
        self.active_context = []

        # Tier 2: Session Memory (file system)
        self.session_files = {}

        # Tier 3: Long-term Memory (persistent storage)
        self.ltm_store = VectorDatabase()

    def retrieve_relevant(self, query, k=5):
        """Retrieval-augmented generation for relevant history"""
        return self.ltm_store.similarity_search(query, k=k)

    def save_to_ltm(self, interaction):
        """Store important interactions for future sessions"""
        embedding = self.embed(interaction)
        self.ltm_store.insert(interaction, embedding)
```

### 5.8 Event Stream Architecture

```python
class EventStream:
    """Chronological log of all interactions (backbone of context)"""

    EVENT_TYPES = ['message', 'action', 'observation', 'plan', 'knowledge', 'datasource']

    def __init__(self):
        self.events = []

    def add_event(self, event_type, content, metadata=None):
        event = {
            'id': uuid4(),
            'type': event_type,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.events.append(event)  # Append-only for KV-cache
        return event

    def get_context_window(self, max_tokens):
        """Build context prioritizing recent events."""
        recent = self.events[-10:]
        older = self.events[:-10]

        if self.count_tokens(older) > max_tokens * 0.3:
            older = self.summarize_events(older)

        return older + recent
```

### 5.9 Error Preservation Pattern

```python
class ErrorPreservation:
    """Leave wrong turns in context for implicit learning"""

    def handle_tool_failure(self, action, error):
        """DON'T remove failed action; DO keep visible for model to learn."""
        observation = {
            'type': 'observation',
            'action': action,
            'success': False,
            'error': str(error),
            'stacktrace': traceback.format_exc()
        }
        self.event_stream.add_event('observation', observation)
        return observation
```

### 5.10 Multi-Agent Patterns

#### Wide Research (100+ Parallel Agents)

```python
class WideResearch:
    """Execute large-scale tasks with 100+ parallel agents"""

    async def execute_wide_research(self, query, num_agents=100):
        research_tracks = self.decompose_query(query, num_agents)

        agents = [
            ResearchAgent(
                context=self.create_isolated_context(track),
                sandbox=self.create_sandbox()
            )
            for track in research_tracks
        ]

        results = await asyncio.gather(*[agent.research() for agent in agents])
        return self.synthesize_with_consensus(results)
```

#### Context Isolation (Go-Lang Pattern)

```python
class AgentCommunication:
    """Share memory by communicating, don't communicate by sharing memory"""

    def __init__(self):
        self.message_queues = defaultdict(asyncio.Queue)

    async def send_to_agent(self, target_agent, message):
        structured_message = {
            'from': self.agent_id,
            'to': target_agent,
            'type': message.type,
            'summary': message.summary,  # Summarized, not raw
            'data_path': message.data_path  # Reference, not content
        }
        await self.message_queues[target_agent].put(structured_message)
```

### 5.11 GAIA Benchmark Performance

| Level | Manus | OpenAI Operator | Previous SOTA |
|-------|-------|-----------------|---------------|
| Level 1 | **86.5%** | 74.3% | 67.9% |
| Level 2 | **70.1%** | 69.1% | 67.4% |
| Level 3 | **57.7%** | 47.6% | 42.3% |

### 5.12 Key Differentiators

| Feature | Manus | Other Frameworks |
|---------|-------|------------------|
| Context Philosophy | File system as memory | In-memory/vector stores |
| Tool Approach | Logit masking (KV-cache aware) | Dynamic tool injection |
| Action Format | CodeAct (Python execution) | JSON function calls |
| Autonomy | Full end-to-end execution | Requires orchestration |
| Core Insight | "Context engineering > Model capabilities" | Model-centric approach |

---

## 6. Comparative Analysis

### 6.1 Philosophy Comparison

| Aspect | Claude SDK | Google ADK | LangChain | Manus |
|--------|------------|------------|-----------|-------|
| **State Model** | Stateless (explicit) | Stateful (sessions) | Flexible (patterns) | Append-only (KV-cache) |
| **Design Philosophy** | Simplicity, control | Hierarchical, structured | Modularity, extensibility | Efficiency, production-first |
| **Primary Focus** | Token efficiency | Multi-agent coordination | Ecosystem integration | Context engineering |

### 6.2 Feature Matrix

| Feature | Claude SDK | Google ADK | LangChain | Manus |
|---------|------------|------------|-----------|-------|
| **Built-in Session Management** | Agent SDK only | Yes | Via integrations | File-based |
| **State Scoping** | Manual | Prefix-based | Via store selection | Hierarchical (3-tier) |
| **Token Counting** | Native API | Via model | Utility functions | Implicit (KV-cache) |
| **Context Editing** | Native (beta) | Manual | trim_messages | Recoverable compression |
| **Memory Persistence** | File-based tool | Session/Memory services | Multiple backends | File system + Vector DB |
| **Multi-Agent Support** | Subagents (isolated) | Sub-agents (shared/isolated) | LangGraph | Wide Research (100+) |
| **Cache Optimization** | Prompt caching | Manual | None | KV-cache first design |

### 6.3 Context Window Management

| Framework | Approach | Automation Level | Cache Awareness |
|-----------|----------|------------------|-----------------|
| **Claude SDK** | Token counting + context editing | High (declarative) | Prompt caching |
| **Google ADK** | State management, include_contents | Medium (explicit) | None |
| **LangChain** | trim_messages, summary memories | High (strategies) | None |
| **Manus** | Recoverable compression + file offload | High (automatic) | Primary design goal |

### 6.4 Multi-Agent Context Patterns

```
Claude SDK Subagents:
┌───────────────┐
│  Orchestrator │ ──► spawns (isolated) ──► Returns summary only
└───────────────┘

Google ADK Sub-agents:
┌───────────────┐
│  Sequential   │ ──► passes (shared context) ──► Writes to state["output_key"]
└───────────────┘

LangChain LangGraph:
┌───────────────┐
│    Graph      │ ──► checkpointed ──► Shared state via checkpointer
└───────────────┘

Manus Wide Research:
┌───────────────┐
│  Orchestrator │ ──► 100+ parallel agents ──► Consensus synthesis
└───────────────┘
```

### 6.5 Code Complexity Comparison

**Simple Chat with Memory:**

```python
# Claude SDK
messages = []
response = client.messages.create(model="...", messages=messages)
messages.append({"role": "assistant", "content": response.content})

# Google ADK
runner = Runner(agent=agent, session_service=session_service)
async for event in runner.run_async(user_id="u1", session_id="s1", new_message=content):
    if event.is_final_response():
        print(event.content)

# LangChain
chain_with_history = RunnableWithMessageHistory(chain, get_session_history, ...)
response = chain_with_history.invoke({"input": "..."}, config={"configurable": {"session_id": "..."}})

# Manus
context.append(user_message)  # Append-only
with open("/workspace/context.txt", "a") as f:
    f.write(response)  # File as extended memory
```

### 6.6 Strengths and Weaknesses

#### Claude SDK

**Strengths:**
- Explicit control over context
- Token counting API for precise management
- Context editing for automatic pruning
- Clean, simple API design

**Weaknesses:**
- Requires manual history management
- Less built-in multi-agent coordination
- Memory tool is still in beta

#### Google ADK

**Strengths:**
- Rich state scoping with prefixes
- Built-in session management
- Excellent multi-agent patterns
- Google Cloud integration

**Weaknesses:**
- Tied to Google ecosystem
- More complex learning curve
- Newer, less community resources

#### LangChain

**Strengths:**
- Extensive memory type options
- Large ecosystem of integrations
- Mature community and documentation
- Multiple LLM support

**Weaknesses:**
- Legacy memory API deprecated
- Multiple patterns can be confusing
- Overhead for simple use cases

#### Manus

**Strengths:**
- Production-optimized (KV-cache first)
- File system as unlimited memory
- Massive parallelization (100+ agents)
- Recoverable compression

**Weaknesses:**
- Closed-source (OpenManus is reconstruction)
- Complex setup (Docker sandbox required)
- Specific to agentic workflows

---

## 7. Recommendations

### 7.1 When to Use Each Framework

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Claude-specific apps | Claude SDK | Native token management, context editing |
| Google Cloud deployment | Google ADK | Vertex AI integration, managed services |
| Multi-LLM applications | LangChain | Provider abstraction, easy switching |
| Complex multi-agent systems | Google ADK or Manus | Rich patterns, state sharing |
| Token-constrained apps | Claude SDK or Manus | Context editing, KV-cache optimization |
| Rapid prototyping | LangChain | Large ecosystem, many examples |
| Production at scale | Manus patterns | 10x cost reduction via caching |

### 7.2 Migration Considerations

**From LangChain to Claude SDK:**
1. Implement explicit message history management
2. Use Token Counting API for context window management
3. Consider Memory Tool for persistence needs

**From LangChain to Google ADK:**
1. Map memory types to state prefixes
2. Implement SessionService for persistence
3. Refactor chains to agent/sub-agent patterns

**Adopting Manus Patterns:**
1. Design for append-only context (never mutate)
2. Use file system for large intermediate results
3. Implement logit masking for tool availability
4. Add `todo.md` pattern for focus maintenance

### 7.3 Best Practices

1. **Always implement persistence for production** - Don't rely on in-memory storage
2. **Use token counting proactively** - Prevent context overflow before it happens
3. **Choose state scoping carefully** - user: vs app: vs session vs temp:
4. **Implement summarization for long conversations** - Preserve important context
5. **Consider KV-cache hit rate** - Structure prompts for stable prefixes
6. **Use file system for large outputs** - Keep only paths in context
7. **Leave errors visible** - Models learn from failures in context

---

## 8. Conclusion

Context management is critical for building effective AI agents. Each framework offers distinct approaches:

- **Claude SDK** provides explicit control with powerful token management and context editing, ideal for applications requiring precise context control.

- **Google ADK** offers a comprehensive hierarchical context system with built-in session management and excellent multi-agent coordination patterns.

- **LangChain** delivers maximum flexibility with extensive memory types and integrations, suitable for diverse deployment scenarios.

- **Manus** introduces production-first context engineering principles, prioritizing KV-cache efficiency and file-based extended memory for autonomous agents.

The choice depends on your specific needs: model preference, deployment environment, complexity requirements, and production scale. All four frameworks continue to evolve rapidly, with context management remaining a key area of innovation.

**Key Insight from Manus:** "Context engineering is not about adding more context—it is about finding the minimal effective context required for the next step."

---

## Sources

### Claude SDK
- [Agent SDK Reference - Python](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python)
- [Context Editing Documentation](https://docs.claude.com/en/docs/build-with-claude/context-editing)
- [Memory Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- [Token Counting Documentation](https://platform.claude.com/docs/en/build-with-claude/token-counting)
- [GitHub: anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python)

### Google ADK
- [ADK Documentation Home](https://google.github.io/adk-docs/)
- [Context Documentation](https://google.github.io/adk-docs/context/)
- [Sessions Overview](https://google.github.io/adk-docs/sessions/session/)
- [State Documentation](https://google.github.io/adk-docs/sessions/state/)
- [Memory Documentation](https://google.github.io/adk-docs/sessions/memory/)
- [GitHub: google/adk-python](https://github.com/google/adk-python)

### LangChain
- [LangChain Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [RunnableWithMessageHistory API](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
- [Memory Migration Guide](https://python.langchain.com/docs/versions/migrating_memory/)

### Manus
- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [In-depth Technical Investigation into Manus AI](https://gist.github.com/renschni/4fbc70b31bad8dd57f3370239dccd58f)
- [OpenManus GitHub Repository](https://github.com/FoundationAgents/OpenManus)
- [CodeAct Research Paper](https://arxiv.org/abs/2402.01030)
- [Wide Research Announcement](https://manus.im/blog/introducing-wide-research)

---

## Appendices

- [Appendix A: Claude SDK Deep Technical Reference](./context-management-deep-technical.md)
- [Appendix B: LangChain Deep Technical Reference](./langchain-technical-deep-dive.md)
