# Context Management in AI Agent SDKs: A Comprehensive Research Report

**Research Date:** January 2026
**Scope:** Claude SDK (Anthropic), Google ADK, and LangChain

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Claude SDK Context Management](#2-claude-sdk-context-management)
3. [Google ADK Context Management](#3-google-adk-context-management)
4. [LangChain Context Management](#4-langchain-context-management)
5. [Comparative Analysis](#5-comparative-analysis)
6. [Recommendations](#6-recommendations)
7. [Conclusion](#7-conclusion)

---

## 1. Executive Summary

Context management is fundamental to building effective AI agents and conversational applications. This report analyzes three leading frameworks for managing conversation context, memory, and state:

| Framework | Philosophy | Key Strength |
|-----------|------------|--------------|
| **Claude SDK** | Stateless API with explicit context passing | Token-aware context editing, memory tools |
| **Google ADK** | Hierarchical context with built-in session management | Rich state scoping, multi-agent patterns |
| **LangChain** | Pluggable memory abstractions with extensive integrations | Flexibility, large ecosystem |

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

### 2.3 Key Features

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

    # Implement: view(), create(), str_replace(), insert(), delete(), rename()
```

#### Subagents for Context Isolation

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

### 2.4 Context Window Specifications

| Model | Standard Context | Extended Context |
|-------|------------------|------------------|
| Claude Sonnet 4 | 200K tokens | 1M tokens |
| Claude Sonnet 4.5 | 200K tokens | 1M tokens |
| Claude Opus 4.5 | 200K tokens | - |

### 2.5 Performance Impact

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
    # Read and write session state
    tool_context.state["user:favorite_city"] = city
    tool_context.state["temp:last_operation"] = "city_saved"

    # Transfer to another agent if needed
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

### 3.4 Session Services

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

### 3.5 Memory Systems (Long-Term)

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
    tools=[load_memory]  # Built-in tool
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

### 3.6 Multi-Agent Context Patterns

#### Sub-Agents (Shared Context)

```python
from google.adk.agents import LlmAgent, SequentialAgent

researcher = LlmAgent(
    name="Researcher",
    model="gemini-2.0-flash",
    instruction="Research the topic.",
    output_key="research_findings"  # Saves to shared state
)

writer = LlmAgent(
    name="Writer",
    model="gemini-2.0-flash",
    instruction="Write based on: {research_findings}"  # Reads from state
)

pipeline = SequentialAgent(
    name="ResearchPipeline",
    sub_agents=[researcher, writer]  # Shared context
)
```

#### AgentTool (Isolated Context)

```python
from google.adk.tools import AgentTool

# This agent runs in isolation
calculator_agent = LlmAgent(
    name="Calculator",
    model="gemini-2.0-flash",
    instruction="Perform calculations."
)

# Wrap as tool - creates isolated execution
calculator_tool = AgentTool(agent=calculator_agent)

main_agent = LlmAgent(
    name="MainAgent",
    tools=[calculator_tool]  # Uses like external API
)
```

### 3.7 Events System

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

### 4.2 Memory Types

#### ConversationBufferMemory
Stores complete conversation history:

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=ChatOpenAI(model="gpt-4"),
    memory=memory
)

conversation.predict(input="Hi, I'm Alice")
conversation.predict(input="What's my name?")  # Remembers "Alice"
```

#### ConversationBufferWindowMemory
Keeps only the last K interactions:

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3)  # Last 3 exchanges
```

#### ConversationSummaryMemory
Maintains a running summary:

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAI

memory = ConversationSummaryMemory(llm=OpenAI(temperature=0))
memory.save_context(
    {"input": "Hi, I'm working on a project about AI"},
    {"output": "That sounds interesting! Tell me more."}
)
# Returns summarized history
memory.load_memory_variables({})
```

#### ConversationSummaryBufferMemory
Hybrid: keeps recent messages + summarizes older ones:

```python
from langchain.memory import ConversationSummaryBufferMemory

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=650  # Summarize when exceeding
)
```

#### VectorStoreRetrieverMemory
Semantic retrieval from vector store:

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
# Retrieve relevant memories
memory.load_memory_variables({"input": "What food do I like?"})
```

#### ConversationEntityMemory
Extracts and tracks named entities:

```python
from langchain.memory import ConversationEntityMemory

memory = ConversationEntityMemory(llm=llm)
memory.save_context(
    {"input": "Deven and Sam are working on a hackathon project"},
    {"output": "That's exciting!"}
)
# Query about specific entity
result = memory.load_memory_variables({"input": "Who is Sam?"})
# {'entities': {'Sam': 'Sam is working on a hackathon project with Deven.'}}
```

### 4.3 Memory Selection Guide

| Memory Type | Best For | Token Usage | Context Retention |
|-------------|----------|-------------|-------------------|
| Buffer | Short conversations | High (grows) | Complete |
| BufferWindow | Recent context only | Fixed | Last K messages |
| Summary | Long conversations | Low (constant) | Summarized |
| SummaryBuffer | Balanced needs | Medium | Recent + Summary |
| VectorStoreRetriever | Semantic recall | Variable | Query-based |
| Entity | Entity tracking | Medium | Entity-focused |

### 4.4 Modern Approach: RunnableWithMessageHistory

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

### 4.5 Token Management

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

### 4.6 LangGraph Persistence (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore

# Checkpointer for thread-scoped persistence
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/db"
)

# Store for cross-thread memory
store = InMemoryStore()

graph = workflow.compile(checkpointer=checkpointer)

result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config={"configurable": {"thread_id": "conversation_123"}}
)
```

### 4.7 Persistence Options

| Backend | Implementation | Best For |
|---------|----------------|----------|
| Redis | `RedisChatMessageHistory` | High-performance, TTL support |
| PostgreSQL | `PostgresChatMessageHistory` | Relational, ACID compliance |
| MongoDB | `MongoDBChatMessageHistory` | Document storage, flexibility |
| SQLite | `SqliteSaver` | Local development |

```python
from langchain_redis import RedisChatMessageHistory

history = RedisChatMessageHistory(
    session_id="user123",
    url="redis://localhost:6379/0",
    ttl=3600  # Expire after 1 hour
)
```

---

## 5. Comparative Analysis

### 5.1 Philosophy Comparison

| Aspect | Claude SDK | Google ADK | LangChain |
|--------|------------|------------|-----------|
| **State Model** | Stateless API (explicit) | Stateful (built-in sessions) | Flexible (multiple patterns) |
| **Design Philosophy** | Simplicity, control | Hierarchical, structured | Modularity, extensibility |
| **Primary Focus** | Token efficiency, context editing | Multi-agent coordination | Ecosystem integration |

### 5.2 Feature Matrix

| Feature | Claude SDK | Google ADK | LangChain |
|---------|------------|------------|-----------|
| **Built-in Session Management** | Agent SDK only | Yes | Via integrations |
| **State Scoping** | Manual | Prefix-based (user:, app:, temp:) | Via store selection |
| **Token Counting** | Native API | Via model | Utility functions |
| **Context Editing** | Native (beta) | Manual | trim_messages |
| **Memory Persistence** | File-based tool | Session/Memory services | Multiple backends |
| **Vector Memory** | Custom implementation | Via MemoryService | Native support |
| **Multi-Agent Support** | Subagents (isolated) | Sub-agents (shared), AgentTool (isolated) | LangGraph |
| **Summarization** | Via context editing | Custom implementation | Native memory types |

### 5.3 Context Window Management

| Framework | Approach | Automation Level |
|-----------|----------|------------------|
| **Claude SDK** | Token counting API + context editing rules | High (declarative rules) |
| **Google ADK** | Manual state management, include_contents param | Medium (explicit control) |
| **LangChain** | trim_messages, summary memories, window memories | High (multiple strategies) |

### 5.4 Multi-Agent Context Patterns

```
Claude SDK Subagents:
┌───────────────┐
│  Orchestrator │
│  (global ctx) │
└───────┬───────┘
        │ spawns (isolated)
┌───────┴───────┐
│   Subagent    │──► Returns summary only
│  (local ctx)  │
└───────────────┘

Google ADK Sub-agents:
┌───────────────┐
│   Sequential  │
│    Agent      │
└───────┬───────┘
        │ passes (shared context)
┌───────┴───────┐
│  Sub-agent 1  │──► Writes to state["output_key"]
│  Sub-agent 2  │──► Reads from state["output_key"]
└───────────────┘

LangChain LangGraph:
┌───────────────┐
│    Graph      │
│  (compiled)   │
└───────┬───────┘
        │ checkpointed
┌───────┴───────┐
│    Nodes      │──► Shared state via checkpointer
│  (functions)  │
└───────────────┘
```

### 5.5 Persistence Architecture

| Claude SDK | Google ADK | LangChain |
|------------|------------|-----------|
| File-based memory tool | SessionService abstraction | ChatMessageHistory abstraction |
| Custom backends via abstract class | InMemory, Database, Vertex AI | Redis, PostgreSQL, MongoDB, etc. |
| Designed for agent-controlled persistence | Designed for application-controlled | Designed for integration flexibility |

### 5.6 Code Complexity Comparison

**Simple Chat with Memory - Claude SDK:**
```python
messages = []  # Developer manages
response = client.messages.create(model="...", messages=messages)
messages.append({"role": "assistant", "content": response.content})
```

**Simple Chat with Memory - Google ADK:**
```python
runner = Runner(agent=agent, session_service=session_service)
async for event in runner.run_async(user_id="u1", session_id="s1", new_message=content):
    if event.is_final_response():
        print(event.content)
```

**Simple Chat with Memory - LangChain:**
```python
chain_with_history = RunnableWithMessageHistory(chain, get_session_history, ...)
response = chain_with_history.invoke({"input": "..."}, config={"configurable": {"session_id": "..."}})
```

### 5.7 Strengths and Weaknesses

#### Claude SDK

**Strengths:**
- Explicit control over context
- Token counting API for precise management
- Context editing for automatic pruning
- Clean, simple API design
- Memory tool for agent-controlled persistence

**Weaknesses:**
- Requires manual history management
- Less built-in multi-agent coordination
- Memory tool is still in beta

#### Google ADK

**Strengths:**
- Rich state scoping with prefixes
- Built-in session management
- Excellent multi-agent patterns
- Template injection in instructions
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
- Flexible architecture
- Multiple LLM support

**Weaknesses:**
- Legacy memory API deprecated
- Multiple patterns can be confusing
- Overhead for simple use cases
- Frequent API changes

---

## 6. Recommendations

### 6.1 When to Use Each Framework

| Scenario | Recommended Framework | Reason |
|----------|----------------------|--------|
| Claude-specific applications | Claude SDK | Native token management, context editing |
| Google Cloud deployment | Google ADK | Vertex AI integration, managed services |
| Multi-LLM applications | LangChain | Provider abstraction, easy switching |
| Complex multi-agent systems | Google ADK | Rich sub-agent patterns, state sharing |
| Token-constrained applications | Claude SDK | Context editing, precise counting |
| Rapid prototyping | LangChain | Large ecosystem, many examples |
| Production with custom backends | LangChain | Extensive persistence options |

### 6.2 Migration Considerations

**From LangChain Legacy Memory to Modern:**
1. Replace `ConversationBufferMemory` with `RunnableWithMessageHistory`
2. Use LangGraph checkpointers for state persistence
3. Implement custom `BaseChatMessageHistory` for specialized storage

**From LangChain to Claude SDK:**
1. Implement explicit message history management
2. Use Token Counting API for context window management
3. Consider Memory Tool for persistence needs

**From LangChain to Google ADK:**
1. Map memory types to state prefixes
2. Implement SessionService for persistence
3. Refactor chains to agent/sub-agent patterns

### 6.3 Best Practices

1. **Always implement persistence for production** - Don't rely on in-memory storage
2. **Use token counting proactively** - Prevent context overflow before it happens
3. **Choose state scoping carefully** - user: vs app: vs session vs temp:
4. **Implement summarization for long conversations** - Preserve important context
5. **Isolate expensive operations** - Use subagents or AgentTool for heavy processing
6. **Version your memory schemas** - Enable migration as requirements evolve

---

## 7. Conclusion

Context management is critical for building effective AI agents. Each framework offers distinct approaches:

- **Claude SDK** provides explicit control with powerful token management and context editing, ideal for applications requiring precise context control.

- **Google ADK** offers a comprehensive hierarchical context system with built-in session management and excellent multi-agent coordination patterns.

- **LangChain** delivers maximum flexibility with extensive memory types and integrations, suitable for diverse deployment scenarios.

The choice depends on your specific needs: model preference, deployment environment, complexity requirements, and team expertise. All three frameworks continue to evolve rapidly, with context management remaining a key area of innovation.

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
