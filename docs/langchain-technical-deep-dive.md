# LangChain Context Management: Deep Technical Reference

**Research Date:** January 2026
**Focus:** Internal implementation details, interfaces, and advanced patterns

---

## Table of Contents

1. [Message Type System](#1-message-type-system)
2. [Chat History Internals](#2-chat-history-internals)
3. [LCEL Integration](#3-lcel-langchain-expression-language-integration)
4. [Memory Variable Management](#4-memory-variable-management)
5. [Token Counting Implementation](#5-token-counting-implementation)
6. [LangGraph State Management](#6-langgraph-state-management)
7. [Vector Memory Technical Details](#7-vector-memory-technical-details)
8. [Callback System](#8-callback-system)

---

## 1. Message Type System

### 1.1 BaseMessage Hierarchy

LangChain's message system is built on a class hierarchy extending from `BaseMessage`:

```
BaseMessage (abstract, extends Serializable)
├── HumanMessage      (type: "human")
├── AIMessage         (type: "ai")
├── SystemMessage     (type: "system")
├── ToolMessage       (type: "tool")
├── FunctionMessage   (type: "function") [deprecated]
└── ChatMessage       (type: "chat", dynamic role)
```

#### BaseMessage Core Structure

```python
from langchain_core.messages.base import BaseMessage
from pydantic import Field

class BaseMessage(Serializable):
    """Base abstract message class."""

    content: Union[str, List[Union[str, Dict]]]
    """The string contents of the message or list of content blocks."""

    additional_kwargs: dict = Field(default_factory=dict, repr=False)
    """Reserved for additional payload data associated with the message.
    For example, for a message from an AI, this could include tool calls
    as encoded by the model provider."""

    response_metadata: dict = Field(default_factory=dict, repr=False)
    """Response metadata. For example: response headers, logprobs,
    token counts, model name."""

    type: str
    """Must be unique to the message type for serialization."""

    name: Optional[str] = None
    """Optional name for the message."""

    id: Optional[str] = None
    """Unique identifier for the message. Should be provided by
    the provider/model which created the message."""
```

### 1.2 Message-Specific Fields

#### AIMessage

```python
class AIMessage(BaseMessage):
    type: Literal["ai"] = "ai"

    tool_calls: list[ToolCall] = Field(default_factory=list)
    """If present, tool calls associated with the message."""

    invalid_tool_calls: list[InvalidToolCall] = Field(default_factory=list)
    """Tool calls with parsing errors."""

    usage_metadata: Optional[UsageMetadata] = None
    """Token usage metadata (if returned by model)."""

# ToolCall structure
class ToolCall(TypedDict):
    name: str              # Tool name
    args: Dict[str, Any]   # Tool arguments
    id: Optional[str]      # Unique call identifier
    type: Literal["tool_call"] = "tool_call"
```

**Example AIMessage with tool_calls:**
```python
AIMessage(
    content='',  # Empty when tool use is forced
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

    tool_call_id: str
    """ID to associate tool call request with response.
    Essential for parallel tool calls."""

    artifact: Optional[Any] = None
    """Tool execution output not meant for the model.
    Use when only a subset of output is sent as content."""

    status: Literal["success", "error"] = "success"
    """Status of the tool invocation."""
```

**Usage Pattern:**
```python
# Process tool calls from AIMessage
for tool_call in ai_msg.tool_calls:
    tool = tool_registry[tool_call["name"]]
    result = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"],  # Must match!
        artifact=result  # Full result if different from content
    ))
```

### 1.3 Message Serialization

#### Type Discriminator System

```python
from typing import Annotated
from langchain_core.messages import AnyMessage
from typing_extensions import Tag

# AnyMessage uses discriminated union for serialization
AnyMessage = Annotated[
    Annotated[AIMessage, Tag(tag="ai")] |
    Annotated[HumanMessage, Tag(tag="human")] |
    Annotated[ChatMessage, Tag(tag="chat")] |
    Annotated[SystemMessage, Tag(tag="system")] |
    Annotated[FunctionMessage, Tag(tag="function")] |
    Annotated[ToolMessage, Tag(tag="tool")],
    Discriminator("type")
]
```

#### Serialization Methods

```python
message = HumanMessage(content="Hello", name="user")

# Dict representation
message.dict()
# {'content': 'Hello', 'name': 'user', 'type': 'human',
#  'additional_kwargs': {}, 'response_metadata': {}}

# JSON representation
message.json()
# '{"content": "Hello", "name": "user", "type": "human", ...}'

# LangChain-specific serialization
message.to_json()
# Includes lc_serializable metadata for reconstruction

# Check if serializable
message.is_lc_serializable()  # True
```

### 1.4 Message Utility Functions

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
    strategy="last",           # Keep most recent
    token_counter=count_tokens_approximately,
    start_on="human",          # Ensure starts with human
    include_system=True,       # Always keep system message
    allow_partial=False        # Don't split messages
)

# Trim by message count
trimmed = trim_messages(
    messages,
    max_tokens=10,
    token_counter=len  # Counts messages, not tokens
)

# Filter by type
filtered = filter_messages(
    messages,
    include_types=[HumanMessage, AIMessage],
    exclude_ids=["msg_123"]
)

# Merge consecutive same-type messages
merged = merge_message_runs(messages)
# Note: ToolMessages are never merged (distinct tool_call_id)
# String contents: concatenated with newline
# List contents: concatenated as list
```

---

## 2. Chat History Internals

### 2.1 BaseChatMessageHistory Interface

```python
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage

class BaseChatMessageHistory(ABC):
    """Abstract base class for chat message history."""

    messages: List[BaseMessage]
    """Property returning all messages in order."""

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the store."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from the store."""
        pass

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Add multiple messages. Default: loop over add_message."""
        for message in messages:
            self.add_message(message)

    def add_user_message(self, message: str) -> None:
        """Convenience method for adding human message."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        """Convenience method for adding AI message."""
        self.add_message(AIMessage(content=message))
```

### 2.2 InMemoryChatMessageHistory Implementation

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """In-memory implementation using a list."""

    def __init__(self, messages: Optional[List[BaseMessage]] = None):
        self._messages: List[BaseMessage] = messages or []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_message(self, message: BaseMessage) -> None:
        self._messages.append(message)

    def clear(self) -> None:
        self._messages = []
```

### 2.3 Redis Implementation

```python
from langchain_redis import RedisChatMessageHistory
from redis import Redis

class RedisChatMessageHistory(BaseChatMessageHistory):
    """Redis-backed chat history using RedisVL for indexing."""

    def __init__(
        self,
        session_id: str,
        url: str = "redis://localhost:6379",
        key_prefix: str = "chat_history:",
        ttl: Optional[int] = None,  # Seconds until expiry
        index_name: Optional[str] = None
    ):
        self.session_id = session_id
        self.key = f"{key_prefix}{session_id}"
        self.ttl = ttl
        self._client = Redis.from_url(url)

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from Redis JSON storage."""
        data = self._client.json().get(self.key)
        if data is None:
            return []
        return [_message_from_dict(m) for m in data["messages"]]

    def add_message(self, message: BaseMessage) -> None:
        """Append message to Redis JSON array."""
        msg_dict = message.dict()
        # Use JSON.ARRAPPEND for atomic append
        if not self._client.exists(self.key):
            self._client.json().set(self.key, "$", {"messages": []})
        self._client.json().arrappend(self.key, "$.messages", msg_dict)
        if self.ttl:
            self._client.expire(self.key, self.ttl)

    def clear(self) -> None:
        """Delete the key entirely."""
        self._client.delete(self.key)
```

### 2.4 PostgreSQL Implementation

```python
from langchain_postgres import PostgresChatMessageHistory
import psycopg

class PostgresChatMessageHistory(BaseChatMessageHistory):
    """PostgreSQL-backed chat history."""

    def __init__(
        self,
        session_id: str,
        connection_string: str,
        table_name: str = "chat_history"
    ):
        self.session_id = session_id
        self.table_name = table_name
        self._conn = psycopg.connect(connection_string)
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create table if not exists."""
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    message JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_session
                    ON {self.table_name}(session_id);
            """)
            self._conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        """Fetch messages ordered by creation time."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT message FROM {self.table_name} "
                f"WHERE session_id = %s ORDER BY created_at",
                (self.session_id,)
            )
            return [_message_from_dict(row[0]) for row in cur.fetchall()]

    def add_message(self, message: BaseMessage) -> None:
        """Insert message as JSONB."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {self.table_name} (session_id, message) "
                f"VALUES (%s, %s)",
                (self.session_id, Json(message.dict()))
            )
            self._conn.commit()
```

### 2.5 MongoDB Implementation

```python
from langchain_mongodb import MongoDBChatMessageHistory
from pymongo import MongoClient

class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """MongoDB-backed chat history."""

    def __init__(
        self,
        connection_string: str,
        session_id: str,
        database_name: str = "chat_history",
        collection_name: str = "message_store",
        history_size: Optional[int] = None  # Fetch only last N
    ):
        self.session_id = session_id
        self.history_size = history_size
        client = MongoClient(connection_string)
        self._collection = client[database_name][collection_name]

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages, optionally limited."""
        cursor = self._collection.find(
            {"session_id": self.session_id}
        ).sort("timestamp", 1)

        if self.history_size:
            # Get last N messages
            cursor = cursor.limit(self.history_size)

        return [_message_from_dict(doc["message"]) for doc in cursor]

    def add_message(self, message: BaseMessage) -> None:
        """Insert document with session_id and timestamp."""
        self._collection.insert_one({
            "session_id": self.session_id,
            "message": message.dict(),
            "timestamp": datetime.utcnow()
        })
```

---

## 3. LCEL (LangChain Expression Language) Integration

### 3.1 RunnableWithMessageHistory Internals

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.chat_history import BaseChatMessageHistory

class RunnableWithMessageHistory(Runnable):
    """Wraps a Runnable to manage chat message history."""

    def __init__(
        self,
        runnable: Runnable,
        get_session_history: Callable[[str], BaseChatMessageHistory],
        input_messages_key: Optional[str] = None,
        output_messages_key: Optional[str] = None,
        history_messages_key: Optional[str] = None,
        history_factory_config: Optional[List[ConfigurableFieldSpec]] = None
    ):
        """
        Args:
            runnable: The underlying chain/model to wrap
            get_session_history: Factory function (session_id) -> history
            input_messages_key: Key for input messages in chain input
            output_messages_key: Key for output messages in chain output
            history_messages_key: Key where history is injected
            history_factory_config: Custom config fields for history factory
        """
        self.runnable = runnable
        self.get_session_history = get_session_history
        self.input_messages_key = input_messages_key
        self.output_messages_key = output_messages_key
        self.history_messages_key = history_messages_key

        # Default config expects "session_id" string
        self.history_factory_config = history_factory_config or [
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description="Unique session identifier"
            )
        ]
```

### 3.2 Config Propagation Flow

```python
def invoke(
    self,
    input: Dict[str, Any],
    config: Optional[RunnableConfig] = None
) -> Any:
    """
    Execution flow:
    1. Extract session_id from config["configurable"]
    2. Get/create chat history for session
    3. Load existing messages from history
    4. Inject history into input
    5. Run underlying chain
    6. Extract output messages
    7. Save input + output to history
    8. Return result
    """
    config = config or {}
    configurable = config.get("configurable", {})

    # Step 1: Extract session identifier
    session_id = configurable.get("session_id")
    if session_id is None:
        raise ValueError("session_id required in config['configurable']")

    # Step 2: Get history instance
    history = self.get_session_history(session_id)

    # Step 3: Load existing messages
    existing_messages = history.messages

    # Step 4: Inject history
    input_with_history = self._inject_history(input, existing_messages)

    # Step 5: Run chain with bound config
    result = self.runnable.invoke(input_with_history, config)

    # Step 6-7: Extract and save messages
    input_messages = self._extract_input_messages(input)
    output_messages = self._extract_output_messages(result)

    for msg in input_messages + output_messages:
        history.add_message(msg)

    return result

def _inject_history(
    self,
    input: Dict,
    history: List[BaseMessage]
) -> Dict:
    """Inject history messages at the configured key."""
    if self.history_messages_key:
        return {**input, self.history_messages_key: history}
    # If no key specified, prepend to messages list
    if "messages" in input:
        return {**input, "messages": history + input["messages"]}
    return input
```

### 3.3 History Injection Points

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter

# Pattern 1: MessagesPlaceholder in prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # <-- Injection point
    ("human", "{input}")
])

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"  # Matches placeholder
)

# Pattern 2: RunnablePassthrough.assign for manual injection
chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables)
                | itemgetter("history")
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Pattern 3: Direct state modification in LangGraph
def inject_history(state: State) -> State:
    """Node that loads and injects history."""
    history = get_session_history(state["session_id"])
    return {
        **state,
        "messages": history.messages + state.get("messages", [])
    }
```

### 3.4 Custom ConfigurableFieldSpec

```python
from langchain_core.runnables import ConfigurableFieldSpec

# Multi-tenant history with user_id + conversation_id
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history=lambda user_id, conversation_id:
        get_history(f"{user_id}:{conversation_id}"),
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique user identifier"
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique conversation within user"
        )
    ]
)

# Invoke with multi-part config
response = chain_with_history.invoke(
    {"input": "Hello"},
    config={
        "configurable": {
            "user_id": "user_123",
            "conversation_id": "conv_456"
        }
    }
)
```

---

## 4. Memory Variable Management

### 4.1 BaseMemory Interface

```python
from abc import ABC, abstractmethod
from langchain_core.memory import BaseMemory

class BaseMemory(ABC):
    """Abstract base for memory classes."""

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Return memory keys that will be injected into chain.
        Example: ["chat_history", "entities"]
        """
        pass

    @abstractmethod
    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Load context variables from memory.

        Args:
            inputs: Current chain inputs (may be used for retrieval)

        Returns:
            Dict mapping memory_variables to their values
        """
        pass

    @abstractmethod
    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Save context from this conversation to buffer.

        Args:
            inputs: User inputs to the chain
            outputs: Chain outputs
        """
        pass

    def clear(self) -> None:
        """Clear memory contents."""
        pass
```

### 4.2 ConversationBufferMemory Implementation

```python
from langchain.memory import ConversationBufferMemory

class ConversationBufferMemory(BaseMemory):
    """Buffer that stores conversation messages."""

    def __init__(
        self,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        return_messages: bool = False,
        chat_memory: Optional[BaseChatMessageHistory] = None
    ):
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.return_messages = return_messages
        self.chat_memory = chat_memory or InMemoryChatMessageHistory()

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return conversation history."""
        if self.return_messages:
            # Return as list of BaseMessage objects
            return {self.memory_key: self.chat_memory.messages}
        else:
            # Return as formatted string
            return {
                self.memory_key: self._format_messages(
                    self.chat_memory.messages
                )
            }

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Format messages as string."""
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"{self.human_prefix}: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"{self.ai_prefix}: {msg.content}")
        return "\n".join(lines)

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Add input and output to chat memory."""
        # Determine input key
        input_key = self.input_key
        if input_key is None:
            if len(inputs) == 1:
                input_key = list(inputs.keys())[0]
            else:
                raise ValueError("Multiple inputs; specify input_key")

        # Determine output key
        output_key = self.output_key
        if output_key is None:
            if len(outputs) == 1:
                output_key = list(outputs.keys())[0]
            else:
                raise ValueError("Multiple outputs; specify output_key")

        # Save to chat memory
        self.chat_memory.add_user_message(inputs[input_key])
        self.chat_memory.add_ai_message(outputs[output_key])
```

### 4.3 Memory Key Mapping

```python
# Custom key configuration example
memory = ConversationBufferMemory(
    memory_key="chat_history",  # Key injected into prompt
    input_key="question",       # Which input field to save
    output_key="answer",        # Which output field to save
    return_messages=True        # Return BaseMessage list vs string
)

# Prompt must have matching placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="chat_history"),  # Matches memory_key
    ("human", "{question}")  # Matches input_key
])

# Chain integration
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    input_key="question",   # Must match memory.input_key
    output_key="answer"     # Must match memory.output_key
)
```

### 4.4 ConversationSummaryMemory Implementation

```python
class ConversationSummaryMemory(BaseMemory):
    """Memory that summarizes conversation over time."""

    def __init__(
        self,
        llm: BaseLanguageModel,
        memory_key: str = "history",
        human_prefix: str = "Human",
        ai_prefix: str = "AI",
        buffer: str = ""  # Running summary
    ):
        self.llm = llm
        self.memory_key = memory_key
        self.buffer = buffer  # Current summary
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix

    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return current summary."""
        return {self.memory_key: self.buffer}

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Update running summary with new exchange."""
        # Format new messages
        new_lines = self._get_buffer_string(inputs, outputs)

        # Generate updated summary using LLM
        self.buffer = self._predict_new_summary(
            existing_summary=self.buffer,
            new_lines=new_lines
        )

    def _predict_new_summary(
        self,
        existing_summary: str,
        new_lines: str
    ) -> str:
        """Use LLM to update summary."""
        prompt = f"""Progressively summarize the lines of conversation provided,
adding onto the previous summary returning a new summary.

EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence.
The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence.
The AI thinks artificial intelligence is a force for good because
it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{existing_summary}

New lines of conversation:
{new_lines}

New summary:"""

        return self.llm.predict(prompt)
```

---

## 5. Token Counting Implementation

### 5.1 Approximate Token Counting

```python
from langchain_core.messages.utils import count_tokens_approximately

def count_tokens_approximately(
    messages: Sequence[BaseMessage],
    *,
    default_token_length: int = 4  # ~4 chars per token
) -> int:
    """
    Fast approximate token counting.

    Rule of thumb: 1 token ~= 4 characters for English text
    This translates to roughly 3/4 of a word (100 tokens ~= 75 words)
    """
    total = 0
    for message in messages:
        if isinstance(message.content, str):
            total += len(message.content) // default_token_length
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, str):
                    total += len(block) // default_token_length
                elif isinstance(block, dict) and "text" in block:
                    total += len(block["text"]) // default_token_length
        # Add overhead for message structure
        total += 4  # Approximate message formatting tokens
    return total
```

### 5.2 Exact Token Counting with tiktoken

```python
import tiktoken
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def count_tokens_exact(
    messages: List[BaseMessage],
    model: str = "gpt-4"
) -> int:
    """
    Exact token counting using tiktoken.

    Encoding mapping:
    - cl100k_base: gpt-4-turbo, gpt-4, gpt-3.5-turbo, embeddings
    - o200k_base: gpt-4o, gpt-4o-mini (newer models)
    - p50k_base: Codex models
    - r50k_base/gpt2: GPT-3 models (davinci, etc.)
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # Token overhead per message (model-specific)
    tokens_per_message = 3  # <|start|>{role/name}\n{content}<|end|>
    tokens_per_name = 1     # If name field is present

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message

        # Count content tokens
        if isinstance(message.content, str):
            num_tokens += len(encoding.encode(message.content))
        elif isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, str):
                    num_tokens += len(encoding.encode(block))
                elif isinstance(block, dict) and "text" in block:
                    num_tokens += len(encoding.encode(block["text"]))

        # Add role tokens
        if isinstance(message, HumanMessage):
            num_tokens += len(encoding.encode("user"))
        elif isinstance(message, AIMessage):
            num_tokens += len(encoding.encode("assistant"))

        # Add name tokens if present
        if message.name:
            num_tokens += tokens_per_name
            num_tokens += len(encoding.encode(message.name))

    # Every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3

    return num_tokens
```

### 5.3 LLM Built-in Token Counting

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Count tokens for messages
messages = [
    HumanMessage(content="What is the meaning of life?"),
    AIMessage(content="42")
]

# Use LLM's built-in counter (most accurate)
token_count = llm.get_num_tokens_from_messages(messages)
print(f"Exact count: {token_count}")

# For streaming with token counts
llm_with_usage = ChatOpenAI(model="gpt-4o", stream_usage=True)
for chunk in llm_with_usage.stream(messages):
    if chunk.usage_metadata:
        print(f"Usage: {chunk.usage_metadata}")
```

### 5.4 Universal Token Counting Callback (2025)

```python
from langchain_core.callbacks import BaseCallbackHandler

class TokenCountingCallback(BaseCallbackHandler):
    """Universal callback for tracking token usage across models."""

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.calls = []

    def on_llm_end(self, response, **kwargs):
        """Extract token usage from response metadata."""
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, 'generation_info'):
                    info = gen.generation_info or {}
                    # OpenAI format
                    if 'token_usage' in info:
                        usage = info['token_usage']
                        self.total_input_tokens += usage.get('prompt_tokens', 0)
                        self.total_output_tokens += usage.get('completion_tokens', 0)
                    # Anthropic format
                    if 'usage' in info:
                        usage = info['usage']
                        self.total_input_tokens += usage.get('input_tokens', 0)
                        self.total_output_tokens += usage.get('output_tokens', 0)

        self.calls.append({
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens
        })

# Usage
callback = TokenCountingCallback()
llm = ChatOpenAI(model="gpt-4o", callbacks=[callback])
response = llm.invoke("Hello!")
print(f"Total tokens: {callback.total_input_tokens + callback.total_output_tokens}")
```

---

## 6. LangGraph State Management

### 6.1 State Schema Definition

```python
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# Basic state schema
class BasicState(TypedDict):
    messages: list[BaseMessage]
    user_input: str

# State with reducer annotations
class ReducerState(TypedDict):
    # add_messages reducer: appends new messages to existing
    messages: Annotated[list[BaseMessage], add_messages]
    # Default behavior: overwrite
    current_step: str
    # Custom accumulator
    visited_nodes: Annotated[list[str], lambda x, y: x + y]
```

### 6.2 Reducer Functions

```python
from typing import Union

def reducer_signature(left: Value, right: Value) -> Value:
    """
    Reducer function signature.

    Args:
        left: Current state value
        right: Update value from node

    Returns:
        New state value
    """
    pass

# Built-in add_messages reducer
def add_messages(
    left: list[BaseMessage],
    right: Union[BaseMessage, list[BaseMessage]]
) -> list[BaseMessage]:
    """
    Append messages to existing list.
    Handles deduplication by message ID.
    """
    if isinstance(right, BaseMessage):
        right = [right]

    # Create ID -> message mapping for dedup
    left_by_id = {m.id: m for m in left if m.id}

    result = list(left)
    for msg in right:
        if msg.id and msg.id in left_by_id:
            # Update existing message
            idx = next(i for i, m in enumerate(result) if m.id == msg.id)
            result[idx] = msg
        else:
            # Append new message
            result.append(msg)

    return result

# Custom reducers
def max_reducer(left: int, right: int) -> int:
    """Keep maximum value."""
    return max(left, right)

def set_union_reducer(left: set, right: set) -> set:
    """Union of sets."""
    return left | right

def last_value_reducer(left: Any, right: Any) -> Any:
    """Always use the newest value (default behavior)."""
    return right
```

### 6.3 MessagesState Convenience Class

```python
from langgraph.graph import MessagesState

# Pre-defined state with messages reducer
class MessagesState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Extend for custom fields
class CustomState(MessagesState):
    user_id: str
    session_data: dict

# Usage in graph
from langgraph.graph import StateGraph

graph = StateGraph(CustomState)
```

### 6.4 Checkpointer Implementation Details

```python
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple
)
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

class BaseCheckpointSaver(ABC):
    """Base class for checkpoint persistence."""

    serde: SerializerProtocol = JsonPlusSerializer()
    """Serializer for encoding/decoding checkpoints."""

    @abstractmethod
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions
    ) -> RunnableConfig:
        """Store a checkpoint with configuration and metadata."""
        pass

    @abstractmethod
    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str
    ) -> None:
        """Store intermediate writes (pending writes)."""
        pass

    @abstractmethod
    def get_tuple(
        self,
        config: RunnableConfig
    ) -> Optional[CheckpointTuple]:
        """Fetch checkpoint tuple by config."""
        pass

    @abstractmethod
    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching criteria."""
        pass

    def get_next_version(
        self,
        current: Optional[str],
        channel: ChannelProtocol
    ) -> str:
        """Generate next version ID (monotonically increasing)."""
        if current is None:
            return "1"
        return str(int(current) + 1)

# Checkpoint data structure
class Checkpoint(TypedDict):
    """Snapshot of graph state at a point in time."""
    v: int                              # Schema version
    id: str                             # Unique checkpoint ID
    ts: str                             # ISO timestamp
    channel_values: Dict[str, Any]      # State values per channel
    channel_versions: Dict[str, str]    # Version per channel
    versions_seen: Dict[str, Dict[str, str]]  # Versions seen by nodes
    pending_sends: List[Any]            # Pending messages to send

class CheckpointTuple(NamedTuple):
    """Complete checkpoint with metadata."""
    config: RunnableConfig
    checkpoint: Checkpoint
    metadata: CheckpointMetadata
    parent_config: Optional[RunnableConfig]
    pending_writes: List[tuple[str, str, Any]]
```

### 6.5 PostgresSaver Implementation

```python
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

# Sync usage
with psycopg.connect(
    "postgresql://user:pass@localhost/db",
    autocommit=True,
    row_factory=dict_row
) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # Create required tables

    graph = workflow.compile(checkpointer=checkpointer)

    result = graph.invoke(
        {"messages": [HumanMessage("Hello")]},
        config={"configurable": {"thread_id": "thread_123"}}
    )

# Async usage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import psycopg_pool

pool = psycopg_pool.AsyncConnectionPool(
    "postgresql://user:pass@localhost/db",
    kwargs={"autocommit": True, "row_factory": dict_row}
)

async with pool.connection() as conn:
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    graph = workflow.compile(checkpointer=checkpointer)
```

### 6.6 Cross-Thread Memory with Store

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# InMemoryStore for development
store = InMemoryStore()

# Store interface
class BaseStore(ABC):
    """Interface for cross-thread persistent storage."""

    @abstractmethod
    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict
    ) -> None:
        """Store value at namespace/key."""
        pass

    @abstractmethod
    def get(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> Optional[dict]:
        """Retrieve value from namespace/key."""
        pass

    @abstractmethod
    def search(
        self,
        namespace: tuple[str, ...],
        *,
        filter: Optional[dict] = None,
        limit: int = 10
    ) -> List[dict]:
        """Search within namespace."""
        pass

    @abstractmethod
    def delete(
        self,
        namespace: tuple[str, ...],
        key: str
    ) -> None:
        """Delete key from namespace."""
        pass

# Usage in graph
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

graph = StateGraph(State)

# Compile with both checkpointer and store
compiled = graph.compile(
    checkpointer=PostgresSaver(...),  # Thread-scoped
    store=InMemoryStore()              # Cross-thread
)

# Access store in nodes
def my_node(state: State, *, store: BaseStore) -> dict:
    # Namespace by user for cross-thread memory
    namespace = ("users", state["user_id"], "preferences")

    # Retrieve user preferences
    prefs = store.get(namespace, "settings")

    # Update preferences
    store.put(namespace, "settings", {"theme": "dark"})

    return {"processed": True}
```

---

## 7. Vector Memory Technical Details

### 7.1 VectorStoreRetrieverMemory Architecture

```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.vectorstores import VectorStore

class VectorStoreRetrieverMemory(BaseMemory):
    """Semantic memory using vector similarity search."""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        return_docs: bool = False
    ):
        self.retriever = retriever
        self.memory_key = memory_key
        self.input_key = input_key
        self.return_docs = return_docs

    def load_memory_variables(
        self,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve semantically relevant memories."""
        # Get query from inputs
        query = inputs.get(self.input_key, "")
        if not query:
            query = str(inputs)

        # Semantic search
        docs = self.retriever.invoke(query)

        if self.return_docs:
            return {self.memory_key: docs}

        # Format as string
        return {
            self.memory_key: "\n".join(doc.page_content for doc in docs)
        }

    def save_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, str]
    ) -> None:
        """Store conversation as searchable documents."""
        # Format conversation turn
        input_str = str(inputs)
        output_str = str(outputs)

        # Create document
        doc = Document(
            page_content=f"Input: {input_str}\nOutput: {output_str}",
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "type": "conversation"
            }
        )

        # Add to vector store
        self.retriever.vectorstore.add_documents([doc])
```

### 7.2 Embedding Generation Pipeline

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

class OpenAIEmbeddings(Embeddings):
    """OpenAI embedding model wrapper."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,  # Reduce dimensions
        chunk_size: int = 1000,  # Batch size for API calls
    ):
        self.model = model
        self.dimensions = dimensions
        self.chunk_size = chunk_size
        self._client = OpenAI()

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """Embed multiple documents with batching."""
        embeddings = []

        # Process in chunks to respect API limits
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i:i + self.chunk_size]

            response = self._client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions
            )

            embeddings.extend([
                data.embedding for data in response.data
            ])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed single query text."""
        return self.embed_documents([text])[0]

# Embedding flow in VectorStore
class VectorStore:
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        1. Extract text from documents
        2. Generate embeddings via embedding model
        3. Store (embedding, document, metadata) tuples
        4. Return document IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Generate embeddings
        embeddings = self.embedding_function.embed_documents(texts)

        # Store in backend
        return self._add_embeddings(texts, embeddings, metadatas)
```

### 7.3 Similarity Search Parameters

```python
from langchain_core.vectorstores import VectorStoreRetriever

retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" or "similarity_score_threshold"
    search_kwargs={
        "k": 4,                    # Number of documents to return
        "score_threshold": 0.5,   # Minimum similarity (0-1)
        "fetch_k": 20,            # Docs to fetch before MMR reranking
        "lambda_mult": 0.5,       # MMR diversity (0=max diversity, 1=max relevance)
        "filter": {"type": "qa"}  # Metadata filter
    }
)

# Search types explained
class SearchType(Enum):
    SIMILARITY = "similarity"
    """Pure cosine/euclidean similarity ranking."""

    MMR = "mmr"
    """Maximum Marginal Relevance: balances relevance and diversity.
    Fetches fetch_k docs, then selects k most diverse."""

    SIMILARITY_SCORE_THRESHOLD = "similarity_score_threshold"
    """Only return docs above score_threshold."""

# Direct similarity search with scores
results = vectorstore.similarity_search_with_relevance_scores(
    query="What is machine learning?",
    k=5
)
# Returns: List[Tuple[Document, float]] where float is 0-1 score
```

### 7.4 Memory Relevance Scoring

```python
from langchain_core.documents import Document

class ScoredRetriever(VectorStoreRetriever):
    """Retriever that adds scores to document metadata."""

    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:
        """Retrieve documents with relevance scores."""

        # Get docs with scores
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.search_kwargs.get("k", 4)
        )

        # Add scores to metadata
        result = []
        for doc, score in docs_and_scores:
            doc.metadata["relevance_score"] = score
            result.append(doc)

        return result

# Score normalization (distance -> similarity)
def normalize_score(distance: float, metric: str = "cosine") -> float:
    """Convert distance to 0-1 similarity score."""
    if metric == "cosine":
        # Cosine distance is 1 - similarity
        return 1 - distance
    elif metric == "euclidean":
        # Normalize euclidean to 0-1 range
        return 1 / (1 + distance)
    elif metric == "inner_product":
        # Inner product can be negative
        return (distance + 1) / 2
    return distance
```

---

## 8. Callback System

### 8.1 CallbackManager Architecture

```python
from langchain_core.callbacks.manager import (
    CallbackManager,
    CallbackManagerForLLMRun,
    CallbackManagerForChainRun
)
from langchain_core.callbacks.base import BaseCallbackHandler

class CallbackManager:
    """Core callback manager coordinating all handlers."""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler] = None,
        inheritable_handlers: List[BaseCallbackHandler] = None,
        parent_run_id: Optional[UUID] = None,
        tags: List[str] = None,
        inheritable_tags: List[str] = None,
        metadata: Dict[str, Any] = None,
        inheritable_metadata: Dict[str, Any] = None
    ):
        self.handlers = handlers or []
        self.inheritable_handlers = inheritable_handlers or []
        self.parent_run_id = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> CallbackManagerForLLMRun:
        """Dispatch to all handlers when LLM starts."""
        run_id = uuid4()

        for handler in self.handlers:
            try:
                handler.on_llm_start(
                    serialized,
                    prompts,
                    run_id=run_id,
                    parent_run_id=self.parent_run_id,
                    tags=self.tags,
                    metadata=self.metadata,
                    **kwargs
                )
            except Exception as e:
                if handler.raise_error:
                    raise

        return CallbackManagerForLLMRun(
            run_id=run_id,
            handlers=self.handlers,
            inheritable_handlers=self.inheritable_handlers,
            parent_run_id=self.parent_run_id,
            tags=self.tags,
            inheritable_tags=self.inheritable_tags,
            metadata=self.metadata,
            inheritable_metadata=self.inheritable_metadata
        )
```

### 8.2 BaseCallbackHandler Interface

```python
from langchain_core.callbacks.base import BaseCallbackHandler

class BaseCallbackHandler(ABC):
    """Base class for callback handlers."""

    raise_error: bool = False
    """Whether to raise exceptions or log them."""

    run_inline: bool = False
    """Whether to run synchronously or in background."""

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Run when LLM starts."""
        pass

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run on each new LLM token (streaming)."""
        pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run when LLM ends."""
        pass

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run on LLM error."""
        pass

    # Chain callbacks
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Run when chain starts."""
        pass

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run when chain ends."""
        pass

    # Tool callbacks
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Run when tool starts."""
        pass

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run when tool ends."""
        pass

    # Retriever callbacks
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """Run when retriever starts."""
        pass

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs
    ) -> Any:
        """Run when retriever ends."""
        pass
```

### 8.3 Memory-Aware Callback Handler

```python
class MemoryTrackingCallback(BaseCallbackHandler):
    """Callback that tracks memory operations."""

    def __init__(self):
        self.memory_loads = []
        self.memory_saves = []
        self.token_counts = []

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Track when memory is loaded."""
        if "memory" in serialized.get("kwargs", {}):
            self.memory_loads.append({
                "chain": serialized.get("name"),
                "timestamp": datetime.utcnow(),
                "input_keys": list(inputs.keys())
            })

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Track when memory is saved."""
        self.memory_saves.append({
            "timestamp": datetime.utcnow(),
            "output_keys": list(outputs.keys())
        })

    def on_llm_end(
        self,
        response: LLMResult,
        **kwargs
    ) -> None:
        """Track token usage for context management."""
        for generation in response.generations:
            for gen in generation:
                if hasattr(gen, 'generation_info') and gen.generation_info:
                    usage = gen.generation_info.get('token_usage', {})
                    self.token_counts.append({
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "timestamp": datetime.utcnow()
                    })

    def get_total_tokens(self) -> int:
        """Get total tokens used."""
        return sum(
            tc["prompt_tokens"] + tc["completion_tokens"]
            for tc in self.token_counts
        )
```

### 8.4 Tracing Integration

```python
import os
from langchain_core.tracers import LangChainTracer

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Automatic tracing is enabled globally
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello")  # Automatically traced

# Manual tracer for fine-grained control
tracer = LangChainTracer(project_name="custom-project")
llm = ChatOpenAI(model="gpt-4", callbacks=[tracer])

# Trace grouping with context manager
from langchain_core.callbacks import trace_as_chain_group

with trace_as_chain_group("my_workflow") as manager:
    # All calls within this block are grouped
    result1 = llm.invoke("Step 1", config={"callbacks": manager})
    result2 = llm.invoke("Step 2", config={"callbacks": manager})
```

### 8.5 Custom Event Dispatch

```python
from langchain_core.callbacks import dispatch_custom_event

async def my_node(state: State, config: RunnableConfig) -> dict:
    """Node that dispatches custom events."""

    # Dispatch custom event to handlers
    dispatch_custom_event(
        name="memory_update",
        data={
            "action": "save",
            "keys": ["user_preference"],
            "values": {"theme": "dark"}
        },
        config=config
    )

    return state

# Handle custom events
class CustomEventHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Handle custom events."""
        if name == "memory_update":
            print(f"Memory updated: {data}")
```

---

## Quick Reference Tables

### Message Type Summary

| Type | `type` Field | Key Fields | Use Case |
|------|-------------|------------|----------|
| `HumanMessage` | `"human"` | `content` | User input |
| `AIMessage` | `"ai"` | `content`, `tool_calls`, `usage_metadata` | Model response |
| `SystemMessage` | `"system"` | `content` | System instructions |
| `ToolMessage` | `"tool"` | `content`, `tool_call_id`, `artifact` | Tool results |
| `FunctionMessage` | `"function"` | `content`, `name` | Deprecated |
| `ChatMessage` | `"chat"` | `content`, `role` | Dynamic role |

### Checkpointer Comparison

| Implementation | Package | Use Case | Async Support |
|----------------|---------|----------|---------------|
| `MemorySaver` | `langgraph` | Dev/testing | Yes |
| `SqliteSaver` | `langgraph-checkpoint-sqlite` | Local persistence | Yes |
| `PostgresSaver` | `langgraph-checkpoint-postgres` | Production | Yes |
| `MongoDBStore` | `langgraph-store-mongodb` | Cross-thread | Yes |
| `RedisStore` | Custom | High-performance | Yes |

### Chat History Backends

| Backend | Package | Key Features |
|---------|---------|--------------|
| InMemory | `langchain-core` | Development, no persistence |
| Redis | `langchain-redis` | TTL support, RedisVL indexing |
| PostgreSQL | `langchain-postgres` | ACID, relational queries |
| MongoDB | `langchain-mongodb` | Document storage, history_size limit |

---

## Sources

### Official Documentation
- [LangChain Messages Reference](https://reference.langchain.com/python/langchain/messages/)
- [BaseChatMessageHistory API](https://python.langchain.com/api_reference/core/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html)
- [RunnableWithMessageHistory API](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html)
- [CallbackManager API](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.manager.CallbackManager.html)
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph Checkpointing](https://reference.langchain.com/python/langgraph/checkpoints/)
- [trim_messages Documentation](https://python.langchain.com/docs/how_to/trim_messages/)

### Package References
- [langgraph-checkpoint PyPI](https://pypi.org/project/langgraph-checkpoint/)
- [langgraph-checkpoint-postgres PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/)
- [langchain-redis GitHub](https://github.com/langchain-ai/langchain-redis)

### Technical Deep Dives
- [Tool Calling with LangChain Blog](https://www.blog.langchain.com/tool-calling-with-langchain/)
- [LangGraph State Management 2025](https://sparkco.ai/blog/mastering-langgraph-state-management-in-2025)
- [Memory in LangChain Deep Dive](https://www.comet.com/site/blog/memory-in-langchain-a-deep-dive-into-persistent-context/)
- [Langfuse Tracing Integration](https://langfuse.com/integrations/frameworks/langchain)

### GitHub Sources
- [LangChain callbacks/manager.py](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/manager.py)
- [LangGraph checkpoint SQLite](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/langgraph/checkpoint/sqlite/__init__.py)
- [LangChain Redis chat_message_history.py](https://github.com/langchain-ai/langchain-redis/blob/main/libs/redis/langchain_redis/chat_message_history.py)
