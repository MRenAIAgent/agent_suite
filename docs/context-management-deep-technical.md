# Claude SDK Context Management: Deep Technical Reference

**Research Date:** January 2026
**Scope:** Advanced technical internals for Claude SDK context management

---

## Table of Contents

1. [Internal Message Handling](#1-internal-message-handling)
2. [Streaming Implementation](#2-streaming-implementation)
3. [Tool Use Protocol](#3-tool-use-protocol)
4. [Context Editing Internals](#4-context-editing-internals)
5. [Memory Tool Implementation](#5-memory-tool-implementation)
6. [Extended Thinking](#6-extended-thinking)
7. [Multi-modal Context](#7-multi-modal-context)

---

## 1. Internal Message Handling

### 1.1 Message Serialization Format

Messages in the Claude API follow a strict JSON structure with role-based validation:

```python
# Message structure for Claude Messages API
message_schema = {
    "role": "user" | "assistant",  # Required, alternating pattern
    "content": str | List[ContentBlock]  # String or array of content blocks
}

# Full request structure
request_payload = {
    "model": "claude-sonnet-4-5-20250929",
    "max_tokens": 4096,
    "system": "Optional system prompt",  # Separate from messages
    "messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What can you do?"}
    ]
}
```

### 1.2 Content Block Types

Claude supports multiple content block types for complex multi-modal interactions:

| Block Type | Direction | Description |
|------------|-----------|-------------|
| `text` | Input/Output | Plain text content |
| `image` | Input | Base64 or URL image data |
| `tool_use` | Output | Tool invocation by Claude |
| `tool_result` | Input | Results returned to Claude |
| `thinking` | Output | Extended thinking blocks (Claude 4+) |
| `document` | Input | PDF documents (base64) |

#### Text Content Block
```json
{
    "type": "text",
    "text": "The content string"
}
```

#### Image Content Block
```json
{
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",  // jpeg, png, gif, webp
        "data": "/9j/4AAQSkZJRgABAQAAAQ..."
    }
}
// OR URL reference
{
    "type": "image",
    "source": {
        "type": "url",
        "url": "https://example.com/image.jpg"
    }
}
```

#### Tool Use Content Block (Output)
```json
{
    "type": "tool_use",
    "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "name": "get_stock_price",
    "input": {
        "ticker": "^GSPC"
    }
}
```

#### Tool Result Content Block (Input)
```json
{
    "type": "tool_result",
    "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
    "content": "259.75 USD",
    "is_error": false  // Optional, defaults to false
}
```

### 1.3 Message Validation Constraints

**Strict Alternation Rule:**
```
user -> assistant -> user -> assistant -> ...
```

**Critical Constraints:**
- Conversations MUST begin with a `user` message
- Messages MUST strictly alternate between `user` and `assistant` roles
- The `system` prompt is separate and does NOT count toward alternation
- Tool results MUST be in a `user` message following the `assistant` tool_use

**Assistant Prefilling:**
```python
# Valid: Prefill assistant response to guide format
messages = [
    {"role": "user", "content": "Analyze this data"},
    {"role": "assistant", "content": "```json\n{"}  # Prefill
]
```

---

## 2. Streaming Implementation

### 2.1 Server-Sent Events (SSE) Protocol

Claude uses SSE for streaming responses. The protocol follows [WHATWG specification](https://html.spec.whatwg.org/multipage/server-sent-events.html):

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

event: message_start
data: {"type":"message_start","message":{"id":"msg_01XFDUDYJgAACzvnptvVoYEL","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-5-20250929","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" there!"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
```

### 2.2 SSE Decoder Implementation

Based on the actual httpx_sse decoder implementation:

```python
class SSEDecoder:
    def __init__(self):
        self._event = ""
        self._data: List[str] = []
        self._last_event_id = ""
        self._retry: Optional[int] = None

    def decode(self, line: str) -> Optional[ServerSentEvent]:
        # Empty line = dispatch event
        if not line:
            if not self._event and not self._data:
                return None

            sse = ServerSentEvent(
                event=self._event,
                data="\n".join(self._data),  # Multi-line data joined
                id=self._last_event_id,
                retry=self._retry,
            )

            # Reset state (except last_event_id per spec)
            self._event = ""
            self._data = []
            self._retry = None
            return sse

        # Comment line - ignore
        if line.startswith(":"):
            return None

        # Parse field:value
        fieldname, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]  # Strip leading space

        if fieldname == "event":
            self._event = value
        elif fieldname == "data":
            self._data.append(value)
        elif fieldname == "id":
            if "\0" not in value:  # Null character = ignore
                self._last_event_id = value
        elif fieldname == "retry":
            try:
                self._retry = int(value)
            except ValueError:
                pass

        return None
```

### 2.3 Delta Accumulation Algorithm

The SDK accumulates deltas into snapshots using this recursive algorithm:

```python
def accumulate_delta(
    acc: dict[object, object],
    delta: dict[object, object]
) -> dict[object, object]:
    """Recursively merge delta into accumulated state."""

    for key, delta_value in delta.items():
        if key not in acc:
            acc[key] = delta_value
            continue

        acc_value = acc[key]
        if acc_value is None:
            acc[key] = delta_value
            continue

        # Special keys: don't accumulate, replace
        if key == "index" or key == "type":
            acc[key] = delta_value
            continue

        # String concatenation
        if isinstance(acc_value, str) and isinstance(delta_value, str):
            acc_value += delta_value

        # Numeric addition
        elif isinstance(acc_value, (int, float)) and isinstance(delta_value, (int, float)):
            acc_value += delta_value

        # Recursive dict merge
        elif is_dict(acc_value) and is_dict(delta_value):
            acc_value = accumulate_delta(acc_value, delta_value)

        # List handling - use index for positioning
        elif is_list(acc_value) and is_list(delta_value):
            for delta_entry in delta_value:
                if is_dict(delta_entry):
                    index = delta_entry["index"]
                    try:
                        acc_entry = acc_value[index]
                        acc_value[index] = accumulate_delta(acc_entry, delta_entry)
                    except IndexError:
                        acc_value.insert(index, delta_entry)

        acc[key] = acc_value

    return acc
```

### 2.4 Complete Event Type Enumeration

| Event Type | When Emitted | Payload |
|------------|--------------|---------|
| `message_start` | Beginning of response | Full message object (empty content) |
| `content_block_start` | New content block begins | Block type, index |
| `content_block_delta` | Partial content | Delta object with type-specific data |
| `content_block_stop` | Content block complete | Block index |
| `message_delta` | Message metadata update | stop_reason, usage |
| `message_stop` | Stream complete | Empty |
| `ping` | Keep-alive | Empty |
| `error` | Error occurred | Error details |

---

## 3. Tool Use Protocol

### 3.1 Tool Schema Format (JSON Schema)

```python
tool_definition = {
    "name": "get_weather",  # a-z, A-Z, 0-9, _, - (max 64 chars)
    "description": "Get current weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    }
}
```

### 3.2 Pydantic to Tool Schema Conversion

The SDK provides automatic schema generation from Pydantic models:

```python
from pydantic import BaseModel, Field
from anthropic.lib._tools import pydantic_function_tool

class WeatherParams(BaseModel):
    """Get current weather for a location."""
    location: str = Field(description="City and state")
    unit: str = Field(default="celsius", description="Temperature unit")

# Automatic conversion with strict mode
tool = pydantic_function_tool(
    WeatherParams,
    name="get_weather",  # Optional, defaults to class name
    description=None     # Optional, defaults to docstring
)
# Result:
# {
#     "type": "function",
#     "function": {
#         "name": "get_weather",
#         "strict": True,
#         "parameters": {...}  # Strict JSON schema from Pydantic
#     }
# }
```

### 3.3 Tool Choice Parameters

| Tool Choice | Behavior | Use Case |
|------------|----------|----------|
| `{"type": "auto"}` | Claude decides whether to use tools (default) | General agent behavior |
| `{"type": "any"}` | Must use one of the provided tools | Force tool usage |
| `{"type": "tool", "name": "X"}` | Must use specific tool X | Deterministic pipelines |
| `{"type": "none"}` | Cannot use any tools | Force text-only response |

```python
# Force specific tool
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[weather_tool],
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "London weather?"}]
)
```

### 3.4 Parallel Tool Calls

Claude 4 models support parallel tool execution by default:

```python
# Prompt to encourage parallel tool use
system_prompt = """
When performing multiple independent operations, invoke all relevant
tools simultaneously rather than sequentially. For example, when reading
3 files, make 3 parallel tool calls.
"""

# Disable parallel tool use
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    tools=tools,
    tool_choice={
        "type": "auto",
        "disable_parallel_tool_use": True  # At most one tool
    },
    # OR
    tool_choice={
        "type": "any",
        "disable_parallel_tool_use": True  # Exactly one tool
    },
    messages=messages
)
```

### 3.5 Tool Result Formatting

```python
# Single tool result
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
            "content": "Temperature: 72F, Sunny"
        }
    ]
})

# Multiple parallel tool results
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01A...",
            "content": "Result 1"
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01B...",
            "content": "Result 2"
        }
    ]
})

# Error result
messages.append({
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01D...",
            "content": "Error: API rate limit exceeded",
            "is_error": True
        }
    ]
})
```

### 3.6 Strict Schema Compliance (Beta)

Enable guaranteed JSON schema compliance:

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[{
        "name": "structured_output",
        "description": "Return structured data",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"],
            "additionalProperties": False  # Strict mode
        }
    }],
    extra_headers={
        "anthropic-beta": "structured-outputs-2025-11-13"
    },
    messages=messages
)
```

---

## 4. Context Editing Internals

### 4.1 Configuration Structure

```python
response = client.beta.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    betas=["context-management-2025-06-27"],
    messages=messages,
    context_management={
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {
                    "type": "input_tokens",
                    "value": 30000  # Trigger when context exceeds 30K tokens
                },
                "keep": {
                    "type": "tool_uses",
                    "value": 3  # Preserve most recent 3 tool uses
                },
                "clear_at_least": {
                    "type": "input_tokens",
                    "value": 5000  # Minimum tokens to clear
                },
                "exclude_tools": ["web_search"],  # Never clear these
                "clear_tool_inputs": False  # Only clear results, not calls
            },
            {
                "type": "clear_thinking_20251015",
                "trigger": {
                    "type": "input_tokens",
                    "value": 50000
                },
                "keep": {
                    "type": "thinking_blocks",
                    "value": 2
                }
            }
        ]
    }
)
```

### 4.2 Edit Strategy Types

| Strategy | Version | What It Clears |
|----------|---------|----------------|
| `clear_tool_uses_20250919` | Sept 2025 | Tool results (optionally tool inputs) |
| `clear_thinking_20251015` | Oct 2025 | Extended thinking blocks |

### 4.3 Processing Logic

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context Edit Processing                        │
├─────────────────────────────────────────────────────────────────┤
│  1. Calculate current input_tokens                               │
│  2. FOR each edit rule:                                          │
│     a. Check if trigger.value < current_tokens                   │
│     b. If triggered:                                             │
│        - Identify clearable content (chronologically oldest)     │
│        - Exclude tools in exclude_tools list                     │
│        - Calculate tokens to clear (>= clear_at_least.value)     │
│        - Preserve keep.value most recent items                   │
│        - Replace cleared content with placeholder text           │
│  3. Invalidate any cached prompt prefixes                        │
│  4. Send modified context to model                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Placeholder Text for Cleared Content

When tool results are cleared, they're replaced with a placeholder that informs Claude:

```
[Tool result cleared to manage context. Original result was for tool_use_id: toolu_01XYZ...]
```

### 4.5 Edge Cases and Limitations

| Limitation | Details |
|------------|---------|
| **Cache Invalidation** | Clearing modifies context structure, invalidating cached prefixes |
| **Minimum Retention** | Cannot clear below `keep.value` items |
| **Order Dependency** | Edit rules processed in order; earlier rules may affect later ones |
| **Tool Exclusion** | `exclude_tools` only prevents clearing; doesn't affect other operations |
| **Streaming** | Context editing applies before streaming begins |

---

## 5. Memory Tool Implementation

### 5.1 File System Structure

Recommended memory organization:

```
/memories/
├── user-profile/
│   ├── name.txt
│   ├── preferences.txt
│   └── communication-style.txt
├── projects/
│   ├── current-task.txt
│   ├── goals.txt
│   └── progress/
│       ├── milestone-1.txt
│       └── milestone-2.txt
├── knowledge/
│   ├── domain-facts.txt
│   └── learned-patterns.txt
└── session-state/
    ├── last-context.txt
    └── pending-tasks.txt
```

### 5.2 Operation Semantics

#### View Operation
```json
{
    "command": "view",
    "path": "/memories/user-profile",
    "view_range": [1, 50]  // Optional: lines 1-50
}

// Directory listing response
{
    "type": "directory",
    "path": "/memories/user-profile",
    "entries": [
        {"name": "name.txt", "type": "file", "size": 24},
        {"name": "preferences.txt", "type": "file", "size": 156}
    ]
}

// File content response
{
    "type": "file",
    "path": "/memories/user-profile/name.txt",
    "content": "User's name: John Smith",
    "total_lines": 1
}
```

#### Create Operation
```json
{
    "command": "create",
    "path": "/memories/notes/meeting-2024.txt",
    "file_text": "Meeting Notes\n- Topic 1\n- Topic 2"
}
```

#### str_replace Operation
```json
{
    "command": "str_replace",
    "path": "/memories/user-profile/preferences.txt",
    "old_str": "Theme: light",
    "new_str": "Theme: dark"
}
// Exact string matching required; fails if old_str not found exactly once
```

#### Insert Operation
```json
{
    "command": "insert",
    "path": "/memories/todo.txt",
    "insert_line": 2,  // 0-indexed line number
    "insert_text": "- New priority task\n"
}
```

#### Delete Operation
```json
{
    "command": "delete",
    "path": "/memories/old-notes/archived.txt"
}
```

#### Rename Operation
```json
{
    "command": "rename",
    "path": "/memories/old-notes",
    "new_path": "/memories/archived-notes"
}
```

### 5.3 Implementation Interface

```python
from anthropic.types.beta import BetaAbstractMemoryTool
from pathlib import Path
import json

class LocalFilesystemMemoryTool(BetaAbstractMemoryTool):
    """Custom memory tool using local filesystem."""

    def __init__(self, base_path: str = "./agent_memory"):
        super().__init__()
        self.root = Path(base_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def view(self, path: str, view_range: list[int] | None = None) -> dict:
        """View directory listing or file contents."""
        full_path = self.root / path.lstrip("/")

        if full_path.is_dir():
            entries = []
            for item in full_path.iterdir():
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            return {"type": "directory", "path": path, "entries": entries}

        elif full_path.is_file():
            lines = full_path.read_text().splitlines()
            if view_range:
                start, end = view_range
                lines = lines[start-1:end]  # 1-indexed
            return {
                "type": "file",
                "path": path,
                "content": "\n".join(lines),
                "total_lines": len(lines)
            }

        raise FileNotFoundError(f"Path not found: {path}")

    def create(self, path: str, file_text: str) -> dict:
        """Create a new file with content."""
        full_path = self.root / path.lstrip("/")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(file_text)
        return {"status": "created", "path": path}

    def str_replace(self, path: str, old_str: str, new_str: str) -> dict:
        """Replace exact string in file."""
        full_path = self.root / path.lstrip("/")
        content = full_path.read_text()

        count = content.count(old_str)
        if count == 0:
            raise ValueError(f"String not found: {old_str}")
        if count > 1:
            raise ValueError(f"String found {count} times; must be unique")

        new_content = content.replace(old_str, new_str, 1)
        full_path.write_text(new_content)
        return {"status": "replaced", "path": path}

    def insert(self, path: str, insert_line: int, insert_text: str) -> dict:
        """Insert text at specific line."""
        full_path = self.root / path.lstrip("/")
        lines = full_path.read_text().splitlines()
        lines.insert(insert_line, insert_text.rstrip("\n"))
        full_path.write_text("\n".join(lines))
        return {"status": "inserted", "path": path, "line": insert_line}

    def delete(self, path: str) -> dict:
        """Delete file or directory."""
        full_path = self.root / path.lstrip("/")
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)
        return {"status": "deleted", "path": path}

    def rename(self, path: str, new_path: str) -> dict:
        """Rename/move file or directory."""
        full_path = self.root / path.lstrip("/")
        new_full_path = self.root / new_path.lstrip("/")
        full_path.rename(new_full_path)
        return {"status": "renamed", "old_path": path, "new_path": new_path}
```

### 5.4 Concurrency Handling

The memory tool operates client-side, so concurrency is your responsibility:

```python
import asyncio
from threading import Lock

class ThreadSafeMemoryTool(LocalFilesystemMemoryTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._locks: dict[str, Lock] = {}
        self._global_lock = Lock()

    def _get_lock(self, path: str) -> Lock:
        with self._global_lock:
            if path not in self._locks:
                self._locks[path] = Lock()
            return self._locks[path]

    def str_replace(self, path: str, old_str: str, new_str: str) -> dict:
        with self._get_lock(path):
            return super().str_replace(path, old_str, new_str)
```

---

## 6. Extended Thinking

### 6.1 Thinking Token Budget

```python
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=16000,
    thinking={
        "type": "enabled",
        "budget_tokens": 10000  # Minimum: 1024
    },
    messages=messages
)
```

| Budget Range | Recommendation |
|--------------|----------------|
| 1,024 - 4,096 | Simple reasoning tasks |
| 4,096 - 16,384 | Moderate complexity |
| 16,384 - 32,768 | Complex analysis |
| > 32,768 | Use batch processing (avoid network timeouts) |

### 6.2 Streaming with Thinking

```python
# Streaming is REQUIRED when max_tokens > 21,333
with client.messages.stream(
    model="claude-sonnet-4-5-20250929",
    max_tokens=25000,
    thinking={"type": "enabled", "budget_tokens": 15000},
    messages=messages
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            if event.content_block.type == "thinking":
                print("=== THINKING ===")
            elif event.content_block.type == "text":
                print("=== RESPONSE ===")

        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
            elif event.delta.type == "signature_delta":
                # Cryptographic signature for verification
                signature = event.delta.signature
```

### 6.3 Thinking Block Structure

```json
{
    "type": "thinking",
    "thinking": "Let me analyze this step by step...",
    "signature": "eyJhbGciOiJSUzI1NiIsInR5cCI6..."  // For verification
}
```

### 6.4 Interleaved Thinking (Claude 4 Only)

```python
response = client.messages.create(
    model="claude-opus-4-5-20251101",
    max_tokens=8000,
    thinking={
        "type": "enabled",
        "budget_tokens": 20000  # Can exceed max_tokens with interleaved
    },
    extra_headers={
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    },
    messages=messages
)

# Response may contain multiple interleaved thinking/text blocks:
# [thinking, text, thinking, tool_use, thinking, text]
```

### 6.5 Tool Use with Extended Thinking

**Limitations:**
- `tool_choice: {"type": "any"}` - NOT supported with thinking
- `tool_choice: {"type": "tool", "name": "..."}` - NOT supported with thinking
- Only `tool_choice: {"type": "auto"}` and `{"type": "none"}` work

**Signature Requirement:**
When returning tool results with thinking enabled, include the signature:

```python
messages = [
    # Original thinking response preserved
    {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "I need to check...",
                "signature": "eyJhbGciOiJSUzI1NiIs..."  # REQUIRED
            },
            {
                "type": "tool_use",
                "id": "toolu_01ABC...",
                "name": "get_data",
                "input": {"query": "..."}
            }
        ]
    },
    # Tool result
    {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01ABC...",
                "content": "Result data"
            }
        ]
    }
]
```

---

## 7. Multi-modal Context

### 7.1 Image Handling

#### Token Cost Calculation

| Image Size | Approximate Tokens |
|------------|-------------------|
| Up to 1568px (long edge) | ~1,600 tokens |
| Larger images | Auto-scaled down |
| Multiple images | Sum of individual costs |

**Optimization Guidelines:**
- Resize to max 1.15 megapixels
- Keep both dimensions under 1568 pixels
- Use JPEG for photos, PNG for diagrams
- Cost: ~$4.80 per 1,000 images (Sonnet 4.5)

#### Detail Levels

```python
# Low detail - faster, fewer tokens
image_block = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64_data
    },
    "detail": "low"  # Fewer tokens, faster processing
}

# High detail - more tokens, better accuracy
image_block = {
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": base64_data
    },
    "detail": "high"  # More tokens, better for text/small details
}

# Auto (default) - Claude decides
image_block = {
    "type": "image",
    "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_data}
    # detail defaults to "auto"
}
```

### 7.2 PDF Processing

```python
import base64

# Load PDF
with open("document.pdf", "rb") as f:
    pdf_base64 = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_base64
                    }
                },
                {
                    "type": "text",
                    "text": "Summarize this document."
                }
            ]
        }
    ]
)
```

#### PDF Constraints

| Constraint | Limit |
|------------|-------|
| Max file size | 32 MB |
| Max pages | 100 per request |
| Encryption | Not supported (must be unencrypted) |
| Token cost | 1,500 - 3,000 per page |

### 7.3 Multiple Images

```python
# Up to 100 images via API (20 in claude.ai)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/1.jpg"}},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/2.jpg"}},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/3.jpg"}},
            {"type": "text", "text": "Compare these three images."}
        ]
    }
]
```

---

## Appendix A: Beta Headers Reference

| Feature | Beta Header | Version |
|---------|-------------|---------|
| Context Management | `context-management-2025-06-27` | June 2025 |
| Structured Outputs | `structured-outputs-2025-11-13` | Nov 2025 |
| Interleaved Thinking | `interleaved-thinking-2025-05-14` | May 2025 |
| Token-Efficient Tools | `token-efficient-tools-2025-02-19` | Feb 2025 |
| Advanced Tool Use | `advanced-tool-use-2025-11-20` | Nov 2025 |

---

## Appendix B: Token Counting API

```python
# Pre-calculate token usage
count = client.messages.count_tokens(
    model="claude-sonnet-4-5-20250929",
    system="You are a helpful assistant",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's the weather?"}
    ],
    tools=[weather_tool]  # Tools also counted
)

print(f"Input tokens: {count.input_tokens}")
# Use this to estimate costs and trigger context editing
```

---

## Sources

- [Context Editing Documentation](https://platform.claude.com/docs/en/build-with-claude/context-editing)
- [Extended Thinking Documentation](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Streaming Messages Documentation](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Memory Tool Documentation](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)
- [Tool Use Implementation Guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use)
- [Vision Documentation](https://platform.claude.com/docs/en/build-with-claude/vision)
- [PDF Support Documentation](https://platform.claude.com/docs/en/build-with-claude/pdf-support)
- [Messages API Reference](https://docs.claude.com/en/api/messages)
