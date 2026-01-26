# Context Management System Design Document

## Overview

This document describes a context management system designed for multi-agent AI applications. The system optimizes for KV-cache efficiency, enables context sharing between agents, and provides configurable compression strategies.

---

## 1. Design Principles

| Principle | Description |
|-----------|-------------|
| **Subagent Isolation** | Each agent maintains focused context for its dedicated task |
| **Selective Sharing** | Shared context layers allow efficient cross-agent communication |
| **KV-Cache Optimization** | Stable prefix ordering maximizes cache reuse |
| **Structured Memory** | Hierarchical memory types for different retention needs |
| **Adaptive Compression** | Multiple algorithms triggered by configurable thresholds |

---

## 2. Context Window Structure

The context window is organized in a fixed order to maximize KV-cache hits:

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM Context Window                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  1. SYSTEM PROMPT (Stable - High Cache)                   │  │
│  │     ├── Agent persona and role definition                 │  │
│  │     ├── Tool definitions and schemas                      │  │
│  │     └── Behavioral constraints and guidelines             │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  2. MEMORY BLOCK (Semi-Stable - Medium Cache)             │  │
│  │     ├── 2.1 User/Group Profile                            │  │
│  │     ├── 2.2 Agent Memory (Summarized Facts)               │  │
│  │     ├── 2.3 Episode Memory (Relevant Past Episodes)       │  │
│  │     ├── 2.4 Knowledge Graph Context                       │  │
│  │     └── 2.5 File Memory References                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  3. CONVERSATION HISTORY (Dynamic - Low Cache)            │  │
│  │     ├── Previous user messages                            │  │
│  │     ├── Previous assistant responses                      │  │
│  │     └── Tool calls and results                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  4. CURRENT INPUT (Per-Request)                           │  │
│  │     ├── Current user message                              │  │
│  │     └── Retrieved context (RAG results)                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### KV-Cache Optimization Strategy

| Section | Stability | Cache Benefit | Update Frequency |
|---------|-----------|---------------|------------------|
| System Prompt | High | ~100% reuse | Per agent definition |
| Memory Block | Medium | ~70-90% reuse | Per session/periodic |
| Conversation History | Low | ~30-50% reuse | Per turn |
| Current Input | None | 0% reuse | Every request |

**Key Insight**: By placing stable content (System Prompt + Memory) before dynamic content (Conversation), we maximize the prefix that can be cached across requests.

---

## 3. Subagent Architecture

### 3.1 Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR AGENT                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Shared Context Pool                                     │    │
│  │  ├── User Profile (read: all agents)                    │    │
│  │  ├── Session Goals (read: all agents)                   │    │
│  │  └── Cross-Agent Memory (write: orchestrator only)      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │  Research   │     │   Code      │     │  Review     │       │
│  │  Subagent   │     │  Subagent   │     │  Subagent   │       │
│  ├─────────────┤     ├─────────────┤     ├─────────────┤       │
│  │ Own Context │     │ Own Context │     │ Own Context │       │
│  │ - Search    │     │ - Code Gen  │     │ - Analysis  │       │
│  │   history   │     │   history   │     │   history   │       │
│  │ - Findings  │     │ - File edits│     │ - Feedback  │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Context Isolation Model

Each subagent receives:
- **Inherited**: Shared context from parent (read-only)
- **Scoped**: Task-specific instructions and tools
- **Isolated**: Own conversation history (not polluted by sibling tasks)

```python
@dataclass
class SubagentContext:
    # Inherited from parent (read-only)
    shared_memory: SharedMemoryView

    # Scoped to this agent's task
    agent_config: AgentConfig
    task_instruction: str
    available_tools: List[Tool]

    # Isolated per subagent
    conversation_history: List[Message]
    local_memory: LocalMemory
```

### 3.3 Context Flow Between Agents

```
┌──────────────────────────────────────────────────────────────┐
│                    Context Flow Patterns                      │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Parent → Child (Delegation):                                │
│  ┌────────┐    shared_context    ┌────────┐                 │
│  │ Parent │ ─────────────────────▶│ Child  │                 │
│  │ Agent  │    task_instruction  │ Agent  │                 │
│  └────────┘                      └────────┘                 │
│                                                              │
│  Child → Parent (Return):                                    │
│  ┌────────┐     result_summary   ┌────────┐                 │
│  │ Child  │ ─────────────────────▶│ Parent │                 │
│  │ Agent  │   (compressed)       │ Agent  │                 │
│  └────────┘                      └────────┘                 │
│                                                              │
│  Sibling → Sibling (Via Parent):                            │
│  ┌────────┐                      ┌────────┐                 │
│  │ Agent A│ ──▶ Parent ──▶       │ Agent B│                 │
│  └────────┘    (filtered)        └────────┘                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Shared Context Design

### 4.1 Context Sharing Levels

```python
class ContextScope(Enum):
    GLOBAL = "global"      # Shared across all agents in system
    SESSION = "session"    # Shared within a user session
    AGENT = "agent"        # Specific to one agent type
    TASK = "task"          # Scoped to current task only
```

### 4.2 Shared Context Pool

```python
@dataclass
class SharedContextPool:
    """Central pool for shareable context across agents."""

    # Global scope - available to all agents
    user_profile: UserProfile
    group_profile: Optional[GroupProfile]
    system_knowledge: KnowledgeGraph

    # Session scope - current session only
    session_goals: List[str]
    session_facts: List[Fact]
    active_files: Dict[str, FileReference]

    # Access control
    permissions: Dict[str, ContextPermission]

    def get_view(self, agent_id: str, scope: ContextScope) -> SharedMemoryView:
        """Return filtered view based on agent permissions."""
        pass
```

### 4.3 Permission Model

| Context Type | Orchestrator | Subagent | Read/Write |
|--------------|--------------|----------|------------|
| User Profile | RW | R | Parent writes, children read |
| Session Goals | RW | R | Parent writes, children read |
| Agent Memory | RW | RW (own) | Each agent owns its memory |
| Knowledge Graph | RW | R | Parent writes, children query |
| Task Results | W | W | Anyone can contribute |

---

## 5. Memory Block Design

### 5.1 Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEMORY BLOCK                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5.1 USER/GROUP PROFILE                                  │    │
│  │      Persistent identity, preferences, permissions       │    │
│  │      Update: Rarely | Source: Auth + Learning           │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5.2 AGENT MEMORY (Summarized Facts)                     │    │
│  │      Learned facts, user preferences, past decisions     │    │
│  │      Update: Per session | Source: Conversation mining  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5.3 EPISODE MEMORY                                      │    │
│  │      Relevant past conversation episodes                 │    │
│  │      Update: Per query | Source: Retrieval (embedding)  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5.4 KNOWLEDGE GRAPH CONTEXT                             │    │
│  │      Structured relationships, entities, constraints     │    │
│  │      Update: Background | Source: Extraction + RAG      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  5.5 FILE MEMORY REFERENCES                              │    │
│  │      Pointers to extended context in files              │    │
│  │      Update: On demand | Source: File system            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Memory Type Definitions

```python
@dataclass
class UserProfile:
    """Persistent user identity and preferences."""
    user_id: str
    name: str
    preferences: Dict[str, Any]        # UI, communication style, etc.
    expertise_level: str               # beginner, intermediate, expert
    permissions: List[str]
    custom_instructions: Optional[str]

    # Token budget: ~200-500 tokens (stable)

@dataclass
class AgentMemory:
    """Summarized facts learned across sessions."""
    facts: List[Fact]                  # "User prefers TypeScript"
    decisions: List[Decision]          # Past choices and rationale
    corrections: List[Correction]      # Things user corrected

    # Token budget: ~500-1000 tokens (grows, then compressed)

@dataclass
class Fact:
    content: str
    confidence: float                  # 0.0 - 1.0
    source: str                        # conversation_id or "inferred"
    created_at: datetime
    last_used: datetime
    use_count: int

@dataclass
class EpisodeMemory:
    """Retrieved relevant past conversation episodes."""
    episodes: List[Episode]
    retrieval_query: str
    relevance_scores: List[float]

    # Token budget: ~500-2000 tokens (dynamic per query)

@dataclass
class Episode:
    episode_id: str
    summary: str                       # Compressed conversation summary
    key_points: List[str]              # Extracted important points
    outcome: str                       # What was accomplished
    timestamp: datetime

@dataclass
class KnowledgeGraphContext:
    """Relevant subgraph for current context."""
    entities: List[Entity]
    relationships: List[Relationship]
    constraints: List[Constraint]

    # Token budget: ~300-800 tokens (query-dependent)

@dataclass
class FileMemory:
    """References to extended context in files."""
    active_files: List[FileReference]
    file_summaries: Dict[str, str]     # path -> summary
    scratchpad_path: Optional[str]     # For reasoning overflow

    # Token budget: ~200-500 tokens (references only)

@dataclass
class FileReference:
    path: str
    purpose: str                       # "working_code", "notes", "data"
    last_modified: datetime
    summary: str                       # Brief content description
```

### 5.3 Memory Rendering Order

Memory is rendered in a specific order for KV-cache optimization:

```python
def render_memory_block(memory: MemoryBlock) -> str:
    """Render memory in stable order for KV-cache optimization."""
    sections = []

    # 1. User Profile (most stable)
    if memory.user_profile:
        sections.append(f"<user_profile>\n{memory.user_profile.render()}\n</user_profile>")

    # 2. Agent Memory (semi-stable)
    if memory.agent_memory and memory.agent_memory.facts:
        sections.append(f"<agent_memory>\n{memory.agent_memory.render()}\n</agent_memory>")

    # 3. Episode Memory (dynamic but ordered by relevance)
    if memory.episode_memory and memory.episode_memory.episodes:
        sections.append(f"<episode_memory>\n{memory.episode_memory.render()}\n</episode_memory>")

    # 4. Knowledge Graph (query-dependent)
    if memory.knowledge_graph:
        sections.append(f"<knowledge_context>\n{memory.knowledge_graph.render()}\n</knowledge_context>")

    # 5. File References (changes with active files)
    if memory.file_memory and memory.file_memory.active_files:
        sections.append(f"<file_memory>\n{memory.file_memory.render()}\n</file_memory>")

    return "\n\n".join(sections)
```

---

## 6. Context Compression System

### 6.1 Compression Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMPRESSION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context Input                                                   │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  THRESHOLD CHECK                                         │    │
│  │  Is context_tokens > threshold?                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │ Yes                                                      │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  COMPRESSION PIPELINE (Configurable Order)               │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ 1. Metadata │──▶│ 2. Dedup   │──▶│ 3. Rule    │      │    │
│  │  │   Extract   │  │   Remove    │  │   Based    │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │         │                                    │           │    │
│  │         │              ┌─────────────┐      │           │    │
│  │         └──────────────▶│ 4. Summary │◀─────┘           │    │
│  │                        │   (LLM)    │                   │    │
│  │                        └─────────────┘                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  Compressed Context                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Compression Strategies

#### Strategy 1: Metadata Extraction
Extract and store metadata separately from content.

```python
@dataclass
class MetadataCompressionConfig:
    enabled: bool = True
    extract_timestamps: bool = True
    extract_tool_schemas: bool = True      # Store schemas once, reference by name
    extract_file_contents: bool = True     # Move to file memory
    min_savings_threshold: float = 0.1     # Only if saves >10%

class MetadataCompressor:
    def compress(self, messages: List[Message]) -> CompressionResult:
        metadata = {}
        compressed_messages = []

        for msg in messages:
            # Extract tool call results to metadata
            if msg.tool_results:
                for result in msg.tool_results:
                    if len(result.content) > 500:
                        ref_id = self._store_metadata(result.content)
                        result.content = f"[See result:{ref_id}]"

            # Extract large code blocks to file memory
            if self._has_large_code_blocks(msg):
                msg = self._extract_to_file_memory(msg)

            compressed_messages.append(msg)

        return CompressionResult(compressed_messages, metadata)
```

#### Strategy 2: Deduplication
Remove redundant content across messages.

```python
@dataclass
class DeduplicationConfig:
    enabled: bool = True
    similarity_threshold: float = 0.9      # Cosine similarity for fuzzy match
    keep_strategy: str = "latest"          # "latest", "first", "most_relevant"
    dedupe_tool_results: bool = True       # Same tool, same args = dedupe
    dedupe_code_blocks: bool = True        # Identical code = reference first

class DeduplicationCompressor:
    def compress(self, messages: List[Message]) -> CompressionResult:
        seen_content = {}  # hash -> (index, content)
        deduplicated = []

        for i, msg in enumerate(messages):
            content_hash = self._semantic_hash(msg.content)

            if content_hash in seen_content:
                # Replace with reference
                original_idx = seen_content[content_hash][0]
                msg.content = f"[Same as message {original_idx}]"
            else:
                seen_content[content_hash] = (i, msg.content)

            deduplicated.append(msg)

        return CompressionResult(deduplicated, {})
```

#### Strategy 3: Rule-Based Compression
Apply deterministic rules for predictable compression.

```python
@dataclass
class RuleBasedConfig:
    enabled: bool = True
    rules: List[CompressionRule] = field(default_factory=list)

@dataclass
class CompressionRule:
    name: str
    condition: Callable[[Message], bool]
    transform: Callable[[Message], Message]
    priority: int = 0

# Built-in rules
DEFAULT_RULES = [
    CompressionRule(
        name="truncate_long_tool_output",
        condition=lambda m: m.role == "tool" and len(m.content) > 2000,
        transform=lambda m: m.with_content(m.content[:1000] + "\n...[truncated]..."),
        priority=10
    ),
    CompressionRule(
        name="remove_thinking_blocks",
        condition=lambda m: "<thinking>" in m.content,
        transform=lambda m: m.with_content(re.sub(r'<thinking>.*?</thinking>', '', m.content, flags=re.DOTALL)),
        priority=20
    ),
    CompressionRule(
        name="collapse_repeated_errors",
        condition=lambda m: m.is_error and m.error_count > 2,
        transform=lambda m: m.with_content(f"[Error repeated {m.error_count} times]: {m.content[:200]}"),
        priority=30
    ),
    CompressionRule(
        name="summarize_file_listings",
        condition=lambda m: m.tool_name == "glob" and m.content.count("\n") > 20,
        transform=lambda m: m.with_content(f"Found {m.content.count(chr(10))} files. First 10:\n" + "\n".join(m.content.split("\n")[:10])),
        priority=40
    ),
]
```

#### Strategy 4: LLM Summarization
Use LLM to intelligently compress conversations.

```python
@dataclass
class SummarizationConfig:
    enabled: bool = True
    model: str = "claude-3-haiku"           # Fast, cheap model
    target_ratio: float = 0.3               # Compress to 30% of original
    preserve_recent_turns: int = 3          # Keep last N turns uncompressed
    summarize_threshold_tokens: int = 4000  # Only summarize if above this

class SummarizationCompressor:
    SUMMARIZATION_PROMPT = """Summarize this conversation segment preserving:
1. Key decisions made
2. Important facts learned
3. Current task context
4. Any unresolved issues

Conversation:
{conversation}

Provide a concise summary (target: {target_tokens} tokens):"""

    async def compress(self, messages: List[Message]) -> CompressionResult:
        # Split: old messages to summarize, recent to preserve
        to_summarize = messages[:-self.config.preserve_recent_turns]
        to_preserve = messages[-self.config.preserve_recent_turns:]

        if self._count_tokens(to_summarize) < self.config.summarize_threshold_tokens:
            return CompressionResult(messages, {})

        # Generate summary
        summary = await self._generate_summary(to_summarize)

        # Create summary message
        summary_msg = Message(
            role="system",
            content=f"<conversation_summary>\n{summary}\n</conversation_summary>"
        )

        return CompressionResult([summary_msg] + to_preserve, {
            "summarized_turn_count": len(to_summarize),
            "original_tokens": self._count_tokens(to_summarize),
            "summary_tokens": self._count_tokens([summary_msg])
        })
```

### 6.3 Compression Pipeline Configuration

```python
@dataclass
class CompressionConfig:
    """Master configuration for compression pipeline."""

    # Thresholds
    soft_threshold: int = 50000        # Start light compression
    hard_threshold: int = 80000        # Aggressive compression
    critical_threshold: int = 100000   # Emergency compression

    # Strategy configs
    metadata: MetadataCompressionConfig = field(default_factory=MetadataCompressionConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    rule_based: RuleBasedConfig = field(default_factory=RuleBasedConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)

    # Pipeline order (executed in sequence)
    pipeline_order: List[str] = field(default_factory=lambda: [
        "metadata",        # First: extract metadata (cheap, fast)
        "deduplication",   # Second: remove duplicates (cheap, fast)
        "rule_based",      # Third: apply rules (cheap, fast)
        "summarization"    # Last: LLM summary (expensive, powerful)
    ])

    # Per-threshold strategy activation
    threshold_strategies: Dict[str, List[str]] = field(default_factory=lambda: {
        "soft": ["metadata", "deduplication"],
        "hard": ["metadata", "deduplication", "rule_based"],
        "critical": ["metadata", "deduplication", "rule_based", "summarization"]
    })

class CompressionPipeline:
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compressors = {
            "metadata": MetadataCompressor(config.metadata),
            "deduplication": DeduplicationCompressor(config.deduplication),
            "rule_based": RuleBasedCompressor(config.rule_based),
            "summarization": SummarizationCompressor(config.summarization),
        }

    async def compress(self, context: Context) -> Context:
        """Run compression pipeline based on current token count."""
        token_count = context.token_count

        # Determine which strategies to apply
        if token_count < self.config.soft_threshold:
            return context  # No compression needed
        elif token_count < self.config.hard_threshold:
            strategies = self.config.threshold_strategies["soft"]
        elif token_count < self.config.critical_threshold:
            strategies = self.config.threshold_strategies["hard"]
        else:
            strategies = self.config.threshold_strategies["critical"]

        # Apply strategies in order
        result = context
        for strategy_name in self.config.pipeline_order:
            if strategy_name in strategies:
                compressor = self.compressors[strategy_name]
                result = await compressor.compress(result)

                # Check if we're under threshold now
                if result.token_count < self.config.soft_threshold:
                    break

        return result
```

---

## 7. Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT MANAGEMENT SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PERSISTENCE LAYER                             │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │    │
│  │  │  User    │  │  Agent   │  │ Episode  │  │  Knowledge       │    │    │
│  │  │  Store   │  │  Memory  │  │  Store   │  │  Graph           │    │    │
│  │  │  (DB)    │  │  (DB)    │  │  (Vector)│  │  (Graph DB)      │    │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      MEMORY MANAGER                                  │    │
│  │  ┌─────────────────────────────────────────────────────────────┐   │    │
│  │  │  Load/Save │ Retrieval │ Update │ Token Budget Management   │   │    │
│  │  └─────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   SHARED CONTEXT POOL                                │    │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐        │    │
│  │  │ User Profile   │  │ Session State  │  │ Cross-Agent    │        │    │
│  │  │ (Global)       │  │ (Session)      │  │ Memory         │        │    │
│  │  └────────────────┘  └────────────────┘  └────────────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│         ┌────────────────────────────┼────────────────────────────┐         │
│         │                            │                            │         │
│         ▼                            ▼                            ▼         │
│  ┌─────────────┐            ┌─────────────┐              ┌─────────────┐   │
│  │ Orchestrator│            │  Subagent   │              │  Subagent   │   │
│  │   Agent     │            │     A       │              │     B       │   │
│  ├─────────────┤            ├─────────────┤              ├─────────────┤   │
│  │┌───────────┐│            │┌───────────┐│              │┌───────────┐│   │
│  ││  Context  ││            ││  Context  ││              ││  Context  ││   │
│  ││  Builder  ││            ││  Builder  ││              ││  Builder  ││   │
│  │└───────────┘│            │└───────────┘│              │└───────────┘│   │
│  │      │      │            │      │      │              │      │      │   │
│  │      ▼      │            │      ▼      │              │      ▼      │   │
│  │┌───────────┐│            │┌───────────┐│              │┌───────────┐│   │
│  ││Compression││            ││Compression││              ││Compression││   │
│  ││ Pipeline  ││            ││ Pipeline  ││              ││ Pipeline  ││   │
│  │└───────────┘│            │└───────────┘│              │└───────────┘│   │
│  │      │      │            │      │      │              │      │      │   │
│  │      ▼      │            │      ▼      │              │      ▼      │   │
│  │┌───────────┐│            │┌───────────┐│              │┌───────────┐│   │
│  ││   LLM     ││            ││   LLM     ││              ││   LLM     ││   │
│  ││  Request  ││            ││  Request  ││              ││  Request  ││   │
│  │└───────────┘│            │└───────────┘│              │└───────────┘│   │
│  └─────────────┘            └─────────────┘              └─────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. API Design

### 8.1 Core Interfaces

```python
class ContextManager:
    """Main entry point for context management."""

    async def create_session(self, user_id: str, config: SessionConfig) -> Session:
        """Create a new session with initialized context."""
        pass

    async def get_context(self, session: Session, agent_id: str) -> Context:
        """Build context for a specific agent in the session."""
        pass

    async def update_memory(self, session: Session, updates: MemoryUpdate) -> None:
        """Update memory (facts, episodes, etc.) for the session."""
        pass

    async def spawn_subagent(
        self,
        parent_session: Session,
        agent_config: AgentConfig,
        shared_context: List[str]  # Keys of context to share
    ) -> SubagentSession:
        """Spawn a subagent with selective context sharing."""
        pass

    async def merge_subagent_result(
        self,
        parent_session: Session,
        subagent_result: SubagentResult,
        merge_strategy: MergeStrategy
    ) -> None:
        """Merge subagent results back to parent context."""
        pass


class Context:
    """Immutable context snapshot for LLM request."""

    system_prompt: str
    memory_block: MemoryBlock
    conversation_history: List[Message]
    current_input: CurrentInput

    @property
    def token_count(self) -> int:
        """Total tokens in context."""
        pass

    def render(self) -> List[Dict]:
        """Render to LLM message format."""
        pass


class MemoryBlock:
    """All memory components for context."""

    user_profile: Optional[UserProfile]
    agent_memory: Optional[AgentMemory]
    episode_memory: Optional[EpisodeMemory]
    knowledge_graph: Optional[KnowledgeGraphContext]
    file_memory: Optional[FileMemory]

    def render(self) -> str:
        """Render memory block as string."""
        pass
```

### 8.2 Usage Example

```python
# Initialize
context_manager = ContextManager(config)

# Create session
session = await context_manager.create_session(
    user_id="user_123",
    config=SessionConfig(
        compression=CompressionConfig(soft_threshold=50000),
        memory_budget=MemoryBudget(
            user_profile=500,
            agent_memory=1000,
            episode_memory=2000,
            knowledge_graph=800,
            file_memory=500
        )
    )
)

# Main agent loop
async def agent_loop(session: Session, user_input: str):
    # Build context
    context = await context_manager.get_context(session, agent_id="main")

    # Check if we need subagent
    if needs_research(user_input):
        # Spawn research subagent with shared profile
        subagent = await context_manager.spawn_subagent(
            parent_session=session,
            agent_config=RESEARCH_AGENT_CONFIG,
            shared_context=["user_profile", "session_goals"]
        )

        # Run subagent (isolated context)
        result = await run_subagent(subagent, task="research X")

        # Merge result back (compressed)
        await context_manager.merge_subagent_result(
            parent_session=session,
            subagent_result=result,
            merge_strategy=MergeStrategy.SUMMARY_ONLY
        )

    # Make LLM request with built context
    response = await llm.complete(context.render())

    # Update memory with new facts
    await context_manager.update_memory(session, MemoryUpdate(
        new_facts=extract_facts(response),
        episode_update=create_episode(user_input, response)
    ))

    return response
```

---

## 9. Token Budget Management

### 9.1 Budget Allocation

```python
@dataclass
class TokenBudget:
    """Token budget allocation across context sections."""

    total_limit: int = 100000           # Model's context window

    # Fixed allocations
    system_prompt: int = 2000           # ~2% - stable
    current_input: int = 4000           # ~4% - per request
    response_reserve: int = 4000        # ~4% - for model output

    # Dynamic allocations (remaining ~90%)
    memory_block: int = 10000           # ~10%
    conversation_history: int = 80000   # ~80%

    # Memory sub-budgets (within memory_block)
    memory_allocation: Dict[str, int] = field(default_factory=lambda: {
        "user_profile": 500,
        "agent_memory": 2000,
        "episode_memory": 4000,
        "knowledge_graph": 2000,
        "file_memory": 1500
    })

    def available_for_conversation(self) -> int:
        """Calculate remaining budget for conversation history."""
        fixed = self.system_prompt + self.current_input + self.response_reserve
        return self.total_limit - fixed - self.memory_block
```

### 9.2 Dynamic Reallocation

```python
class TokenBudgetManager:
    """Dynamically manage token budget based on context needs."""

    def reallocate(self, context: Context, priorities: Dict[str, int]) -> TokenBudget:
        """Reallocate budget based on current needs and priorities."""
        budget = self.base_budget.copy()

        # If episode memory is highly relevant, give it more space
        if context.episode_memory and context.episode_memory.max_relevance > 0.9:
            budget.memory_allocation["episode_memory"] += 1000
            budget.memory_allocation["agent_memory"] -= 500
            budget.memory_allocation["knowledge_graph"] -= 500

        # If conversation is short, give more to memory
        if context.conversation_token_count < 10000:
            surplus = 10000 - context.conversation_token_count
            budget.memory_allocation["episode_memory"] += surplus // 2
            budget.memory_allocation["knowledge_graph"] += surplus // 2

        return budget
```

---

## 10. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Context data models (Message, MemoryBlock, etc.)
- [ ] Token counting utilities
- [ ] Context rendering pipeline
- [ ] Basic persistence (SQLite/PostgreSQL)

### Phase 2: Memory System
- [ ] User/Group profile management
- [ ] Agent memory (fact storage)
- [ ] Episode memory with vector retrieval
- [ ] Knowledge graph integration
- [ ] File memory tools

### Phase 3: Subagent Architecture
- [ ] Subagent spawning
- [ ] Context isolation
- [ ] Shared context pool
- [ ] Result merging

### Phase 4: Compression Pipeline
- [ ] Metadata extraction
- [ ] Deduplication
- [ ] Rule-based compression
- [ ] LLM summarization
- [ ] Threshold-based triggering

### Phase 5: Optimization
- [ ] KV-cache optimization verification
- [ ] Token budget tuning
- [ ] Performance benchmarking
- [ ] Cost analysis

---

## Appendix A: Configuration Examples

### Minimal Configuration
```python
config = ContextConfig(
    compression=CompressionConfig(
        soft_threshold=50000,
        pipeline_order=["deduplication", "rule_based"]
    ),
    memory=MemoryConfig(
        enable_episode_memory=False,
        enable_knowledge_graph=False
    )
)
```

### Full-Featured Configuration
```python
config = ContextConfig(
    compression=CompressionConfig(
        soft_threshold=40000,
        hard_threshold=70000,
        critical_threshold=90000,
        summarization=SummarizationConfig(
            model="claude-3-haiku",
            preserve_recent_turns=5
        )
    ),
    memory=MemoryConfig(
        enable_user_profile=True,
        enable_agent_memory=True,
        enable_episode_memory=True,
        enable_knowledge_graph=True,
        enable_file_memory=True,
        episode_retrieval_count=5,
        knowledge_graph_depth=2
    ),
    subagent=SubagentConfig(
        max_concurrent=3,
        default_shared_context=["user_profile"],
        result_merge_strategy="summary"
    )
)
```
