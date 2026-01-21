# Context Management System Design

A context management system for multi-agent AI applications optimized for KV-cache efficiency, context sharing, and adaptive compression.

---

## 1. Design Principles

| Principle | Goal |
|-----------|------|
| **Subagent Isolation** | Each agent maintains focused context for its task—no pollution from sibling tasks |
| **Selective Sharing** | Parent agents share relevant context (profiles, goals) with children |
| **KV-Cache Optimization** | Stable prefix ordering maximizes cache reuse (up to 10x cost savings) |
| **Structured Memory** | Hierarchical memory types for different retention needs |
| **Adaptive Compression** | Threshold-triggered compression preserves recent context while compacting old |

---

## 2. Context Window Structure

Context is ordered from **most stable** to **most dynamic** to maximize KV-cache hits:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM CONTEXT WINDOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. SYSTEM PROMPT                        [Stable · High Cache] │  │
│  │     • Agent persona and role definition                        │  │
│  │     • Tool definitions and schemas                             │  │
│  │     • Behavioral constraints and guidelines                    │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  2. MEMORY BLOCK                      [Semi-Stable · Med Cache] │  │
│  │     ┌─────────────────────────────────────────────────────┐   │  │
│  │     │ 2.1 User/Group Profile                               │   │  │
│  │     │     Identity, preferences, permissions               │   │  │
│  │     └─────────────────────────────────────────────────────┘   │  │
│  │     ┌─────────────────────────────────────────────────────┐   │  │
│  │     │ 2.2 Agent Memory (Summarized Facts)                  │   │  │
│  │     │     Learned facts, past decisions, corrections       │   │  │
│  │     └─────────────────────────────────────────────────────┘   │  │
│  │     ┌─────────────────────────────────────────────────────┐   │  │
│  │     │ 2.3 Episode Memory                                   │   │  │
│  │     │     Relevant past conversation summaries             │   │  │
│  │     └─────────────────────────────────────────────────────┘   │  │
│  │     ┌─────────────────────────────────────────────────────┐   │  │
│  │     │ 2.4 Knowledge Graph Context                          │   │  │
│  │     │     Relevant entities and relationships              │   │  │
│  │     └─────────────────────────────────────────────────────┘   │  │
│  │     ┌─────────────────────────────────────────────────────┐   │  │
│  │     │ 2.5 File Memory References                           │   │  │
│  │     │     Paths to extended context in files               │   │  │
│  │     └─────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  3. CONVERSATION HISTORY                 [Dynamic · Low Cache] │  │
│  │     • Previous user messages                                   │  │
│  │     • Previous assistant responses                             │  │
│  │     • Tool calls and results                                   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  4. CURRENT INPUT                           [Per-Request · 0%] │  │
│  │     • Current user message                                     │  │
│  │     • Retrieved context (RAG results)                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Cache Efficiency by Section

| Section | Update Frequency | Cache Reuse |
|---------|------------------|-------------|
| System Prompt | Per agent definition | ~100% |
| Memory Block | Per session | ~70-90% |
| Conversation History | Per turn | ~30-50% |
| Current Input | Every request | 0% |

**Key Rule**: Never mutate content—always append. Modifying earlier content invalidates the entire cache prefix.

---

## 3. Subagent Architecture

### Agent Hierarchy

```
┌──────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR AGENT                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Shared Context Pool                                        │  │
│  │  • User Profile        (read: all agents)                  │  │
│  │  • Session Goals       (read: all agents)                  │  │
│  │  • Cross-Agent Results (write: orchestrator only)          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                               │                                   │
│            ┌──────────────────┼──────────────────┐               │
│            ▼                  ▼                  ▼               │
│     ┌───────────┐      ┌───────────┐      ┌───────────┐         │
│     │  Research │      │   Code    │      │  Review   │         │
│     │  Agent    │      │   Agent   │      │  Agent    │         │
│     ├───────────┤      ├───────────┤      ├───────────┤         │
│     │ Isolated  │      │ Isolated  │      │ Isolated  │         │
│     │ Context:  │      │ Context:  │      │ Context:  │         │
│     │ • Search  │      │ • Edits   │      │ • Analysis│         │
│     │   history │      │   history │      │   history │         │
│     │ • Findings│      │ • Files   │      │ • Feedback│         │
│     └───────────┘      └───────────┘      └───────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

### Context Flow

```
Parent → Child (Delegation)
┌────────┐  shared_context + task  ┌────────┐
│ Parent │ ────────────────────────▶ │ Child  │
└────────┘                          └────────┘

Child → Parent (Return)
┌────────┐    result_summary       ┌────────┐
│ Child  │ ────────────────────────▶ │ Parent │
└────────┘    (compressed)          └────────┘
```

### Isolation Benefits

| Aspect | Without Isolation | With Isolation |
|--------|-------------------|----------------|
| Context Size | 150K+ tokens (all history) | 30-50K tokens per agent |
| Focus | Diluted across tasks | Dedicated to single task |
| Cost | High (full context each call) | Lower (relevant context only) |
| Errors | Cross-contamination risk | Contained per agent |

---

## 4. Context Sharing Model

### Sharing Scopes

```python
class ContextScope(Enum):
    GLOBAL = "global"      # System-wide (user identity)
    SESSION = "session"    # Current session (goals, state)
    AGENT = "agent"        # Agent-specific (own memory)
    TASK = "task"          # Current task only (temporary)
```

### Shared Context Pool

```python
@dataclass
class SharedContextPool:
    # Global scope
    user_profile: UserProfile

    # Session scope
    session_goals: List[str]
    session_facts: List[Fact]

    def get_view(self, agent_id: str) -> ContextView:
        """Return read-only view for subagent."""
        return ContextView(
            user_profile=self.user_profile,      # Always shared
            goals=self.session_goals,             # Always shared
            facts=self._filter_relevant(agent_id) # Filtered
        )
```

### Permission Matrix

| Context Type | Orchestrator | Subagent |
|--------------|--------------|----------|
| User Profile | Read/Write | Read |
| Session Goals | Read/Write | Read |
| Agent Memory | Read/Write | Read/Write (own only) |
| Task Results | Write | Write |

---

## 5. Memory Block Design

### 5.1 User/Group Profile

Persistent identity and preferences.

```python
@dataclass
class UserProfile:
    user_id: str
    name: str
    preferences: Dict[str, Any]     # Communication style, etc.
    expertise_level: str            # beginner | intermediate | expert
    custom_instructions: str        # User-provided guidelines

    # Token budget: ~200-500 tokens
```

### 5.2 Agent Memory (Summarized Facts)

Learned facts accumulated across sessions.

```python
@dataclass
class AgentMemory:
    facts: List[Fact]               # "User prefers TypeScript"
    decisions: List[Decision]       # Past choices and rationale
    corrections: List[Correction]   # Things user corrected

    # Token budget: ~500-2000 tokens

@dataclass
class Fact:
    content: str                    # The fact itself
    confidence: float               # 0.0 - 1.0
    source: str                     # Where learned
    last_used: datetime             # For relevance scoring
    use_count: int                  # Usage frequency
```

### 5.3 Episode Memory

Relevant past conversation episodes retrieved by similarity.

```python
@dataclass
class EpisodeMemory:
    episodes: List[Episode]         # Retrieved relevant episodes

    # Token budget: ~500-2000 tokens (dynamic)

@dataclass
class Episode:
    episode_id: str
    summary: str                    # Compressed conversation
    key_points: List[str]           # Important extracted points
    outcome: str                    # What was accomplished
    timestamp: datetime
    embedding: List[float]          # For similarity search
```

### 5.4 Knowledge Graph Context

Relevant subgraph for current query.

```python
@dataclass
class KnowledgeGraphContext:
    entities: List[Entity]          # People, concepts, files
    relationships: List[Relationship]

    # Token budget: ~300-800 tokens

@dataclass
class Entity:
    name: str
    type: str                       # person | concept | file | etc.
    properties: Dict[str, Any]

@dataclass
class Relationship:
    source: str                     # Entity name
    target: str                     # Entity name
    relation: str                   # "uses", "depends_on", etc.
```

### 5.5 File Memory References

Pointers to extended context stored in files.

```python
@dataclass
class FileMemory:
    active_files: List[FileRef]
    scratchpad: Optional[str]       # Path to scratch space

    # Token budget: ~200-500 tokens (references only)

@dataclass
class FileRef:
    path: str
    purpose: str                    # "working_code" | "notes" | "data"
    summary: str                    # Brief content description
```

### Memory Rendering Example

```xml
<memory>
  <user_profile>
    Name: Alice Chen
    Expertise: Senior developer
    Preferences: Concise responses, TypeScript, functional style
  </user_profile>

  <agent_memory>
    Facts:
    - User's project uses React 18 with Next.js (confidence: 0.95)
    - User prefers Tailwind CSS over styled-components (confidence: 0.90)
    - Authentication uses NextAuth.js (confidence: 0.85)
  </agent_memory>

  <episode_memory>
    Relevant past conversation (2024-01-15):
    - Discussed API rate limiting implementation
    - Decided on token bucket algorithm
    - Outcome: Implemented in src/middleware/rateLimit.ts
  </episode_memory>

  <knowledge_graph>
    Entities: [User, Project, RateLimiter, API]
    Relations: User -owns-> Project, Project -contains-> RateLimiter
  </knowledge_graph>

  <file_memory>
    Active files:
    - src/middleware/rateLimit.ts (rate limiting logic)
    - docs/api-design.md (API documentation)
    Scratchpad: /tmp/workspace/scratch.md
  </file_memory>
</memory>
```

---

## 6. Context Compression

### Compression Pipeline

Compression activates when context exceeds thresholds, applied in order from cheapest to most expensive:

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPRESSION PIPELINE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context Tokens                                                  │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  CHECK THRESHOLD                                         │    │
│  │  < 50K tokens? → No compression needed                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │ Exceeds threshold                                        │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  STAGE 1: DEDUPLICATION                    [Cost: Free] │    │
│  │  • Remove duplicate tool results                        │    │
│  │  • Collapse repeated content → references               │    │
│  │  • Deduplicate identical code blocks                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  STAGE 2: RULE-BASED                       [Cost: Free] │    │
│  │  • Truncate tool outputs > 2000 chars                   │    │
│  │  • Remove <thinking> blocks                             │    │
│  │  • Collapse repeated errors                             │    │
│  │  • Summarize long file listings                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  STAGE 3: FILE OFFLOAD                 [Cost: Minimal] │    │
│  │  • Move large content to files                          │    │
│  │  • Keep only metadata + file path in context            │    │
│  │  • Content retrievable on demand                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  STAGE 4: SUMMARIZATION               [Cost: LLM Call] │    │
│  │  • Keep last N turns raw                                │    │
│  │  • Summarize older turns with fast model                │    │
│  │  • Preserve: decisions, facts, unresolved issues        │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  Compressed Context                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Threshold Configuration

```python
@dataclass
class CompressionConfig:
    # Thresholds (for 200K context window)
    soft_threshold: int = 50_000      # Start light compression
    hard_threshold: int = 80_000      # Aggressive compression
    critical_threshold: int = 100_000 # Emergency compression

    # What to apply at each threshold
    strategies = {
        "soft": ["deduplication"],
        "hard": ["deduplication", "rule_based", "file_offload"],
        "critical": ["deduplication", "rule_based", "file_offload", "summarization"]
    }

    # Summarization settings
    preserve_recent_turns: int = 5    # Never compress last N turns
    summarization_model: str = "claude-3-haiku"  # Fast, cheap
```

### Compression Strategies

#### Stage 1: Deduplication

```python
def deduplicate(messages: List[Message]) -> List[Message]:
    seen = {}
    result = []

    for i, msg in enumerate(messages):
        hash = semantic_hash(msg.content)
        if hash in seen:
            # Replace with reference
            msg.content = f"[Same as turn {seen[hash]}]"
        else:
            seen[hash] = i
        result.append(msg)

    return result

# Savings: 10-30%
```

#### Stage 2: Rule-Based

```python
COMPRESSION_RULES = [
    # Truncate long tool outputs
    Rule(
        condition=lambda m: m.role == "tool" and len(m.content) > 2000,
        transform=lambda m: m.content[:1000] + "\n...[truncated]..."
    ),

    # Remove thinking blocks
    Rule(
        condition=lambda m: "<thinking>" in m.content,
        transform=lambda m: re.sub(r'<thinking>.*?</thinking>', '', m.content)
    ),

    # Collapse repeated errors
    Rule(
        condition=lambda m: m.is_error and m.repeat_count > 2,
        transform=lambda m: f"[Error repeated {m.repeat_count}x]: {m.content[:200]}"
    ),

    # Summarize file listings
    Rule(
        condition=lambda m: m.tool == "glob" and m.content.count('\n') > 20,
        transform=lambda m: f"Found {m.line_count} files:\n" + first_10_lines(m)
    ),
]

# Savings: 20-35%
```

#### Stage 3: File Offload

```python
def offload_to_file(content: str, context_id: str) -> str:
    """Move large content to file, return reference."""
    if len(content) < 2000:
        return content

    # Save to file
    path = f"/workspace/.context/{context_id}/{uuid4()}.txt"
    write_file(path, content)

    # Return metadata reference
    return f"[Content saved to {path}] Preview: {content[:200]}..."

# Savings: 20-40%
# Content retrievable via file_read tool
```

#### Stage 4: Summarization

```python
async def summarize_old_turns(
    messages: List[Message],
    preserve_recent: int = 5
) -> List[Message]:
    """Summarize old turns, preserve recent ones."""

    to_summarize = messages[:-preserve_recent]
    to_preserve = messages[-preserve_recent:]

    if not to_summarize:
        return messages

    # Generate summary with fast model
    summary = await llm.complete(
        model="claude-3-haiku",
        prompt=f"""Summarize this conversation preserving:
        1. Key decisions made
        2. Important facts learned
        3. Current task context
        4. Unresolved issues

        Conversation:
        {render_messages(to_summarize)}
        """
    )

    # Replace old turns with summary
    summary_msg = Message(
        role="system",
        content=f"<conversation_summary>\n{summary}\n</conversation_summary>"
    )

    return [summary_msg] + to_preserve

# Savings: 60-80%
```

### Compression Decision Flow

```
START: Check context size
│
├─ < 50K tokens → No compression
│
├─ 50K - 80K tokens (Soft)
│   └─ Apply: Deduplication only
│
├─ 80K - 100K tokens (Hard)
│   └─ Apply: Dedup + Rules + File Offload
│
└─ > 100K tokens (Critical)
    └─ Apply: All strategies including Summarization
```

---

## 7. API Overview

### Core Classes

```python
class ContextManager:
    """Main entry point."""

    async def create_session(self, user_id: str) -> Session:
        """Create session with user context loaded."""

    async def build_context(self, session: Session, agent_id: str) -> Context:
        """Build context for specific agent."""

    async def spawn_subagent(
        self,
        parent: Session,
        agent_config: AgentConfig,
        shared: List[str]  # ["user_profile", "session_goals"]
    ) -> SubagentSession:
        """Spawn subagent with selective sharing."""

    async def merge_result(
        self,
        parent: Session,
        result: SubagentResult
    ) -> None:
        """Merge subagent result back (compressed)."""


class Context:
    """Immutable context for LLM request."""

    system_prompt: str
    memory: MemoryBlock
    conversation: List[Message]
    current_input: str

    @property
    def token_count(self) -> int: ...

    def render(self) -> List[Dict]: ...
```

### Usage Example

```python
# Initialize
ctx_manager = ContextManager(config)
session = await ctx_manager.create_session(user_id="user_123")

# Main agent loop
async def handle_request(user_input: str):
    # Build context (auto-compressed if needed)
    context = await ctx_manager.build_context(session, agent_id="main")

    # Need specialized work? Spawn subagent
    if needs_research(user_input):
        subagent = await ctx_manager.spawn_subagent(
            parent=session,
            agent_config=RESEARCH_AGENT,
            shared=["user_profile", "session_goals"]
        )

        # Subagent has isolated context
        result = await run_agent(subagent, task="research X")

        # Merge summary back to parent
        await ctx_manager.merge_result(session, result)

    # Make LLM call
    response = await llm.complete(context.render())

    # Update session
    session.add_turn(user_input, response)

    return response
```

---

## 8. Token Budget Guidelines

### Allocation (200K window)

| Section | Budget | % |
|---------|--------|---|
| System Prompt | 2,000 | 1% |
| Memory Block | 8,000 | 4% |
| Conversation History | 150,000 | 75% |
| Current Input | 4,000 | 2% |
| Response Reserve | 36,000 | 18% |

### Memory Sub-Budget

| Memory Type | Tokens | Notes |
|-------------|--------|-------|
| User Profile | 500 | Stable, rarely changes |
| Agent Memory | 2,000 | Grows, then compressed |
| Episode Memory | 3,000 | Dynamic per query |
| Knowledge Graph | 1,500 | Query-dependent |
| File References | 1,000 | Paths + summaries |

---

## 9. Implementation Phases

### Phase 1: Foundation
- [ ] Context data models
- [ ] Token counting
- [ ] Basic rendering pipeline
- [ ] Simple persistence (SQLite)

### Phase 2: Memory System
- [ ] User profile management
- [ ] Agent memory (fact storage)
- [ ] Episode memory with embedding retrieval
- [ ] File memory tool

### Phase 3: Subagent Support
- [ ] Subagent spawning
- [ ] Context isolation
- [ ] Shared context pool
- [ ] Result merging

### Phase 4: Compression
- [ ] Deduplication
- [ ] Rule-based compression
- [ ] File offloading
- [ ] LLM summarization
- [ ] Threshold triggers

### Phase 5: Optimization
- [ ] KV-cache verification
- [ ] Performance benchmarking
- [ ] Cost analysis

---

## 10. Summary

| Component | Purpose | Key Benefit |
|-----------|---------|-------------|
| **Ordered Structure** | System → Memory → History → Input | 10x cost savings via KV-cache |
| **Subagent Isolation** | Dedicated context per task | No context dilution |
| **Shared Pool** | Selective context sharing | Efficient cross-agent data |
| **Memory Block** | Profile + Facts + Episodes + KG + Files | Structured long-term memory |
| **Compression Pipeline** | Dedup → Rules → Offload → Summarize | Handles unbounded conversations |
