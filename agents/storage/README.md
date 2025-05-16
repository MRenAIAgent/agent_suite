# Intermediate Step Storage for Agents

This package provides a flexible, configurable system for managing intermediate reasoning steps and tool results in agent executions. The implementation offers different storage strategies to optimize for different performance characteristics.

## Overview

Agent execution often involves multiple iterations of reasoning steps and tool calls. Efficiently managing these intermediate steps is critical for:

1. **Performance**: Reducing token usage by intelligently managing context
2. **Memory Efficiency**: Avoiding excessive memory consumption for long-running agents
3. **Flexibility**: Supporting different storage strategies for different use cases

## Components

### `IntermediateStorageBase`

Abstract base class defining the interface for storing and retrieving intermediate steps:

```python
class IntermediateStorageBase(ABC):
    @abstractmethod
    def add_step(self, step_type: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an intermediate step to storage."""
        
    @abstractmethod
    def add_tool_call(self, tool_name: str, tool_input: Any, tool_output: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a tool call result to storage."""
        
    @abstractmethod
    def get_steps(self, step_types: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve stored intermediate steps."""
        
    @abstractmethod
    def get_formatted_context(self, format_type: str = "default") -> str:
        """Get formatted context string for inclusion in LLM prompts."""
        
    @abstractmethod
    def get_as_messages(self, max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get stored steps as message objects for LLM context."""
```

### Implementations

#### `InMemoryIntermediateStorage`

Basic implementation that stores all steps in memory with no compression or summarization.

Characteristics:
- Full fidelity: keeps all steps exactly as they occurred
- Simple implementation
- Higher token usage for LLM context
- Suitable for shorter agent executions or when full detail is required

#### `TokenEfficientIntermediateStorage`

Optimized implementation focusing on reducing token usage:

Characteristics:
- Maintains detailed records of only the N most recent steps
- Summarizes older steps to preserve key information with reduced tokens
- Intelligently compresses tool outputs and other verbose content
- Suitable for long-running agents or when optimizing for token efficiency

### `IntermediateStepManager`

High-level manager class that simplifies working with the storage system:

```python
class IntermediateStepManager:
    def __init__(
        self, 
        storage_implementation: Optional[IntermediateStorageBase] = None,
        token_efficient: bool = False,
        max_detailed_steps: int = 5
    ):
        # ...
        
    def add_thought(self, thought: str): # ...
    def add_action(self, action: str, action_input: Any = None): # ...
    def add_observation(self, observation: Any): # ...
    def add_tool_result(self, tool_name: str, tool_input: Any, tool_output: Any): # ...
    def add_final_answer(self, final_answer: str): # ...
    def get_as_messages(self, max_tokens: Optional[int] = None): # ...
    # ...
```

## Integration with Agent Patterns

The storage system integrates with the agent pattern system through:

### `StorageAwareLLMExecutionPattern`

An execution pattern that uses the `IntermediateStepManager` to track and manage steps:

```python
class StorageAwareLLMExecutionPattern(LLMExecutionPattern):
    def __init__(
        self, 
        step_manager: Optional[IntermediateStepManager] = None,
        token_efficient: bool = False
    ):
        # ...
```

### `StorageAwareAgentPattern`

Agent patterns that use the storage-aware execution patterns:

```python
class StorageAwareReActAgentPattern(StorageAwareAgentPattern):
    # ReAct-specific implementation with storage awareness
```

## Using the Storage System

### Basic Usage with Factory Functions

The easiest way to get started is to use the factory functions:

```python
from agents.storage_aware_agent_pattern import create_token_efficient_react_pattern

# Create a token-efficient ReAct pattern
agent_pattern = create_token_efficient_react_pattern(
    max_detailed_steps=5,
    prompt_template=my_prompt_template
)
```

### Using with the Storage-Aware Agent

```python
from agents.storage_aware_react_agent import StorageAwareReActAgent

agent = StorageAwareReActAgent(
    llm=llm,
    role="You are an assistant...",
    task="Help the user...",
    guide="Always think step by step...",
    tools=my_tools,
    token_efficient=True,  # Enable token-efficient storage
    max_detailed_steps=3   # Keep only 3 most recent steps in detail
)

result = await agent.arun("What is the population of France squared?", "gpt-3.5-turbo")
```

## Benefits of Token-Efficient Storage

1. **Reduced Token Usage**: By summarizing older steps, the token count for context can be significantly reduced.

2. **Improved Performance**: Less token usage can result in faster LLM responses and lower costs.

3. **Better Scale**: Allows agents to handle more complex, multi-step tasks without exceeding context limits.

4. **Detailed Statistics**: Get insights into storage efficiency with the execution summary:
   ```python
   summary = agent.get_execution_summary()
   print(f"Compression ratio: {summary['compression_ratio']}")
   ```

## Customization

You can create custom storage implementations by subclassing `IntermediateStorageBase`:

```python
class MyCustomStorage(IntermediateStorageBase):
    # Implement the required methods with your custom logic
```

Or extend the existing implementations:

```python
class EnhancedTokenEfficientStorage(TokenEfficientIntermediateStorage):
    # Add your enhancements...
``` 