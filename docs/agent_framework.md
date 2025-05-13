# Agent Framework Documentation

## Architecture Overview

The agent framework is designed with a modular, extensible architecture that allows for the creation of AI agents with various capabilities. The framework follows object-oriented design principles with abstract base classes defining interfaces that concrete implementations must follow.

```
┌──────────────┐       ┌───────────────┐       ┌──────────────┐
│   BaseAgent  │◄──────│     Agent     │──────►│    Tools     │
└──────────────┘       └───────────────┘       └──────────────┘
        ▲                      │                       ▲
        │                      │                       │
        │                      ▼                       │
┌──────────────┐       ┌───────────────┐       ┌──────────────┐
│ThinkingPattern│◄─────│PromptManager  │       │ToolAdapters  │
└──────────────┘       └───────────────┘       └──────────────┘
        │                      │                       │
        │                      │                       │
        ▼                      ▼                       ▼
┌──────────────┐       ┌───────────────┐       ┌──────────────┐
│AnalyticalThink│       │MemoryManager │       │LangChainTools│
└──────────────┘       └───────────────┘       └──────────────┘
```

## Core Components

### 1. Agent System

#### BaseAgent (Abstract Class)
- **Purpose**: Defines the interface that all agent implementations must follow
- **Key Methods**: 
  - `__init__`: Initialize agent with LLM, prompt manager, tools, thinking pattern, memory
  - `arun`: Process user input asynchronously
  - `run`: Process user input synchronously
  - `think`: Process input using agent's thinking process
  - `handle_tool_calls`: Execute tools based on LLM-generated tool calls
  - Memory operations: `save_memory`, `load_memory`

#### Agent (Concrete Implementation)
- **Purpose**: Standard implementation of the BaseAgent interface
- **Key Features**:
  - Tool handling
  - Memory management
  - Thought processing
  - Logging interactions
  - Asynchronous and synchronous execution modes

### 2. Thinking Patterns

#### AgentThinkPattern (Abstract Class)
- **Purpose**: Defines how agents process information and structure reasoning
- **Key Methods**:
  - `process_input`: Transform input data according to thinking pattern
  - `format_thoughts`: Format thoughts according to pattern

#### Concrete Implementations
- **AnalyticalThinkPattern**: Structured, logical analysis of information
- **CreativeThinkPattern**: Generates novel connections and ideas
- **CriticalThinkPattern**: Evaluates information skeptically

### 3. Tool System

#### Tool (Base Class)
- **Purpose**: Defines interface for tools that agents can use
- **Key Methods**:
  - `run`: Synchronous execution
  - `arun`: Asynchronous execution
  - `convert_to_function_call`: Convert tool to format for LLM function calls

#### Tool Adapters
- **LangChainToolAdapter**: Adapts LangChain tools to work within the framework

### 4. Memory Management

#### MemoryManager
- **Purpose**: Manages agent's conversation history and state
- **Key Methods**:
  - `add`: Add new message to history
  - `get_history`: Retrieve conversation history
  - `save_memory`: Save memory to persistence layer
  - `load_memory`: Load memory from persistence layer

### 5. Prompt Management

#### PromptManager
- **Purpose**: Manages system prompts and message formatting
- **Key Methods**:
  - `get_messages`: Format messages for LLM with proper history and user input

## Call Flow Sequence

### Creating and Using an Agent

1. **Initialization**:
   ```python
   llm = OpenAILLM.create_llm()
   prompt_manager = PromptManager(system_prompt)
   tools = [SomeTool()]
   agent = Agent(llm, prompt_manager, tools)
   ```

2. **Processing User Input (Async)**:
   ```python
   response = await agent.arun(user_input, model="gpt-3.5-turbo")
   ```

3. **Internal Call Flow**:
   1. Agent calls `prompt_manager.get_messages()` to prepare formatted messages
   2. Agent calls `llm.chat_completion()` to get response
   3. If tool calls exist in response:
      - Agent calls `handle_tool_calls()` to execute tools
      - Results are added to messages
      - Process repeats until no more tool calls
   4. Agent updates memory with `memory_manager.add()`
   5. Agent logs interaction with `log_manager.log_interaction()`

### Tool Execution Flow

1. LLM generates tool calls in response
2. Agent extracts tool name and arguments
3. Agent finds corresponding tool in its tools list
4. Agent calls `tool.arun()` with extracted arguments
5. Results are added to conversation context for next LLM call

## Example Usage

### Basic Agent with Tool

```python
from agents.agent import Agent
from agents.prompt import PromptManager
from llm.openai.openai_llm import OpenAILLM
from tools.tool import Tool

# Create LLM
llm = OpenAILLM.create_llm()

# Create prompt manager with system prompt
system_prompt = "You are a helpful assistant..."
prompt_manager = PromptManager(system_prompt)

# Create tools
class CalculatorTool(Tool):
    def run(self, a: int, b: int, operation: str) -> int:
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        # etc.
    
    async def arun(self, a: int, b: int, operation: str) -> int:
        return self.run(a, b, operation)

# Create agent
tools = [CalculatorTool()]
agent = Agent(llm, prompt_manager, tools)

# Use agent
response = await agent.arun("What is 42 * 56?", model="gpt-3.5-turbo")
print(response)
```

## Best Practices

1. **Creating New Agents**:
   - Use the Agent class directly rather than subclassing when possible
   - Pass a PromptManager with appropriate system prompts

2. **Creating Tools**:
   - Implement both `run` and `arun` methods
   - Use descriptive Pydantic Fields with clear descriptions
   - Keep tools focused on a single responsibility

3. **Thinking Patterns**:
   - Select appropriate thinking patterns based on the task
   - Combine thinking patterns for complex reasoning

4. **Memory Management**:
   - For persistent agents, use save_memory and load_memory
   - Consider context window limitations when retrieving history

## Extension Points

The framework can be extended in several ways:

1. **New Tool Implementations**:
   - Implement the Tool interface for new capabilities
   - Create adapters for existing tool ecosystems

2. **Custom Thinking Patterns**:
   - Implement AgentThinkPattern for specialized reasoning approaches
   - Combine patterns for multi-step reasoning

3. **Memory Backends**:
   - Implement custom persistence layers for MemoryManager
   - Create specialized retrieval strategies

4. **Agent Implementations**:
   - Implement BaseAgent for specialized agent types
   - Create domain-specific agents for particular use cases 