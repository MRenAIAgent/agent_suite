# Agent Framework Class Diagram

Below is a detailed class diagram showing the relationships between the main components in the agent framework.

```mermaid
classDiagram
    %% Abstract Classes
    class BaseAgent {
        <<abstract>>
        +__init__(llm, prompt_manager, tools, thinking_pattern, memory_manager, log_manager)
        +arun(user_input, model, **kwargs)* str
        +run(user_input, model, **kwargs)* str
        +think(user_input, model)* dict
        +handle_tool_calls(tool_calls)* list
        +handle_single_tool_call(tool_name, tool_input)* any
        +add_tool(tool)* void
        +remove_tool(tool_name)* bool
        +save_memory(context)* void
        +load_memory()* void
    }
    
    class AgentThinkPattern {
        <<abstract>>
        +name: str
        +description: str
        +metadata: dict
        +__init__(name, description)
        +process_input(input_data)* dict
        +format_thoughts(thoughts)* str
        +add_metadata(key, value) void
    }
    
    class Tool {
        +run(...)* any
        +arun(...)* any
        +convert_to_function_call() dict
    }
    
    %% Concrete Implementations
    class Agent {
        -llm: LLMBase
        -prompt_manager: PromptManager
        -memory_manager: MemoryManager
        -cache_manager: CacheManager
        -tools: List[Tool]
        -log_manager: LogManager
        -max_iterations: int
        -thinking_pattern: AgentThinkPattern
        -metadata: dict
        +__init__(llm, prompt_manager, tools, thinking_pattern, memory_manager, log_manager)
        +arun(user_input, model, **kwargs) str
        +run(user_input, model, **kwargs) str
        +think(user_input, model) dict
        +handle_tool_calls(tool_calls) list
        +handle_single_tool_call(tool_name, tool_input) any
        +add_tool(tool) void
        +remove_tool(tool_name) bool
        +save_memory(context) void
        +load_memory() void
        +log_interaction(user_input, response, model, **kwargs) void
    }
    
    class AnalyticalThinkPattern {
        +__init__()
        +process_input(input_data) dict
        +format_thoughts(thoughts) str
    }
    
    class CreativeThinkPattern {
        +__init__()
        +process_input(input_data) dict
        +format_thoughts(thoughts) str
    }
    
    class CriticalThinkPattern {
        +__init__()
        +process_input(input_data) dict
        +format_thoughts(thoughts) str
    }
    
    class LangChainToolAdapter {
        -action: any
        +__init__(action)
        +run(**kwargs) any
        +arun(**kwargs) any
    }
    
    class DriverLicenseExtractor {
        +license_number: str
        +last_name: str
        +first_name: str
        +middle_name: str
        +birth_date: str
        +gender: str
        +expiration_date: str
        +state: str
        +run(fields, text) dict
        +arun(fields, text) dict
    }
    
    %% Helper Classes
    class PromptManager {
        -system_prompt: str
        +__init__(system_prompt)
        +get_messages(user_input, history) list
    }
    
    class MemoryManager {
        -history: list
        +__init__()
        +add(message) void
        +get_history() list
        +clear() void
        +save_memory(context) void
        +load_memory() void
    }
    
    class LLMBase {
        <<abstract>>
        +chat_completion(model, messages, tools)* any
    }
    
    %% Relationships
    BaseAgent <|-- Agent
    AgentThinkPattern <|-- AnalyticalThinkPattern
    AgentThinkPattern <|-- CreativeThinkPattern
    AgentThinkPattern <|-- CriticalThinkPattern
    Tool <|-- LangChainToolAdapter
    Tool <|-- DriverLicenseExtractor
    
    Agent o-- LLMBase : uses
    Agent o-- PromptManager : uses
    Agent o-- MemoryManager : uses
    Agent o-- AgentThinkPattern : uses
    Agent o-- Tool : uses many
    
    %% Composition/Aggregation
    class CacheManager {
        +get(key) any
        +set(key, value) void
    }
    
    Agent *-- CacheManager : has
```

## Key Relationships Explained

1. **Inheritance Relationships**:
   - `Agent` inherits from `BaseAgent` - implements the abstract interface
   - `AnalyticalThinkPattern`, `CreativeThinkPattern`, and `CriticalThinkPattern` inherit from `AgentThinkPattern`
   - Both `LangChainToolAdapter` and `DriverLicenseExtractor` inherit from `Tool`

2. **Composition Relationships**:
   - `Agent` has a `CacheManager` (composition - lifecycle managed by Agent)
   - `Agent` uses various objects (aggregation - independent lifecycles):
     - `LLMBase` - for language model interaction
     - `PromptManager` - for message formatting
     - `MemoryManager` - for conversation history
     - `AgentThinkPattern` - for thought processing
     - `Tool` - for extending capabilities

## Usage Flow

1. Client code creates dependencies (LLM, PromptManager, Tools)
2. Client creates an Agent with these dependencies
3. Client calls agent.arun() or agent.run() with user input
4. Agent processes input, potentially using tools, thinking patterns, and memory
5. Agent returns response to client 