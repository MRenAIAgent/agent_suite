# Agent Framework Documentation

Welcome to the documentation for the Agent Framework! This directory contains comprehensive guides, references, and examples to help you understand and use the framework effectively.

## Documentation Overview

### For New Users

Start here if you're new to the framework:

- [**Developer Quick Reference**](developer_quickstart.md) - Quick start guide with code examples
- [**Agent Framework Overview**](agent_framework.md) - Comprehensive overview of the framework architecture

### Technical Reference

For developers looking to understand the structure in detail:

- [**Class Diagram**](class_diagram.md) - Detailed class hierarchy and relationships
- [API Reference](../README.md) - Main README with API details

## Framework Architecture

The Agent Framework is built around a few core concepts:

1. **Agents** - Central entities that process user input and generate responses
2. **Tools** - Extensions that provide agents with capabilities for specific tasks
3. **Thinking Patterns** - Different reasoning approaches an agent can use
4. **Memory Management** - Systems for storing and retrieving conversation history
5. **Prompt Management** - Systems for formatting prompts and messages

## Key Components

The framework is organized into several key components:

```
agents/
├── base_classes/      # Abstract interfaces
│   ├── base_agent.py  # BaseAgent abstract class
│   └── base_think_pattern.py  # AgentThinkPattern abstract class
├── agent.py           # Main Agent implementation
├── prompt.py          # Prompt management
├── memory_manager.py  # Memory management
└── examples/          # Example implementations
```

## Example Usage

Here's a simple example of creating and using an agent:

```python
from agents.agent import Agent
from agents.prompt import PromptManager
from llm.openai.openai_llm import OpenAILLM

# Create components
llm = OpenAILLM.create_llm()
prompt_manager = PromptManager("You are a helpful assistant.")

# Create agent
agent = Agent(llm, prompt_manager)

# Use the agent
async def main():
    response = await agent.arun("Hello, can you help me?", model="gpt-3.5-turbo")
    print(response)

import asyncio
asyncio.run(main())
```

## Getting Help

If you encounter issues or have questions:

1. Check the documentation in this directory
2. Look at the example implementations in `agents/examples/`
3. Review the abstract interfaces in `agents/base_classes/` to understand expected behavior

## Contributing

When contributing to the framework:

1. Follow the established patterns and class hierarchies
2. Ensure new components implement the appropriate abstract interfaces
3. Add thorough documentation for new features
4. Include examples demonstrating the usage of new components 