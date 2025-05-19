# Agent Suite

A comprehensive toolkit for building AI agents with various capabilities.

## Project Structure

The Agent Suite is organized into the following components:

```
agent_suite/
├── agent_suite/
│   ├── agents/
│   │   ├── react/
│   │   │   ├── agent.py
│   │   │   ├── patterns.py
│   │   │   └── prompts.py
│   │   └── storage/
│   │       └── intermediate_storage.py
│   ├── tools/
│   │   ├── providers/
│   │   │   ├── basic/
│   │   │   ├── web/
│   │   │   └── data/
│   │   ├── adapters/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── selector.py
│   │   └── types.py
│   ├── integrations/
│   │   ├── mcp/
│   │   │   ├── client.py
│   │   │   ├── mock_server.py
│   │   │   ├── tools.py
│   │   │   └── wrapper.py
│   │   └── zapier/
│   ├── llm/
│   │   ├── litellm/
│   │   ├── anthropic/
│   │   ├── openai/
│   │   └── base.py
│   ├── memory/
│   │   └── stores/
│   ├── utils/
│   └── storage/
│       └── providers/
├── examples/
│   ├── simple_react/
│   ├── mcp_integration/
│   └── advanced/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── docs/
    ├── api/
    ├── examples/
    └── guides/
```

## Components

### Agents

The core agent implementations, including:

- **React Agent**: Implementation of the ReAct (Reasoning + Acting) pattern
- **Storage**: Components for storing intermediate results during agent execution

### Tools

Tools that agents can use to interact with the world:

- **Basic**: Simple utility tools like calculators
- **Web**: Tools for searching and retrieving web information
- **Data**: Tools for data analysis and processing

### Integrations

Integrations with external platforms:

- **MCP**: Model Composition Platform for accessing diverse tools
- **Zapier**: Integration with Zapier automations

### LLM

Large language model implementations:

- **LiteLLM**: Unified interface for multiple LLM providers
- **Anthropic**: Anthropic Claude models
- **OpenAI**: OpenAI models

## Getting Started

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from agent_suite.agents.react.agent import ReActAgent
from agent_suite.llm.litellm.litellm import LiteLLM

# Create LLM
llm = LiteLLM.create_llm()

# Create React agent
agent = ReActAgent(
    llm=llm,
    role="You are a helpful AI assistant.",
    task="Help the user with their request.",
    guide="Think step by step and use the available tools when necessary."
)

# Run the agent
response = agent.run("What's the weather like in New York today?")
print(response)
```

## Examples

Check out the `examples/` directory for more detailed examples of how to use the Agent Suite.

## Testing

Run the tests with:

```bash
pytest tests/
```

## License

MIT 