# Agent Framework: Developer Quick Reference

This guide provides a quick overview of how to use the agent framework for common tasks.

## Setup and Installation

```bash
# Clone the repository
git clone <repository-url>
cd agent_suite

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Common Usage Patterns

### 1. Creating a Basic Agent

```python
from agents.agent import Agent
from agents.prompt import PromptManager
from llm.openai.openai_llm import OpenAILLM

# Initialize LLM
llm = OpenAILLM.create_llm()

# Create a prompt manager with system instructions
system_prompt = "You are a helpful AI assistant..."
prompt_manager = PromptManager(system_prompt)

# Create the agent
agent = Agent(llm, prompt_manager)

# Use the agent
async def main():
    response = await agent.arun("Tell me about AI agents", model="gpt-3.5-turbo")
    print(response)

# Run the async function
import asyncio
asyncio.run(main())
```

### 2. Creating a Tool

```python
from tools.tool import Tool
from pydantic import Field
from typing import Dict, Any

class WeatherTool(Tool):
    """Tool for getting weather information."""
    
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    date: str = Field(description="The date to get weather for, e.g. 2023-07-25")
    
    def run(self, location: str, date: str) -> Dict[str, Any]:
        """Get weather for a location on a specific date."""
        # Implementation would connect to a weather API
        return {
            "location": location,
            "date": date,
            "temperature": 72,
            "conditions": "Sunny"
        }
    
    async def arun(self, location: str, date: str) -> Dict[str, Any]:
        """Async version of run method."""
        return self.run(location=location, date=date)
```

### 3. Using Tools with an Agent

```python
# Create tools
weather_tool = WeatherTool()

# Create agent with tools
agent = Agent(llm, prompt_manager, tools=[weather_tool])

# The agent can now use the weather tool
response = await agent.arun(
    "What's the weather like in San Francisco today?", 
    model="gpt-3.5-turbo"
)
```

### 4. Using a Thinking Pattern

```python
from agents.base_classes.base_think_pattern import AnalyticalThinkPattern

# Create a thinking pattern
thinking_pattern = AnalyticalThinkPattern()

# Create agent with thinking pattern
agent = Agent(
    llm=llm,
    prompt_manager=prompt_manager,
    tools=[weather_tool],
    thinking_pattern=thinking_pattern
)

# Get thoughts using the thinking pattern
thoughts = await agent.think(
    "How should I approach solving climate change?",
    model="gpt-4"
)
```

### 5. Memory Management

```python
# Agent memory is managed automatically
agent = Agent(llm, prompt_manager)

# First interaction
await agent.arun("My name is John", model="gpt-3.5-turbo")

# Second interaction (agent will remember previous context)
await agent.arun("What's my name?", model="gpt-3.5-turbo")

# Save and load memory
await agent.save_memory("john_session")
await agent.load_memory("john_session")
```

## Common Patterns

### 1. Tool-Using Agent Pattern

```python
tools = [WeatherTool(), SearchTool(), CalculatorTool()]
agent = Agent(llm, prompt_manager, tools=tools)

# Agent will automatically use appropriate tools based on the query
response = await agent.arun(
    "I need to know if I should bring an umbrella to my meeting in Seattle tomorrow, and also calculate how long it will take to drive there if it's 100 miles away and I drive 55 mph.",
    model="gpt-4"
)
```

### 2. Multi-Step Reasoning Pattern

```python
from agents.base_classes.base_think_pattern import AnalyticalThinkPattern

thinking_pattern = AnalyticalThinkPattern()
agent = Agent(llm, prompt_manager, thinking_pattern=thinking_pattern)

# Agent will break down complex problems step by step
thoughts = await agent.think(
    "How can we improve city transportation systems?",
    model="gpt-4"
)

# Format the thoughts according to the thinking pattern
formatted_thoughts = thinking_pattern.format_thoughts(thoughts)
```

### 3. Field Extraction Pattern

```python
from agents.examples.fields_extraction_agent import DriverLicenseExtractor

# Create tool for extraction
extractor = DriverLicenseExtractor()

# Create agent with extraction tool
agent = Agent(llm, prompt_manager, tools=[extractor])

# Extract fields from text
license_text = """
DRIVER LICENSE
NAME: DOE, JOHN M
DOB: 01/01/1980
EXP: 01/01/2025
LIC#: D12345678
STATE: CA
"""

response = await agent.arun(
    f"Extract all fields from this driver's license: {license_text}",
    model="gpt-3.5-turbo"
)
```

## Troubleshooting

### Common Issues

1. **Tool not being used**: Ensure tool descriptions are clear and the user prompt explicitly requires information that would trigger tool use.

2. **Memory issues**: If the agent doesn't seem to remember context, check that MemoryManager is functioning correctly.

3. **Import errors**: Ensure your project structure matches the expected imports or adjust imports to match your structure.

4. **LLM API errors**: Verify API keys are correctly set in your .env file.

### Debugging

Enable debug logging to see what's happening under the hood:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now create and use your agent
agent = Agent(...)
```

## API Reference Quick Links

- [Agent](../agents/agent.py) - Main agent implementation
- [BaseAgent](../agents/base_classes/base_agent.py) - Abstract interface for agents
- [Tool](../tools/tool.py) - Base class for tool implementations
- [AgentThinkPattern](../agents/base_classes/base_think_pattern.py) - Base class for thinking patterns
- [PromptManager](../agents/prompt.py) - Class for managing prompts and messages 