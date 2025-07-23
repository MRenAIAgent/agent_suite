# Agent Suite Repository Summary

## Overview
**Agent Suite** is a comprehensive AI framework for building intelligent agents with enhanced tool capabilities, personalized learning systems, and sophisticated benchmarking infrastructure.

## Key Features

### 🤖 Agent System
- **Modular Architecture**: Pattern-based agent design with BaseAgent interface
- **ReAct Implementation**: Reasoning and Acting pattern for intelligent decision-making
- **Specialized Agents**: Math tutoring, image recognition, general-purpose agents
- **Memory Management**: Conversation history and context tracking
- **Async Support**: Full asynchronous operation support

### 🛠️ Tool System
- **Intelligent Selection**: Context-aware tool selection based on queries, roles, tasks
- **Rich Metadata**: Categories, capabilities, domains, usage statistics
- **Registry Pattern**: Centralized tool management with namespacing
- **Adapter Support**: LangChain and MCP tool integration
- **Auto-Registration**: Tools automatically register with the system

### 📚 Math Learning System
- **Knowledge Graph**: 180+ K-12 algebra concepts with prerequisite relationships
- **Learning Graph**: Individual progress tracking with Bayesian mastery calculation
- **Personalization**: Adaptive learning paths based on strengths/weaknesses
- **Error Analysis**: Diagnostic system for identifying misconceptions
- **AI Tutoring**: Integrated AI agent for personalized math instruction

### 🔍 RAG System
- **Multi-Storage**: Graph, vector, and key-value storage backends
- **Flexible Architecture**: Configurable storage combinations
- **Graph-First Design**: Optimized for knowledge relationships
- **Context Management**: Sophisticated context tracking and retrieval

### 📊 Benchmarking Framework
- **Agent Comparison**: ReAct vs LangChain vs custom implementations
- **TauBench Integration**: Standardized agent evaluation
- **RAG Evaluation**: Retrieval accuracy, answer quality, factual correctness
- **Comprehensive Metrics**: Success rates, performance, accuracy tracking

### 🌐 LLM Integration
- **Multi-Provider**: OpenAI, Anthropic, DeepSeek, LiteLLM support
- **Abstract Interface**: Unified API across different LLM providers
- **Function Calling**: Native tool integration with LLMs
- **Async Operations**: Efficient async LLM interactions

## Architecture Highlights

### Design Patterns
- **Pattern-Based**: Agents use patterns to define behavior
- **Registry**: Centralized tool and component management
- **Adapter**: External tool integration (MCP, LangChain)
- **Factory**: Component creation and initialization
- **Observer**: Usage statistics and performance tracking

### Key Technologies
- **Python 3.8+** with AsyncIO
- **Pydantic** for data validation
- **Multiple LLM APIs** (OpenAI, Anthropic, etc.)
- **Vector/Graph Databases** (Qdrant, Neo4j)
- **MCP Protocol** for tool integration

## Directory Structure
```
agent_suite/
├── agents/                 # Core agent implementations
├── tools/                  # Tool system and registry
├── math_learning/          # Personalized learning platform
├── benchmark/              # Evaluation frameworks
├── llm/                    # LLM provider integrations
├── docs/                   # Documentation
├── examples/               # Usage examples
└── tests/                  # Test suites
```

## Getting Started

### Quick Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with API keys

# Run tests
pytest tests/
```

### Basic Usage
```python
from agents.react_agent import ReActAgent
from llm.openai.openai_llm import OpenAILLM

# Create agent
llm = OpenAILLM.create_llm()
agent = ReActAgent(
    llm=llm,
    role="You are a helpful assistant",
    task="Help users with questions",
    guide="Be helpful and accurate"
)

# Use agent
response = await agent.arun("Hello!", model="gpt-3.5-turbo")
```

## Key Strengths

1. **Modular Design**: Easy to extend and customize
2. **Intelligent Tool Selection**: Context-aware tool matching
3. **Comprehensive Benchmarking**: Thorough evaluation capabilities
4. **Production Ready**: Robust error handling and testing
5. **Multi-Modal**: Text, image, and structured data support
6. **Personalization**: Adaptive learning and recommendation systems

## Use Cases

- **Educational Platforms**: Personalized math tutoring and learning
- **Research**: Agent behavior analysis and comparison
- **Enterprise**: Intelligent assistants with tool integration
- **Development**: Framework for building custom AI agents
- **Benchmarking**: Evaluating agent performance and capabilities

## Status
- **Core Framework**: Production ready
- **Math Learning**: Fully implemented with comprehensive testing
- **Benchmarking**: Complete evaluation suite
- **Documentation**: Extensive guides and examples
- **Testing**: High test coverage across all components

This repository represents a mature, well-architected framework for building sophisticated AI agents with real-world applications in education, research, and enterprise environments. 