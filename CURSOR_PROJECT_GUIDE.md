# Agent Suite Project Guide for Cursor IDE

## Project Overview

**Agent Suite** is a comprehensive framework for building AI agents with enhanced tool capabilities, personalized learning systems, and sophisticated benchmarking infrastructure. The project combines multiple AI/ML technologies including LLMs, RAG (Retrieval-Augmented Generation), knowledge graphs, and intelligent tool selection.

## High-Level Architecture

### Core Components

1. **Agent System** (`agents/`) - Modular agent implementations with pattern-based architecture
2. **Tool System** (`tools/`) - Intelligent tool registry and selection framework
3. **Math Learning System** (`math_learning/`) - Personalized K-12 algebra learning platform
4. **RAG System** (`agents/rag/`) - Retrieval-Augmented Generation with multi-storage backends
5. **Benchmarking Framework** (`benchmark/`) - Comprehensive evaluation and comparison tools
6. **LLM Integration** (`llm/`) - Abstract interfaces for multiple LLM providers

### Key Design Patterns

- **Pattern-Based Architecture**: Agents use patterns to define behavior (ReAct, custom patterns)
- **Registry Pattern**: Centralized tool registry with intelligent selection
- **Adapter Pattern**: Integration with external tools (MCP, LangChain)
- **Factory Pattern**: Component creation and initialization
- **Observer Pattern**: Usage statistics and performance tracking

## Directory Structure & Key Files

### Core Agent System (`agents/`)

```
agents/
├── base_classes/
│   ├── base_agent.py           # Abstract base class for all agents
│   └── base_think_pattern.py   # Abstract thinking pattern interface
├── agent.py                    # Standard agent implementation
├── react_agent.py              # ReAct (Reasoning and Acting) agent
├── algebra_learning_agent.py   # Specialized math tutoring agent
├── enhanced_image_recognition_agent.py  # Image analysis agent
├── agent_pattern.py            # Agent behavior patterns
├── llm_execute_pattern.py      # LLM interaction patterns
├── memory/                     # Memory management system
├── rag/                        # RAG implementation
├── mcp_client/                 # MCP (Model Context Protocol) client
├── storage/                    # Storage backends
└── eval/                       # Evaluation frameworks
```

**Key Classes:**
- `BaseAgent`: Abstract interface for all agents
- `Agent`: Standard implementation with tools and memory
- `ReActAgent`: Reasoning and Acting pattern implementation
- `AlgebraLearningAgent`: Specialized math tutoring with image recognition

### Tool System (`tools/`)

```
tools/
├── base.py                     # Enhanced tool base class
├── tool.py                     # Original tool interface
├── registry.py                 # Centralized tool registry
├── tool_selector.py            # Intelligent tool selection
├── tool_types.py               # Core type definitions
├── selection.py                # Tool selection service
├── adapters/                   # Tool adapters (LangChain, MCP)
└── examples/                   # Example tool implementations
```

**Key Features:**
- **Intelligent Selection**: Context-aware tool selection based on queries, roles, tasks
- **Rich Metadata**: Categories, capabilities, domains, usage statistics
- **Multiple Selection Methods**: Keyword, category, capability, domain, hybrid
- **Adapter Support**: LangChain, MCP tool integration

### Math Learning System (`math_learning/`)

```
math_learning/
├── knowledge_graph/
│   ├── algebra_graph.py        # K-12 algebra knowledge graph
│   └── concept.py              # Concept representation
├── learning_graph/
│   ├── user_model.py           # Individual learning tracking
│   └── personalized_learning.py # Personalization algorithms
├── exercises/
│   └── exercise_bank.py        # Exercise management
├── recommendation/
│   ├── gap_analyzer.py         # Knowledge gap detection
│   └── recommender.py          # Exercise recommendations
├── ai_agent/
│   └── math_tutoring_agent.py  # AI tutoring integration
└── testing/                    # Comprehensive testing system
```

**Key Features:**
- **Knowledge Graph**: 180+ algebra concepts with prerequisite relationships
- **Learning Graph**: Individual progress tracking with Bayesian mastery calculation
- **Personalization**: Adaptive learning paths based on strengths/weaknesses
- **Error Analysis**: Diagnostic system for identifying misconceptions
- **RAG Integration**: Enhanced with retrieval-augmented generation

### RAG System (`agents/rag/`)

```
agents/rag/
├── api/
│   ├── rag_service.py          # Main RAG service interface
│   ├── rag_store.py            # Storage interface
│   └── rag_retrieval.py        # Retrieval interface
├── storage/
│   ├── graph/                  # Graph database storage
│   ├── vector/                 # Vector database storage
│   └── key_value/              # Key-value storage
├── models/
│   ├── document.py             # Document representation
│   ├── knowledge.py            # Knowledge graph models
│   └── context.py              # Context management
└── middleware/
    ├── storage_router.py       # Storage routing logic
    └── retrieval_orchestrator.py # Retrieval coordination
```

**Key Features:**
- **Multi-Storage**: Graph, vector, and key-value storage backends
- **Flexible Architecture**: Configurable storage combinations
- **Graph-First Design**: Optimized for knowledge relationships
- **Context Management**: Sophisticated context tracking

### Benchmarking Framework (`benchmark/`)

```
benchmark/
├── agent/
│   ├── code/
│   │   ├── agent_benchmark.py  # Agent comparison tool
│   │   └── taubench/           # TauBench evaluation
│   └── results/                # Benchmark results
├── rag/
│   ├── code/                   # RAG benchmarking
│   └── results/                # RAG evaluation results
└── run_all_benchmarks.py      # Unified benchmark runner
```

**Key Features:**
- **Agent Comparison**: ReAct vs LangChain vs custom implementations
- **TauBench Integration**: Standardized agent evaluation
- **RAG Evaluation**: Retrieval accuracy, answer quality, factual correctness
- **Comprehensive Metrics**: Success rates, performance, accuracy

### LLM Integration (`llm/`)

```
llm/
├── llm.py                      # Abstract LLM interface
├── openai/
│   └── openai_llm.py          # OpenAI integration
├── anthropic/
│   └── claud.py               # Anthropic Claude integration
├── deepseek/
│   └── deepseek_llm.py        # DeepSeek integration
└── litellm/
    └── litellm.py             # LiteLLM multi-provider support
```

## Key Technologies & Dependencies

### Core Technologies
- **Python 3.8+**: Primary language
- **Pydantic**: Data validation and settings management
- **AsyncIO**: Asynchronous programming
- **JSON/JSONL**: Data serialization
- **SQLite**: Local database storage

### LLM Integration
- **OpenAI API**: GPT models
- **Anthropic API**: Claude models
- **LiteLLM**: Multi-provider LLM access
- **Function Calling**: Tool integration

### Storage & Retrieval
- **Qdrant**: Vector database (optional)
- **Neo4j**: Graph database (optional)
- **Redis**: Key-value storage (optional)
- **Memory Storage**: Testing and development

### External Integrations
- **MCP (Model Context Protocol)**: Tool integration standard
- **LangChain**: Tool and agent framework
- **Graphiti**: Graph memory management
- **TauBench**: Agent evaluation framework

## Development Workflow

### Setting Up Development Environment

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd agent_suite
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run Tests**
   ```bash
   pytest tests/
   ```

### Common Development Tasks

#### Creating a New Agent
```python
from agents.react_agent import ReActAgent
from llm.openai.openai_llm import OpenAILLM

# Create LLM instance
llm = OpenAILLM.create_llm()

# Create agent
agent = ReActAgent(
    llm=llm,
    role="You are a helpful assistant",
    task="Help users with their questions",
    guide="Be helpful and accurate",
    tools=[],  # Add tools here
    max_iterations=5
)

# Use agent
response = await agent.arun("Hello!", model="gpt-3.5-turbo")
```

#### Creating a New Tool
```python
from tools.base import EnhancedTool
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain

class MyTool(EnhancedTool):
    def __init__(self):
        super().__init__()
        self.metadata.name = "my_tool"
        self.metadata.description = "Does something useful"
        self.metadata.categories = [ToolCategory.UTILITY]
        self.metadata.capabilities = [ToolCapability.COMPUTE]
        self.metadata.domains = [ToolDomain.GENERAL]
    
    async def arun(self, input_text: str) -> str:
        # Tool implementation
        return f"Processed: {input_text}"
```

#### Running Benchmarks
```bash
# Run all benchmarks
python benchmark/run_all_benchmarks.py

# Run specific benchmark
python benchmark/agent/code/agent_benchmark.py --model gpt-4

# Run math learning tests
python -m math_learning.tests.test_full_integration
```

## Configuration & Environment

### Environment Variables
```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Storage Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
NEO4J_URI=bolt://localhost:7687
REDIS_URL=redis://localhost:6379

# Development Settings
DEBUG=true
LOG_LEVEL=INFO
```

### Configuration Files
- `pyproject.toml`: Project metadata and dependencies
- `agents/rag/config/rag_config.py`: RAG system configuration
- `math_learning/config/`: Math learning system configuration

## Testing Strategy

### Test Structure
```
tests/
├── unit/                       # Unit tests
├── integration/                # Integration tests
├── benchmark/                  # Benchmark tests
└── fixtures/                   # Test fixtures
```

### Running Tests
```bash
# All tests
pytest

# Specific test suite
pytest tests/unit/agents/
pytest tests/integration/

# With coverage
pytest --cov=agents --cov=tools --cov=math_learning
```

## Common Patterns & Best Practices

### Agent Development
1. **Inherit from BaseAgent**: Use abstract base class
2. **Implement Patterns**: Use AgentPattern for behavior
3. **Tool Integration**: Register tools with registry
4. **Memory Management**: Use MemoryManager for conversation history
5. **Error Handling**: Implement robust error handling

### Tool Development
1. **Use EnhancedTool**: Inherit from enhanced base class
2. **Rich Metadata**: Provide comprehensive metadata
3. **Async Implementation**: Implement async methods
4. **Parameter Validation**: Use Pydantic for validation
5. **Auto-Registration**: Tools auto-register with registry

### Performance Optimization
1. **Async Operations**: Use async/await for I/O operations
2. **Caching**: Implement caching for expensive operations
3. **Batch Processing**: Process multiple items together
4. **Memory Management**: Clean up resources properly
5. **Monitoring**: Track usage statistics

## Troubleshooting

### Common Issues
1. **Import Errors**: Check Python path and dependencies
2. **API Key Issues**: Verify environment variables
3. **Storage Connection**: Check database connectivity
4. **Memory Issues**: Monitor memory usage with large datasets
5. **Async Issues**: Ensure proper async/await usage

### Debug Tools
1. **Logging**: Use structured logging throughout
2. **Benchmarks**: Performance and accuracy measurement
3. **Tests**: Comprehensive test coverage
4. **Profiling**: Performance profiling tools
5. **Monitoring**: Usage and error tracking

## Future Roadmap

### Planned Features
1. **Multi-Agent Systems**: Coordinated agent interactions
2. **Enhanced Memory**: Sophisticated memory management
3. **Tool Composition**: Chaining tools together
4. **Distributed Architecture**: Scalable deployment
5. **Advanced Personalization**: ML-based adaptation

### Extension Points
1. **New Agent Types**: Custom agent implementations
2. **Storage Backends**: Additional storage options
3. **LLM Providers**: New LLM integrations
4. **Tool Sources**: Additional tool providers
5. **Evaluation Metrics**: Custom evaluation frameworks

## Getting Help

### Documentation
- Architecture docs in `docs/`
- API documentation in code comments
- Examples in `examples/` and `demos/`
- Test cases for usage patterns

### Key Contact Points
- Agent system: `agents/` directory
- Tool system: `tools/` directory  
- Math learning: `math_learning/` directory
- Benchmarking: `benchmark/` directory
- Issues and bugs: GitHub issues

This guide provides a comprehensive overview for working with the Agent Suite project in Cursor IDE. The modular architecture, comprehensive testing, and extensive documentation make it suitable for both research and production use cases. 