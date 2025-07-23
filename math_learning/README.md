# Math Learning System

A comprehensive AI-powered personalized learning platform for mathematics education, featuring advanced knowledge graphs, RAG (Retrieval-Augmented Generation) integration, and real-time adaptive learning.

## 🚀 Quick Start

```bash
# Install development dependencies
make install-dev

# Run standard tests
make test

# Run all tests including real database tests
make setup-db && make test-all
```

## 📁 Project Structure

```
math_learning/
├── README.md                    # This file
├── Makefile                     # Build and test automation
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Database setup
├── pytest.ini                  # Test configuration
│
├── ai_agent/                    # AI tutoring agent components
├── config/                      # Configuration files
├── exercises/                   # Exercise generation system
├── geometry/                    # Geometry learning components
├── knowledge_graph/             # Knowledge graph management
├── learning_graph/              # Personal learning tracking
├── recommendation/              # Learning recommendation engine
├── simulation_output/           # Learning simulation results
│
├── tests/                       # All test files (organized)
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── system/                  # System & E2E tests
│   └── real_db/                 # Real database tests
│
├── examples/                    # Demo and example files
│   ├── demos/                   # Demo scripts
│   ├── tutorials/               # Tutorial examples
│   └── integration/             # Integration examples
│
├── docs/                        # Documentation
│   ├── guides/                  # User guides and READMEs
│   ├── analysis/                # Analysis documents
│   └── reports/                 # Test reports and summaries
│
├── scripts/                     # Utility scripts
│   ├── setup/                   # Setup scripts
│   ├── utils/                   # Utility scripts
│   └── debug/                   # Debug tools
│
├── data/                        # Data files
│   ├── exercises/               # Exercise data
│   ├── knowledge/               # Knowledge graph data
│   └── samples/                 # Sample data
│
└── apps/                        # Standalone applications
    ├── chat/                    # Math chat applications
    ├── cli/                     # Command line interfaces
    └── web/                     # Web applications
```

## 🧪 Testing

The project includes comprehensive testing infrastructure with multiple test types:

### Test Types

- **Unit Tests**: Fast, isolated component testing
- **Integration Tests**: Component interaction validation
- **System Tests**: End-to-end workflow verification
- **Real Database Tests**: Actual Neo4j and Qdrant integration

### Running Tests

```bash
# Quick tests (unit only)
make test-unit

# Integration tests
make test-integration

# System/E2E tests
make test-system

# Real database tests (requires Docker)
make test-real-db

# All tests
make test-all

# With coverage report
make coverage
```

### Database Testing

For real database tests, you need to start the databases first:

```bash
# Setup and start databases
make setup-db

# Check database status
make check-db

# Run database tests
make test-real-db

# Clean up databases
make clean-db
```

## 🛠️ Development

### Setup Development Environment

```bash
# Complete development setup
make dev-setup

# Or step by step:
make install-dev    # Install dependencies
make setup-db       # Start databases
make test          # Run tests
```

### Development Workflow

```bash
# Quick tests during development
make test-quick

# Code quality checks
make lint
make format
make check

# Clean up temporary files
make clean
```

## 📚 Key Features

### 🧠 AI-Powered Learning
- **Intelligent Tutoring Agent**: Provides personalized guidance and explanations
- **Adaptive Learning**: Adjusts difficulty based on student performance
- **Error Analysis**: Identifies misconceptions and provides targeted remediation

### 📊 Knowledge Management
- **Knowledge Graphs**: Structured representation of mathematical concepts
- **Prerequisite Tracking**: Ensures proper learning sequence
- **Concept Relationships**: Maps dependencies between topics

### 🎯 Personalized Learning
- **Individual Learning Graphs**: Tracks each student's progress
- **Mastery Calculation**: Bayesian approach to assess understanding
- **Recommendation Engine**: Suggests next steps based on performance

### 🗄️ Advanced Storage
- **RAG Integration**: Retrieval-Augmented Generation for enhanced AI responses
- **Vector Database**: Semantic search with Qdrant
- **Graph Database**: Complex relationships with Neo4j
- **Multi-Modal Storage**: Supports various data types and queries

### 🧮 Mathematics Coverage
- **Algebra**: From basic arithmetic to advanced equations
- **Geometry**: Visual learning with spatial reasoning
- **Problem Solving**: Real-world application scenarios
- **Assessment**: Comprehensive diagnostic capabilities

## 🔧 Configuration

### Environment Variables

```bash
# Database configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"
```

### Database Setup

The system uses Docker Compose for easy database management:

- **Qdrant**: Vector database for semantic search (ports 6333, 6334)
- **Neo4j**: Graph database for relationships (ports 7474, 7687)

## 📖 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Testing Guide](docs/guides/README_TESTING.md)**: Complete testing instructions
- **[Database Testing](docs/guides/README_REAL_DB_TESTS.md)**: Real database integration
- **[RAG Integration](docs/guides/README_RAG_Integration.md)**: RAG system overview
- **[Analysis Reports](docs/analysis/)**: System analysis and reviews
- **[Test Reports](docs/reports/)**: Test execution summaries

## 🤝 Contributing

1. **Setup**: Run `make dev-setup` for complete environment setup
2. **Testing**: Ensure all tests pass with `make test-all`
3. **Code Quality**: Run `make check` before committing
4. **Documentation**: Update relevant docs for new features

### Adding Tests

- **Unit tests**: Add to `tests/unit/`
- **Integration tests**: Add to `tests/integration/`
- **System tests**: Add to `tests/system/`
- **Database tests**: Add to `tests/real_db/`

## 🎯 Usage Examples

### Basic Usage

```python
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.knowledge_graph.algebra_graph import AlgebraGraph

# Create learning system
algebra = AlgebraGraph()
student = LearningGraph("student_001", "Alice")

# Record learning activity
student.record_exercise_attempt("ex_001", "basic_arithmetic", True, 0.5, 1.0)

# Get recommendations
mastery = student.get_mastery("basic_arithmetic")
recommendations = student.get_struggling_concepts()
```

### Advanced Features

```python
# RAG-enhanced learning
from math_learning.config.rag_config import get_memory_config
from math_learning.knowledge_graph.graph_rag_algebra_graph import GraphRagAlgebraGraph

config = get_memory_config()
enhanced_algebra = await GraphRagAlgebraGraph(config)
await enhanced_algebra.initialize()

# Semantic search for concepts
results = await enhanced_algebra.search_concepts("solving equations")
```

## 📊 Performance

- **Test Coverage**: >90% for core components
- **Test Execution**: <60 seconds for full suite
- **Database Performance**: <100ms per operation
- **Memory Usage**: Optimized for concurrent users

## 🔗 Related Projects

This system integrates with the broader Agent Suite ecosystem:

- **Agent Framework**: Core AI agent infrastructure
- **RAG System**: Advanced retrieval and generation
- **Storage Adapters**: Multi-database support
- **Tool Integration**: MCP-compatible tool system

## 📄 License

Part of the Agent Suite project. See the main repository for license information.

## 🆘 Support

For questions, issues, or contributions:

1. Check the documentation in `docs/`
2. Run `make help` for available commands
3. Use `make env-check` to verify your setup
4. Refer to test files for usage examples

---

**Made with ❤️ for mathematics education** 