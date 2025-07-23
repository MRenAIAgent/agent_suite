# RAG-Enhanced Math Learning System

This directory contains a RAG (Retrieval-Augmented Generation) enhanced version of the math learning system that integrates the repository's sophisticated RAG components with knowledge graphs and learning graphs for intelligent, personalized education.

## 🚀 Overview

The RAG-enhanced math learning system combines:

- **RAG-Enhanced Knowledge Graphs**: Semantic search, intelligent concept relationships, and personalized teaching recommendations
- **RAG-Enhanced Learning Graphs**: Advanced analytics, pattern recognition, and adaptive learning paths
- **Intelligent Tutoring System**: Complete integration that provides personalized, context-aware math education

## 📁 File Structure

```
math_learning/
├── knowledge_graph/
│   ├── rag_enhanced_algebra_graph.py    # RAG-enhanced knowledge graph
│   └── algebra_graph.py                 # Original knowledge graph
├── learning_graph/
│   ├── rag_enhanced_user_model.py       # RAG-enhanced learning graph
│   └── user_model.py                    # Original learning graph
├── config/
│   └── rag_config.py                    # RAG configuration settings
├── examples/
│   └── rag_integration_example.py       # Complete integration example
└── README_RAG_Integration.md            # This file
```

## 🛠️ Setup

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export QDRANT_HOST="localhost"  # Optional, defaults to localhost
   export QDRANT_PORT="6333"       # Optional, defaults to 6333
   ```

3. **Start Qdrant (Optional for Vector Storage)**:
   ```bash
   # Using Docker
   docker run -p 6333:6333 qdrant/qdrant
   
   # Or use memory storage for testing (no setup required)
   ```

### Quick Start

```python
import asyncio
from math_learning.examples.rag_integration_example import IntelligentMathTutor

async def main():
    # Create an intelligent tutor
    tutor = IntelligentMathTutor("student_001", "Alice")
    
    # Initialize with RAG capabilities
    await tutor.initialize()
    
    # Start a learning session
    session_id = await tutor.start_learning_session([
        "Master basic number sense",
        "Build confidence with counting"
    ])
    
    # Get personalized exercise
    exercise = await tutor.get_next_exercise()
    print(f"Exercise: {exercise['concept_id']}")
    print(f"Difficulty: {exercise['difficulty']}")
    print(f"Tips: {exercise['personalized_tips']}")
    
    # Submit result and get feedback
    feedback = await tutor.submit_exercise_result(
        exercise, result=True, time_spent=120, confidence=4
    )
    print(f"Feedback: {feedback['feedback_message']}")
    
    # Get learning insights
    insights = await tutor.get_learning_insights()
    print(f"Strengths: {len(insights['strengths'])}")
    print(f"Focus areas: {insights['recommendations']['focus_areas']}")
    
    # End session
    summary = await tutor.end_session(mood_rating=4, difficulty_rating=3)
    print(f"Session success rate: {summary['success_rate']:.1%}")
    
    await tutor.close()

asyncio.run(main())
```

## 🧠 Key Features

### RAG-Enhanced Knowledge Graph

The `RagEnhancedAlgebraGraph` provides:

- **Semantic Concept Search**: Find related concepts using natural language
- **Intelligent Learning Paths**: AI-generated learning sequences
- **Personalized Teaching Recommendations**: Adaptive strategies based on student profiles
- **Enhanced Concept Relationships**: Vector-based similarity and prerequisite mapping

```python
from math_learning.knowledge_graph.rag_enhanced_algebra_graph import create_rag_enhanced_algebra_graph

# Create knowledge graph
kg = await create_rag_enhanced_algebra_graph()

# Find similar concepts
similar = await kg.find_similar_concepts("basic number understanding", top_k=3)

# Get learning path
path = await kg.get_learning_path_suggestions("NS-01", "understand fractions")

# Get teaching recommendations
student_profile = {
    "learning_style": "visual",
    "difficulty_preference": "moderate",
    "interests": ["games", "real-world applications"]
}
recommendations = await kg.get_teaching_recommendations("NS-02", student_profile)
```

### RAG-Enhanced Learning Graph

The `RagEnhancedLearningGraph` provides:

- **Advanced Analytics**: Pattern recognition and learning insights
- **Adaptive Difficulty**: AI-powered difficulty adjustment
- **Personalized Recommendations**: Context-aware learning suggestions
- **Enhanced Progress Tracking**: Multi-dimensional learning analytics

```python
from math_learning.learning_graph.rag_enhanced_user_model import RagEnhancedLearningGraph

# Create learning graph
lg = RagEnhancedLearningGraph("student_001", "Alice's Learning Journey")

# Start session with intelligent goals
session_id = await lg.start_learning_session(["Master counting", "Build confidence"])

# Get personalized recommendations
recommendations = await lg.get_personalized_recommendations(session_id)

# Record enhanced exercise attempt
await lg.record_enhanced_exercise_attempt(
    session_id=session_id,
    exercise_id="ex_001",
    concept_id="NS-01",
    result=True,
    difficulty=0.6,
    time_spent=120,
    hints_used=1,
    student_confidence=4
)

# Generate learning insights
insights = await lg.generate_learning_insights()
```

## 🎯 Usage Examples

### 1. Basic RAG Features Demo

```python
from math_learning.examples.rag_integration_example import demo_rag_features

# Demonstrate RAG capabilities
await demo_rag_features()
```

### 2. Complete Intelligent Tutoring Demo

```python
from math_learning.examples.rag_integration_example import demo_intelligent_tutoring

# Full tutoring system demonstration
await demo_intelligent_tutoring()
```

### 3. Custom Integration

```python
import asyncio
from math_learning.knowledge_graph.rag_enhanced_algebra_graph import create_rag_enhanced_algebra_graph
from math_learning.learning_graph.rag_enhanced_user_model import RagEnhancedLearningGraph

async def custom_integration():
    # Initialize components
    kg = await create_rag_enhanced_algebra_graph()
    lg = RagEnhancedLearningGraph("custom_student", "Custom Learning")
    
    # Your custom logic here
    # ...
    
    # Cleanup
    await kg.close()
    await lg.close()

asyncio.run(custom_integration())
```

## ⚙️ Configuration

### RAG Configuration

Modify `math_learning/config/rag_config.py` to customize:

```python
from math_learning.config.rag_config import RagConfig

config = RagConfig(
    vector_storage_type="memory",  # Use memory for testing
    embedding_provider="dummy",    # Use dummy embeddings for testing
    concept_similarity_threshold=0.8,
    learning_path_max_depth=7,
    personalization_enabled=True
)
```

### Storage Options

- **Vector Storage**: `qdrant` (recommended) or `memory` (testing)
- **Graph Storage**: `memory` (default) or `neo4j` (if available)
- **Key-Value Storage**: `memory` (default) or `redis` (if available)
- **Embeddings**: `openai` (recommended), `sentence_transformer`, or `dummy` (testing)

## 🔧 Advanced Features

### Custom Student Profiles

```python
student_profile = {
    "learning_style": "kinesthetic",      # visual, auditory, kinesthetic
    "difficulty_preference": "challenging", # easy, moderate, challenging
    "interests": ["sports", "technology"],
    "attention_span": "short",            # short, medium, long
    "motivation_level": "high"            # low, medium, high
}

recommendations = await kg.get_teaching_recommendations(concept_id, student_profile)
```

### Learning Analytics

```python
# Get comprehensive learning insights
insights = await lg.generate_learning_insights()

# Filter by insight type
strengths = [i for i in insights if i.insight_type == "strength"]
weaknesses = [i for i in insights if i.insight_type == "weakness"]
patterns = [i for i in insights if i.insight_type == "pattern"]

# Get adaptive difficulty
difficulty = await lg.get_adaptive_difficulty("NS-01")
```

### Semantic Search

```python
# Find concepts by natural language description
concepts = await kg.find_similar_concepts(
    "help with basic arithmetic operations", 
    top_k=5
)

# Search with specific filters
filtered_concepts = await kg.find_similar_concepts(
    "visual learning strategies for fractions",
    top_k=3,
    filters={"difficulty_level": "beginner"}
)
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Qdrant Connection Issues**:
   - Use memory storage for testing: `vector_storage_type="memory"`
   - Check Qdrant is running: `docker ps`

3. **Import Errors**:
   - Ensure you're in the correct directory
   - Check all dependencies are installed

### Testing Without External Services

```python
from math_learning.config.rag_config import RagConfig

# Configuration for testing without external services
test_config = RagConfig(
    vector_storage_type="memory",
    embedding_provider="dummy",
    graph_storage_type="memory",
    kv_storage_type="memory"
)
```

## 📊 Performance Considerations

- **Memory Usage**: RAG components use additional memory for vector storage
- **Initialization Time**: First startup may take longer due to embedding generation
- **API Costs**: OpenAI embeddings incur API costs (use dummy provider for testing)
- **Scalability**: Consider using persistent storage (Qdrant, Redis) for production

## 🤝 Contributing

When extending the RAG-enhanced system:

1. Follow the existing async/await patterns
2. Add proper error handling and logging
3. Include type hints and docstrings
4. Test with both real and dummy providers
5. Update this README with new features

## 📝 License

This RAG-enhanced math learning system follows the same license as the main repository.

---

For more information about the underlying RAG system, see `agents/rag/README.md`. 