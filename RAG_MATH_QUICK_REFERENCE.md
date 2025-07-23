# RAG & Math Learning Systems - Quick Reference Guide

## 🚀 Getting Started

### RAG System Setup

```python
# 1. Basic RAG Service Creation
from agents.rag.factory import create_rag_service

config = {
    "vector": {
        "type": "qdrant",
        "collection_name": "documents",
        "vector_size": 384,
        "embedding": {
            "type": "sentence-transformer",
            "model": "all-MiniLM-L6-v2"
        }
    },
    "graph": {
        "type": "neo4j",
        "uri": "bolt://localhost:7687",
        "username": "neo4j",
        "password": "password"
    }
}

rag_service = await create_rag_service(config)
```

### Math Learning System Setup

```python
# 1. Initialize Core Components
from math_learning.knowledge_graph.algebra_graph import build_algebra_knowledge_graph
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.recommender import Recommender
from math_learning.ai_agent.math_tutoring_agent import MathTutoringAgent

# 2. Create Knowledge Graph
knowledge_graph = build_algebra_knowledge_graph()

# 3. Create Learning Graph for Student
learning_graph = LearningGraph(user_id="student_123", name="John Doe")

# 4. Initialize AI Tutor
ai_tutor = MathTutoringAgent(
    llm=your_llm,
    learning_graph=learning_graph,
    user_model=learning_graph,
    recommender=recommender,
    gap_analyzer=gap_analyzer
)
```

---

## 📚 RAG System - Common Operations

### Storing Data

```python
# Store Document
from agents.rag.models.document import Document

document = Document(
    content="Linear equations are fundamental to algebra...",
    metadata={"subject": "math", "difficulty": "medium"}
)
doc_id = await rag_service.store_document(document)

# Store Knowledge
from agents.rag.models.knowledge import Entity, Relationship

entity = Entity(
    type="math_concept",
    properties={
        "name": "Linear Equations",
        "difficulty": 3,
        "prerequisites": ["variables", "basic_operations"]
    }
)
entity_id = await rag_service.store_knowledge(entity)

# Store Context
from agents.rag.models.context import ContextItem

context = ContextItem(
    key="user_session_123",
    value={"current_topic": "algebra", "progress": 0.7},
    metadata={"timestamp": datetime.now()}
)
context_id = await rag_service.store_context(context)
```

### Retrieving Data

```python
# Semantic Search
results = await rag_service.retrieve(
    query="How to solve linear equations?",
    top_k=5,
    filter_criteria={"subject": "math"}
)

# Document-Specific Retrieval
doc_results = await rag_service.retrieve_documents(
    query="quadratic formula examples",
    top_k=3,
    filter_criteria={"difficulty": "medium"}
)

# Knowledge Graph Queries
knowledge_results = await rag_service.retrieve_knowledge(
    query="prerequisites for quadratic equations",
    top_k=5
)

# Retrieve by ID
item = await rag_service.retrieve_by_id("doc_123")
```

---

## 🎓 Math Learning System - Common Operations

### Working with Learning Graphs

```python
# Track Student Progress
learning_graph.record_exercise_attempt(
    exercise_id="ex_001",
    concept_id="LF-05",
    result=True,
    difficulty=0.7,
    concept_weight=0.9
)

# Get Mastery Level
mastery = learning_graph.get_mastery("LF-05")  # Returns 0.0-1.0
confidence = learning_graph.get_confidence("LF-05")

# Find Struggling Areas
struggling = learning_graph.get_struggling_concepts(threshold=0.3)
mastered = learning_graph.get_mastered_concepts(threshold=0.7)
```

### AI Tutoring Sessions

```python
# Start Tutoring Session
await ai_tutor.start_tutoring_session(
    student_id="student_123",
    context={
        "current_topic": "linear_equations",
        "goal": "master_solving_techniques",
        "session_type": "practice"
    }
)

# Interactive Tutoring
response = await ai_tutor.tutor(
    student_input="I don't understand how to solve 2x + 3 = 11",
    model="gpt-4"
)

# Generate Learning Plan
plan = await ai_tutor.generate_learning_plan(
    student_id="student_123",
    target_concepts=["LF-05", "LF-06", "LF-07"],
    timeframe_days=14
)

# Get Recommendations
recommendations = await ai_tutor.recommend_next_steps("student_123")
```

### Exercise Recommendations

```python
# Get Exercise Recommendations
from math_learning.recommendation.recommender import Recommender

recommender = Recommender(knowledge_graph, exercise_bank)
recommendations = recommender.recommend_exercises(
    learning_graph,
    max_exercises=5
)

# Generate Learning Path
learning_path = recommender.get_learning_path(
    learning_graph,
    max_concepts=10
)
```

---

## 🔧 Configuration & Customization

### RAG System Configuration

```python
# Custom Storage Router
from agents.rag.middleware.storage_router import StorageRouter

router = StorageRouter()
router.add_rule(
    condition=lambda data, metadata: "math" in metadata.get("subject", ""),
    storage_type=StorageType.GRAPH
)

# Custom Retrieval Strategy
from agents.rag.middleware.retrieval_orchestrator import RetrievalStrategy

class MathRetrievalStrategy(RetrievalStrategy):
    def determine_sources(self, query: str, metadata: Dict[str, Any]) -> Set[StorageType]:
        if "equation" in query.lower():
            return {StorageType.GRAPH}
        return {StorageType.VECTOR}
```

### Math Learning Customization

```python
# Custom Learning Pattern Analysis
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem

personalized_system = PersonalizedLearningSystem(knowledge_graph)
patterns = personalized_system.analyze_learning_patterns(learning_graph)

# Custom Weakness Detection
weaknesses = personalized_system.detect_weaknesses(learning_graph)
for weakness in weaknesses:
    print(f"Concept: {weakness.concept_id}, Severity: {weakness.severity}")
```

---

## 🐛 Debugging & Troubleshooting

### Common RAG Issues

```python
# Check Storage Backends
print(f"Available storage adaptors: {rag_service.storage_adaptors.keys()}")

# Debug Retrieval
import logging
logging.getLogger("agents.rag").setLevel(logging.DEBUG)

# Test Individual Backends
vector_adaptor = rag_service.storage_adaptors.get(StorageType.VECTOR)
if vector_adaptor:
    results = await vector_adaptor.semantic_search("test query", top_k=3)
```

### Common Math Learning Issues

```python
# Check Knowledge Graph
print(f"Total concepts: {len(knowledge_graph.concepts)}")
print(f"Total relationships: {len(knowledge_graph.relationships)}")

# Debug Learning Graph
print(f"Concepts with mastery: {len(learning_graph.concept_mastery)}")
print(f"Exercise history: {len(learning_graph.exercise_history)}")

# Validate Prerequisites
missing_prereqs = gap_analyzer.find_knowledge_gaps(learning_graph)
print(f"Missing prerequisites: {missing_prereqs}")
```

---

## 📊 Monitoring & Analytics

### RAG System Metrics

```python
# Monitor Retrieval Performance
import time

start_time = time.time()
results = await rag_service.retrieve("query", top_k=10)
retrieval_time = time.time() - start_time

print(f"Retrieved {len(results)} results in {retrieval_time:.2f}s")

# Storage Usage
for storage_type, adaptor in rag_service.storage_adaptors.items():
    if hasattr(adaptor, 'get_collection_info'):
        info = await adaptor.get_collection_info()
        print(f"{storage_type}: {info}")
```

### Math Learning Analytics

```python
# Student Progress Analytics
progress_summary = ai_tutor.get_student_progress_summary("student_123")
print(f"Overall mastery: {progress_summary['overall_mastery']:.2%}")
print(f"Concepts mastered: {progress_summary['mastered_count']}")

# Learning Velocity Analysis
patterns = personalized_system.analyze_learning_patterns(learning_graph)
for concept_id, pattern in patterns.items():
    print(f"{concept_id}: velocity={pattern.learning_velocity:.2f}")
```

---

## 🔄 Integration Patterns

### RAG + Math Learning Integration

```python
# Store Math Concepts in RAG
async def store_math_concept_in_rag(concept, rag_service):
    entity = Entity(
        type="math_concept",
        properties={
            "concept_id": concept.concept_id,
            "name": concept.name,
            "difficulty": concept.difficulty,
            "prerequisites": concept.prerequisites
        }
    )
    return await rag_service.store_knowledge(entity)

# Retrieve Related Concepts
async def find_related_concepts(concept_id, rag_service):
    return await rag_service.retrieve_knowledge(
        query=f"concepts related to {concept_id}",
        top_k=5
    )

# Enhanced Tutoring with RAG
async def enhanced_tutoring_response(student_input, ai_tutor, rag_service):
    # Get context from RAG
    context = await rag_service.retrieve(
        query=student_input,
        top_k=3,
        filter_criteria={"type": "educational_content"}
    )
    
    # Generate response with enhanced context
    response = await ai_tutor.tutor(
        student_input=student_input,
        additional_context=context
    )
    
    return response
```

---

## 🚀 Performance Optimization

### RAG System Optimization

```python
# Batch Operations
documents = [doc1, doc2, doc3]
tasks = [rag_service.store_document(doc) for doc in documents]
doc_ids = await asyncio.gather(*tasks)

# Connection Pooling
config = {
    "vector": {
        "type": "qdrant",
        "pool_size": 10,
        "max_retries": 3
    }
}

# Caching
from functools import lru_cache

@lru_cache(maxsize=128)
async def cached_retrieval(query, top_k):
    return await rag_service.retrieve(query, top_k)
```

### Math Learning Optimization

```python
# Batch Exercise Processing
exercises = [ex1, ex2, ex3]
for exercise in exercises:
    learning_graph.record_exercise_attempt(
        exercise.id, exercise.concept_id, exercise.result
    )

# Efficient Mastery Calculation
# Use batch updates instead of individual calls
batch_updates = [
    ("LF-01", True, 0.7),
    ("LF-02", False, 0.8),
    ("LF-03", True, 0.6)
]

for concept_id, result, difficulty in batch_updates:
    learning_graph.update_mastery(concept_id, result, difficulty, 1.0)
```

---

## 📖 Additional Resources

### Key Files to Explore

**RAG System:**
- `agents/rag/api/rag_service.py` - Main service implementation
- `agents/rag/factory.py` - Service creation and configuration
- `agents/rag/middleware/` - Routing and orchestration logic

**Math Learning System:**
- `math_learning/ai_agent/math_tutoring_agent.py` - AI tutor implementation
- `math_learning/learning_graph/user_model.py` - Student progress tracking
- `math_learning/knowledge_graph/algebra_graph.py` - Knowledge graph structure

### Example Projects

- `math_learning/examples/rag_integration_example.py` - Complete integration example
- `examples/simple_react/` - Basic agent examples
- `benchmark/rag/` - Performance benchmarking examples

### Testing

- `tests/integration/` - Integration tests
- `math_learning/testing/` - Math learning system tests
- `benchmark/` - Performance benchmarks 