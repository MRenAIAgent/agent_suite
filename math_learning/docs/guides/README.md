# Math Learning System

A personalized learning system using knowledge graphs to identify and address knowledge gaps in mathematics.

## Overview

This system uses dual graph structures to accelerate math learning:

1. **Knowledge Graph**: Models math topics as interconnected concepts
2. **Learning Graph**: Tracks each user's mastery of concepts
3. **Gap Analysis Engine**: Identifies knowledge gaps and priorities
4. **Exercise Bank**: Links exercises to multiple concepts
5. **Recommendation Engine**: Suggests optimal exercises

## Features

- Intelligent knowledge gap detection based on graph analysis
- Personalized exercise recommendations based on mastery level
- Learning path generation respecting concept dependencies
- Bayesian knowledge tracing for mastery estimation
- Multi-concept tagging of exercises

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the demo:

```
python -m math_learning
```

This will:
1. Create a sample knowledge graph for basic geometry
2. Initialize a set of exercises
3. Simulate a user with some prior knowledge
4. Identify knowledge gaps
5. Recommend appropriate exercises
6. Show a recommended learning path

Results will be saved to the `output` directory.

## System Components

### Knowledge Graph

The knowledge graph represents the domain knowledge structure, with:
- Concepts as nodes
- Prerequisite and related relationships as edges
- Metadata about difficulty, time to master, etc.

### Exercise Bank

The exercise bank contains problems that test understanding of concepts:
- Each exercise can be linked to multiple concepts
- Relationships have weights indicating how strongly they test each concept
- Exercises have difficulty levels and format types

### Learning Graph

The learning graph represents a user's current knowledge state:
- Tracks mastery level for each concept (0.0-1.0)
- Maintains confidence in mastery estimates
- Records history of exercise attempts

### Gap Analyzer

The gap analyzer identifies knowledge gaps by:
- Comparing the learning graph to the knowledge graph
- Calculating impact scores for missing concepts
- Considering concept centrality and dependencies

### Recommender

The recommender suggests exercises and learning paths by:
- Matching exercise difficulty to current mastery levels
- Prioritizing high-impact knowledge gaps
- Finding the optimal learning boundary

## Example

```python
from math_learning.knowledge_graph.graph import KnowledgeGraph
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.recommendation.recommender import Recommender

# Load knowledge graph and exercise bank
knowledge_graph = KnowledgeGraph.load_from_file("geometry_graph.json")
exercise_bank = ExerciseBank.load_from_file("geometry_exercises.json")

# Create a learning graph for a user
learning_graph = LearningGraph(user_id="user1")

# Set initial knowledge
learning_graph.set_mastery("concept1", 0.8)
learning_graph.set_mastery("concept2", 0.5)

# Create a recommender
recommender = Recommender(knowledge_graph, exercise_bank)

# Get exercise recommendations
recommendations = recommender.recommend_exercises(learning_graph)

# Get a learning path
path = recommender.get_learning_path(learning_graph)
```

## Extending the System

To extend the system for your own domain:

1. Create a knowledge graph for your subject area
2. Add exercises linked to concepts
3. Initialize a learning graph for each user
4. Use the recommender to suggest personalized learning material

## License

MIT 