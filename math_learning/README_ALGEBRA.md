# K-12 Algebra Learning System

A comprehensive learning system for K-12 algebra with a knowledge graph of concepts and an exercise bank.

## Features

- **Comprehensive Knowledge Graph**: 25+ algebra concepts from elementary to high school
- **Structured Learning Paths**: Find optimal paths to learn advanced concepts
- **Extensive Exercise Bank**: 60+ exercises with varying difficulty levels
- **Concept Relationships**: Prerequisites and related concepts
- **Command-line Interface**: Explore concepts and practice exercises

## Concept Structure

The knowledge graph organizes algebra concepts into three main levels:

1. **Elementary School (K-5)**
   - Counting, Addition, Subtraction
   - Multiplication, Division
   - Fractions, Patterns, Place Value

2. **Middle School (6-8)**
   - Variables and Expressions
   - One-Step and Two-Step Equations
   - Inequalities, Coordinate Plane
   - Proportional Relationships
   - Linear Functions, Systems of Linear Equations

3. **High School (9-12)**
   - Exponents and Roots
   - Polynomials, Factoring
   - Quadratic Functions, Quadratic Formula
   - Rational Expressions
   - Exponential and Logarithmic Functions
   - Function Transformations

## Getting Started

### Using the CLI

```bash
# List all algebra concepts
python -m math_learning.algebra_cli list

# Filter concepts by category
python -m math_learning.algebra_cli list --category "Number Sense"

# Filter concepts by difficulty (1-5)
python -m math_learning.algebra_cli list --difficulty 3

# Explore a specific concept
python -m math_learning.algebra_cli explore "Variables"

# Practice exercises for a concept
python -m math_learning.algebra_cli practice "Quadratic Functions"

# Show a learning path to a target concept
python -m math_learning.algebra_cli path "Logarithmic Functions"
```

### Using the API

```python
from math_learning.algebra_learning import AlgebraLearningSystem

# Initialize the system
system = AlgebraLearningSystem()

# Find a concept
concept = system.get_concept_by_name("Linear Functions")

# Get learning path to a concept
learning_path = system.get_learning_path(concept.id)

# Get exercises for a concept
exercises = system.exercise_bank.get_exercises_for_concept(concept.id)

# Get recommended next concepts based on what's already learned
learned_concepts = ["addition", "subtraction", "multiplication"]
recommendations = system.recommend_next_concepts(learned_concepts)
```

## Knowledge Graph Structure

The algebra knowledge graph has the following structure:

- **Concepts**: Nodes representing mathematical topics
- **Prerequisites**: Directed edges showing dependencies
- **Related Concepts**: Undirected edges showing related topics

Each concept contains:
- Name and description
- Difficulty level (1-5)
- Time to master (in minutes)
- Category
- Examples

## Exercise Bank

The exercise bank contains problems of varying difficulty for each concept:

- **Multiple formats**: Free-response, multiple-choice, fill-in-blank, graphing
- **Difficulty levels**: 1-5 scale matching concept difficulty
- **Hints**: Progressive hints for students who need help
- **Solutions**: Complete solutions for each problem
- **Concept relationships**: Primary, secondary, and foundational relationships

## Extending the System

You can extend the system by:

1. Adding new concepts to the knowledge graph
2. Creating new exercises for existing concepts
3. Building custom learning paths for specific educational goals
4. Implementing personalized recommendation algorithms

## Technical Structure

The system consists of several key components:

- `algebra_graph.py`: Builds the K-12 algebra knowledge graph
- `algebra_exercises.py`, `middle_algebra_exercises.py`, `high_algebra_exercises.py`: Exercise definitions
- `algebra_learning.py`: Core system integrating graph and exercises
- `algebra_cli.py`: Command-line interface 