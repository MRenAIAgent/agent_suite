# TauBench Evaluation Framework

TauBench is a framework for evaluating agent capabilities across a diverse set of reasoning, planning, and tool use tasks.

## Overview

The TauBench evaluation framework focuses on evaluating an agent's ability to:

1. Solve reasoning tasks that involve logical and analytical thinking
2. Create and execute plans to achieve complex goals
3. Use a variety of tools to accomplish tasks
4. Answer questions across different domains
5. Generate code based on requirements

## Minimal Implementation

This directory includes a minimal implementation of the TauBench framework (`taubench_minimal.py`) that can be used without external dependencies. This implementation:

- Provides a `DatasetLoader` class that can load and generate sample datasets
- Includes an `Evaluator` class for comparing agent responses against references
- Calculates success metrics for tasks based on response quality

## Sample Tasks

The minimal implementation includes sample tasks across five categories:

1. **Reasoning**: Logical reasoning problems that test deductive thinking
2. **Planning**: Tasks requiring step-by-step plan creation
3. **Tool Use**: Tasks that require using specific tools to accomplish goals
4. **QA**: General knowledge question answering
5. **Code Generation**: Tasks requiring code implementation based on specifications

## Running Evaluations

You can run TauBench evaluations using the main evaluation script:

```bash
python run_agent_eval.py --model "your-model-name" --frameworks taubench --env-limit 3 --task-limit 5
```

Options:
- `--model`: The LLM model to evaluate
- `--frameworks`: Which frameworks to use (taubench, agentbench, agentboard, all)
- `--env-limit`: Number of categories to evaluate
- `--task-limit`: Maximum number of tasks per category
- `--output-dir`: Where to save evaluation results

## Evaluation Results

Results are saved in JSON format and include:
- Per-task responses and metrics
- Category-level success rates
- Overall success metrics

Example output structure:
```json
{
  "model": "gpt-4",
  "agent_type": "ReActAgent",
  "categories": {
    "reasoning": {
      "total_tasks": 5,
      "tasks": [
        {
          "id": "reasoning_0",
          "prompt": "...",
          "reference": "...",
          "response": "...",
          "metrics": {
            "success": true,
            "overlap_score": 0.75,
            "word_count": 42,
            "relevance": 0.85
          }
        }
      ],
      "successful_tasks": 4,
      "success_rate": 0.8
    }
  },
  "summary": {
    "total_tasks": 15,
    "successful_tasks": 12,
    "overall_success_rate": 0.8
  }
}
```

## Adding Custom Tasks

To add custom tasks, create JSONL files in the `datasets/<category>/test.jsonl` format with the following structure:

```json
{
  "id": "unique_task_id",
  "instruction": "Task instruction",
  "input": "Additional input (optional)",
  "reference": "Expected output",
  "evaluation_criteria": {"success": true},
  "tools": [
    {
      "name": "tool_name",
      "description": "Tool description",
      "parameters": {"param1": "string"}
    }
  ]
}
```

## Extending The Framework

The minimal TauBench implementation can be extended by:

1. Adding more sophisticated evaluation metrics
2. Creating custom task datasets
3. Implementing comparative evaluation against reference models

## Installation

To use TauBench evaluation, you need to install the TauBench library:

```bash
# From the taubench directory
pip install -r requirements.txt

# Or install directly
pip install tau-bench
```

## Usage

### Command Line

You can run TauBench evaluation from the command line:

```bash
# Run only TauBench evaluation
python -m agents.eval.run_evaluations --frameworks taubench

# Customize the evaluation
python -m agents.eval.run_evaluations --frameworks taubench --model gpt-4 --env-limit 2 --task-limit 3
```

### Programmatic Usage

You can also use TauBench evaluation directly in your code:

```python
import asyncio
from agents.react_agent import ReActAgent
from agents.eval.taubench.taubench_eval import TauBenchEvaluation
from llm.openai.openai_llm import OpenAILLM

async def evaluate_with_taubench():
    # Set up agent
    llm = OpenAILLM.create_llm()
    agent = ReActAgent(
        llm=llm,
        role="You are a helpful AI assistant.",
        task="Complete tasks accurately and efficiently.",
        guide="Approach each task methodically and think step-by-step.",
        tools=[],
        max_iterations=5
    )
    
    # Create evaluator with custom settings
    evaluator = TauBenchEvaluation(
        agent=agent,
        model="gpt-4",
        categories=["reasoning", "planning", "tool_use"],
        task_limit=5,
        name="custom_taubench_eval",
        use_reference_model=True,
        reference_model="gpt-3.5-turbo"
    )
    
    # Run evaluation
    results = await evaluator.run_evaluation()
    
    # Print summary
    summary = evaluator.get_summary()
    print(f"Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    return results

# Run the evaluation
results = asyncio.run(evaluate_with_taubench())
```

## Task Categories

TauBench includes the following task categories:

- **reasoning**: Logical reasoning tasks
- **planning**: Sequential planning tasks
- **tool_use**: Tool manipulation and usage
- **code_generation**: Code writing tasks
- **qa**: Question answering
- **math**: Mathematical problem solving
- **science**: Scientific reasoning
- **instruction**: Following complex instructions

## Comparative Evaluation

TauBench supports comparing your agent's performance against a reference model:

```python
evaluator = TauBenchEvaluation(
    agent=agent,
    model="your-model",
    use_reference_model=True,
    reference_model="gpt-4"
)
```

This will run a subset of the evaluation tasks on both your model and the reference model, then compare their performance.

## Data Loader

The `TauBenchDataLoader` class provides access to TauBench datasets:

```python
from agents.eval.taubench.taubench_eval import TauBenchDataLoader

# Initialize data loader
data_loader = TauBenchDataLoader(data_dir="custom/datasets/path")

# List available categories
categories = data_loader.list_available_categories()

# Load tasks for a specific category
tasks = data_loader.load_dataset("reasoning", version="latest", split="test")

# Sample a limited number of tasks
sample_tasks = data_loader.sample_tasks("tool_use", num_samples=5)
```

## Customizing Evaluation

You can customize the TauBench evaluation by:

1. **Selecting categories**: Focus on specific task types
2. **Limiting task count**: Control the number of tasks per category
3. **Using a reference model**: Compare against a baseline model
4. **Custom datasets**: Provide your own directory of compatible TauBench datasets

## Evaluation Results

The evaluation produces detailed metrics for each task and category, including:

- Success rates per category and overall
- Task completion times
- Detailed responses and their metrics
- Comparative performance when using a reference model

Results are saved in JSON format and include both detailed task data and summary metrics. 