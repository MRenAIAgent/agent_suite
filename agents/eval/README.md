# Agent Evaluation Framework

This module provides tools for evaluating and benchmarking agent performance using established evaluation frameworks. The implementation currently supports:

1. **AgentBench** - A comprehensive benchmark for evaluating LLMs as agents across diverse environments
2. **AgentBoard** - An analytical evaluation framework for multi-turn agent tasks in partially-observable environments

## Installation

The evaluation framework is part of the agent_suite package and uses its dependencies. Make sure you have all required environment variables set in your `.env` file, particularly for OpenAI API access.

## Usage

### Quick Start

To run a quick evaluation on a ReActAgent:

```bash
# Run both evaluation frameworks with default settings
python -m agents.eval.run_evaluations

# Run a specific framework
python -m agents.eval.run_evaluations --frameworks agentbench

# Customize the evaluation
python -m agents.eval.run_evaluations --model gpt-4 --env-limit 2 --task-limit 1 --turn-limit 3
```

### Command Line Options

The following options are available for the evaluation runner:

- `--model`: The LLM model to use (default: "gpt-3.5-turbo")
- `--frameworks`: The evaluation frameworks to use (choices: "agentbench", "agentboard")
- `--output-dir`: Directory to save results (default: "evaluation_results")
- `--env-limit`: Limit number of environments/domains per framework (default: 3)
- `--task-limit`: Limit number of tasks per environment/domain (default: 2)
- `--turn-limit`: Limit number of turns per task for multi-turn evaluations (default: 5)

### Programmatic Usage

You can also use the evaluation frameworks directly in your code:

```python
import asyncio
from agents.react_agent import ReActAgent
from agents.eval.agentbench.agentbench_eval import AgentBenchEvaluation
from llm.openai.openai_llm import OpenAILLM
from tools.serper_search import SerperSearchTool

# Set up agent
llm = OpenAILLM.create_llm()
search_tool = SerperSearchTool(query="")
agent = ReActAgent(
    llm=llm,
    role="You are a helpful AI assistant.",
    task="Complete tasks accurately and efficiently.",
    guide="Approach each task methodically and think step-by-step.",
    tools=[search_tool],
    max_iterations=5
)

# Run evaluation
async def evaluate_agent():
    # Create evaluator
    evaluator = AgentBenchEvaluation(
        agent=agent,
        model="gpt-3.5-turbo",
        environments=["os", "db"],
        task_limit=2
    )
    
    # Run evaluation
    results = await evaluator.run_evaluation()
    
    # Print summary
    summary = evaluator.get_summary()
    print(f"Overall success rate: {summary.get('overall_success_rate', 0):.2%}")
    
    return results

# Run the evaluation
results = asyncio.run(evaluate_agent())
```

## Evaluation Frameworks

### AgentBench

AgentBench evaluates agents across diverse environments:

- **Operating System (OS)**: Basic command line tasks
- **Database (DB)**: SQL queries and database operations
- **Knowledge Graph (KG)**: Knowledge graph querying and manipulation
- **Digital Card Game (DCG)**: Strategic decision making in games
- **Lateral Thinking Puzzles (LTP)**: Creative problem solving

Each environment contains tasks that test different agent capabilities. The evaluation produces metrics on:

- Success rate per environment and overall
- Task completion time
- Iterations required per task

### AgentBoard

AgentBoard focuses on analytical evaluation of multi-turn agent behavior across domains:

- **Navigation**: Spatial navigation and path finding
- **Reasoning**: Logical reasoning and puzzle solving
- **Knowledge**: Knowledge retrieval and explanation
- **Planning**: Task planning and sequencing
- **Translation**: Language translation tasks
- **Coding**: Programming and code generation
- **Arithmetic**: Mathematical reasoning
- **Creativity**: Creative content generation
- **Interaction**: Multi-step user interactions

This framework emphasizes:

- Progress rates through multi-turn conversations
- Grounding accuracy (using available information correctly)
- Success in completing complex, multi-step tasks

## Evaluation Results

Results are saved in the specified output directory as JSON files:

- Individual framework results: `{framework}_{timestamp}.json`
- Overall evaluation summary: `evaluation_summary_{timestamp}.json`

The summary provides high-level metrics while the detailed result files contain complete information about each task, including the agent's responses.

## Extending the Framework

### Adding New Evaluation Frameworks

To add a new evaluation framework:

1. Create a new directory under `agents/eval/`
2. Implement a class that inherits from `BaseEvaluation` 
3. Implement the required methods, especially `run_evaluation()` and `get_summary()`
4. Update `run_evaluations.py` to include your new framework

### Customizing Existing Frameworks

You can customize the existing frameworks by:

- Adding new environments/domains
- Creating more realistic tasks
- Implementing more sophisticated evaluation metrics
- Connecting to real-world APIs instead of simulated environments

## Comparing Different Agents

To compare different agent types (e.g., one-shot vs. ReAct):

```python
async def compare_agents():
    # Set up agents
    one_shot_agent = setup_one_shot_agent()
    react_agent = setup_react_agent()
    
    # Run evaluations on each agent
    one_shot_results = await run_evaluations(one_shot_agent, "gpt-3.5-turbo")
    react_results = await run_evaluations(react_agent, "gpt-3.5-turbo")
    
    # Compare results
    print("One-Shot Agent:")
    print(f"- Success Rate: {one_shot_results['frameworks']['agentbench']['summary']['overall_success_rate']:.2%}")
    
    print("ReAct Agent:")
    print(f"- Success Rate: {react_results['frameworks']['agentbench']['summary']['overall_success_rate']:.2%}")
``` 