# Agent Comparison Tool

This tool compares the performance of your custom React agent against LangChain's ReAct agent using the TauBench evaluation framework. It generates detailed metrics and visualizations to help you understand how your agent performs relative to a standard implementation.

## Features

- Evaluates both agents on the same set of tasks for fair comparison
- Tests across multiple categories (reasoning, planning, tool use, etc.)
- Generates visualizations of performance metrics
- Provides detailed task-by-task comparison
- Exports results in multiple formats (JSON, CSV, HTML)

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install langchain langchain_openai langchain_community pandas matplotlib
```

## Running the Comparison

To run a basic comparison with default settings:

```bash
python compare_agents.py
```

### Advanced Options

The script provides several command-line options for customizing the evaluation:

```bash
python compare_agents.py --model "gpt-4" --categories reasoning planning tool_use --task-limit 5 --output-dir "./my_comparison"
```

- `--model`: Specify which LLM to use (default: "gpt-4")
- `--categories`: Select which task categories to evaluate (default: reasoning, planning, tool_use)
- `--task-limit`: Set the maximum number of tasks per category (default: 3)
- `--output-dir`: Directory where results will be saved (default: "agent_comparison")

## Understanding the Results

After running the comparison, you'll find the following outputs in the specified directory:

1. **JSON Results**: Raw evaluation data for each agent
2. **Visualizations**:
   - Overall success rate comparison chart
   - Category-level success rate comparison chart
3. **Summary Tables**:
   - CSV file with detailed metrics
   - HTML table for easy viewing in a browser

## Example Output

The script will generate comparative metrics showing how each agent performed:

```
===== AGENT COMPARISON SUMMARY =====
Custom React success rate: 85.7%
LangChain React success rate: 71.4%
Visualizations saved to: agent_comparison/visualizations
```

The visualizations directory will contain bar charts comparing performance across different categories, making it easy to identify strengths and weaknesses of each implementation.

## Interpreting the Metrics

- **Overall Success Rate**: Percentage of tasks completed successfully across all categories
- **Category Success Rates**: Performance breakdown by task type
- **Task-level Results**: Detailed comparison of how each agent handled specific tasks

## Tips for Fair Comparison

1. Use the same model for both agents to ensure differences are due to agent implementation, not model capability
2. Run evaluations with higher task limits for more statistically significant results
3. Focus on categories relevant to your agent's intended use case
4. Remember that some categories may favor certain agent architectures 