# LangChain ReAct Agent Evaluation

This tool evaluates a LangChain ReAct agent using the TauBench minimal implementation. It provides detailed performance metrics across different reasoning, planning, and tool use tasks.

## Features

- Evaluates LangChain's ReAct agent on standard TauBench tasks
- Tests multiple categories including reasoning, planning, and tool use
- Automatically handles tool initialization and fallbacks
- Provides detailed performance metrics
- Saves results in JSON format for further analysis

## Prerequisites

Before running the evaluation, ensure you have installed:

```bash
pip install langchain langchain_openai langchain_community
```

For OpenAI models, set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Running the Evaluation

To run with default settings:

```bash
./run_langchain_eval.sh
```

### Customizing the Evaluation

You can customize the evaluation with several options:

```bash
./run_langchain_eval.sh --model "gpt-4" --categories "reasoning planning" --task-limit 5 --output-dir "./results" --verbose
```

Available options:

- `--model`: The LLM model to use (default: "gpt-4")
- `--categories`: Task categories to test (default: "reasoning planning tool_use")
- `--task-limit`: Maximum tasks per category (default: 3)
- `--output-dir`: Where to save results (default: "langchain_eval_results")
- `--verbose`: Enable verbose agent execution output (useful for debugging)

## Understanding Results

The evaluation results include:

- **Overall success rate**: Percentage of tasks completed successfully
- **Category breakdown**: Performance metrics per task category
- **Task-level details**: Individual responses and success metrics for each task

Example output:

```
===== EVALUATION SUMMARY =====
Model: gpt-4
Overall success rate: 78.6%
Successful tasks: 11/14
Results saved to: langchain_eval_results/langchain_results_20250503_180142.json

--- Category Breakdown ---
Reasoning: 85.0%
Planning: 75.0%
Tool_use: 67.0%
```

## Troubleshooting

### Missing Tools

The script will automatically handle missing tools by creating mock versions:

1. If Wikipedia API is unavailable, it will create a mock Wikipedia tool
2. If DuckDuckGo search is unavailable, it will create a mock search tool
3. Calculator tool is always included

### OpenAI API Issues

If you encounter errors related to OpenAI API, check:

1. Your API key is properly set
2. You have sufficient credits available
3. The specified model name is correct

### Package Dependencies

If you encounter import errors, install the required packages:

```bash
pip install langchain langchain_openai langchain_community openai
```

## Extending the Evaluation

You can extend the evaluation by:

1. Adding custom tasks to the datasets:
   - Create JSONL files in `agents/eval/taubench/datasets/<category>/test.jsonl`

2. Adding more tools to the LangChainReActAgent class:
   - Modify the `_setup_tools` method in the `langchain_react_eval.py` script

3. Customizing the evaluation metrics:
   - Edit `agents/eval/taubench/taubench_minimal.py` to adjust the metrics calculation 