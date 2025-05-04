# Agent Comparison with TauBench Minimal

This guide explains how to compare your custom React agent with LangChain's ReAct agent using the minimal TauBench implementation.

## Background

The TauBench minimal implementation enables agent evaluation without requiring external dependencies. It provides sample tasks across multiple categories:

- **Reasoning**: Logical reasoning problems
- **Planning**: Step-by-step planning tasks
- **Tool use**: Tasks requiring tool manipulation
- **QA**: Question answering tasks
- **Code generation**: Tasks requiring code implementation

## Quick Start

Run the comparison with default settings:

```bash
./run_agent_comparison.sh
```

This will:
1. Evaluate your custom React agent on sample tasks
2. Evaluate LangChain's ReAct agent on the same tasks
3. Generate visualizations and a comparison report

## Options

Customize the evaluation with these options:

```bash
./run_agent_comparison.sh --model "gpt-4" --categories "reasoning planning" --task-limit 5 --output-dir "./my_comparison"
```

- `--model`: The LLM model to use (default: "gpt-4")
- `--categories`: Task categories to test (default: "reasoning planning tool_use")
- `--task-limit`: Maximum tasks per category (default: 3)
- `--output-dir`: Where to save results (default: "agent_comparison_results")
- `--skip-langchain`: Only evaluate your custom agent (skip LangChain's agent)
- `--langchain-only`: Only evaluate the LangChain agent (skip custom agent)

## Evaluation Modes

The script supports three evaluation modes:

1. **Comparison Mode** (default): Evaluates both agents and generates comparison visuals
   ```bash
   ./run_agent_comparison.sh
   ```

2. **Custom Agent Only**: Evaluates only your custom React agent
   ```bash
   ./run_agent_comparison.sh --skip-langchain
   ```

3. **LangChain Only**: Evaluates only the LangChain React agent
   ```bash
   ./run_agent_comparison.sh --langchain-only
   ```

## Troubleshooting

### LangChain Agent Issues

If you encounter issues with the LangChain agent, check the following:

1. Make sure `langchain`, `langchain_openai`, and `langchain_community` packages are installed:
   ```bash
   pip install langchain langchain_openai langchain_community
   ```

2. If the agent fails to initialize, try running with only your custom agent:
   ```bash
   ./run_agent_comparison.sh --skip-langchain
   ```

3. If you want to test only the LangChain agent:
   ```bash
   ./run_agent_comparison.sh --langchain-only
   ```

### OpenAI API Issues

If using OpenAI models, ensure you have:
1. Set the `OPENAI_API_KEY` environment variable
2. Have an active subscription with sufficient credits

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Interpreting Results

After running the comparison, you'll find:

1. **HTML Report**: A detailed comparison report with metrics and visualizations
2. **JSON Results**: Raw evaluation data for each agent
3. **Visualizations**: Charts comparing success rates across categories

The report highlights:
- Overall success rates for both agents
- Category-by-category performance comparison
- Detailed metrics on successful tasks

## Example Output

Success rates are calculated based on how well each agent's responses match the expected reference answers:

```
===== AGENT COMPARISON SUMMARY =====
Custom React success rate: 85.7%
LangChain React success rate: 71.4%
Visualizations saved to: agent_comparison_results/visualizations
```

## Extending the Evaluation

You can extend the evaluation by:

1. Adding custom tasks to the datasets:
   - Create JSONL files in `agents/eval/taubench/datasets/<category>/test.jsonl`

2. Modifying evaluation metrics:
   - Edit `agents/eval/taubench/taubench_minimal.py` to customize the `calculate_metrics` function

3. Adding new task categories:
   - Create a new directory in `agents/eval/taubench/datasets/`
   - Add sample tasks in the JSONL format 