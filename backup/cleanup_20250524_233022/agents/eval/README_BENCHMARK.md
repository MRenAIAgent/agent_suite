# Agent Benchmark

This tool benchmarks three types of agents using the TauBench evaluation framework:

1. Custom ReActAgent from this repo
2. LangChain ReAct agent (ZERO_SHOT_REACT_DESCRIPTION)
3. LangChain Structured Chat agent (using create_structured_chat_agent)

## Requirements

- Python 3.9+
- Required packages (install with `pip install -r requirements.txt`):
  - langchain
  - langchain_openai
  - langchain_anthropic (if using Anthropic models)
  - pandas
  - matplotlib
  - numpy

## Environment Variables

Set the following environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic (optional)
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For Serper search (optional)
export SERPER_API_KEY=your_serper_api_key
```

## Usage

Run the benchmark script:

```bash
# Run with all TauBench test cases
python agent_benchmark.py --llm-provider openai --model-name gpt-4.1 --output-dir results

# Run with specific categories and limited number of cases
python agent_benchmark.py --categories math planning tool_use --max-cases 5
```

### Command-line Arguments

- `--llm-provider`: LLM provider to use (openai or anthropic, default: openai)
- `--model-name`: Model name to use (default: gpt-4.1)
- `--max-cases`: Maximum number of test cases to evaluate (optional, default: all)
- `--categories`: Categories of test cases to use (optional, specify one or more)
- `--output-dir`: Directory to save reports to (default: benchmark_results)
- `--verbose`: Enable verbose logging

## Output

The benchmark generates the following outputs in the specified directory:

1. `agent_metrics.csv`: CSV file with metrics for each agent
2. `accuracy_comparison.png`: Bar chart comparing agent accuracy
3. `execution_time_comparison.png`: Bar chart comparing agent execution time
4. `detailed_results.json`: Detailed JSON results for each test case
5. `report.html`: HTML report with summary and visualizations

## Tools Included

The benchmark includes the following tools for testing:

1. Calculator: For arithmetic operations
2. Weather: For weather information (mock implementation)
3. Translation: For text translation (mock implementation)
4. Search: For web search (uses Serper API if available, otherwise mock)
5. Calendar: For event scheduling (mock implementation)

## Implementation Details

- All three agent types use the same set of tools for fair comparison
- The benchmark evaluates agents on a subset of TauBench test cases
- Metrics include accuracy, timeout rate, and average execution time
- Results are visualized in charts and summarized in an HTML report

## Extending the Benchmark

- Add new tools by creating classes that inherit from `Tool`
- Add new agent types by creating initialization functions
- Add new evaluation metrics by modifying the evaluation functions 