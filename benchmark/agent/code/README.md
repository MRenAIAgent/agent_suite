# Agent Benchmark Code

This directory contains the implementation code for agent benchmarking frameworks.

## Agent Benchmark

The main benchmark implementation is provided in `agent_benchmark.py`. This tool benchmarks three types of agents using the TauBench evaluation framework:

1. Custom ReActAgent from this repo
2. LangChain ReAct agent (ZERO_SHOT_REACT_DESCRIPTION)
3. LangChain Structured Chat agent (using create_structured_chat_agent)

### Requirements

- Python 3.9+
- Required packages (install with `pip install -r requirements.txt`):
  - langchain
  - langchain_openai
  - langchain_anthropic (if using Anthropic models)
  - pandas
  - matplotlib
  - numpy

### Usage

Run the benchmark script:

```bash
# Run with all TauBench test cases
python agent_benchmark.py --llm-provider openai --model-name gpt-4 --output-dir ../results

# Run with specific categories and limited number of cases
python agent_benchmark.py --categories math planning tool_use --max-cases 5
```

### Command-line Arguments

- `--llm-provider`: LLM provider to use (openai or anthropic, default: openai)
- `--model-name`: Model name to use (default: gpt-4)
- `--max-cases`: Maximum number of test cases to evaluate (optional, default: all)
- `--categories`: Categories of test cases to use (optional, specify one or more)
- `--output-dir`: Directory to save reports to (default: ../results)
- `--verbose`: Enable verbose logging

### Tools Included

The benchmark includes the following tools for testing:

1. Calculator: For arithmetic operations
2. Weather: For weather information (mock implementation)
3. Translation: For text translation (mock implementation)
4. Search: For web search (uses Serper API if available, otherwise mock)
5. Calendar: For event scheduling (mock implementation)

### Implementation Details

- All three agent types use the same set of tools for fair comparison
- The benchmark evaluates agents on a subset of TauBench test cases
- Metrics include accuracy, timeout rate, and average execution time
- Results are visualized in charts and summarized in an HTML report 