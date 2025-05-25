# Scripts Directory

This directory contains utility scripts for running agent evaluations, benchmarks, and comparisons. These scripts have been moved from the root directory to keep the codebase organized.

## Available Scripts

### Agent Evaluation Scripts

- `run_agent_eval.py` - Run evaluations on the custom ReAct agent
- `run_agent_eval_with_logs.py` - Run evaluations with detailed logging enabled
- `run_langchain_eval.py` - Run evaluations on LangChain agents
- `run_agent_comparison.sh` - Shell script to run comparative evaluations across different agent types
- `run_langchain_eval.sh` - Shell script to run LangChain evaluations

### Benchmarking Scripts

- `run_taubench_comparison.py` - Run the TauBench comparison across different agent types
- `compare_agents.py` - Script to compare different agent implementations

## Usage

Most scripts can be run directly from this directory. For example:

```bash
cd scripts
python run_taubench_comparison.py --model-name gpt-4 --task-limit 5
```

However, note that these scripts may need to be updated to use the newer benchmark structure in the `benchmark/` directory. The preferred approach is to use the scripts in the `examples/` directory which use the newer benchmark structure.

## Deprecation Notice

These scripts are maintained for historical and reference purposes, but new code should use the more structured benchmark implementations in the `benchmark/` directory. 