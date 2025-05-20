# SSH Key Management Evaluation

This module provides specialized tasks for evaluating agents' ability to manage SSH keys.

## Overview

The SSH Key Management evaluation extends the TauBench framework to assess how well an agent can handle common SSH key operations, including:

- Generating SSH key pairs
- Adding keys to authorized_keys
- Managing the SSH agent
- Configuring SSH clients
- Securely handling SSH keys

## Installation

Ensure you have the TauBench framework installed:

```bash
pip install -r agents/eval/taubench/requirements.txt
```

## Usage

### Command Line

Run the SSH key evaluation directly from the command line:

```bash
# Run the evaluation with default settings
python -m agents.eval.taubench.run_ssh_key_eval

# Customize the evaluation
python -m agents.eval.taubench.run_ssh_key_eval --model gpt-4 --task-limit 3
```

### Available Options

- `--model`: The LLM model to use (default: "gpt-3.5-turbo")
- `--output-dir`: Directory to save results (default: "evaluation_results")
- `--task-limit`: Maximum number of SSH key tasks to run (default: 5)

### Programmatic Usage

You can also integrate the SSH key evaluation into your own code:

```python
import asyncio
from agents.react_agent import ReActAgent
from agents.eval.taubench.run_ssh_key_eval import run_ssh_key_evaluation, setup_agent
from llm.openai.openai_llm import OpenAILLM

async def evaluate_ssh_key_management():
    # Set up agent (or use your own custom agent)
    agent = setup_agent(model="gpt-4")
    
    # Run evaluation
    results = await run_ssh_key_evaluation(
        agent=agent,
        model="gpt-4",
        task_limit=5
    )
    
    # Access results
    print(f"Success rate: {results['summary']['overall_success_rate']:.2%}")
    return results

# Run the evaluation
results = asyncio.run(evaluate_ssh_key_management())
```

## SSH Key Tasks

The evaluation includes the following tasks:

1. **Generate a new SSH key pair** using Ed25519 algorithm
2. **Add a public key to authorized_keys** for the current user
3. **Check available SSH keys** in the SSH agent
4. **Add a private SSH key to the SSH agent**
5. **Configure SSH** to use a specific key for GitHub
6. **Copy a public SSH key to clipboard** for use with web services
7. **Test SSH connection** to GitHub
8. **Generate an SSH key with custom comment**
9. **Change the passphrase** for an existing SSH key
10. **Back up SSH keys** to a secure location

## Validation

Each task comes with validation criteria to assess whether the agent's response correctly addresses the task. The validator looks at:

- Command syntax and structure
- Security best practices
- Key existence and properties
- Configuration changes

## Adding Custom Tasks

To add your own SSH key tasks, modify the `SSH_KEY_TASKS` list in `ssh_key_tasks.py`:

```python
# Example custom task
{
    "id": "ssh_key_custom",
    "category": "tool_use",
    "instruction": "Your custom task instruction",
    "reference": "Expected command to accomplish the task",
    "evaluation_criteria": {
        "custom_criterion": True
    }
}
```

Then update the `SSHKeyValidator.validate_operation()` method to handle your custom criteria.

## Results

Results are saved in JSON format and include:

- Overall success rate
- Detailed task results
- Agent responses
- Validation details
- Timing information

Example output file: `ssh_key_eval_20230501_120000.json` 