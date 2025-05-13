"""
Sequential thinking pattern implementation for MCP client integration.
"""
from typing import Dict, Any, List, Optional

from agents.base_classes.base_think_pattern import AgentThinkPattern


class SequentialThinkPattern(AgentThinkPattern):
    """
    Sequential thinking pattern that breaks down complex problems into a sequence of steps.
    
    This thinking pattern is designed to work with MCP servers that support step-by-step
    reasoning and can be used to solve complex problems by breaking them down into
    a sequence of smaller, manageable steps.
    """
    
    def __init__(self):
        super().__init__(
            name="Sequential Thinking",
            description="Process complex problems step by step, with each step building on previous ones. "
                       "Allows for revisions, branching, and verification of intermediate steps."
        )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with sequential thinking."""
        processed = {
            "problem_statement": input_data.get("input", ""),
            "current_thought_number": 1,
            "total_thoughts": 5,  # Default to 5 steps, but can be adjusted
            "thoughts": [],
            "need_more_thoughts": True,
            "revisions": [],
            "branches": [],
            "conclusion": ""
        }
        
        # Add any existing thoughts from the response
        if "response" in input_data and "thoughts" in input_data["response"]:
            processed["thoughts"] = input_data["response"]["thoughts"]
            processed["current_thought_number"] = len(processed["thoughts"]) + 1
        
        return processed
    
    def format_thoughts(self, thoughts: Dict[str, Any]) -> str:
        """Format thoughts sequentially."""
        formatted = "# Sequential Thinking Process\n\n"
        
        # Add problem statement
        formatted += f"## Problem Statement\n{thoughts.get('problem_statement', '')}\n\n"
        
        # Add thoughts
        formatted += "## Thought Sequence\n"
        for i, thought in enumerate(thoughts.get("thoughts", [])):
            formatted += f"### Thought {i+1}/{thoughts.get('total_thoughts', 5)}\n"
            formatted += f"{thought}\n\n"
        
        # Add revisions if any
        if thoughts.get("revisions", []):
            formatted += "## Revisions\n"
            for i, revision in enumerate(thoughts.get("revisions", [])):
                formatted += f"### Revision {i+1}\n"
                formatted += f"**Revising Thought {revision.get('revises_thought', '')}**\n"
                formatted += f"{revision.get('content', '')}\n\n"
        
        # Add branches if any
        if thoughts.get("branches", []):
            formatted += "## Alternative Branches\n"
            for i, branch in enumerate(thoughts.get("branches", [])):
                formatted += f"### Branch {i+1}\n"
                formatted += f"**Branching from Thought {branch.get('branch_from', '')}**\n"
                formatted += f"{branch.get('content', '')}\n\n"
        
        # Add conclusion if available
        if thoughts.get("conclusion", ""):
            formatted += f"## Conclusion\n{thoughts.get('conclusion', '')}\n"
        elif not thoughts.get("need_more_thoughts", True):
            formatted += "## Conclusion\nAnalysis complete.\n"
        else:
            formatted += "## Status\nThinking in progress...\n"
        
        return formatted

    def get_prompt_format(self) -> str:
        """Get the prompt format for sequential thinking."""
        return f"""
Thinking Pattern: {self.name}

Description:
{self.description}

When processing this task, follow these guidelines:

1. Break down the problem into sequential steps
2. Explicitly state your current thought number and estimated total thoughts
3. Build on previous thoughts in your sequence
4. When appropriate, revise earlier thoughts if you gain new insights
5. Consider alternative branches when useful
6. Verify intermediate conclusions before proceeding
7. Provide a clear conclusion when the thought sequence is complete

Current thought number: {{current_thought_number}}
Total estimated thoughts: {{total_thoughts}}
""" 