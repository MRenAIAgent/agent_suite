from abc import ABC
from typing import List, Dict, Any, Optional

from enum import Enum

class PromptFormatType(Enum):
    """Enum defining supported prompt formatting styles."""
    MARKDOWN = "markdown"
    XML = "xml"
    PLAIN = "plain"


class PromptFormatter(ABC):
    """A template for generating structured prompts with defined sections."""
    
    def __init__(self,
                 role: str = None,
                 task: str = None,
                 guide: str = None,
                 output_format: str = None,
                 examples: List[Dict[str, str]] = None,
                 format_type: PromptFormatType = PromptFormatType.MARKDOWN):
        """Initialize prompt template with optional sections.
        
        Args:
            role: Description of the AI assistant's role
            task: The specific task or objective
            guide: Guidelines or constraints for the task
            output_format: Expected format of the response
            examples: List of example input/output pairs
        """
        self.role: Optional[str] = role
        self.task: Optional[str] = task
        self.guide: Optional[str] = guide
        self.output_format: Optional[str] = output_format
        self.examples: List[Dict[str, str]] = examples or []
        self.format_type: PromptFormatType = format_type

    def format_prompt(self, user_input: str, variables: Dict[str, str] = None) -> str:
        """Format a prompt by substituting variables.
        
        Args:
            user_input: The user's input/question
            variables: Optional variables to substitute in the prompt
            
        Returns:
            str: Formatted prompt with all sections and variables substituted
        """
        # Initialize ordered sections dictionary
        if self.examples:
            examples_str = "Examples:\n"
            for i, example in enumerate(self.examples, 1):
                examples_str += f"Example {i}\nInput: {example['input']}\nOutput: {example['output']}\n"
    
        prompt_str = ""
        sections = {
            "ROLE": self.role,
            "TASK": self.task,
            "GUIDE": self.guide,
            "OUTPUT_FORMAT": self.output_format,
            "EXAMPLES": examples_str,
            "USER_INPUT": user_input,
        }
        
        # Filter out None values while preserving order
        sections = {k: v for k, v in sections.items() if v is not None}

        # Join sections with double newlines
        if self.format_type == PromptFormatType.PLAIN:
            prompt_str = "\n\n".join(sections)
        elif self.format_type == PromptFormatType.MARKDOWN:
            prompt_str = "\n\n".join([f"## {key}\n{value}" for key, value in sections.items()])
        elif self.format_type == PromptFormatType.XML:
            prompt_str = "\n\n".join([f"<{key.lower()}>{value}</{key.lower()}>" for key, value in sections.items()])

        
        # Substitute variables if provided
        if variables:
            try:
                # Use str.format() to substitute {variable} placeholders with values from variables dict
                prompt_str = prompt_str.format(**variables)
            except KeyError as exc:
                raise ValueError(f"Variables missing: {variables}") from exc
                
        return prompt_str


class PromptManager:
    """Manages system and user prompts for the agent."""
    
    def __init__(self, system_prompt: str, variables: Dict[str, str] = None):
        """Initialize with system prompt and optional variables.
        
        Args:
            system_prompt: Base system prompt template
            variables: Dictionary of variable names and values to substitute
        """
        self.system_prompt = system_prompt
        self.variables = variables or {}
        self.default_messages = [
            {"role": "system", "content": self._format_prompt(system_prompt, self.variables)}
        ]

    def get_messages(self, user_input: str, history: List[Dict] = None, variables: Dict[str, str] = None) -> List[Dict]:
        """Construct messages list with system prompt, history and user input.
        
        Args:
            user_input: User input template
            history: Optional conversation history
            variables: Optional variables to substitute in user input
        """
        messages = self.default_messages.copy()
        if history:
            messages.extend(history)
        
        # Merge instance variables with method variables
        all_vars = {**self.variables}
        if variables:
            all_vars.update(variables)
            
        formatted_input = self._format_prompt(user_input, all_vars)
        messages.append({"role": "user", "content": formatted_input})
        return messages
        
    def _format_prompt(self, prompt: str, variables: Dict[str, str] = None) -> str:
        """Format a prompt by substituting variables.
        
        Args:
            prompt: Prompt template with {variable} placeholders
            variables: Variables to substitute, defaults to instance variables
        """
        vars_to_use = variables if variables is not None else self.variables
        try:
            return prompt.format(**vars_to_use)
        except KeyError:
            # Return unformatted if variables missing
            return prompt