from abc import ABC
from agents.prompt import PromptFormatter
from agents.llm_execute_pattern import LLMExecutionPattern

class AgentPattern(ABC):
    """Base class defining the core components of an agent pattern."""
    prompt_template: PromptFormatter = None
    llm_execution_pattern: LLMExecutionPattern = None
