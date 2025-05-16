from agents.agent_pattern import AgentPattern
from agents.memory_react_execute_pattern import MemoryReActLLMExecutionPattern
from agents.prompt import PromptFormatter


class MemoryReactAgentPattern(AgentPattern):
    """
    ReAct agent pattern that implements thought-action-observation cycle
    with InMemoryIntermediateStorage for storing steps.
    """
    
    def __init__(self, prompt_template: PromptFormatter = None):
        """
        Initialize the memory-enhanced ReAct agent pattern.
        
        Args:
            prompt_template: Optional prompt formatter to use
        """
        self.prompt_template = prompt_template
        self.llm_execution_pattern = MemoryReActLLMExecutionPattern() 