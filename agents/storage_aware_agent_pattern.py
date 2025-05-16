from typing import Optional, Dict, Any

from agents.agent_pattern import AgentPattern
from agents.prompt import PromptFormatter
from agents.storage.storage_manager import IntermediateStepManager
from agents.storage_aware_llm_execute_pattern import (
    StorageAwareLLMExecutionPattern,
    StorageAwareReActExecutionPattern
)


class StorageAwareAgentPattern(AgentPattern):
    """
    Base agent pattern with storage-aware execution.
    
    This pattern uses the StorageAwareLLMExecutionPattern to enable
    more controlled and optimized intermediate step storage.
    """
    
    def __init__(
        self, 
        step_manager: Optional[IntermediateStepManager] = None,
        token_efficient: bool = False,
        prompt_template: Optional[PromptFormatter] = None
    ):
        """
        Initialize the storage-aware agent pattern.
        
        Args:
            step_manager: Optional pre-configured step manager
            token_efficient: Whether to use token-efficient storage
            prompt_template: Optional prompt formatter
        """
        self.prompt_template = prompt_template
        self.llm_execution_pattern = StorageAwareLLMExecutionPattern(
            step_manager=step_manager,
            token_efficient=token_efficient
        )


class StorageAwareReActAgentPattern(StorageAwareAgentPattern):
    """
    ReAct agent pattern with storage-aware execution.
    
    This pattern uses the StorageAwareReActExecutionPattern for ReAct-specific
    logic combined with optimized storage for intermediate steps.
    """
    
    def __init__(
        self, 
        step_manager: Optional[IntermediateStepManager] = None,
        token_efficient: bool = False,
        prompt_template: Optional[PromptFormatter] = None
    ):
        """
        Initialize the storage-aware ReAct agent pattern.
        
        Args:
            step_manager: Optional pre-configured step manager
            token_efficient: Whether to use token-efficient storage
            prompt_template: Optional prompt formatter
        """
        self.prompt_template = prompt_template
        self.llm_execution_pattern = StorageAwareReActExecutionPattern(
            step_manager=step_manager,
            token_efficient=token_efficient
        )


# Factory functions for creating pre-configured patterns

def create_token_efficient_react_pattern(
    max_detailed_steps: int = 5,
    prompt_template: Optional[PromptFormatter] = None
) -> StorageAwareReActAgentPattern:
    """
    Create a token-efficient ReAct agent pattern.
    
    This pre-configured pattern uses the TokenEfficientIntermediateStorage
    implementation to optimize token usage during agent execution.
    
    Args:
        max_detailed_steps: Number of detailed steps to preserve
        prompt_template: Optional prompt formatter
    
    Returns:
        Pre-configured StorageAwareReActAgentPattern
    """
    # Create a custom step manager with token-efficient storage
    step_manager = IntermediateStepManager(
        token_efficient=True,
        max_detailed_steps=max_detailed_steps
    )
    
    # Return the configured pattern
    return StorageAwareReActAgentPattern(
        step_manager=step_manager,
        prompt_template=prompt_template
    )


def create_in_memory_react_pattern(
    prompt_template: Optional[PromptFormatter] = None
) -> StorageAwareReActAgentPattern:
    """
    Create a standard in-memory ReAct agent pattern.
    
    This pattern uses the default in-memory storage implementation
    which keeps all intermediate steps in full detail.
    
    Args:
        prompt_template: Optional prompt formatter
    
    Returns:
        Pre-configured StorageAwareReActAgentPattern
    """
    # Create a default step manager with in-memory storage
    step_manager = IntermediateStepManager(token_efficient=False)
    
    # Return the configured pattern
    return StorageAwareReActAgentPattern(
        step_manager=step_manager,
        prompt_template=prompt_template
    ) 