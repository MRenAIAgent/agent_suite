from agents.agent_pattern import AgentPattern
from agents.react_prompt_template import ReActPromptTemplate
from agents.react_llm_execute_pattern import ReActLLMExecutionPattern
from agents.prompt import PromptFormatter
from agents.llm_execute_pattern import LLMExecutionPattern


class ReactAgentPattern(AgentPattern):
    """ReAct agent pattern that implements thought-action-observation cycle."""
    llm_execution_pattern: LLMExecutionPattern = ReActLLMExecutionPattern()
