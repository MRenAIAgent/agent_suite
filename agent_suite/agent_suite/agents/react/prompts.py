from agents.prompt import PromptFormatter, PromptFormatType
from typing import Dict, Any, List
from enum import Enum



class ReActPromptTemplate(PromptFormatter):
    """Predefined template for ReAct agent prompts with thought-action-observation format.
    Schema for formatting ReAct agent responses with thought-action-observation pattern.
    This class defines the response format for a ReAct agent that:
    1. Breaks down tasks into steps through reasoning
    2. Takes actions based on thoughts
    3. Observes results of actions
    4. Iterates through thought-action-observation until reaching a solution
    5. Provides a final answer
    """
    def __init__(self, role: str, task: str, guide: str, examples: List[str], prompt_format_type: PromptFormatType = PromptFormatType.MARKDOWN):
        """Initialize ReAct prompt template with predefined format."""
        approach = "\nBreaks down tasks into steps through reasoning. Takes actions based on thoughts and observations."

        self.output_format = """
            Break down the user's request into a series of steps. 
            Break into steps if you need using tools to get information. 
            If you don't need to use tools, just think and do one step.
            For each step:
                - Think about what information you need
                - Choose an appropriate action to take
                - Use the observation from action to plan your next step
            Continue until you can provide a complete answer.

            Use the following format:
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of tools
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question                             
        """
        super().__init__(
            role = role,
            task = task,
            guide = guide + approach,
            examples = examples,
            output_format = self.output_format,
            prompt_format_type = prompt_format_type)

    def format_thought(self, thought: str) -> str:
        """Format a thought/reasoning step.
        Used when agent is analyzing the task or planning next steps.
        
        Args:
            thought: The agent's reasoning or analysis text
            
        Returns:
            str: Formatted thought prefixed with "Thought:"
        """
        return f"Thought: {thought}"

    def format_action(self, action: str, action_input: Dict[str, Any]) -> str:
        """Format an action step with its input parameters.
        Used when agent decides to use a tool/action.
        
        Args:
            action: Name of the tool/action to execute
            action_input: Parameters needed for the action
            
        Returns:
            str: Formatted action and input on separate lines
        """
        return f"Action: {action}\nAction Input: {action_input}"

    def format_observation(self, observation: str) -> str:
        """Format the observation from an action's result.
        Used to capture output from executed actions.
        
        Args:
            observation: Result/output from the action
            
        Returns:
            str: Formatted observation prefixed with "Observation:"
        """
        return f"Observation: {observation}"

    def format_final_answer(self, answer: str) -> str:
        """Format the final solution/response.
        Used when agent has completed all necessary steps.
        
        Args:
            answer: The final solution/response
            
        Returns:
            str: Formatted answer with [FINAL ANSWER] tag
        """
        return f"[FINAL ANSWER] {answer}"

    def format_response(self, response: Dict[str, Any]) -> str:
        """Format a complete ReAct response combining all components.
        Assembles the full response following the ReAct pattern:
        Thought -> Action -> Observation -> (repeat) -> Final Answer
        
        Args:
            response: Dict containing any of:
                     thought, action, action_input, observation, final_answer
            
        Returns:
            str: Complete formatted response with all included components
        """
        sections = []
        
        if "thought" in response:
            sections.append(self.format_thought(response["thought"]))
            
        if "action" in response and "action_input" in response:
            sections.append(self.format_action(response["action"], response["action_input"]))
            
        if "observation" in response:
            sections.append(self.format_observation(response["observation"]))
            
        if "final_answer" in response:
            sections.append(self.format_final_answer(response["final_answer"]))
            
        return "\n".join(sections)

