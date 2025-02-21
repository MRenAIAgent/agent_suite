from datetime import datetime
from typing import List, Dict, Any

from agents.agent_pattern import AgentPattern
from agents.agent import Agent
from agents.prompt import PromptFormatter
from agents.llm_execute_pattern import LLMExecutionPattern
from llm.llm import LLMBase
from tools.tool import Tool


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
    def __init__(self):
        """Initialize ReAct prompt template with predefined format."""
        super().__init__(
            role="You are an AI assistant that carefully approaches tasks step-by-step:\n"
                 "1. First THINK about what needs to be done\n"
                 "2. Then decide on an ACTION to take\n"
                 "3. Execute the action and observe the OBSERVATION\n"
                 "4. Repeat steps 1-3 until the task is complete\n"
                 "5. Finally, provide your [FINAL ANSWER]",
            
            task="Break down the user's request into a series of steps. For each step:\n"
                 "- Think about what information you need\n"
                 "- Choose an appropriate action to take\n"
                 "- Use the observation to plan your next step\n"
                 "Continue until you can provide a complete answer.",
            
            guide="- Always start with a Thought about what you need to do\n"
                  "- Format your steps as:\n"
                  "  Thought: your reasoning\n"
                  "  Action: tool_name\n"
                  "  Action Input: {\"param\": \"value\"}\n"
                  "  Observation: tool output\n"
                  "- Use [FINAL ANSWER] when you have the complete solution",
            
            examples=[{
                "input": "What's the weather in New York?",
                "output": "Thought: I need to check the current weather in New York\n"
                         "Action: get_weather\n"
                         "Action Input: {\"location\": \"New York\"}\n"
                         "Observation: 72°F, Partly cloudy\n"
                         "Thought: I have the weather information now\n"
                         "[FINAL ANSWER] The current weather in New York is 72°F and partly cloudy."
            }]
        )
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


class ReActLLMExecutionPattern(LLMExecutionPattern):
    """ReAct agent pattern that implements thought-action-observation cycle."""
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse raw LLM response into structured format."""
        # Implement ReAct parsing logic here
        pass

    


class ReactAgentPattern(AgentPattern):
    """ReAct agent pattern that implements thought-action-observation cycle."""
    prompt_template: PromptFormatter = ReActPromptTemplate()
    llm_execution_pattern: LLMExecutionPattern = ReActLLMExecutionPattern()


class ReactAgent(Agent):
    """Agent that uses ReAct (Reasoning and Acting) approach to solve tasks."""

    def __init__(self, llm: LLMBase, system_prompt: str, tools: List[Tool] = None):
        super().__init__(llm, system_prompt, tools)

    def process(self, user_input: str, model: str) -> str:
        """Process user input using ReAct approach.
        
        The agent thinks and acts in steps, using tools when needed,
        until it reaches a final answer.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            
        Returns:
            str: The agent's final response
        """
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        while True:
            response = self.llm.chat_completion(
                model=model,
                messages=messages
            )
            
            # Check if this is the final answer
            if "[FINAL ANSWER]" in response:
                # Extract final answer (remove the flag)
                final_answer = response.replace("[FINAL ANSWER]", "").strip()
                
                # Update history
                self.memory_manager.add({"role": "user", "content": user_input})
                self.memory_manager.add({"role": "assistant", "content": final_answer})
                
                # Log interaction
                self.log_manager.log_interaction(
                    user_input=user_input,
                    agent_response=final_answer,
                    model=model,
                    timestamp=datetime.now().isoformat()
                )
                
                return final_answer
            
            # Add the thought/action to messages for next iteration
            messages.append({
                "role": "assistant",
                "content": response
            })

    async def aprocess(self, user_input: str, model: str) -> str:
        """Async version of process method."""
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        
        while True:
            response = await self.llm.chat_completion(
                model=model,
                messages=messages,
                tools=self.tools
            )
            
            # Check if this is the final answer
            if "[FINAL ANSWER]" in response.content:
                # Extract final answer (remove the flag)
                final_answer = response.content.replace("[FINAL ANSWER]", "").strip()
                
                # Update history
                self.memory_manager.add({"role": "user", "content": user_input})
                self.memory_manager.add({"role": "assistant", "content": final_answer})
                
                # Log interaction
                self.log_manager.log_interaction(
                    user_input=user_input,
                    agent_response=final_answer,
                    model=model,
                    timestamp=datetime.now().isoformat()
                )
                
                return final_answer
            
            if response.tool_calls:
                tool_results = await self.handle_tool_calls(response.tool_calls)
                
                # Add the thought/action and tool results to messages
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.tool_calls
                })
                messages.append({
                    "role": "tool",
                    "content": str(tool_results)
                })
            else:
                # Add just the thought/action to messages
                messages.append({
                    "role": "assistant",
                    "content": response.content
                })
