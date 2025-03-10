import asyncio
import json
from datetime import datetime
from typing import List

from agents.agent import Agent
from agents.prompt import PromptManager
from agents.react_prompt_template import ReActPromptTemplate
from agents.react_agent_fc.react_execute_pattern_fc import ReActFunction
from agents.utils.function_call import convert_to_function_call
from llm.llm import LLMBase
from tools.tool import Tool


class ReactAgentFC(Agent):
    """React Agent that uses function calling for structured reasoning and actions.
    
    This agent extends ReactAgent by using OpenAI function calling to enforce
    a structured format for the agent's reasoning process.
    """

    def __init__(
            self,
            llm: LLMBase,
            role: str,
            task: str,
            guide: str,
            examples: list[str] = None,
            tools: List[Tool] = None,
            max_iterations: int = 5):
        react_prompt_template = ReActPromptTemplate(
            role=role,
            task=task,
            guide=guide,
            examples=examples
        )
        system_prompt = react_prompt_template.format_prompt()
        prompt_manager = PromptManager(system_prompt)
        super().__init__(llm, prompt_manager, tools)

        self.agent_pattern = ReActFunction(
            thought="",
            action="",
            action_input=None,
            tool_calls=None,
            observation="",
            final_answer=""
        )
        self.max_iterations = max_iterations

        
    def format_execution_pattern(self):
        """Format the execution pattern as a function call.
        
        Returns:
            Dict: The execution pattern formatted as an OpenAI function call
        """
        return convert_to_function_call(ReActFunction, ReActFunction.model_json_schema())
    
    async def arun(self, user_input: str, model: str) -> str:
        """Process user input using ReAct approach with function calling.
        
        The agent thinks and acts in steps, using tools when needed,
        until it reaches a final answer. Uses function calling for structured reasoning.
        
        Args:
            user_input: The user's question or request
            model: The LLM model to use
            
        Returns:
            str: The agent's final response
        """
        GREEN = "\033[92m"
        RESET = "\033[0m"
        
        def pretty_json(obj) -> str:
            return json.dumps(obj, indent=2, ensure_ascii=False)
        
        messages = self.prompt_manager.get_messages(
            user_input,
            self.memory_manager.get_history()
        )
        self.log_manager.log_debug("# Initial Messages\n```json\n" + pretty_json(messages) + "\n```")
        
        iterations = 0
        try:
            while iterations < self.max_iterations:
                self.log_manager.log_debug(f"\n## ReAct Iteration {iterations + 1}/{self.max_iterations}")
                # Get the ReAct function definition using the format_execution_pattern method
                react_function = self.format_execution_pattern()
                # Add all tools plus the ReAct function to the tools list
                self.tools = self.tools or []
                all_tools = [tool.convert_to_function_call() for tool in self.tools] + [react_function]
                self.log_manager.log_debug(f"\n### Tools: {all_tools}")
                
                # Force the model to use the ReAct function
                response = self.llm.chat_completion(
                    model=model,
                    messages=messages,
                    tools=all_tools,
                    tool_choice={"type": "function", "function": {"name": "reactfunction"}}
                )
                
                self.log_manager.log_debug(f"\n### LLM Response {GREEN}\n```json\n{(response.choices[0].message)}\n```{RESET}")
                
                # parse reponse into self.agent_pattern
                self.agent_pattern.parse_response(response.choices[0].message)

                iterations += 1
                if iterations >= self.max_iterations-1:
                    # last iteration
                    final_answer = self.agent_pattern.get_final_answer()
                    observation = self.agent_pattern.get_observation()
                    if final_answer:
                        return final_answer
                    elif observation:
                        return observation
                    else:
                        return "No final answer or observation"


                tool, tool_input = self.agent_pattern.get_tool_call()
                if tool:
                    tool_results = await self.handle_single_tool_call(tool, tool_input)
                    self.log_manager.log_debug("\n### Tool Results\n```json\n" + pretty_json(tool_results) + "\n```")

                    # Format the tool calls properly with required fields
                    formatted_tool_calls = self.agent_pattern.format_tool_call()
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": formatted_tool_calls
                    })
                    # Add tool results
                    messages.append({
                        "role": "tool",
                        "tool_call_id": formatted_tool_calls[0]["id"],
                        "content": json.dumps(tool_results)
                    })
                    continue #do not support direct return from tool calls


                if self.agent_pattern.should_continue():
                    formatted_steps = self.agent_pattern.format_intermediate_steps()
                    messages.append({
                        "role": "system",
                        "content": formatted_steps
                    })
                    self.log_manager.log_debug(f"\n### Intermediate Steps {GREEN}\n```\n{formatted_steps}\n```{RESET}")
                else:
                    final_answer = self.agent_pattern.get_final_answer()
                    self.log_manager.log_debug(f"\n### Final Answer {GREEN}\n```\n{final_answer}\n```{RESET}")
                    
                    # Update history
                    self.memory_manager.add({"role": "user", "content": user_input})
                    self.memory_manager.add({"role": "assistant", "content": final_answer})
                    
                    # Log interaction summary
                    self.log_manager.log_interaction(
                        user_input=user_input,
                        agent_response=final_answer,
                        model=model,
                        timestamp=datetime.now().isoformat()
                    )
                    return final_answer
            
                
        except Exception as e:
            self.log_manager.log_debug("\n### Error\n```\n" + str(e) + "\n```")
            raise e
        finally:
            self.log_manager.log_debug("\n## Interaction Complete")
            self.log_manager.print_logs_in_pretty_format()
    
    def run(self, user_input: str, model: str) -> str:
        """Run the agent synchronously.
        
        Args:
            user_input (str): The input to the agent
            model (str): The model to use for the agent
            
        Returns:
            str: The output of the agent
        """
        return asyncio.run(self.arun(user_input, model))
    
    
    
