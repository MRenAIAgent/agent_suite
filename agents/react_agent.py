import asyncio
import json
from datetime import datetime
from typing import List

from agents.agent import Agent
from agents.prompt import PromptManager
from agents.react_agent_pattern import ReactAgentPattern
from agents.react_prompt_template import ReActPromptTemplate
from llm.llm import LLMBase
from log.logging import LogManager
from tools.tool import Tool

class ReactAgent(Agent):
    """Agent that uses ReAct (Reasoning and Acting) approach to solve tasks."""

    def __init__(
            self,
            llm: LLMBase,
            role: str,
            task: str,
            guide: str,
            examples: list[str] = None,
            tools: List[Tool] = None,
            log_manager: LogManager = None,
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

        self.agent_pattern = ReactAgentPattern()
        self.execution_pattern = self.agent_pattern.llm_execution_pattern
        self.log_manager = log_manager
        self.max_iterations = max_iterations

    async def arun(self, user_input: str, model: str) -> str:
        """Process user input using ReAct approach.
        
        The agent thinks and acts in steps, using tools when needed,
        until it reaches a final answer.
        
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
            while True:
                self.log_manager.log_debug(f"\n## ReAct Iteration {iterations + 1}/{self.max_iterations}")
                self.log_manager.log_debug(f"\n### Tools: {[tool.convert_to_function_call() for tool in self.tools]}")
                response = self.llm.chat_completion(
                    model=model,
                    messages=messages,
                    tools=[tool.convert_to_function_call() for tool in self.tools],
                    tool_choice="auto"
                )

                self.log_manager.log_debug(f"\n### LLM Response {GREEN}\n```json\n{pretty_json(response.choices[0].message.content)}\n```{RESET}")
                # Parse the LLM response using the agent pattern
                parsed_response = self.execution_pattern.parse_llm_response(response)

                iterations += 1
                if iterations >= self.max_iterations-1:
                    # last iteration
                    final_answer = parsed_response.get('final_answer', '')
                    observation = parsed_response.get('observation', '')
                    if final_answer:
                        return final_answer
                    elif observation:
                        return observation
                    else:
                        return "No final answer or observation"

                #handle tool calls
                if "tool_calls" in parsed_response:
                    tool_results = await self.handle_tool_calls(parsed_response["tool_calls"])
                    self.log_manager.log_debug("\n### Tool Results\n```json\n" + pretty_json(tool_results) + "\n```")

                    # Format the tool calls properly with required fields
                    formatted_tool_calls = []
                    for i, tool_call in enumerate(parsed_response["tool_calls"]):
                        formatted_tool_calls.append({
                            "id": tool_call.id if hasattr(tool_call, 'id') else f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        })

                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": formatted_tool_calls
                    })
                    
                    # Add tool results
                    for i, result in enumerate(tool_results):
                        messages.append({
                            "role": "tool",
                            "tool_call_id": formatted_tool_calls[i]["id"],
                            "content": json.dumps(result)
                        })
                    continue #do not support direct return from tool calls

                self.log_manager.log_debug(f"\n### Parsed Response {GREEN}\n```json\n{pretty_json(parsed_response)}\n```{RESET}")
                # Check if we should continue with more steps
                # Format the intermediate steps for the next iteration
                if self.execution_pattern.should_continue(parsed_response):
                    formatted_steps = self.execution_pattern.format_intermediate_steps(parsed_response)
                    messages.append({
                        "role": "system",
                        "content": formatted_steps
                    })
                    self.log_manager.log_debug(f"\n### Intermediate Steps {GREEN}\n```\n{formatted_steps}\n```{RESET}")
                else:
                    final_answer = self.execution_pattern.get_final_answer(parsed_response)
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
        return asyncio.run(self.arun(user_input, model))