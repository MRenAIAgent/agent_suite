from datetime import datetime
import json
from typing import List, Dict, Any, Optional
import textwrap
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class LogManager:
    """Manages logging of agent interactions."""
    
    def __init__(self):
        self.logs: List[Dict] = []
        self.thought_step_counter = 0
        self.action_step_counter = 0
        self.current_execution = {
            "thoughts": [],
            "actions": [],
            "results": [],
            "final_answer": None
        }
        
    def log_interaction(self, user_input: str, agent_response: str, 
                       model: str, timestamp: str):
        """Log an interaction."""
        log_entry = {
            "timestamp": timestamp,
            "model": model,
            "user_input": user_input,
            "agent_response": agent_response
        }
        self.logs.append(log_entry)
        
        # Reset the execution counters and data for the next interaction
        self._reset_execution_data()

    def log_debug(self, message: str, data: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None):
        """Log detailed debug information for agent steps.
        
        Args:
            message: Debug message describing the step
            data: Optional dictionary with additional debug data
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        debug_entry = {
            "type": "debug",
            "timestamp": timestamp,
            "message": message,
            "data": data or {}
        }
        self.logs.append(debug_entry)

    def log_thought(self, thought: str, timestamp: Optional[str] = None):
        """Log an agent's thought process step.
        
        Args:
            thought: The agent's thought content
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        self.thought_step_counter += 1
        
        thought_entry = {
            "type": "thought",
            "step": self.thought_step_counter,
            "timestamp": timestamp,
            "content": thought
        }
        
        self.current_execution["thoughts"].append(thought_entry)
        self.logs.append(thought_entry)
        
        return self.thought_step_counter

    def log_action(self, action_name: str, action_input: Any, timestamp: Optional[str] = None):
        """Log an agent's action.
        
        Args:
            action_name: Name of the action/tool being used
            action_input: Input parameters for the action
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        self.action_step_counter += 1
        
        action_entry = {
            "type": "action",
            "step": self.action_step_counter,
            "timestamp": timestamp,
            "name": action_name,
            "input": action_input,
            "related_thought_step": self.thought_step_counter  # Link to the thought that led to this action
        }
        
        self.current_execution["actions"].append(action_entry)
        self.logs.append(action_entry)
        
        return self.action_step_counter

    def log_result(self, action_step: int, result: Any, timestamp: Optional[str] = None):
        """Log the result of an action.
        
        Args:
            action_step: The step number of the action this is a result for
            result: The result of the action
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        result_entry = {
            "type": "result",
            "timestamp": timestamp,
            "content": result,
            "related_action_step": action_step
        }
        
        self.current_execution["results"].append(result_entry)
        self.logs.append(result_entry)

    def log_final_answer(self, answer: str, timestamp: Optional[str] = None):
        """Log the agent's final answer.
        
        Args:
            answer: The agent's final response
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
            
        final_answer_entry = {
            "type": "final_answer",
            "timestamp": timestamp,
            "content": answer,
            "total_thought_steps": self.thought_step_counter,
            "total_action_steps": self.action_step_counter
        }
        
        self.current_execution["final_answer"] = final_answer_entry
        self.logs.append(final_answer_entry)

    def _reset_execution_data(self):
        """Reset the execution data for a new interaction."""
        self.thought_step_counter = 0
        self.action_step_counter = 0
        self.current_execution = {
            "thoughts": [],
            "actions": [],
            "results": [],
            "final_answer": None
        }

    def get_logs(self) -> List[Dict]:
        """Get all logged interactions."""
        return self.logs
    
    def get_current_execution(self) -> Dict:
        """Get the current execution data."""
        return self.current_execution
    
    def get_thought_count(self) -> int:
        """Get the total number of thought steps."""
        return self.thought_step_counter
    
    def clear_logs(self):
        """Clear all logs."""
        self.logs = []
        self._reset_execution_data()

    def print_logs(self):
        """Print all logs."""
        for log in self.logs:
            print(log)

    def print_logs_in_pretty_format(self):
        """Print all logs in pretty format."""
        for log in self.logs:
            # Print all log entries without checking log_type
            print(f"[LOG] {log.get('timestamp', 'No timestamp')}")
            
            # Print all available fields in the log
            for key, value in log.items():
                if key != 'timestamp':  # Already printed timestamp above
                    print(f"  {key}: {value}")
            
            print("-" * 50)  # Separator between log entries
            
    def print_execution_summary(self):
        """Print a summary of the current execution in a readable format."""
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 30} AGENT EXECUTION SUMMARY {'=' * 30}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")
        
        # Print thoughts
        if self.current_execution["thoughts"]:
            print(f"{Fore.GREEN}THOUGHT PROCESS: ({self.thought_step_counter} steps){Style.RESET_ALL}")
            for thought in self.current_execution["thoughts"]:
                step = thought.get("step", "?")
                content = thought.get("content", "")
                print(f"{Fore.YELLOW}[Thought {step}]{Style.RESET_ALL}")
                print(textwrap.indent(content, "  "))
                print()
        
        # Print actions and their results
        if self.current_execution["actions"]:
            print(f"{Fore.GREEN}ACTIONS & RESULTS: ({self.action_step_counter} actions){Style.RESET_ALL}")
            
            for action in self.current_execution["actions"]:
                step = action.get("step", "?")
                name = action.get("name", "Unknown action")
                related_thought = action.get("related_thought_step", "?")
                
                print(f"{Fore.BLUE}[Action {step}] {name} (from thought {related_thought}){Style.RESET_ALL}")
                
                # Pretty print the input
                input_data = action.get("input", "")
                if isinstance(input_data, dict):
                    print("  Input:")
                    for k, v in input_data.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  Input: {input_data}")
                
                # Find and print the corresponding result
                result = None
                for r in self.current_execution["results"]:
                    if r.get("related_action_step") == step:
                        result = r.get("content", "")
                        break
                
                if result is not None:
                    print(f"{Fore.MAGENTA}  Result:{Style.RESET_ALL}")
                    if isinstance(result, dict):
                        for k, v in result.items():
                            print(f"    {k}: {v}")
                    else:
                        # Indent the result for better readability
                        print(textwrap.indent(str(result), "    "))
                print()
        
        # Print final answer
        if self.current_execution["final_answer"]:
            final_answer = self.current_execution["final_answer"]
            print(f"{Fore.GREEN}FINAL ANSWER:{Style.RESET_ALL}")
            print(textwrap.indent(final_answer.get("content", ""), "  "))
            print()
            print(f"{Fore.CYAN}Total Thought Steps: {final_answer.get('total_thought_steps', 0)}{Style.RESET_ALL}")
            print(f"{Fore.CYAN}Total Action Steps: {final_answer.get('total_action_steps', 0)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
