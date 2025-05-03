from datetime import datetime
import json
import time
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
            "final_answer": None,
            "latency": {
                "llm": [],       # List of LLM call latencies in seconds
                "actions": [],   # List of action latencies in seconds
                "total_start": None,  # Start time of execution
                "total_end": None,    # End time of execution
                "total": None         # Total execution time in seconds
            }
        }
        
    def start_execution(self):
        """Mark the start of an execution for latency tracking."""
        self.current_execution["latency"]["total_start"] = time.time()
        
    def end_execution(self):
        """Mark the end of an execution and calculate total latency."""
        if self.current_execution["latency"]["total_start"] is not None:
            self.current_execution["latency"]["total_end"] = time.time()
            self.current_execution["latency"]["total"] = (
                self.current_execution["latency"]["total_end"] - 
                self.current_execution["latency"]["total_start"]
            )
        
    def log_llm_latency(self, latency: float):
        """Log the latency of an LLM call.
        
        Args:
            latency: The latency in seconds
        """
        self.current_execution["latency"]["llm"].append(latency)
        
    def log_action_latency(self, action_step: int, latency: float):
        """Log the latency of an action.
        
        Args:
            action_step: The step number of the action
            latency: The latency in seconds
        """
        # Store as tuple of (action_step, latency) for reference
        self.current_execution["latency"]["actions"].append((action_step, latency))
        
        # Also add latency to the action entry itself
        for action in self.current_execution["actions"]:
            if action.get("step") == action_step:
                action["latency"] = latency
                break
        
    def log_interaction(self, user_input: str, agent_response: str, 
                       model: str, timestamp: str):
        """Log an interaction."""
        log_entry = {
            "timestamp": timestamp,
            "model": model,
            "user_input": user_input,
            "agent_response": agent_response
        }
        
        # Add latency information if available
        if self.current_execution["latency"]["total"] is not None:
            log_entry["total_latency"] = self.current_execution["latency"]["total"]
            
            # Calculate average LLM latency if we have any
            llm_latencies = self.current_execution["latency"]["llm"]
            if llm_latencies:
                log_entry["avg_llm_latency"] = sum(llm_latencies) / len(llm_latencies)
                log_entry["total_llm_latency"] = sum(llm_latencies)
                
            # Calculate average action latency if we have any
            action_latencies = [latency for _, latency in self.current_execution["latency"]["actions"]]
            if action_latencies:
                log_entry["avg_action_latency"] = sum(action_latencies) / len(action_latencies)
                log_entry["total_action_latency"] = sum(action_latencies)
        
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
            "related_thought_step": self.thought_step_counter,  # Link to the thought that led to this action
            "start_time": time.time(),  # Record start time for latency calculation
            "latency": None  # Will be filled when action completes
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
        
        # Calculate latency for this action
        for action in self.current_execution["actions"]:
            if action.get("step") == action_step and action.get("start_time") is not None:
                latency = time.time() - action.get("start_time")
                action["latency"] = latency
                self.log_action_latency(action_step, latency)
                break
            
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
            
        # End the execution to calculate total latency
        self.end_execution()
        
        # Calculate latency metrics
        latency_metrics = {}
        
        # LLM latency stats
        llm_latencies = self.current_execution["latency"]["llm"]
        if llm_latencies:
            latency_metrics["llm_total"] = sum(llm_latencies)
            latency_metrics["llm_avg"] = sum(llm_latencies) / len(llm_latencies)
            latency_metrics["llm_max"] = max(llm_latencies)
            latency_metrics["llm_min"] = min(llm_latencies)
            
        # Action latency stats
        action_latencies = [latency for _, latency in self.current_execution["latency"]["actions"]]
        if action_latencies:
            latency_metrics["action_total"] = sum(action_latencies)
            latency_metrics["action_avg"] = sum(action_latencies) / len(action_latencies)
            latency_metrics["action_max"] = max(action_latencies)
            latency_metrics["action_min"] = min(action_latencies)
            
        # Total execution time
        latency_metrics["total"] = self.current_execution["latency"]["total"]
        
        final_answer_entry = {
            "type": "final_answer",
            "timestamp": timestamp,
            "content": answer,
            "total_thought_steps": self.thought_step_counter,
            "total_action_steps": self.action_step_counter,
            "latency_metrics": latency_metrics
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
            "final_answer": None,
            "latency": {
                "llm": [],
                "actions": [],
                "total_start": None,
                "total_end": None,
                "total": None
            }
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
    
    def get_latency_metrics(self) -> Dict:
        """Get latency metrics for the current execution."""
        if self.current_execution["final_answer"]:
            return self.current_execution["final_answer"].get("latency_metrics", {})
            
        # If no final answer yet, calculate preliminary metrics
        metrics = {}
        
        # LLM latency stats
        llm_latencies = self.current_execution["latency"]["llm"]
        if llm_latencies:
            metrics["llm_total"] = sum(llm_latencies)
            metrics["llm_avg"] = sum(llm_latencies) / len(llm_latencies)
            
        # Action latency stats
        action_latencies = [latency for _, latency in self.current_execution["latency"]["actions"]]
        if action_latencies:
            metrics["action_total"] = sum(action_latencies)
            metrics["action_avg"] = sum(action_latencies) / len(action_latencies)
            
        # Total execution time so far
        if self.current_execution["latency"]["total_start"] is not None:
            metrics["total_so_far"] = time.time() - self.current_execution["latency"]["total_start"]
            
        return metrics
    
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
                latency = action.get("latency")
                latency_str = f" - {latency:.2f}s" if latency is not None else ""
                
                print(f"{Fore.BLUE}[Action {step}] {name} (from thought {related_thought}){latency_str}{Style.RESET_ALL}")
                
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
            
            # Print latency metrics if available
            latency_metrics = final_answer.get("latency_metrics", {})
            if latency_metrics:
                print(f"\n{Fore.GREEN}EXECUTION LATENCY:{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}Total Execution Time: {latency_metrics.get('total', 0):.2f}s{Style.RESET_ALL}")
                
                if "llm_total" in latency_metrics:
                    llm_percent = (latency_metrics["llm_total"] / latency_metrics["total"]) * 100 if latency_metrics["total"] > 0 else 0
                    print(f"  LLM Processing: {latency_metrics['llm_total']:.2f}s ({llm_percent:.1f}% of total)")
                    print(f"    Average: {latency_metrics.get('llm_avg', 0):.2f}s per call")
                    print(f"    Max: {latency_metrics.get('llm_max', 0):.2f}s")
                    print(f"    Min: {latency_metrics.get('llm_min', 0):.2f}s")
                
                if "action_total" in latency_metrics:
                    action_percent = (latency_metrics["action_total"] / latency_metrics["total"]) * 100 if latency_metrics["total"] > 0 else 0
                    print(f"  Action Execution: {latency_metrics['action_total']:.2f}s ({action_percent:.1f}% of total)")
                    print(f"    Average: {latency_metrics.get('action_avg', 0):.2f}s per action")
                    print(f"    Max: {latency_metrics.get('action_max', 0):.2f}s")
                    print(f"    Min: {latency_metrics.get('action_min', 0):.2f}s")
                
                # Calculate overhead (time not spent in LLM or actions)
                if "llm_total" in latency_metrics and "action_total" in latency_metrics:
                    overhead = latency_metrics["total"] - (latency_metrics["llm_total"] + latency_metrics["action_total"])
                    overhead_percent = (overhead / latency_metrics["total"]) * 100 if latency_metrics["total"] > 0 else 0
                    print(f"  Overhead: {overhead:.2f}s ({overhead_percent:.1f}% of total)")
        
        print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
