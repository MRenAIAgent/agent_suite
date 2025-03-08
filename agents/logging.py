from typing import List, Dict, Any, Optional
from datetime import datetime


class LogManager:
    """Manages logging of agent interactions."""
    
    def __init__(self):
        self.logs: List[Dict] = []
        
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

    def get_logs(self) -> List[Dict]:
        """Get all logged interactions."""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs."""
        self.logs = []

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
