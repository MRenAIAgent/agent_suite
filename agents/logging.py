from typing import List, Dict, Any, Optional


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
        
    def get_logs(self) -> List[Dict]:
        """Get all logged interactions."""
        return self.logs
    
    def clear_logs(self):
        """Clear all logs."""
        self.logs = []
