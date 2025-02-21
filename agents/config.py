from typing import Dict, Any, Optional

class AgentConfig:
    """Configuration manager for agent settings."""
    
    def __init__(self, agent_mode: str = "chat"):
        """Initialize agent configuration.
        
        Args:
            agent_mode: Operating mode for the agent (e.g. "chat", "analysis", etc)
        """
        self.agent_mode = agent_mode
        self._validate_mode(agent_mode)
        
    def _validate_mode(self, mode: str) -> None:
        """Validate the agent mode is supported.
        
        Args:
            mode: Agent mode to validate
            
        Raises:
            ValueError: If mode is not supported
        """
        valid_modes = ["chat", "analysis", "task", "extraction", "tool_use", "react"]
        if mode not in valid_modes:
            raise ValueError(f"Agent mode '{mode}' not supported. Must be one of: {valid_modes}")
            
    def get_mode(self) -> str:
        """Get current agent mode.
        
        Returns:
            Current agent mode
        """
        return self.agent_mode
        
    def set_mode(self, mode: str) -> None:
        """Set agent mode.
        
        Args:
            mode: New agent mode
            
        Raises:
            ValueError: If mode is not supported
        """
        self._validate_mode(mode)
        self.agent_mode = mode
