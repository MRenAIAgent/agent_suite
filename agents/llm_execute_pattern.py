from typing import Dict, Any

class LLMExecutionPattern:
    """Base class defining the execution pattern for agent invocation loop.
    
    This class specifies:
    1. How to parse LLM responses
    2. When to continue/break the invocation loop
    3. How to handle intermediate steps vs final answers
    """

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse raw LLM response into structured format.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Dict containing parsed components (thought, action, observation etc)
        """
        raise NotImplementedError
        
    def should_continue(self, parsed_response: Dict[str, Any]) -> bool:
        """Determine if agent should continue the invocation loop.
        
        Args:
            parsed_response: Parsed response from LLM
            
        Returns:
            bool: True if should continue, False if should break loop
        """
        raise NotImplementedError
        
    def get_final_answer(self, parsed_response: Dict[str, Any]) -> str:
        """Extract final answer when loop is complete.
        
        Args:
            parsed_response: Parsed response containing final answer
            
        Returns:
            str: The final answer to return to user
        """
        raise NotImplementedError
        
    def format_intermediate_steps(self, parsed_response: Dict[str, Any]) -> str:
        """Format intermediate reasoning/action steps.
        
        Args:
            parsed_response: Parsed response with intermediate steps
            
        Returns:
            str: Formatted intermediate steps to add to conversation
        """
        raise NotImplementedError

