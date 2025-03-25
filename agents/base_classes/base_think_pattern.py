from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class AgentThinkPattern(ABC):
    """
    Abstract base class for different agent thinking patterns.
    
    This class defines how an agent processes information and makes decisions.
    Different thinking patterns can represent different cognitive styles,
    reasoning approaches, or decision-making frameworks.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a thinking pattern.
        
        Args:
            name: The name of the thinking pattern
            description: A description of how this thinking pattern works
        """
        self.name = name
        self.description = description
        self.metadata = {}
    
    @abstractmethod
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data according to this thinking pattern.
        
        Args:
            input_data: The input data to process
            
        Returns:
            Processed data according to this thinking pattern
        """
        pass
    
    @abstractmethod
    def format_thoughts(self, thoughts: Dict[str, Any]) -> str:
        """
        Format the agent's thoughts according to this thinking pattern.
        
        Args:
            thoughts: The thoughts to format
            
        Returns:
            Formatted thoughts as a string
        """
        pass
    
    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to this thinking pattern.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from this thinking pattern.
        
        Args:
            key: Metadata key
            default: Default value if key not found
            
        Returns:
            Metadata value or default
        """
        return self.metadata.get(key, default)
    
    def get_prompt_format(self) -> str:
        """
        Get the prompt format for this thinking pattern.
        
        Returns:
            Prompt format string
        """
        return f"""
Thinking Pattern: {self.name}

Description:
{self.description}

When responding, follow this thinking pattern.
"""


class AnalyticalThinkPattern(AgentThinkPattern):
    """
    Analytical thinking pattern that breaks down problems logically.
    """
    
    def __init__(self):
        super().__init__(
            name="Analytical Thinking",
            description="Break down complex problems into smaller parts, analyze each part systematically, and form logical conclusions based on evidence."
        )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with analytical thinking."""
        # In a real implementation, this would contain logic to structure analysis
        processed = {
            "problem_statement": input_data.get("query", ""),
            "components": [],
            "analyses": [],
            "evidence": [],
            "conclusions": []
        }
        return processed
    
    def format_thoughts(self, thoughts: Dict[str, Any]) -> str:
        """Format thoughts analytically."""
        formatted = "## Analytical Thinking Process\n\n"
        
        # Add problem statement
        formatted += f"### Problem Statement\n{thoughts.get('problem_statement', '')}\n\n"
        
        # Add components
        formatted += "### Component Analysis\n"
        for i, component in enumerate(thoughts.get("components", [])):
            formatted += f"{i+1}. {component}\n"
        formatted += "\n"
        
        # Add analyses
        formatted += "### Detailed Analysis\n"
        for analysis in thoughts.get("analyses", []):
            formatted += f"- {analysis}\n"
        formatted += "\n"
        
        # Add evidence
        formatted += "### Evidence Considered\n"
        for evidence in thoughts.get("evidence", []):
            formatted += f"- {evidence}\n"
        formatted += "\n"
        
        # Add conclusions
        formatted += "### Conclusions\n"
        for conclusion in thoughts.get("conclusions", []):
            formatted += f"- {conclusion}\n"
        
        return formatted


class CreativeThinkPattern(AgentThinkPattern):
    """
    Creative thinking pattern that explores novel ideas and connections.
    """
    
    def __init__(self):
        super().__init__(
            name="Creative Thinking",
            description="Generate novel ideas, make unexpected connections, and explore innovative solutions without being constrained by conventional approaches."
        )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with creative thinking."""
        # In a real implementation, this would contain logic to encourage creativity
        processed = {
            "initial_prompt": input_data.get("query", ""),
            "inspirations": [],
            "ideas": [],
            "connections": [],
            "novel_solutions": []
        }
        return processed
    
    def format_thoughts(self, thoughts: Dict[str, Any]) -> str:
        """Format thoughts creatively."""
        formatted = "🌟 Creative Exploration 🌟\n\n"
        
        # Add initial prompt
        formatted += f"Starting point: {thoughts.get('initial_prompt', '')}\n\n"
        
        # Add inspirations
        formatted += "✨ Inspirations:\n"
        for inspiration in thoughts.get("inspirations", []):
            formatted += f"* {inspiration}\n"
        formatted += "\n"
        
        # Add ideas
        formatted += "💡 Ideas that emerged:\n"
        for idea in thoughts.get("ideas", []):
            formatted += f"* {idea}\n"
        formatted += "\n"
        
        # Add connections
        formatted += "🔄 Unexpected connections:\n"
        for connection in thoughts.get("connections", []):
            formatted += f"* {connection}\n"
        formatted += "\n"
        
        # Add novel solutions
        formatted += "🚀 Novel approaches:\n"
        for solution in thoughts.get("novel_solutions", []):
            formatted += f"* {solution}\n"
        
        return formatted


class CriticalThinkPattern(AgentThinkPattern):
    """
    Critical thinking pattern that evaluates information skeptically.
    """
    
    def __init__(self):
        super().__init__(
            name="Critical Thinking",
            description="Evaluate information skeptically, identify biases and assumptions, and assess the validity of arguments and evidence."
        )
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input with critical thinking."""
        # In a real implementation, this would contain logic for critical evaluation
        processed = {
            "claim": input_data.get("query", ""),
            "assumptions": [],
            "evidence_evaluation": [],
            "counterarguments": [],
            "biases_identified": [],
            "conclusions": []
        }
        return processed
    
    def format_thoughts(self, thoughts: Dict[str, Any]) -> str:
        """Format thoughts critically."""
        formatted = "Critical Evaluation\n=================\n\n"
        
        # Add claim
        formatted += f"Claim under examination: \"{thoughts.get('claim', '')}\"\n\n"
        
        # Add assumptions
        formatted += "Underlying assumptions:\n"
        for assumption in thoughts.get("assumptions", []):
            formatted += f"- {assumption}\n"
        formatted += "\n"
        
        # Add evidence evaluation
        formatted += "Evaluation of evidence:\n"
        for evaluation in thoughts.get("evidence_evaluation", []):
            formatted += f"- {evaluation}\n"
        formatted += "\n"
        
        # Add counterarguments
        formatted += "Possible counterarguments:\n"
        for counterargument in thoughts.get("counterarguments", []):
            formatted += f"- {counterargument}\n"
        formatted += "\n"
        
        # Add biases identified
        formatted += "Biases and limitations identified:\n"
        for bias in thoughts.get("biases_identified", []):
            formatted += f"- {bias}\n"
        formatted += "\n"
        
        # Add conclusions
        formatted += "Reasoned conclusions:\n"
        for conclusion in thoughts.get("conclusions", []):
            formatted += f"- {conclusion}\n"
        
        return formatted 