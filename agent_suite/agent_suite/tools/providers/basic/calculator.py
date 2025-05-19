"""
Calculator Tool for mathematical operations.

This tool enables agents to perform basic and advanced mathematical operations.
"""
import asyncio
import math
import re
from typing import Union, Dict, Any, List, Optional

from pydantic import Field

from tools.base import EnhancedTool
from tools.tool_types import (
    ToolMetadata, 
    ToolCategory, 
    ToolCapability, 
    ToolDomain,
    ToolSource
)


class CalculatorTool(EnhancedTool):
    """
    Tool for performing mathematical calculations.
    
    This tool can handle basic arithmetic operations (addition, subtraction,
    multiplication, division) as well as more advanced functions like
    exponentiation, logarithms, trigonometry, etc.
    """
    
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate (e.g., '2 + 2 * 3')"
    )
    
    # Define the tool metadata for selection
    metadata: ToolMetadata = ToolMetadata(
        name="CalculatorTool",
        display_name="Calculator",
        description="Performs mathematical calculations and evaluates expressions",
        categories=[ToolCategory.UTILITY],
        capabilities=[ToolCapability.COMPUTE],
        domains=[ToolDomain.GENERAL, ToolDomain.SCIENCE],
        source=ToolSource.SYSTEM,
        keywords=["calculate", "math", "arithmetic", "computation", "formula", 
                  "algebra", "equation", "solve", "number", "calculator"],
        examples=[
            "What is 2 + 2?",
            "Calculate 15% of 85",
            "What is the square root of 144?",
            "Compute 3 raised to the power of 4",
            "Solve 5 * (3 + 2) / 4"
        ],
        applicable_roles=["researcher", "analyst", "scientist", "programmer", "mathematician"],
        task_patterns=["solve", "calculate", "compute", "find the value", "what is"]
    )
    
    # Set the source
    source: ToolSource = ToolSource.SYSTEM
    
    async def arun(self, expression: str) -> str:
        """
        Evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Result of the calculation as a string with explanation
        """
        try:
            # Preprocess the expression to handle common natural language patterns
            processed_expression = self._preprocess_expression(expression)
            
            # Create a safe evaluation environment with limited functions
            safe_env = {
                'abs': abs,
                'round': round,
                'max': max,
                'min': min,
                'sum': sum,
                'pow': pow,
                'math': math
            }
            
            # Evaluate the expression safely
            result = eval(processed_expression, {"__builtins__": {}}, safe_env)
            
            # Format the result
            if isinstance(result, (int, float)):
                if result == int(result):
                    formatted_result = str(int(result))
                else:
                    formatted_result = str(round(result, 8)).rstrip('0').rstrip('.')
                
                return f"{result} (= {formatted_result})"
            else:
                return str(result)
        
        except Exception as e:
            return f"Error calculating '{expression}': {str(e)}"
    
    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess the expression to handle common natural language patterns.
        
        Args:
            expression: Raw mathematical expression
            
        Returns:
            Processed expression ready for evaluation
        """
        # Handle percentage calculations
        if "%" in expression:
            # Replace patterns like "X% of Y" with "(X/100)*Y"
            expression = re.sub(r'(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', r'(\1/100)*\2', expression)
            # Replace standalone percentages with division by 100
            expression = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', expression)
        
        # Handle square root expressions
        expression = re.sub(r'square\s+root\s+of\s+(\d+(?:\.\d+)?)', r'math.sqrt(\1)', expression)
        
        # Handle powers
        expression = re.sub(r'(\d+(?:\.\d+)?)\s+to\s+the\s+power\s+of\s+(\d+(?:\.\d+)?)', r'pow(\1, \2)', expression)
        expression = re.sub(r'(\d+(?:\.\d+)?)\s+raised\s+to\s+(\d+(?:\.\d+)?)', r'pow(\1, \2)', expression)
        
        # Handle basic operations in text form
        expression = expression.replace('plus', '+')
        expression = expression.replace('minus', '-')
        expression = expression.replace('times', '*')
        expression = expression.replace('multiplied by', '*')
        expression = expression.replace('divided by', '/')
        
        # Remove any text that's not part of the mathematical expression
        expression = re.sub(r'[a-zA-Z]+\s*', '', expression, flags=re.IGNORECASE)
        
        # Clean any remaining text
        expression = re.sub(r'[^0-9+\-*/().^ ]', '', expression)
        
        # Handle the ^ operator as power
        expression = expression.replace('^', '**')
        
        # Remove all spaces
        expression = expression.replace(' ', '')
        
        return expression


class CalculatorToolLegacy:
    """Legacy version of the Calculator Tool for compatibility."""
    
    name = "calculator"
    description = "Useful for performing mathematical calculations."
    
    def run(self, expression: str) -> str:
        """Run the calculator synchronously."""
        calculator = CalculatorTool()
        return asyncio.run(calculator.arun(expression=expression))
    
    async def arun(self, expression: str) -> str:
        """Run the calculator asynchronously."""
        calculator = CalculatorTool()
        return await calculator.arun(expression=expression) 