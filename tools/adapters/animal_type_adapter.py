"""
Adapter to convert the original AnimalType tool to use the enhanced tool system.

This module provides an adapter for the original AnimalType tool to make it
compatible with the enhanced tool registration system.
"""
import logging
from typing import Any

from tools.types import (
    ToolCategory, ToolCapability, ToolDomain, ToolMetadata, ToolSource
)
from tools.base import EnhancedTool
from tools.registry import registry
from tools.animal_type import AnimalType

logger = logging.getLogger(__name__)


class EnhancedAnimalTypeTool(EnhancedTool):
    """Enhanced tool for checking the type of animal."""
    
    animal_name: str
    
    def __init__(self, **data):
        """Initialize with enhanced metadata."""
        if "metadata" not in data:
            data["metadata"] = ToolMetadata(
                categories=[ToolCategory.UTILITY],
                domains=[ToolDomain.EDUCATION, ToolDomain.SCIENCE],
                capabilities=[ToolCapability.ANALYZE],
                applicable_roles=["biologist", "teacher", "student", "assistant"],
                task_patterns=["animal type", "classify animal", "identify animal species"],
                examples=[
                    "Check what type of animal a dog is",
                    "Identify the classification of an elephant"
                ],
                is_cacheable=True,
                keywords=["animal", "classification", "species", "type", "biology"]
            )
        
        super().__init__(**data)
        
        # Initialize the original AnimalType tool
        self.animal_tool = AnimalType()
    
    async def arun(self, animal_name: str) -> Any:
        """
        Check the type of animal.
        
        Args:
            animal_name: Name of the animal to check
            
        Returns:
            Information about the animal type
        """
        return await self.animal_tool.arun(animal_name)


def register_animal_tools():
    """Register all animal-related tools with the registry."""
    # Create and register the enhanced AnimalType tool
    registry.register_tool(EnhancedAnimalTypeTool())
    
    logger.info("Registered animal tools with the registry") 