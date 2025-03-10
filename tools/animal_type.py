from typing import Any, Optional
from pydantic import Field

from tools.tool import Tool

class AnimalType(Tool):
    """ animal type tool
    This tool is useful for checking the type of animal.
    """

    animal_name: str = Field(
        description="Animal name"
    )

    async def arun(self, animal_name: str) -> Any:
        """Asynchronously return the input parameters.
        
        Args:
            animal_name: Animal name
            
        Returns:
            dict: Dictionary containing the input parameters
        """
        animal_type = "unknown"
        if animal_name == "dog":
            animal_type = "mammal"
        
        return {
            "message": f"{animal_name} is a {animal_type}!"
        }


    def run(self, animal_name: str) -> Any:
        """Synchronously return the input parameters.
        
        Args:
            animal_name: Animal name
            
        Returns:
            dict: Dictionary containing the input parameters
        """
        animal_type = "unknown"
        if animal_name == "dog":
            animal_type = "mammal"
        
        return {
            "message": f"{animal_name} is a {animal_type}!"
        }