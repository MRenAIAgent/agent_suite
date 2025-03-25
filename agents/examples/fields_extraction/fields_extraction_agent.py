from typing import Dict, Any, List, Optional
import asyncio
import os
from pathlib import Path

from agents.agent import Agent
from agents.prompt import PromptManager
from llm.llm import LLMBase
from tools.tool import Tool
from pydantic import Field


class DriverLicenseExtractor(Tool):
    """Tool for extracting information from driver's licenses.
    
    Extracts standardized fields like license number, name components, dates,
    and other identifying information from driver's license text while handling
    variations across different US states.
    """
    license_number: str = Field(description="The driver's license number")
    last_name: str = Field(description="The driver's last name")
    first_name: str = Field(description="The driver's first name")
    middle_name: str = Field(description="The driver's middle name")
    birth_date: str = Field(description="The driver's birth date")
    gender: str = Field(description="The driver's gender")
    expiration_date: str = Field(description="The driver's license expiration date")
    state: str = Field(description="The state the driver's license was issued in")

    def run(self, fields: List[str], text: str) -> Dict[str, Any]:
        """Extract requested fields from text.
        
        Args:
            fields: List of field names to extract
            text: Text to extract from
            
        Returns:
            Dict mapping field names to extracted values
        """
        return {"fields": fields, "text": text}

    async def arun(self, fields: List[str], text: str) -> Dict[str, Any]:
        """Async version of run method."""
        return self.run(fields=fields, text=text)


class FieldsExtractionService:
    """Service for extracting structured fields from text or files.
    
    This service uses an Agent to extract information from documents,
    supporting both direct text input and file uploads.
    """

    def __init__(self, llm: LLMBase):
        """Initialize the extraction service.
        
        Args:
            llm: Language model for driving the extraction
        """
        # Create system prompt
        system_prompt = """You are an AI assistant specialized in extracting information from documents.
        When given text, you will use the DriverLicenseExtractor tool to identify and extract the requested fields.
        
        You understand that:
        - Different documents may format and present information differently
        - Field names and labels may vary between formats (e.g. "DL No", "License #", "Operator No")
        - Dates may appear in different formats (MM/DD/YYYY, DD-MM-YYYY etc)
        - Some documents may include additional fields or omit certain fields
        
        Extract the information accurately while accounting for these variations.
        Present the extracted information in a clear, standardized format using the defined field names.
        If a field is not found in the document, indicate this with "Not found" as the value."""
        
        # Create prompt manager and tools
        self.prompt_manager = PromptManager(system_prompt)
        self.tools = [DriverLicenseExtractor(
            license_number="license_number",
            last_name="last_name",
            first_name="first_name",
            middle_name="middle_name",
            birth_date="birth_date",
            gender="gender",
            state="state",
            expiration_date="expiration_date",
        )]

        
        # Create agent
        self.agent = Agent(llm, self.prompt_manager, self.tools)
        
        # Default fields to extract
        self.default_fields = [
            "license_number", "last_name", "first_name", 
            "middle_name", "birth_date", "gender", 
            "expiration_date", "state"
        ]
    
    async def extract_from_text(self, text: str, fields: Optional[List[str]] = None, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Extract fields from provided text.
        
        Args:
            text: The text to extract information from
            fields: List of fields to extract (uses default fields if None)
            model: The LLM model to use
            
        Returns:
            Dictionary of extracted fields and values
        """
        fields = fields or self.default_fields
        
        user_input = f"Extract these fields {fields} from this text: {text}"
        response = await self.agent.arun(user_input=user_input, model=model)
        
        return {
            "extracted_data": response,
            "source": "text",
            "fields_requested": fields
        }
    
    async def extract_from_file(self, file_path: str, fields: Optional[List[str]] = None, model: str = "gpt-3.5-turbo") -> Dict[str, Any]:
        """Extract fields from a file.
        
        Args:
            file_path: Path to the file to extract from
            fields: List of fields to extract (uses default fields if None)
            model: The LLM model to use
            
        Returns:
            Dictionary of extracted fields and values
        """
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file content
        file_extension = Path(file_path).suffix.lower()
        if file_extension in ['.txt', '.md', '.csv', '.html']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .txt, .md, .csv, .html")
        
        # Extract from text
        result = await self.extract_from_text(text, fields, model)
        result["source"] = "file"
        result["filename"] = os.path.basename(file_path)
        
        return result
    
    def set_default_fields(self, fields: List[str]) -> None:
        """Set the default fields to extract.
        
        Args:
            fields: List of field names to use as defaults
        """
        self.default_fields = fields


async def main():
    from llm.openai.openai_llm import OpenAILLM
    from dotenv import load_dotenv
    load_dotenv()
    
    # Create LLM
    llm = OpenAILLM.create_llm()
    
    # Create service
    extraction_service = FieldsExtractionService(llm)
    
    # Example text
    text = """
    DRIVER LICENSE
    NAME: DOE, JOHN MICHAEL
    DOB: 01/01/1980
    EXP: 01/01/2025
    LICENSE #: D12345678
    GENDER: M
    STATE: CA
    """
    
    # Extract from text
    result = await extraction_service.extract_from_text(text)
    print("Extracted Information from Text:")
    print(result)
    
    # Create example file for extraction
    example_file = "example_license.txt"
    with open(example_file, "w") as f:
        f.write(text)
    
    # Extract from file
    try:
        file_result = await extraction_service.extract_from_file(example_file)
        print("\nExtracted Information from File:")
        print(file_result)
    finally:
        # Clean up the example file
        if os.path.exists(example_file):
            os.remove(example_file)


if __name__ == "__main__":
    asyncio.run(main())
