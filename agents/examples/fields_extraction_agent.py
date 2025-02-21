from typing import Dict, Any, List
from agents.agent import Agent
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


class FieldsExtractionAgent(Agent):
    """Agent that extracts structured fields from text."""

    def __init__(self, llm: LLMBase):
        system_prompt = """You are an AI assistant specialized in extracting information from driver's licenses across different US states.
        When given text from a driver's license, you will use the DriverLicenseExtractor tool to identify and extract the requested fields.
        
        You understand that:
        - Different states may format and present information differently
        - Field names and labels may vary between states (e.g. "DL No", "License #", "Operator No")
        - Dates may appear in different formats (MM/DD/YYYY, DD-MM-YYYY etc)
        - Some states may include additional fields or omit certain fields
        
        Extract the information accurately while accounting for these state-specific variations.
        Present the extracted information in a clear, standardized format using the defined field names."""
        
        tools = [DriverLicenseExtractor()]
        super().__init__(llm, system_prompt, tools)


def main():
    from llm.openai.openai_llm import OpenAILLM
    from dotenv import load_dotenv
    load_dotenv()
    
    llm = OpenAILLM.create_llm()
    agent = FieldsExtractionAgent(llm)
    
    text = """
    Name: John Smith. Age: 35 years old. 
    Occupation: Software Engineer.
    Location: San Francisco, CA.
    """
    
    fields_to_extract = ["Name", "Age", "Occupation", "Location"]
    response = agent.process(
        user_input=f"Extract these fields {fields_to_extract} from this text: {text}",
        model="gpt-3.5-turbo"
    )
    print("Extracted Information:")
    print(response)


if __name__ == "__main__":
    main()
