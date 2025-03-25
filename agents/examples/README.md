# Field Extraction Agent Examples

This directory contains examples of using the Field Extraction Agent to extract structured information from text and files.

## Overview

The Field Extraction Agent is built on the agent framework and is designed to extract specific fields from unstructured text, especially from documents like driver's licenses. It demonstrates how to:

1. Create a specialized Tool for extraction tasks
2. Use the Agent class directly as a component
3. Process both text input and files
4. Handle various file formats and document structures

## Files

- `fields_extraction_agent.py` - Main implementation (now moved to `agents/fields_extraction/`)
- `test_fields_extraction.py` - Test cases for the extraction agent
- `run_extraction.py` - Command-line tool for running extractions
- `sample_data/` - Sample files in various formats for testing

## Usage

### Command-line Tool

The simplest way to use the extraction agent is through the command-line tool:

```bash
# Extract from a file
python run_extraction.py --file sample_data/standard_license.txt

# Extract specific fields
python run_extraction.py --file sample_data/standard_license.txt --fields first_name,last_name,license_number

# Extract from text
python run_extraction.py --text "NAME: DOE, JOHN M, DOB: 01/01/1980"

# Output to a file in text format
python run_extraction.py --file sample_data/markdown_license.md --format text --output results.txt

# Use a different model
python run_extraction.py --file sample_data/csv_license.csv --model gpt-4
```

### Programmatic Usage

To use the agent in your code:

```python
from agents.fields_extraction.fields_extraction_agent import FieldsExtractionAgent
from llm.openai.openai_llm import OpenAILLM

async def extract_data():
    # Create LLM and agent
    llm = OpenAILLM.create_llm()
    extraction_agent = FieldsExtractionAgent(llm)
    
    # Extract from text
    text_result = await extraction_agent.extract_from_text(
        "NAME: DOE, JOHN M, DOB: 01/01/1980",
        fields=["first_name", "last_name", "birth_date"]
    )
    
    # Extract from file
    file_result = await extraction_agent.extract_from_file("path/to/license.txt")
    
    return text_result, file_result
```

## Running Tests

To run the tests:

```bash
python -m unittest agents/examples/test_fields_extraction.py
```

## Sample Data

The `sample_data` directory contains example files in various formats:

- `standard_license.txt` - Standard format driver's license
- `markdown_license.md` - Driver's license info in Markdown format
- `csv_license.csv` - Driver's license info in CSV format
- `simple_format.txt` - Simple single-line format

These can be used to test the extraction agent with different formats and structures. 