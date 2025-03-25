#!/usr/bin/env python3
"""
Command-line tool for extracting information from files using the FieldsExtractionService.

Usage:
    python run_extraction.py --file path/to/file.txt [--fields field1,field2,field3] [--model model-name]
    python run_extraction.py --text "Text content here" [--fields field1,field2,field3] [--model model-name]
"""

import argparse
import asyncio
import json
import sys
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from llm.openai.openai_llm import OpenAILLM
from agents.examples.fields_extraction.fields_extraction_agent import FieldsExtractionService


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract information from files or text using an AI agent."
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--file", 
        help="Path to a file to extract information from"
    )
    input_group.add_argument(
        "--text", 
        help="Text content to extract information from"
    )
    
    parser.add_argument(
        "--fields",
        help="Comma-separated list of fields to extract (default: all fields)"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="LLM model to use (default: gpt-3.5-turbo)"
    )
    
    parser.add_argument(
        "--output",
        help="Output file for the results (default: stdout)"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )
    
    return parser.parse_args()


def format_result(result: Dict[str, Any], output_format: str) -> str:
    """Format the result according to the specified format.
    
    Args:
        result: Extraction result
        output_format: Format to use ('json' or 'text')
        
    Returns:
        Formatted result as a string
    """
    if output_format == "json":
        return json.dumps(result, indent=2)
    else:
        # Text format
        output = []
        output.append("=== Extraction Results ===")
        
        if "source" in result:
            output.append(f"Source: {result['source']}")
        
        if "filename" in result:
            output.append(f"Filename: {result['filename']}")
        
        if "fields_requested" in result:
            output.append(f"Fields requested: {', '.join(result['fields_requested'])}")
        
        if "extracted_data" in result:
            output.append("\nExtracted Data:")
            output.append(str(result["extracted_data"]))
        
        return "\n".join(output)


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Create the extraction service
    llm = OpenAILLM.create_llm()
    extraction_service = FieldsExtractionService(llm)
    
    # Parse fields if provided
    fields = None
    if args.fields:
        fields = [field.strip() for field in args.fields.split(",")]
    
    # Extract information
    try:
        if args.file:
            result = await extraction_service.extract_from_file(
                file_path=args.file,
                fields=fields,
                model=args.model
            )
        else:
            result = await extraction_service.extract_from_text(
                text=args.text,
                fields=fields,
                model=args.model
            )
        
        # Format the result
        formatted_result = format_result(result, args.format)
        
        # Output the result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_result)
            print(f"Results written to {args.output}")
        else:
            print(formatted_result)
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 