from typing import Dict

def convert_to_function_call(cls, schema) -> Dict:
    """Convert the tool to OpenAI function call format.
    
    Returns:
        Dict: Function definition in OpenAI format
    """
    # Get model fields from Pydantic
    # schema = cls.model_json_schema()

    return {
        "type": "function",
        "function": {
            "name": cls.__name__.lower(),
            "description": cls.__doc__.strip() if cls.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }