"""
Environment Variables Setup for Knowledge Module

This file shows how to set up environment variables for the knowledge module.
You can either set these variables in your environment or create a .env file.
"""

# Example .env file content:
"""
# API keys for vector database and embedding services

# Qdrant Cloud API key (if using cloud service)
QDRANT_API_KEY=your_qdrant_api_key_here

# Cohere API key (required for Cohere embeddings)
COHERE_API_KEY=your_cohere_api_key_here
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_env_vars():
    """
    Function to help set up environment variables.
    
    You can call this function or copy this code to set
    environment variables programmatically.
    """
    # Set environment variables if not already set
    if not os.getenv("COHERE_API_KEY"):
        os.environ["COHERE_API_KEY"] = "your_cohere_api_key_here"
        
    if not os.getenv("QDRANT_API_KEY"):
        os.environ["QDRANT_API_KEY"] = "your_qdrant_api_key_here"
        
    print("Environment variables have been set.")

def check_env_vars():
    """Check if environment variables are set."""
    cohere_key = os.getenv("COHERE_API_KEY")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if cohere_key:
        print("Cohere API key is set.")
    else:
        print("Cohere API key is not set.")
        
    if qdrant_key:
        print("Qdrant API key is set.")
    else:
        print("Qdrant API key is not set.")
        
    return cohere_key is not None, qdrant_key is not None


if __name__ == "__main__":
    # Check if environment variables are set
    cohere_set, qdrant_set = check_env_vars()
    
    if not cohere_set or not qdrant_set:
        print("\nTo set up environment variables:")
        print("1. Create a .env file in your project root with the following content:")
        print("""
# API keys for vector database and embedding services

# Qdrant Cloud API key (if using cloud service)
QDRANT_API_KEY=your_qdrant_api_key_here

# Cohere API key (required for Cohere embeddings)
COHERE_API_KEY=your_cohere_api_key_here
""")
        print("2. Replace the placeholders with your actual API keys.")
        print("3. Make sure the python-dotenv package is installed: pip install python-dotenv")
        print("4. Load the environment variables using load_dotenv() in your script.") 