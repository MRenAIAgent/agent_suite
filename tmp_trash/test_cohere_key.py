#!/usr/bin/env python
"""
Simple script to test a Cohere API key.

Usage:
    python test_cohere_key.py YOUR_API_KEY
"""

import sys
import cohere

def test_cohere_key(api_key):
    """Test if a Cohere API key is valid by making a simple API call."""
    print(f"Testing Cohere API key...")
    
    try:
        # Initialize client
        co = cohere.Client(api_key)
        
        # Make a simple API call to test the key
        response = co.embed(
            texts=["This is a test."],
            model="embed-english-v3.0",
            input_type="search_document"
        )
        
        # Check response
        if hasattr(response, 'embeddings') and len(response.embeddings) > 0:
            print("✅ SUCCESS: API key is valid!")
            print(f"Model: embed-english-v3.0")
            print(f"Embedding dimensions: {len(response.embeddings[0])}")
            return True
        else:
            print("❌ ERROR: Received response but couldn't find embeddings")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_cohere_key.py YOUR_API_KEY")
        sys.exit(1)
        
    api_key = sys.argv[1]
    success = test_cohere_key(api_key)
    
    if not success:
        print("\nTips:")
        print("1. Make sure your API key is valid")
        print("2. Check if you've reached your API key's rate limits")
        print("3. Verify your internet connection")
        print("\nGet a new API key at: https://dashboard.cohere.com/api-keys")
        sys.exit(1) 