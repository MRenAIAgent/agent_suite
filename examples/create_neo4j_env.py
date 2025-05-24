#!/usr/bin/env python
"""
Helper script to create a .env file with Neo4j credentials.

This script prompts the user for Neo4j connection details and
creates a .env file that can be used to run the Neo4j tests.

Usage:
    python examples/create_neo4j_env.py
"""

import os
import sys

def get_input(prompt, default=None):
    """Get input from the user with an optional default value."""
    if default:
        result = input(f"{prompt} [{default}]: ")
        return result if result else default
    else:
        return input(f"{prompt}: ")

def main():
    """Create a .env file with Neo4j credentials."""
    print("Neo4j Connection Setup Helper")
    print("=============================")
    print()
    print("This script will help you create a .env file with your Neo4j credentials")
    print("for testing the Neo4jStore implementation.")
    print()
    print("You can get these details from your Neo4j AuraDB Free dashboard.")
    print("Sign up at https://console.neo4j.io/ if you don't have an account.")
    print()
    
    # Get Neo4j connection details
    neo4j_uri = get_input("Enter your Neo4j URI (e.g., neo4j+s://xxx.databases.neo4j.io)")
    if not neo4j_uri:
        print("Error: Neo4j URI is required.")
        sys.exit(1)
    
    neo4j_username = get_input("Enter your Neo4j username", "neo4j")
    
    neo4j_password = get_input("Enter your Neo4j password")
    if not neo4j_password:
        print("Error: Neo4j password is required.")
        sys.exit(1)
    
    neo4j_database = get_input("Enter your Neo4j database name", "neo4j")
    
    # Create .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    
    with open(env_path, "w") as f:
        f.write(f"NEO4J_URI={neo4j_uri}\n")
        f.write(f"NEO4J_USERNAME={neo4j_username}\n")
        f.write(f"NEO4J_PASSWORD={neo4j_password}\n")
        f.write(f"NEO4J_DATABASE={neo4j_database}\n")
    
    print()
    print(f".env file created at {env_path}")
    print("You can now run the Neo4j tests with:")
    print("  source .env && ./examples/run_neo4j_tests.sh")
    
    # Create a shell helper script
    run_with_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_with_env.sh")
    
    with open(run_with_env_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Helper script to run tests with .env file\n\n")
        f.write("# Load environment variables from .env file\n")
        f.write("set -a\n")
        f.write("source \"$(dirname \"$(dirname \"$0\")\")/.env\"\n")
        f.write("set +a\n\n")
        f.write("# Run the tests\n")
        f.write("\"$(dirname \"$0\")/run_neo4j_tests.sh\"\n")
    
    os.chmod(run_with_env_path, 0o755)
    
    print("  OR")
    print("  ./examples/run_with_env.sh")

if __name__ == "__main__":
    main() 