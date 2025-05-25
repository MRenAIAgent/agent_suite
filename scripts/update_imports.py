#!/usr/bin/env python3
"""
Update Imports Script

This script updates import paths in the moved script files to use
the new benchmark directory structure.
"""

import os
import re
import glob
import argparse


def update_file_imports(file_path):
    """
    Update imports in a file to use new benchmark paths.
    
    Args:
        file_path: Path to the file to update
    """
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update various imports
    replacements = [
        (r'from agents\.eval\.taubench', r'from benchmark.agent.code.taubench'),
        (r'import agents\.eval\.taubench', r'import benchmark.agent.code.taubench'),
        (r'from agents\.eval\.agent_benchmark', r'from benchmark.agent.code.agent_benchmark'),
        (r'import agents\.eval\.agent_benchmark', r'import benchmark.agent.code.agent_benchmark'),
    ]
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)
    
    # Write the updated content back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {file_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update import paths in script files")
    parser.add_argument("--directory", default="scripts", 
                       help="Directory containing scripts to update")
    args = parser.parse_args()
    
    # Get all Python files in the directory
    py_files = glob.glob(os.path.join(args.directory, "*.py"))
    
    for file_path in py_files:
        update_file_imports(file_path)
    
    print(f"Updated {len(py_files)} files in {args.directory}")


if __name__ == "__main__":
    main() 