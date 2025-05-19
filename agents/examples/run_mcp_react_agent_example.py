#!/usr/bin/env python3
"""
Simple Test for Mocked GitHub MCP Tool

This script tests a mocked GitHub repository search to simulate MCP tool functionality.
"""
import os
import sys
import asyncio
import json
import logging

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Custom mock for GitHub search
async def mock_github_search(query="language:Python stars:>10000"):
    """Mock GitHub repository search function."""
    print(f"Searching for: {query}")
    
    # Generate a realistic mock response
    is_python = "python" in query.lower() or "language:python" in query.lower()
    is_stars = "stars" in query.lower() or "sort:stars" in query.lower()
    
    mock_repositories = []
    
    if is_python:
        mock_repositories.extend([
            {
                "name": "python/cpython", 
                "full_name": "python/cpython",
                "description": "The Python programming language", 
                "stars": 56000,
                "url": "https://github.com/python/cpython",
                "language": "Python",
                "forks": 25000,
                "created_at": "2017-02-10T12:31:04Z",
                "updated_at": "2023-04-30T10:15:22Z"
            },
            {
                "name": "vinta/awesome-python",
                "full_name": "vinta/awesome-python",
                "description": "A curated list of awesome Python frameworks, libraries, software and resources", 
                "stars": 170000,
                "url": "https://github.com/vinta/awesome-python",
                "language": "Python",
                "forks": 30000,
                "created_at": "2014-06-27T21:00:06Z",
                "updated_at": "2023-05-01T14:23:44Z"
            },
            {
                "name": "django/django",
                "full_name": "django/django",
                "description": "The Web framework for perfectionists with deadlines", 
                "stars": 72000,
                "url": "https://github.com/django/django",
                "language": "Python",
                "forks": 30000,
                "created_at": "2012-04-28T02:47:18Z",
                "updated_at": "2023-04-29T22:13:59Z"
            },
            {
                "name": "pallets/flask",
                "full_name": "pallets/flask",
                "description": "The Python micro framework for building web applications", 
                "stars": 64000,
                "url": "https://github.com/pallets/flask",
                "language": "Python",
                "forks": 16000,
                "created_at": "2010-04-06T11:11:59Z",
                "updated_at": "2023-05-01T08:44:01Z"
            },
            {
                "name": "tensorflow/tensorflow",
                "full_name": "tensorflow/tensorflow",
                "description": "An Open Source Machine Learning Framework for Everyone", 
                "stars": 174000,
                "url": "https://github.com/tensorflow/tensorflow",
                "language": "C++",
                "forks": 88000,
                "created_at": "2015-11-07T01:19:20Z",
                "updated_at": "2023-05-01T17:01:37Z"
            }
        ])
    else:
        mock_repositories.extend([
            {
                "name": "freeCodeCamp/freeCodeCamp",
                "full_name": "freeCodeCamp/freeCodeCamp",
                "description": "freeCodeCamp.org's open-source codebase and curriculum", 
                "stars": 370000,
                "url": "https://github.com/freeCodeCamp/freeCodeCamp",
                "language": "TypeScript",
                "forks": 33000,
                "created_at": "2014-12-24T17:49:19Z",
                "updated_at": "2023-05-01T16:41:05Z"
            },
            {
                "name": "EbookFoundation/free-programming-books",
                "full_name": "EbookFoundation/free-programming-books",
                "description": "Freely available programming books", 
                "stars": 280000,
                "url": "https://github.com/EbookFoundation/free-programming-books",
                "language": None,
                "forks": 54000,
                "created_at": "2013-10-11T06:25:46Z",
                "updated_at": "2023-05-01T15:21:27Z"
            }
        ])
    
    # If specifically looking for stars, sort by stars
    if is_stars:
        mock_repositories.sort(key=lambda x: x["stars"], reverse=True)
        
    return {
        "total_count": len(mock_repositories),
        "incomplete_results": False,
        "query": query,
        "repositories": mock_repositories
    }


async def main():
    """Run a direct mock test of the GitHub search functionality."""
    print("=== Testing Mocked GitHub Repository Search ===")
    
    # Run search and output results
    print("\nSearching for most starred Python repositories...")
    result = await mock_github_search("language:Python stars:>10000")
    
    # Output the result
    print("\nResults:")
    print(json.dumps(result, indent=2))
    
    # Format the results into a readable summary
    print("\nTop Python Repositories by Stars:")
    for idx, repo in enumerate(sorted(result["repositories"], key=lambda x: x["stars"], reverse=True)):
        print(f"{idx+1}. {repo['full_name']} ({repo['stars']:,} stars)")
        print(f"   {repo['description']}")
        print(f"   Language: {repo['language'] or 'N/A'}")
        print(f"   URL: {repo['url']}")
        print()


if __name__ == "__main__":
    asyncio.run(main()) 