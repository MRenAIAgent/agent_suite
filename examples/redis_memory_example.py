#!/usr/bin/env python3
"""
Example of using Redis-backed memory for persisting chat history.

This example demonstrates how to create and use a memory manager 
that stores conversation history in Redis.

Requirements:
- Redis server running locally or accessible via network
- redis-py package installed (pip install redis)
"""

import json
from agents.memory.factory import create_redis_memory_manager


def main():
    """Demonstrate Redis-backed memory usage."""
    print("Redis Memory Example")
    print("===================")
    
    # Create a memory manager with Redis backing
    # For remote Redis server, specify redis_config={'host': 'your-redis-host', 'port': 6379}
    memory = create_redis_memory_manager(
        max_history=10,  # Keep 10 messages in context
        redis_config={
            'prefix': 'chat:',  # Custom Redis key prefix
            'expire_seconds': 86400  # Cache expires after 24 hours
        }
    )
    
    # Check if we already have history (persistent across runs)
    history = memory.get_history()
    if history:
        print(f"\nFound {len(history)} existing messages:")
        for msg in history:
            print(f"- {msg['role']}: {msg['content']}")
    else:
        print("\nNo existing messages found.")
    
    print("\nLet's add some messages to the conversation:")
    
    # Add messages to the conversation
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Tell me about Redis memory in this system."},
        {"role": "assistant", "content": "This system is using Redis to store chat history, "
                                         "which provides persistence across sessions."}
    ]
    
    for msg in messages:
        memory.add(msg)
        print(f"Added: {msg['role']}: {msg['content']}")
    
    # Display current history
    print("\nCurrent conversation history:")
    history = memory.get_history()
    for msg in history:
        print(f"- {msg['role']}: {msg['content']}")
    
    # Get formatted history (for prompt construction)
    print("\nFormatted history (default):")
    print(memory.get_formatted_history())
    
    print("\nFormatted history (compact):")
    print(memory.get_formatted_history("compact"))
    
    # Search functionality
    query = "Redis"
    print(f"\nSearching for '{query}':")
    results = memory.search(query)
    for msg in results:
        print(f"- Match: {msg['role']}: {msg['content']}")
    
    print("\nNow if you restart this script, the history will be loaded from Redis.")
    print("Redis keys will expire after the time specified in expire_seconds (24h in this example).")


if __name__ == "__main__":
    main() 