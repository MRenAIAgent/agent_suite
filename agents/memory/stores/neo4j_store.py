"""Neo4j implementation of the Store interface.

This module provides a Neo4j-backed implementation of the Store base class,
offering persistent storage for conversation history and cache in Neo4j AuraDB.
"""

import json
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver

from agents.memory.stores.store import Store


class Neo4jStore(Store):
    """Neo4j-backed implementation of Store.
    
    This implementation stores conversation history and cache in Neo4j,
    providing persistence across sessions and potential sharing between instances.
    """
    
    def __init__(
        self, 
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        prefix: str = "agent:"
    ):
        """Initialize the Neo4j store.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j+s://xxx.databases.neo4j.io)
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            prefix: Key prefix for all Neo4j nodes
        """
        super().__init__()  # Initialize empty in-memory history and cache
        
        # Initialize Neo4j client
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password)
        )
        
        self.database = database
        self.prefix = prefix
        
        # Initialize schema (create constraints and indexes if they don't exist)
        self._initialize_schema()
        
        # Load any existing data
        self._load_from_neo4j()
    
    def add_history(self, message: Dict[str, str]) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Dict with 'role' and 'content' keys for LLM conversation
        """
        if "role" in message and "content" in message:
            # Add to in-memory history
            self.history.append(message)
            
            # Update in Neo4j
            self._save_history_to_neo4j()
    
    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        # Update in-memory cache
        self.cache[key] = value
        
        # Update in Neo4j
        self._save_cache_to_neo4j()
    
    def get_history(self, limit: int = 20) -> list:
        """Get conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of conversation messages
        """
        return self.history[-limit:] if limit > 0 else self.history
    
    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get value(s) from cache.
        
        Args:
            key: Optional key to get specific value. If None, returns all values.
            
        Returns:
            Cached value(s)
        """
        if key is None:
            return self.cache
        return self.cache.get(key)
    
    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear all items from the store."""
        # Clear in-memory storage
        if history:
            self.history = []
        if cache:
            self.cache = {}
        
        # Clear in Neo4j
        with self.driver.session(database=self.database) as session:
            if history:
                session.execute_write(
                    lambda tx: tx.run(
                        f"MATCH (h:History {{prefix: '{self.prefix}'}}) DETACH DELETE h"
                    )
                )
            if cache:
                session.execute_write(
                    lambda tx: tx.run(
                        f"MATCH (c:Cache {{prefix: '{self.prefix}'}}) DETACH DELETE c"
                    )
                )
    
    def save_history(self) -> bool:
        """Save conversation history to Neo4j.
        
        Returns:
            Success status
        """
        return self._save_history_to_neo4j()
    
    def load_history(self) -> List[Dict[str, Any]]:
        """Load conversation history from Neo4j.
        
        Returns:
            List of conversation messages
        """
        self._load_from_neo4j(history_only=True)
        return self.history
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    f"MATCH (h:History {{prefix: '{self.prefix}'}}) DETACH DELETE h"
                )
            )
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run(
                    f"MATCH (c:Cache {{prefix: '{self.prefix}'}}) DETACH DELETE c"
                )
            )
    
    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()
    
    # Async methods
    
    async def async_add_history(self, message: Dict[str, str]) -> None:
        """Asynchronously add a message to conversation history."""
        # For simplicity, using sync implementation
        self.add_history(message)
    
    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache."""
        # For simplicity, using sync implementation
        self.set_cache(key, value)
    
    async def async_get_history(self) -> list:
        """Asynchronously get conversation history."""
        return self.history
    
    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache."""
        return self.fetch_cache(key)
    
    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""
        self.clear_history()
    
    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""
        self.clear_cache()
    
    # Helper methods
    
    def _initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        with self.driver.session(database=self.database) as session:
            try:
                # Create constraint on History nodes
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (h:History) 
                        REQUIRE h.prefix IS NOT NULL
                    """)
                )
                
                # Create constraint on Cache nodes
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (c:Cache) 
                        REQUIRE c.prefix IS NOT NULL
                    """)
                )
            except Exception as e:
                print(f"Warning: Could not create constraints: {e}")
    
    def _save_history_to_neo4j(self) -> bool:
        """Save history to Neo4j.
        
        Returns:
            Success status
        """
        try:
            with self.driver.session(database=self.database) as session:
                # First delete existing history
                session.execute_write(
                    lambda tx: tx.run(
                        f"MATCH (h:History {{prefix: '{self.prefix}'}}) DETACH DELETE h"
                    )
                )
                
                # Then create new history node with all messages as a JSON property
                serialized = json.dumps(self.history)
                session.execute_write(
                    lambda tx: tx.run(
                        f"""
                        CREATE (h:History {{
                            prefix: '{self.prefix}',
                            messages: '{serialized.replace("'", "\\'")}'
                        }})
                        """
                    )
                )
                
            return True
        except Exception as e:
            print(f"Error saving history to Neo4j: {e}")
            return False
    
    def _save_cache_to_neo4j(self) -> bool:
        """Save cache to Neo4j.
        
        Returns:
            Success status
        """
        try:
            with self.driver.session(database=self.database) as session:
                # First delete existing cache
                session.execute_write(
                    lambda tx: tx.run(
                        f"MATCH (c:Cache {{prefix: '{self.prefix}'}}) DETACH DELETE c"
                    )
                )
                
                # Then create new cache node with all items as a JSON property
                serialized = json.dumps(self.cache)
                session.execute_write(
                    lambda tx: tx.run(
                        f"""
                        CREATE (c:Cache {{
                            prefix: '{self.prefix}',
                            data: '{serialized.replace("'", "\\'")}'
                        }})
                        """
                    )
                )
                
            return True
        except Exception as e:
            print(f"Error saving cache to Neo4j: {e}")
            return False
    
    def _load_from_neo4j(self, history_only: bool = False) -> None:
        """Load data from Neo4j.
        
        Args:
            history_only: If True, only load history
        """
        with self.driver.session(database=self.database) as session:
            # Load history
            try:
                result = session.execute_read(
                    lambda tx: tx.run(
                        f"MATCH (h:History {{prefix: '{self.prefix}'}}) RETURN h.messages AS messages"
                    ).single()
                )
                
                if result and result["messages"]:
                    try:
                        self.history = json.loads(result["messages"])
                    except json.JSONDecodeError:
                        self.history = []
            except Exception:
                # No history found or other error
                self.history = []
            
            # Load cache
            if not history_only:
                try:
                    result = session.execute_read(
                        lambda tx: tx.run(
                            f"MATCH (c:Cache {{prefix: '{self.prefix}'}}) RETURN c.data AS data"
                        ).single()
                    )
                    
                    if result and result["data"]:
                        try:
                            self.cache = json.loads(result["data"])
                        except json.JSONDecodeError:
                            self.cache = {}
                except Exception:
                    # No cache found or other error
                    self.cache = {} 