"""GraphitiNeo4jStore: Integration of Graphiti with Neo4j backend.

This module provides a combined implementation of the Graphiti memory model
with Neo4j's database backend, offering the best of both worlds:
- Graphiti's advanced graph modeling for conversational memory
- Neo4j's robust, scalable database features and AuraDB cloud hosting
"""

import json
import uuid
from typing import Dict, List, Any, Optional
import time

from agents.memory.stores.store import Store
from agents.memory.stores.neo4j_store import Neo4jStore
from neo4j import GraphDatabase, Driver

try:
    import graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


class GraphitiNeo4jStore(Store):
    """Store that combines Graphiti's graph modeling with Neo4j backend.
    
    This implementation:
    - Uses Graphiti's graph model for messages, entities, and relationships
    - Stores the actual data in Neo4j for persistence and cloud capabilities
    - Provides advanced search and query capabilities across the memory graph
    """
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        namespace: str = "agent",
        prefix: str = "graphiti:"
    ):
        """Initialize the GraphitiNeo4j store.
        
        Args:
            uri: Neo4j connection URI (e.g., neo4j+s://xxx.databases.neo4j.io)
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            namespace: Namespace for grouping memory nodes
            prefix: Key prefix for Neo4j nodes
        
        Raises:
            ImportError: If Graphiti is not installed
        """
        super().__init__()
        
        # Check Graphiti availability
        if not GRAPHITI_AVAILABLE:
            raise ImportError(
                "Graphiti is required for GraphitiNeo4jStore. "
                "Install it with: pip install graphiti"
            )
        
        # Initialize Neo4j connection
        self.driver = GraphDatabase.driver(
            uri, 
            auth=(username, password)
        )
        
        self.database = database
        self.namespace = namespace
        self.prefix = prefix
        
        # Also create a Neo4jStore instance for basic storage operations
        self.neo4j_store = Neo4jStore(
            uri=uri,
            username=username,
            password=password,
            database=database,
            prefix=f"{prefix}{namespace}:"
        )
        
        # Initialize schema
        self._initialize_schema()
        
        # Load existing data
        self.load_history()
    
    def add_history(self, message: Dict) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Message dictionary to add
        """
        # Add to in-memory cache
        self.history.append(message)
        
        # Create a message node in the graph
        msg_id = message.get("id", str(uuid.uuid4()))
        
        # Add ID to message if not present
        if "id" not in message:
            message["id"] = msg_id
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = self._current_timestamp()
        
        with self.driver.session(database=self.database) as session:
            # Create the message node
            session.execute_write(self._create_message_node, message, self.namespace, self.prefix)
            
            # Link to previous message if it exists
            if len(self.history) > 1:
                prev_msg = self.history[-2]
                prev_msg_id = prev_msg.get("id")
                if prev_msg_id:
                    session.execute_write(
                        self._link_messages, 
                        prev_msg_id, 
                        msg_id, 
                        self.namespace, 
                        self.prefix
                    )
            
            # Extract and link entities 
            content = message.get("content", "")
            if content:
                session.execute_write(
                    self._extract_and_link_entities, 
                    msg_id, 
                    content, 
                    self.namespace, 
                    self.prefix
                )
        
        # Also update history in Neo4jStore for basic storage
        self.neo4j_store.add_history(message)
    
    def get_history(self, limit: int = 20) -> List:
        """Get conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of messages
        """
        if limit <= 0:
            return self.history
        return self.history[-limit:]
    
    def set_cache(self, key: str, value: Any) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
        """
        self.cache[key] = value
        self.neo4j_store.set_cache(key, value)
    
    def fetch_cache(self, key: Optional[str] = None) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key. If None, returns all cache
            
        Returns:
            Cached value or None if not found
        """
        if key is None:
            return self.cache
        return self.cache.get(key)
    
    def clear(self, history: bool = True, cache: bool = True) -> None:
        """Clear history and/or cache.
        
        Args:
            history: Whether to clear history
            cache: Whether to clear cache
        """
        if history:
            self.clear_history()
        
        if cache:
            self.clear_cache()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.history = []
        
        # Clear graph nodes
        with self.driver.session(database=self.database) as session:
            # Delete message and entity nodes for this namespace
            session.execute_write(
                lambda tx: tx.run(
                    f"""
                    MATCH (n)
                    WHERE n.namespace = $namespace AND n.prefix = $prefix
                    DETACH DELETE n
                    """,
                    namespace=self.namespace,
                    prefix=self.prefix
                )
            )
        
        # Also clear in Neo4jStore
        self.neo4j_store.clear_history()
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.neo4j_store.clear_cache()
    
    def save_history(self) -> bool:
        """Save history to Neo4j (already done on add).
        
        Returns:
            Success status
        """
        # We save on add, but this ensures Neo4jStore is in sync
        return self.neo4j_store.save_history()
    
    def load_history(self) -> List[Dict]:
        """Load conversation history from Neo4j.
        
        Returns:
            List of conversation messages
        """
        # Load basic history from Neo4jStore
        self.history = self.neo4j_store.load_history()
        return self.history
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search the memory graph for relevant messages.
        
        This method uses the graph structure to find relevant messages based on:
        - Direct content matching
        - Entity relationships
        - Temporal proximity
        
        Args:
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        results = []
        
        with self.driver.session(database=self.database) as session:
            # Search by content match
            content_matches = session.execute_read(
                lambda tx: tx.run(
                    f"""
                    MATCH (m:Message)
                    WHERE 
                        m.namespace = $namespace AND 
                        m.prefix = $prefix AND
                        m.content CONTAINS $query
                    RETURN m
                    ORDER BY m.timestamp DESC
                    LIMIT $limit
                    """,
                    namespace=self.namespace,
                    prefix=self.prefix,
                    query=query,
                    limit=limit
                ).data()
            )
            
            for match in content_matches:
                message = self._message_from_node(match["m"])
                if message and message not in results:
                    results.append(message)
            
            # If we need more results, search by entity relationships
            if len(results) < limit:
                remaining = limit - len(results)
                entity_matches = session.execute_read(
                    lambda tx: tx.run(
                        f"""
                        MATCH (m:Message)-[:MENTIONS]->(e:Entity)
                        WHERE 
                            m.namespace = $namespace AND 
                            m.prefix = $prefix AND
                            e.value CONTAINS $query
                        RETURN m
                        ORDER BY m.timestamp DESC
                        LIMIT $remaining
                        """,
                        namespace=self.namespace,
                        prefix=self.prefix,
                        query=query,
                        remaining=remaining
                    ).data()
                )
                
                for match in entity_matches:
                    message = self._message_from_node(match["m"])
                    if message and message not in results:
                        results.append(message)
        
        return results
    
    def close(self) -> None:
        """Close the store connections."""
        if self.driver:
            self.driver.close()
        self.neo4j_store.close()
    
    # Helper methods
    
    def _initialize_schema(self) -> None:
        """Initialize the Neo4j schema for Graphiti-style graph."""
        with self.driver.session(database=self.database) as session:
            try:
                # Create constraints and indexes
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (m:Message) 
                        REQUIRE m.id IS NOT NULL
                    """)
                )
                
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) 
                        REQUIRE e.id IS NOT NULL
                    """)
                )
                
                # Create index on message timestamp for efficient retrieval
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE INDEX IF NOT EXISTS FOR (m:Message) ON (m.timestamp)
                    """)
                )
                
                # Create index on entity value for efficient search
                session.execute_write(
                    lambda tx: tx.run("""
                        CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.value)
                    """)
                )
                
            except Exception as e:
                print(f"Warning: Could not create all schema elements: {e}")
    
    def _create_message_node(self, tx, message: Dict, namespace: str, prefix: str) -> None:
        """Create a message node in the graph.
        
        Args:
            tx: Neo4j transaction
            message: Message dictionary
            namespace: Namespace for the node
            prefix: Prefix for the node
        """
        # Extract message properties
        msg_id = message.get("id")
        role = message.get("role", "unknown")
        content = message.get("content", "")
        timestamp = message.get("timestamp", self._current_timestamp())
        metadata = json.dumps(message.get("metadata", {}))
        
        # Create the message node
        tx.run(
            """
            MERGE (m:Message {id: $id, namespace: $namespace, prefix: $prefix})
            SET 
                m.role = $role,
                m.content = $content,
                m.timestamp = $timestamp,
                m.metadata = $metadata
            RETURN m
            """,
            id=msg_id,
            namespace=namespace,
            prefix=prefix,
            role=role,
            content=content,
            timestamp=timestamp,
            metadata=metadata
        )
    
    def _link_messages(self, tx, prev_msg_id: str, curr_msg_id: str, namespace: str, prefix: str) -> None:
        """Link two message nodes with a NEXT relationship.
        
        Args:
            tx: Neo4j transaction
            prev_msg_id: ID of the previous message
            curr_msg_id: ID of the current message
            namespace: Namespace for the nodes
            prefix: Prefix for the nodes
        """
        tx.run(
            """
            MATCH 
                (prev:Message {id: $prev_id, namespace: $namespace, prefix: $prefix}),
                (curr:Message {id: $curr_id, namespace: $namespace, prefix: $prefix})
            MERGE (prev)-[:NEXT {timestamp: $timestamp}]->(curr)
            """,
            prev_id=prev_msg_id,
            curr_id=curr_msg_id,
            namespace=namespace,
            prefix=prefix,
            timestamp=self._current_timestamp()
        )
    
    def _extract_and_link_entities(self, tx, msg_id: str, content: str, namespace: str, prefix: str) -> None:
        """Extract entities from content and link them to the message.
        
        Args:
            tx: Neo4j transaction
            msg_id: ID of the message
            content: Content to extract entities from
            namespace: Namespace for the nodes
            prefix: Prefix for the nodes
        """
        # Simple keyword extraction (demo purposes only)
        # A real implementation would use NER or other NLP techniques
        keywords = [word for word in content.lower().split() 
                   if len(word) > 4 and word not in ["about", "there", "their", "would", "should"]]
        
        for keyword in keywords:
            # Create a unique ID for the entity
            entity_id = f"{namespace}:{keyword}"
            
            # Create or merge the entity node
            tx.run(
                """
                MERGE (e:Entity {id: $id, namespace: $namespace, prefix: $prefix})
                SET 
                    e.value = $value,
                    e.type = 'keyword'
                RETURN e
                """,
                id=entity_id,
                namespace=namespace,
                prefix=prefix,
                value=keyword
            )
            
            # Link message to entity
            tx.run(
                """
                MATCH 
                    (m:Message {id: $msg_id, namespace: $namespace, prefix: $prefix}),
                    (e:Entity {id: $entity_id, namespace: $namespace, prefix: $prefix})
                MERGE (m)-[:MENTIONS {confidence: 1.0}]->(e)
                """,
                msg_id=msg_id,
                entity_id=entity_id,
                namespace=namespace,
                prefix=prefix
            )
    
    def _message_from_node(self, node_data: Dict) -> Dict:
        """Convert a Neo4j node to a message dictionary.
        
        Args:
            node_data: Neo4j node data
            
        Returns:
            Message dictionary
        """
        try:
            message = {
                "id": node_data.get("id"),
                "role": node_data.get("role", "unknown"),
                "content": node_data.get("content", ""),
                "timestamp": node_data.get("timestamp")
            }
            
            # Parse metadata if it exists
            metadata_str = node_data.get("metadata")
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str)
                    message["metadata"] = metadata
                except json.JSONDecodeError:
                    pass
            
            return message
        except Exception as e:
            print(f"Error converting node to message: {e}")
            return {}
    
    def _current_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
        
    # Async methods implementations
    
    async def async_add_history(self, message: Dict) -> None:
        """Asynchronously add a message to conversation history."""
        # Use the synchronous implementation
        self.add_history(message)
    
    async def async_get_history(self, limit: int = -1) -> List:
        """Asynchronously get conversation history."""
        return self.get_history(limit)
    
    async def async_set_cache(self, key: str, value: Any) -> None:
        """Asynchronously set a value in the cache."""
        self.set_cache(key, value)
    
    async def async_fetch_cache(self, key: Optional[str] = None) -> Any:
        """Asynchronously get value(s) from cache."""
        return self.fetch_cache(key)
    
    async def async_clear_history(self) -> None:
        """Asynchronously clear conversation history."""
        self.clear_history()
    
    async def async_clear_cache(self) -> None:
        """Asynchronously clear the cache."""
        self.clear_cache() 