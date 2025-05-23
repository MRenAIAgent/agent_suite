"""Graphiti-based store for graph memory management.

This module provides a store implementation that uses Graphiti for
storing conversation history in a graph database, enabling more
advanced relationship queries and memory organization.
"""

import json
import uuid
from typing import Dict, List, Any, Optional

from agents.memory.stores.store import Store

try:
    import graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False


class GraphitiStore(Store):
    """Store implementation using Graphiti for conversation history.
    
    This store allows for more complex memory structures and querying
    by leveraging graph relationships between messages, topics, and entities.
    """
    
    def __init__(
        self,
        db_path: str = "memory.graph",
        namespace: str = "agent",
        auto_connect: bool = True,
        create_if_missing: bool = True,
    ):
        """Initialize the Graphiti store.
        
        Args:
            db_path: Path to the graph database file
            namespace: Namespace for grouping memory nodes
            auto_connect: Whether to automatically connect to the graph
            create_if_missing: Whether to create the database if it doesn't exist
        
        Raises:
            ImportError: If Graphiti is not installed
        """
        super().__init__()
        
        if not GRAPHITI_AVAILABLE:
            raise ImportError(
                "Graphiti is required for GraphitiStore. "
                "Install it with: pip install graphiti"
            )
        
        self.db_path = db_path
        self.namespace = namespace
        self.graph = None
        
        if auto_connect:
            self.connect(create_if_missing)
    
    def connect(self, create_if_missing: bool = True) -> bool:
        """Connect to the graph database.
        
        Args:
            create_if_missing: Whether to create the database if it doesn't exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.graph = graphiti.Graph(self.db_path, create_if_missing=create_if_missing)
            self._ensure_schema()
            return True
        except Exception as e:
            print(f"Error connecting to graph: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the graph database.
        
        Returns:
            True if successful, False otherwise
        """
        if self.graph:
            try:
                self.graph.close()
                self.graph = None
                return True
            except Exception as e:
                print(f"Error disconnecting from graph: {e}")
        return False
    
    def _ensure_schema(self) -> None:
        """Ensure the required node and edge types exist in the graph."""
        # Define node types if they don't exist
        if "Message" not in self.graph.node_types:
            self.graph.create_node_type("Message", {
                "id": str,
                "role": str,
                "content": str,
                "namespace": str,
                "timestamp": int,
                "metadata": str  # JSON serialized
            })
        
        if "Entity" not in self.graph.node_types:
            self.graph.create_node_type("Entity", {
                "value": str,
                "type": str,
                "namespace": str
            })
        
        # Define edge types if they don't exist
        if "NEXT" not in self.graph.edge_types:
            self.graph.create_edge_type("NEXT", {
                "timestamp": int
            })
        
        if "MENTIONS" not in self.graph.edge_types:
            self.graph.create_edge_type("MENTIONS", {
                "confidence": float
            })
        
        if "REFERENCES" not in self.graph.edge_types:
            self.graph.create_edge_type("REFERENCES", {
                "relationship": str
            })
    
    def add_history(self, message: Dict) -> None:
        """Add a message to conversation history.
        
        Args:
            message: Message dictionary to add
        """
        if not self.graph:
            self.connect()
        
        # Store in local history first
        self.history.append(message)
        
        # Create a message node in the graph
        msg_id = message.get("id", str(uuid.uuid4()))
        properties = {
            "id": msg_id,
            "role": message.get("role", "unknown"),
            "content": message.get("content", ""),
            "namespace": self.namespace,
            "timestamp": message.get("timestamp", self._current_timestamp()),
            "metadata": json.dumps(message.get("metadata", {}))
        }
        
        with self.graph.transaction() as tx:
            # Create the message node
            node = tx.add_node("Message", properties)
            
            # Link to previous message if it exists
            if len(self.history) > 1:
                prev_msg_id = self.history[-2].get("id", None)
                if prev_msg_id:
                    # Find the previous message node
                    query = f'MATCH (m:Message) WHERE m.id = "{prev_msg_id}" RETURN m'
                    result = tx.run(query)
                    if result:
                        for record in result:
                            prev_node = record["m"]
                            # Create NEXT relationship from previous to current
                            tx.add_edge("NEXT", prev_node, node, {"timestamp": self._current_timestamp()})
            
            # Extract and link entities (simplified implementation)
            # In a real implementation, you might use NER tools
            self._extract_and_link_entities(tx, node, message.get("content", ""))
    
    def _extract_and_link_entities(self, tx, message_node, content: str) -> None:
        """Extract entities from content and link them to the message.
        
        This is a simplified implementation. A production version would use
        NLP techniques for entity extraction.
        
        Args:
            tx: Graph transaction
            message_node: The message node to link entities to
            content: Message content to extract entities from
        """
        # Simple keyword extraction (demo purposes only)
        # A real implementation would use NER or other NLP techniques
        keywords = [word for word in content.lower().split() 
                   if len(word) > 4 and word not in ["about", "there", "their", "would", "should"]]
        
        for keyword in keywords:
            # Check if entity exists
            query = f'MATCH (e:Entity) WHERE e.value = "{keyword}" AND e.namespace = "{self.namespace}" RETURN e'
            result = tx.run(query)
            entity_node = None
            
            # Create or reuse entity
            if not result:
                entity_props = {
                    "value": keyword,
                    "type": "keyword",
                    "namespace": self.namespace
                }
                entity_node = tx.add_node("Entity", entity_props)
            else:
                for record in result:
                    entity_node = record["e"]
                    break
            
            if entity_node:
                # Link message to entity
                tx.add_edge("MENTIONS", message_node, entity_node, {"confidence": 1.0})
    
    def get_history(self, limit: int = -1) -> List:
        """Get conversation history.
        
        Args:
            limit: Maximum number of history items to return
            
        Returns:
            List of messages
        """
        # First check local cache
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
    
    def fetch_cache(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
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
        
        if self.graph:
            with self.graph.transaction() as tx:
                # Delete all message nodes for this namespace
                query = f'MATCH (m:Message) WHERE m.namespace = "{self.namespace}" DETACH DELETE m'
                tx.run(query)
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        self.cache = {}
    
    def save_history(self) -> bool:
        """Save history to permanent storage.
        
        Returns:
            True if successful, False otherwise
        """
        # The graph is updated in real-time, so we don't need to explicitly save
        return True
    
    def load_history(self) -> List:
        """Load history from permanent storage.
        
        Returns:
            List of messages
        """
        if not self.graph:
            self.connect()
        
        if not self.history:
            # Rebuild history from graph
            with self.graph.transaction() as tx:
                # Get all messages for this namespace, ordered by timestamp
                query = f'''
                MATCH (m:Message) 
                WHERE m.namespace = "{self.namespace}" 
                RETURN m 
                ORDER BY m.timestamp
                '''
                result = tx.run(query)
                
                if result:
                    self.history = []
                    for record in result:
                        node = record["m"]
                        message = self._message_from_node(node)
                        self.history.append(message)
        
        return self.history
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search for messages based on content or entities.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching messages
        """
        if not self.graph:
            self.connect()
        
        results = []
        
        with self.graph.transaction() as tx:
            # Search by direct content match
            content_query = f'''
            MATCH (m:Message) 
            WHERE m.namespace = "{self.namespace}" AND m.content CONTAINS "{query}"
            RETURN m
            ORDER BY m.timestamp DESC
            LIMIT {limit}
            '''
            content_result = tx.run(content_query)
            
            if content_result:
                for record in content_result:
                    node = record["m"]
                    message = self._message_from_node(node)
                    message["match_type"] = "content"
                    results.append(message)
            
            # If we need more results, search by entity relationship
            if len(results) < limit:
                remaining = limit - len(results)
                entity_query = f'''
                MATCH (m:Message)-[:MENTIONS]->(e:Entity)
                WHERE m.namespace = "{self.namespace}" AND e.value CONTAINS "{query.lower()}"
                RETURN DISTINCT m
                ORDER BY m.timestamp DESC
                LIMIT {remaining}
                '''
                entity_result = tx.run(entity_query)
                
                if entity_result:
                    for record in entity_result:
                        node = record["m"]
                        message = self._message_from_node(node)
                        message["match_type"] = "entity"
                        
                        # Avoid duplicates
                        if not any(r["id"] == message["id"] for r in results):
                            results.append(message)
        
        return results
    
    def _message_from_node(self, node) -> Dict:
        """Convert a graph node to a message dictionary.
        
        Args:
            node: Graph node representing a message
            
        Returns:
            Message dictionary
        """
        message = {
            "id": node.properties["id"],
            "role": node.properties["role"],
            "content": node.properties["content"],
            "timestamp": node.properties["timestamp"]
        }
        
        # Add metadata if available
        try:
            metadata = json.loads(node.properties["metadata"])
            if metadata:
                message["metadata"] = metadata
        except:
            pass
            
        return message
    
    def _current_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        import time
        return int(time.time() * 1000) 