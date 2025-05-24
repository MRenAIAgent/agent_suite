"""Neo4j graph storage implementation.

This module provides a GraphStorage implementation using Neo4j,
a popular graph database for storing and querying graph data.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union, Set

from agents.storage.graph_storage import GraphStorage

try:
    from neo4j import GraphDatabase, basic_auth
    # Import Result for type hinting
    from neo4j.work.result import Result
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    # Define a dummy Result type for type hinting when neo4j isn't available
    class Result:
        """Dummy Result class when neo4j isn't available."""
        pass


class Neo4jStorage(GraphStorage):
    """Neo4j implementation of the graph storage interface.
    
    Provides graph storage and traversal capabilities using
    Neo4j as the backend.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        database: Optional[str] = None,
        namespace: str = "default",
        **kwargs
    ):
        """Initialize the Neo4j storage.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name (None for default)
            namespace: Namespace for grouping nodes and relationships
            **kwargs: Additional connection parameters
            
        Raises:
            ImportError: If Neo4j client is not installed
        """
        if not NEO4J_AVAILABLE:
            raise ImportError(
                "Neo4j client is required for Neo4jStorage. "
                "Install it with: pip install neo4j"
            )
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.namespace = namespace
        
        # Initialize driver
        self.driver = GraphDatabase.driver(
            uri, 
            auth=basic_auth(username, password),
            **kwargs
        )
        
        # Initialize indexes for better performance
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Ensure indexes exist for better query performance."""
        with self.driver.session(database=self.database) as session:
            # Index on node IDs
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.id)")
            # Index on node namespace
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.namespace)")
            # Index on node type
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Node) ON (n.type)")
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        self.driver.close()
    
    def _process_result(self, result: Result) -> List[Dict[str, Any]]:
        """Process Neo4j result into a standardized format.
        
        Args:
            result: Neo4j query result
            
        Returns:
            List of results as dictionaries
        """
        processed_results = []
        
        for record in result:
            record_dict = {}
            for key, value in record.items():
                # Handle Neo4j node types
                if hasattr(value, "id") and hasattr(value, "labels") and hasattr(value, "items"):
                    # It's a Neo4j node
                    node_dict = dict(value.items())
                    node_dict["_labels"] = list(value.labels)
                    node_dict["_id"] = value.id
                    record_dict[key] = node_dict
                # Handle Neo4j relationship types
                elif hasattr(value, "id") and hasattr(value, "type") and hasattr(value, "items"):
                    # It's a Neo4j relationship
                    rel_dict = dict(value.items())
                    rel_dict["_type"] = value.type
                    rel_dict["_id"] = value.id
                    record_dict[key] = rel_dict
                # Handle Neo4j paths
                elif hasattr(value, "start_node") and hasattr(value, "end_node"):
                    # It's a Neo4j path
                    record_dict[key] = {
                        "start": dict(value.start_node.items()),
                        "end": dict(value.end_node.items()),
                        "segments": [
                            {
                                "start": dict(segment.start_node.items()),
                                "relationship": {
                                    "type": segment.relationship.type,
                                    **dict(segment.relationship.items())
                                },
                                "end": dict(segment.end_node.items())
                            }
                            for segment in value
                        ]
                    }
                else:
                    record_dict[key] = value
            
            processed_results.append(record_dict)
        
        return processed_results
    
    async def store(self, key: str, data: Dict[str, Any], **kwargs) -> str:
        """Store data with the given key.
        
        Args:
            key: Unique identifier for the data
            data: The data to store
            **kwargs: Additional storage parameters
            
        Returns:
            Identifier for the stored data
        """
        # This is a generic method - we implement specific node/relationship creation separately
        if data.get("type") == "node":
            return await self.create_node(
                data.get("node_type", "Generic"),
                data.get("properties", {})
            )
        elif data.get("type") == "relationship":
            return await self.create_relationship(
                data.get("source_id"),
                data.get("target_id"),
                data.get("relationship_type", "RELATED_TO"),
                data.get("properties", {})
            )
        else:
            # Not a specific graph entity type, store as a property node
            properties = {
                "id": key,
                "namespace": self.namespace,
                "data": json.dumps(data)
            }
            
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MERGE (n:Node:Data {id: $id, namespace: $namespace})
                    SET n.data = $data
                    RETURN n.id as id
                    """,
                    id=key,
                    namespace=self.namespace,
                    data=properties["data"]
                )
                
                for record in result:
                    return record["id"]
            
            return key
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: Unique identifier for the data
            **kwargs: Additional retrieval parameters
            
        Returns:
            The stored data or None if not found
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (n:Node {id: $id, namespace: $namespace})
                RETURN n
                LIMIT 1
                """,
                id=key,
                namespace=self.namespace
            )
            
            processed = self._process_result(result)
            if not processed:
                return None
                
            node_data = processed[0].get("n", {})
            
            # Convert stored JSON data if present
            if "data" in node_data and isinstance(node_data["data"], str):
                try:
                    node_data["data"] = json.loads(node_data["data"])
                except json.JSONDecodeError:
                    pass
            
            return node_data
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: Unique identifier for the data to delete
            **kwargs: Additional deletion parameters
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (n:Node {id: $id, namespace: $namespace})
                    DETACH DELETE n
                    RETURN count(n) as deleted
                    """,
                    id=key,
                    namespace=self.namespace
                )
                
                for record in result:
                    return record["deleted"] > 0
            
            return False
        except Exception:
            return False
    
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update data associated with the key.
        
        Args:
            key: Unique identifier for the data to update
            data: New data to store
            **kwargs: Additional update parameters
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # If it's properties update for a specific node
            if "properties" in data:
                prop_updates = []
                params = {
                    "id": key,
                    "namespace": self.namespace
                }
                
                for prop_key, prop_value in data["properties"].items():
                    prop_param = f"prop_{prop_key}"
                    params[prop_param] = (
                        json.dumps(prop_value) 
                        if isinstance(prop_value, (dict, list)) 
                        else prop_value
                    )
                    prop_updates.append(f"n.{prop_key} = ${prop_param}")
                
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        f"""
                        MATCH (n:Node {{id: $id, namespace: $namespace}})
                        SET {', '.join(prop_updates)}
                        RETURN count(n) as updated
                        """,
                        **params
                    )
                    
                    for record in result:
                        return record["updated"] > 0
            else:
                # Generic update - replace the entire data
                with self.driver.session(database=self.database) as session:
                    result = session.run(
                        """
                        MATCH (n:Node {id: $id, namespace: $namespace})
                        SET n.data = $data
                        RETURN count(n) as updated
                        """,
                        id=key,
                        namespace=self.namespace,
                        data=json.dumps(data)
                    )
                    
                    for record in result:
                        return record["updated"] > 0
            
            return False
        except Exception:
            return False
    
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List keys matching a pattern.
        
        Args:
            pattern: Pattern to match keys against
            **kwargs: Additional parameters for the listing operation
            
        Returns:
            List of matching keys
        """
        # Node types to filter by
        node_types = kwargs.get("node_types", [])
        limit = kwargs.get("limit", 100)
        
        # Convert glob pattern to Cypher-compatible
        if pattern == "*":
            cypher_pattern = ".*"
        else:
            # Simple conversion for basic patterns
            cypher_pattern = pattern.replace("*", ".*").replace("?", ".")
        
        # Build the Cypher query
        cypher_query = """
        MATCH (n:Node {namespace: $namespace})
        WHERE n.id =~ $pattern
        """
        
        # Add node type filter if specified
        if node_types:
            type_labels = " OR ".join([f"n:{t}" for t in node_types])
            cypher_query += f"AND ({type_labels})"
        
        cypher_query += """
        RETURN n.id as id
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                cypher_query,
                namespace=self.namespace,
                pattern=cypher_pattern,
                limit=limit
            )
            
            return [record["id"] for record in result]
    
    async def clear(self, **kwargs) -> bool:
        """Clear all data from the storage.
        
        Args:
            **kwargs: Additional parameters for the clear operation
            
        Returns:
            True if clear was successful, False otherwise
        """
        try:
            # By default, only clear nodes for the current namespace
            only_namespace = kwargs.get("only_namespace", True)
            
            with self.driver.session(database=self.database) as session:
                if only_namespace:
                    session.run(
                        """
                        MATCH (n:Node {namespace: $namespace})
                        DETACH DELETE n
                        """,
                        namespace=self.namespace
                    )
                else:
                    # Clear all nodes (use with caution!)
                    session.run("MATCH (n) DETACH DELETE n")
            
            return True
        except Exception:
            return False
    
    async def create_node(
        self, 
        node_type: str, 
        properties: Dict[str, Any],
        **kwargs
    ) -> str:
        """Create a node in the graph.
        
        Args:
            node_type: Type of node to create
            properties: Properties for the node
            **kwargs: Additional node creation parameters
            
        Returns:
            Identifier for the created node
        """
        # Generate ID if not provided
        node_id = properties.get("id", str(uuid.uuid4()))
        properties["id"] = node_id
        properties["namespace"] = self.namespace
        
        # Prepare properties - serialize complex objects
        params = {
            "props": {
                k: (
                    json.dumps(v) 
                    if isinstance(v, (dict, list)) 
                    else v
                )
                for k, v in properties.items()
            }
        }
        
        # Make node_type acceptable for Neo4j
        safe_node_type = node_type.replace(" ", "_")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                CREATE (n:Node:{safe_node_type} $props)
                RETURN n.id as id
                """,
                **params
            )
            
            for record in result:
                return record["id"]
        
        return node_id
    
    async def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Create a relationship between nodes.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: Properties for the relationship
            **kwargs: Additional relationship creation parameters
            
        Returns:
            Identifier for the created relationship
        """
        properties = properties or {}
        relationship_id = properties.get("id", f"{source_id}_{relationship_type}_{target_id}")
        properties["id"] = relationship_id
        properties["namespace"] = self.namespace
        
        # Prepare properties - serialize complex objects
        rel_props = {
            k: (
                json.dumps(v) 
                if isinstance(v, (dict, list)) 
                else v
            )
            for k, v in properties.items()
        }
        
        # Make relationship type acceptable for Neo4j
        safe_rel_type = relationship_type.replace(" ", "_").upper()
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                f"""
                MATCH (src:Node {{id: $source_id, namespace: $namespace}})
                MATCH (tgt:Node {{id: $target_id, namespace: $namespace}})
                CREATE (src)-[r:{safe_rel_type} $rel_props]->(tgt)
                RETURN r.id as id
                """,
                source_id=source_id,
                target_id=target_id,
                namespace=self.namespace,
                rel_props=rel_props
            )
            
            for record in result:
                return record["id"]
        
        return relationship_id
    
    async def get_nodes(
        self, 
        node_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get nodes matching criteria.
        
        Args:
            node_type: Type of nodes to retrieve
            properties: Filter by properties
            limit: Maximum number of nodes to return
            **kwargs: Additional query parameters
            
        Returns:
            List of matching nodes
        """
        # Build the query
        query_parts = ["MATCH (n:Node {namespace: $namespace})"]
        params = {"namespace": self.namespace, "limit": limit}
        
        # Add node type filter if specified
        if node_type:
            safe_node_type = node_type.replace(" ", "_")
            query_parts.append(f"WHERE n:{safe_node_type}")
        
        # Add property filters if specified
        if properties:
            if node_type:
                query_parts[-1] += " AND "
            else:
                query_parts.append("WHERE ")
            
            filter_parts = []
            for idx, (key, value) in enumerate(properties.items()):
                param_name = f"prop_{idx}"
                params[param_name] = value
                filter_parts.append(f"n.{key} = ${param_name}")
            
            query_parts[-1] += " AND ".join(filter_parts)
        
        # Finalize the query
        query_parts.append("RETURN n LIMIT $limit")
        query = " ".join(query_parts)
        
        # Execute the query
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            return self._process_result(result)
    
    async def get_relationships(
        self, 
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get relationships matching criteria.
        
        Args:
            source_id: Source node identifier
            target_id: Target node identifier
            relationship_type: Type of relationship
            properties: Filter by properties
            limit: Maximum number of relationships to return
            **kwargs: Additional query parameters
            
        Returns:
            List of matching relationships
        """
        # Build the query
        params = {"namespace": self.namespace, "limit": limit}
        
        # Match pattern based on what's provided
        if source_id and target_id:
            match_pattern = "MATCH (src:Node {id: $source_id, namespace: $namespace})-[r]->(tgt:Node {id: $target_id, namespace: $namespace})"
            params["source_id"] = source_id
            params["target_id"] = target_id
        elif source_id:
            match_pattern = "MATCH (src:Node {id: $source_id, namespace: $namespace})-[r]->(tgt:Node {namespace: $namespace})"
            params["source_id"] = source_id
        elif target_id:
            match_pattern = "MATCH (src:Node {namespace: $namespace})-[r]->(tgt:Node {id: $target_id, namespace: $namespace})"
            params["target_id"] = target_id
        else:
            match_pattern = "MATCH (src:Node {namespace: $namespace})-[r]->(tgt:Node {namespace: $namespace})"
        
        query_parts = [match_pattern]
        where_parts = []
        
        # Add relationship type filter if specified
        if relationship_type:
            safe_rel_type = relationship_type.replace(" ", "_").upper()
            where_parts.append(f"type(r) = '{safe_rel_type}'")
        
        # Add property filters if specified
        if properties:
            for idx, (key, value) in enumerate(properties.items()):
                param_name = f"prop_{idx}"
                params[param_name] = value
                where_parts.append(f"r.{key} = ${param_name}")
        
        # Add WHERE clause if needed
        if where_parts:
            query_parts.append("WHERE " + " AND ".join(where_parts))
        
        # Finalize the query
        query_parts.append("RETURN r, src, tgt LIMIT $limit")
        query = " ".join(query_parts)
        
        # Execute the query
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            processed = self._process_result(result)
            
            # Format the results to include source and target
            relationships = []
            for item in processed:
                relationships.append({
                    "relationship": item.get("r", {}),
                    "source": item.get("src", {}),
                    "target": item.get("tgt", {})
                })
            
            return relationships
    
    async def query(
        self, 
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Execute a graph query.
        
        Args:
            query: The query to execute (Cypher for Neo4j)
            parameters: Query parameters
            **kwargs: Additional query execution parameters
            
        Returns:
            Query results
        """
        parameters = parameters or {}
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **parameters)
            return self._process_result(result)
    
    async def traverse(
        self, 
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "outgoing",
        max_depth: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Traverse the graph from a starting node.
        
        Args:
            start_node_id: Starting node identifier
            relationship_types: Types of relationships to traverse
            direction: Direction of traversal ("outgoing", "incoming", "both")
            max_depth: Maximum traversal depth
            **kwargs: Additional traversal parameters
            
        Returns:
            Traversal results
        """
        # Determine relationship direction in Cypher
        if direction == "outgoing":
            path_pattern = "-[r:REL]->"
        elif direction == "incoming":
            path_pattern = "<-[r:REL]-"
        else:  # "both"
            path_pattern = "-[r:REL]-"
        
        # Build relationship type filter
        if relationship_types:
            rel_types = [r.replace(" ", "_").upper() for r in relationship_types]
            rel_filter = "|".join(rel_types)
            path_pattern = path_pattern.replace("REL", rel_filter)
        else:
            path_pattern = path_pattern.replace(":REL", "")
        
        # Build the query
        params = {
            "start_id": start_node_id,
            "namespace": self.namespace,
            "max_depth": max_depth
        }
        
        query = f"""
        MATCH path = (start:Node {{id: $start_id, namespace: $namespace}})
        {path_pattern}*1..{max_depth} (related:Node {{namespace: $namespace}})
        RETURN path
        """
        
        with self.driver.session(database=self.database) as session:
            result = session.run(query, **params)
            paths = self._process_result(result)
            
            # Format the result
            traversal_result = {
                "start_node": {"id": start_node_id},
                "paths": paths
            }
            
            return traversal_result 