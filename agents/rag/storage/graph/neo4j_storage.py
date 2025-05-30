"""Neo4j graph storage adaptor for RAG system.

This module provides a Neo4j-based implementation of the GraphStorageAdaptor
interface for storing and querying graph data in the RAG system.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Union

from agents.rag.storage.graph.base import GraphStorageAdaptor
from agents.rag.models.knowledge import Entity, Relationship

try:
    from neo4j import GraphDatabase, basic_auth
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class Neo4jGraphStorageAdaptor(GraphStorageAdaptor):
    """Neo4j implementation of the graph storage adaptor for RAG."""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "neo4j",
        database: Optional[str] = None,
        namespace: str = "rag_math_learning",
        **kwargs
    ):
        """Initialize the Neo4j graph storage adaptor.
        
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
                "Neo4j client is required for Neo4jGraphStorageAdaptor. "
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
            # Index on entity IDs
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.id)")
            # Index on entity namespace
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.namespace)")
            # Index on entity type
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)")
            # Index on entity name for text search
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    
    async def store_entity(self, entity: Entity) -> str:
        """Store an entity in the graph.
        
        Args:
            entity: The entity to store
            
        Returns:
            The entity ID
        """
        # Extract name and description from properties
        name = entity.properties.get("name", "")
        description = entity.properties.get("description", "")
        
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MERGE (e:Entity {id: $id, namespace: $namespace})
                SET e.type = $type,
                    e.name = $name,
                    e.description = $description,
                    e.properties = $properties,
                    e.updated_at = datetime()
                RETURN e.id as id
                """,
                id=entity.id,
                namespace=self.namespace,
                type=entity.type,
                name=name,
                description=description,
                properties=json.dumps(entity.properties)
            )
            
            for record in result:
                return record["id"]
        
        return entity.id
    
    async def store_relationship(self, relationship: Relationship) -> str:
        """Store a relationship in the graph.
        
        Args:
            relationship: The relationship to store
            
        Returns:
            The relationship ID
        """
        with self.driver.session(database=self.database) as session:
            # First ensure both entities exist
            session.run(
                """
                MERGE (source:Entity {id: $source_id, namespace: $namespace})
                MERGE (target:Entity {id: $target_id, namespace: $namespace})
                """,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                namespace=self.namespace
            )
            
            # Create the relationship
            result = session.run(
                f"""
                MATCH (source:Entity {{id: $source_id, namespace: $namespace}})
                MATCH (target:Entity {{id: $target_id, namespace: $namespace}})
                MERGE (source)-[r:{relationship.type} {{id: $id}}]->(target)
                SET r.properties = $properties,
                    r.updated_at = datetime()
                RETURN r.id as id
                """,
                id=relationship.id,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                namespace=self.namespace,
                properties=json.dumps(relationship.properties)
            )
            
            for record in result:
                return record["id"]
        
        return relationship.id
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID.
        
        Args:
            entity_id: The entity ID
            
        Returns:
            The entity if found, None otherwise
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (e:Entity {id: $id, namespace: $namespace})
                RETURN e.id as id, e.type as type, e.name as name, 
                       e.description as description, e.properties as properties
                """,
                id=entity_id,
                namespace=self.namespace
            )
            
            for record in result:
                properties = json.loads(record["properties"]) if record["properties"] else {}
                # Add name and description to properties if they exist as separate fields
                if record.get("name"):
                    properties["name"] = record["name"]
                if record.get("description"):
                    properties["description"] = record["description"]
                
                return Entity(
                    id=record["id"],
                    type=record["type"],
                    properties=properties
                )
        
        return None
    
    async def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Retrieve a relationship by ID.
        
        Args:
            relationship_id: The relationship ID
            
        Returns:
            The relationship if found, None otherwise
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH (source:Entity)-[r {id: $id}]->(target:Entity)
                WHERE source.namespace = $namespace AND target.namespace = $namespace
                RETURN r.id as id, type(r) as type, source.id as source_id, 
                       target.id as target_id, r.properties as properties
                """,
                id=relationship_id,
                namespace=self.namespace
            )
            
            for record in result:
                properties = json.loads(record["properties"]) if record["properties"] else {}
                return Relationship(
                    id=record["id"],
                    type=record["type"],
                    source_id=record["source_id"],
                    target_id=record["target_id"],
                    properties=properties
                )
        
        return None
    
    async def query_graph(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a graph query.
        
        Args:
            query: The query string
            parameters: Query parameters
            
        Returns:
            Query results
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            
            processed_results = []
            for record in result:
                record_dict = {}
                for key, value in record.items():
                    # Handle Neo4j node types
                    if hasattr(value, "id") and hasattr(value, "labels") and hasattr(value, "items"):
                        # It's a Neo4j node
                        node_dict = dict(value.items())
                        node_dict["_labels"] = list(value.labels)
                        node_dict["_neo4j_id"] = value.id
                        record_dict[key] = node_dict
                    # Handle Neo4j relationship types
                    elif hasattr(value, "id") and hasattr(value, "type") and hasattr(value, "items"):
                        # It's a Neo4j relationship
                        rel_dict = dict(value.items())
                        rel_dict["_type"] = value.type
                        rel_dict["_neo4j_id"] = value.id
                        record_dict[key] = rel_dict
                    else:
                        record_dict[key] = value
                
                processed_results.append(record_dict)
            
            return processed_results
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search entities and relationships.
        
        Args:
            query: Search query (can be text or structured)
            limit: Maximum number of results
            
        Returns:
            Search results
        """
        # Handle different query types
        if isinstance(query, str):
            # Text search across entity names and descriptions
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (e:Entity {namespace: $namespace})
                    WHERE e.name CONTAINS $query OR e.description CONTAINS $query
                    RETURN e.id as id, e.type as type, e.name as name, 
                           e.description as description, e.properties as properties
                    LIMIT $limit
                    """,
                    namespace=self.namespace,
                    query=query,
                    limit=limit
                )
                
                results = []
                for record in result:
                    properties = json.loads(record["properties"]) if record["properties"] else {}
                    results.append({
                        "type": "entity",
                        "id": record["id"],
                        "entity_type": record["type"],
                        "name": record["name"],
                        "description": record["description"],
                        "properties": properties
                    })
                
                return results
        
        return []
    
    async def store(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store data in the graph storage.
        
        Args:
            data: The data to store
            metadata: Optional metadata
            
        Returns:
            Storage identifier
        """
        # Determine if this is an entity or relationship
        if "source_id" in data and "target_id" in data:
            # It's a relationship
            relationship = Relationship(
                id=data.get("id", str(uuid.uuid4())),
                type=data.get("type", "RELATED_TO"),
                source_id=data["source_id"],
                target_id=data["target_id"],
                properties=data.get("properties", {})
            )
            return await self.store_relationship(relationship)
        else:
            # It's an entity
            entity = Entity(
                id=data.get("id", str(uuid.uuid4())),
                type=data.get("type", "Generic"),
                properties={
                    "name": data.get("name", ""),
                    "description": data.get("description", ""),
                    **data.get("properties", {})
                }
            )
            return await self.store_entity(entity)
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Retrieve data by key.
        
        Args:
            key: The data key
            **kwargs: Additional parameters
            
        Returns:
            The retrieved data
        """
        # Try to get as entity first
        entity = await self.get_entity(key)
        if entity:
            return {
                "id": entity.id,
                "type": entity.type,
                "name": entity.name,
                "description": entity.description,
                "properties": entity.properties
            }
        
        # Try to get as relationship
        relationship = await self.get_relationship(key)
        if relationship:
            return {
                "id": relationship.id,
                "type": relationship.type,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": relationship.properties
            }
        
        return None
    
    async def update(self, key: str, data: Dict[str, Any], **kwargs) -> bool:
        """Update data by key.
        
        Args:
            key: The data key
            data: Updated data
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        # Store will merge/update existing data
        await self.store({"id": key, **data})
        return True
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key.
        
        Args:
            key: The data key
            **kwargs: Additional parameters
            
        Returns:
            True if successful
        """
        with self.driver.session(database=self.database) as session:
            # Delete entity and all its relationships
            result = session.run(
                """
                MATCH (e:Entity {id: $id, namespace: $namespace})
                DETACH DELETE e
                RETURN count(e) as deleted
                """,
                id=key,
                namespace=self.namespace
            )
            
            for record in result:
                return record["deleted"] > 0
        
        return False
    
    async def close(self) -> None:
        """Close the storage connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
    
    async def find_paths(self, start_entity_id: str, end_entity_id: str, 
                       max_depth: int = 3) -> List[List[Union[Entity, Relationship]]]:
        """Find paths between entities.
        
        Args:
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each a list of alternating entities and relationships)
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                """
                MATCH path = (start:Entity {id: $start_id, namespace: $namespace})
                -[*1..$max_depth]->
                (end:Entity {id: $end_id, namespace: $namespace})
                RETURN path
                LIMIT 10
                """,
                start_id=start_entity_id,
                end_id=end_entity_id,
                namespace=self.namespace,
                max_depth=max_depth
            )
            
            paths = []
            for record in result:
                path = record["path"]
                path_elements = []
                
                # Extract nodes and relationships from the path
                nodes = path.nodes
                relationships = path.relationships
                
                # Alternate between nodes and relationships
                for i, node in enumerate(nodes):
                    # Add entity
                    properties = json.loads(node.get("properties", "{}")) if node.get("properties") else dict(node)
                    # Add name and description to properties if they exist as separate fields
                    if node.get("name"):
                        properties["name"] = node["name"]
                    if node.get("description"):
                        properties["description"] = node["description"]
                    
                    entity = Entity(
                        id=node["id"],
                        type=node.get("type", "Unknown"),
                        properties=properties
                    )
                    path_elements.append(entity)
                    
                    # Add relationship if not the last node
                    if i < len(relationships):
                        rel = relationships[i]
                        rel_properties = json.loads(rel.get("properties", "{}")) if rel.get("properties") else dict(rel)
                        relationship = Relationship(
                            id=rel.get("id", str(uuid.uuid4())),
                            type=rel.type,
                            source_id=rel.start_node["id"],
                            target_id=rel.end_node["id"],
                            properties=rel_properties
                        )
                        path_elements.append(relationship)
                
                paths.append(path_elements)
            
            return paths
    
    async def store_knowledge_graph(self, graph: 'KnowledgeGraph') -> str:
        """Store a complete knowledge graph.
        
        Args:
            graph: The knowledge graph to store
            
        Returns:
            The ID of the stored graph
        """
        # Store all entities first
        for entity in graph.entities:
            await self.store_entity(entity)
        
        # Store all relationships
        for relationship in graph.relationships:
            await self.store_relationship(relationship)
        
        # Return a graph identifier (could be based on entities/relationships)
        graph_id = f"graph_{len(graph.entities)}_{len(graph.relationships)}_{uuid.uuid4().hex[:8]}"
        
        # Optionally store graph metadata
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                CREATE (g:KnowledgeGraph {
                    id: $graph_id,
                    namespace: $namespace,
                    entity_count: $entity_count,
                    relationship_count: $relationship_count,
                    created_at: datetime()
                })
                """,
                graph_id=graph_id,
                namespace=self.namespace,
                entity_count=len(graph.entities),
                relationship_count=len(graph.relationships)
            )
        
        return graph_id 