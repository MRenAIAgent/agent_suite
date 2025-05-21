"""
Graphiti knowledge graph implementation for long-term memory.
"""

import uuid
import datetime
from typing import Any, Dict, List, Optional, Union

from agent_suite.knowledge.graphiti import GraphitiClient
from agent_suite.memory.long_term.graph.base import GraphMemory


class GraphitiMemory(GraphMemory):
    """
    Long-term memory implementation using Graphiti knowledge graph.
    
    This implementation uses the Graphiti knowledge graph service to
    store and retrieve memories as graph entities and relationships.
    """
    
    def __init__(
        self,
        graphiti_client: Optional[GraphitiClient] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """
        Initialize with either an existing GraphitiClient or credentials.
        
        Args:
            graphiti_client: Existing GraphitiClient instance
            endpoint: Graphiti API endpoint URL
            api_key: Graphiti API key
            user_id: Optional user ID for user-specific memory
        """
        if graphiti_client:
            self.client = graphiti_client
        elif endpoint and api_key:
            self.client = GraphitiClient(endpoint=endpoint, api_key=api_key)
        else:
            raise ValueError(
                "Either provide a GraphitiClient instance or endpoint and api_key"
            )
        
        self.user_id = user_id
    
    def store(self, data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store structured data in the graph.
        
        Args:
            data: Dictionary containing entity data and relationships
            metadata: Optional metadata
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
        
        # Add memory_id to metadata
        metadata["memory_id"] = memory_id
        
        # Add user_id if available
        if self.user_id:
            metadata["user_id"] = self.user_id
        
        # Process the data based on its structure
        if "entity" in data:
            # Store an entity with its properties
            entity_data = data["entity"]
            entity_type = entity_data.get("type", "Memory")
            properties = entity_data.get("properties", {})
            
            # Add metadata to properties
            for key, value in metadata.items():
                if key not in properties:
                    properties[key] = value
            
            # Store the entity
            result = self.client.add_entity(entity_type, properties)
            return result.get("id", memory_id)
            
        elif "entities" in data and "relationships" in data:
            # Store multiple entities and relationships
            entity_ids = {}
            
            # First, store all entities
            for entity in data["entities"]:
                entity_type = entity.get("type", "Memory")
                properties = entity.get("properties", {}).copy()
                
                # Add metadata to first entity only
                if not entity_ids:
                    for key, value in metadata.items():
                        if key not in properties:
                            properties[key] = value
                
                # Store entity
                result = self.client.add_entity(entity_type, properties)
                entity_id = result.get("id")
                temp_id = entity.get("id", str(uuid.uuid4()))
                entity_ids[temp_id] = entity_id
            
            # Then, store all relationships
            for rel in data["relationships"]:
                from_id = entity_ids.get(rel["from"])
                to_id = entity_ids.get(rel["to"])
                rel_type = rel.get("type", "RELATED_TO")
                properties = rel.get("properties", {})
                
                if from_id and to_id:
                    self.client.add_relationship(from_id, rel_type, to_id, properties)
            
            # Return the first entity's ID as the memory ID
            return list(entity_ids.values())[0] if entity_ids else memory_id
        
        else:
            # Simple entity creation with data as properties
            result = self.client.add_entity("Memory", {**data, **metadata})
            return result.get("id", memory_id)
    
    def add_entity(self, entity_type: str, properties: Dict[str, Any]) -> str:
        """
        Add an entity node to the graph.
        
        Args:
            entity_type: Type of entity
            properties: Entity properties
            
        Returns:
            entity_id: Unique identifier for the stored entity
        """
        # Add user_id if available
        if self.user_id and "user_id" not in properties:
            properties = properties.copy()
            properties["user_id"] = self.user_id
        
        result = self.client.add_entity(entity_type, properties)
        return result.get("id", str(uuid.uuid4()))
    
    def add_relationship(self, from_entity: str, relationship_type: str, to_entity: str, properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a relationship between entities.
        
        Args:
            from_entity: Source entity ID
            relationship_type: Type of relationship
            to_entity: Target entity ID
            properties: Optional relationship properties
            
        Returns:
            relationship_id: Unique identifier for the relationship
        """
        if properties is None:
            properties = {}
        
        result = self.client.add_relationship(from_entity, relationship_type, to_entity, properties)
        return result.get("id", str(uuid.uuid4()))
    
    def query(self, query: Union[str, Dict], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results
            
        Returns:
            List of matching subgraphs or paths
        """
        # Process the query
        if isinstance(query, str):
            # Natural language query
            results = self.client.query(query, limit=limit, user_id=self.user_id)
            return results
        else:
            # Structured query
            if "cypher" in query:
                # Raw Cypher query with user_id filter
                cypher_query = query["cypher"]
                
                # Add user filter if user_id is set and not already in query
                if self.user_id and "user_id" not in cypher_query:
                    # This is a simplified approach - proper implementation would parse and modify Cypher
                    # Using a parameter would be better in practice
                    if "WHERE" in cypher_query:
                        cypher_query = cypher_query.replace(
                            "WHERE",
                            f"WHERE n.user_id = '{self.user_id}' AND "
                        )
                    else:
                        # Add WHERE clause before RETURN
                        cypher_query = cypher_query.replace(
                            "RETURN",
                            f"WHERE n.user_id = '{self.user_id}' RETURN"
                        )
                
                results = self.client.run_cypher(cypher_query, limit=limit)
                return results
            
            elif "entity_type" in query:
                # Query by entity type and properties
                entity_type = query["entity_type"]
                properties = query.get("properties", {})
                
                # Add user_id filter if set
                if self.user_id and "user_id" not in properties:
                    properties = properties.copy()
                    properties["user_id"] = self.user_id
                
                results = self.client.query_entities(entity_type, properties, limit=limit)
                return results
            
            else:
                # Default to client's query method
                results = self.client.query(query, limit=limit, user_id=self.user_id)
                return results
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a specific memory item.
        
        Args:
            memory_id: The unique identifier of the memory
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            result = self.client.delete_entity(memory_id)
            return result.get("success", False)
        except Exception:
            return False
    
    def clear(self) -> None:
        """
        Clear all stored memories.
        
        If user_id is set, only clear that user's memories.
        """
        if self.user_id:
            # Query entities with this user_id
            query = f"MATCH (n) WHERE n.user_id = '{self.user_id}' RETURN n"
            results = self.client.run_cypher(query)
            
            # Delete each entity
            for result in results:
                if "id" in result.get("n", {}):
                    self.client.delete_entity(result["n"]["id"])
        else:
            # Dangerous! Will clear all entities in the graph
            # In a real implementation, this should require confirmation
            # or be limited to a specific namespace/subgraph
            self.client.clear() 