"""Memory-based graph storage adaptor for the RAG system.

This module provides an in-memory implementation of the graph storage adaptor
interface, suitable for testing and development.
"""

import uuid
from typing import Dict, Any, List, Optional, Union, Set
from collections import defaultdict

from agents.rag.storage.graph.base import GraphStorageAdaptor
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph


class MemoryGraphStorageAdaptor(GraphStorageAdaptor):
    """In-memory implementation of graph storage adaptor."""
    
    def __init__(self):
        """Initialize the memory graph storage."""
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.knowledge_graphs: Dict[str, KnowledgeGraph] = {}
        
        # Index for efficient querying
        self.entity_by_type: Dict[str, List[str]] = defaultdict(list)
        self.relationships_by_source: Dict[str, List[str]] = defaultdict(list)
        self.relationships_by_target: Dict[str, List[str]] = defaultdict(list)
        self.relationships_by_type: Dict[str, List[str]] = defaultdict(list)
    
    async def store_entity(self, entity: Entity) -> str:
        """
        Store an entity in the graph.
        
        Args:
            entity: The entity to store
            
        Returns:
            The ID of the stored entity
        """
        entity_id = entity.id or str(uuid.uuid4())
        entity.id = entity_id
        
        self.entities[entity_id] = entity
        self.entity_by_type[entity.type].append(entity_id)
        
        return entity_id
    
    async def store_relationship(self, relationship: Relationship) -> str:
        """
        Store a relationship in the graph.
        
        Args:
            relationship: The relationship to store
            
        Returns:
            The ID of the stored relationship
        """
        rel_id = relationship.id or str(uuid.uuid4())
        relationship.id = rel_id
        
        self.relationships[rel_id] = relationship
        self.relationships_by_source[relationship.source_id].append(rel_id)
        self.relationships_by_target[relationship.target_id].append(rel_id)
        self.relationships_by_type[relationship.type].append(rel_id)
        
        return rel_id
    
    async def store_knowledge_graph(self, graph: KnowledgeGraph) -> str:
        """
        Store a complete knowledge graph.
        
        Args:
            graph: The knowledge graph to store
            
        Returns:
            The ID of the stored graph
        """
        graph_id = graph.id or str(uuid.uuid4())
        graph.id = graph_id
        
        # Store all entities and relationships from the graph
        for entity in graph.entities:
            await self.store_entity(entity)
        
        for relationship in graph.relationships:
            await self.store_relationship(relationship)
        
        self.knowledge_graphs[graph_id] = graph
        return graph_id
    
    async def query_graph(self, query_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a graph query.
        
        Args:
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query results
        """
        if query_type == "get_entities":
            return await self._query_entities(parameters)
        elif query_type == "get_relationships":
            return await self._query_relationships(parameters)
        elif query_type == "get_neighbors":
            return await self._query_neighbors(parameters)
        elif query_type == "find_paths":
            return await self._query_paths(parameters)
        else:
            return []
    
    async def find_paths(self, start_entity_id: str, end_entity_id: str, 
                       max_depth: int = 3) -> List[List[Union[Entity, Relationship]]]:
        """
        Find paths between entities.
        
        Args:
            start_entity_id: Start entity ID
            end_entity_id: End entity ID
            max_depth: Maximum path depth
            
        Returns:
            List of paths (each a list of alternating entities and relationships)
        """
        if start_entity_id not in self.entities or end_entity_id not in self.entities:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current_id: str, target_id: str, current_path: List[Union[Entity, Relationship]], depth: int):
            if depth > max_depth:
                return
            
            if current_id == target_id and len(current_path) > 1:
                paths.append(current_path.copy())
                return
            
            if current_id in visited:
                return
            
            visited.add(current_id)
            
            # Find outgoing relationships
            for rel_id in self.relationships_by_source.get(current_id, []):
                relationship = self.relationships[rel_id]
                target_entity_id = relationship.target_id
                
                if target_entity_id in self.entities:
                    target_entity = self.entities[target_entity_id]
                    new_path = current_path + [relationship, target_entity]
                    dfs(target_entity_id, target_id, new_path, depth + 1)
            
            visited.remove(current_id)
        
        start_entity = self.entities[start_entity_id]
        dfs(start_entity_id, end_entity_id, [start_entity], 0)
        
        return paths
    
    async def _query_entities(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query entities based on parameters."""
        entity_type = parameters.get("type")
        properties = parameters.get("properties", {})
        limit = parameters.get("limit", 100)
        
        results = []
        entity_ids = self.entity_by_type.get(entity_type, list(self.entities.keys())) if entity_type else list(self.entities.keys())
        
        for entity_id in entity_ids[:limit]:
            entity = self.entities[entity_id]
            
            # Check if entity matches property filters
            if self._matches_properties(entity.properties, properties):
                results.append({
                    "id": entity.id,
                    "type": entity.type,
                    "properties": entity.properties
                })
        
        return results
    
    async def _query_relationships(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query relationships based on parameters."""
        source_id = parameters.get("source_id")
        target_id = parameters.get("target_id")
        rel_type = parameters.get("type")
        properties = parameters.get("properties", {})
        limit = parameters.get("limit", 100)
        
        results = []
        rel_ids = []
        
        if source_id:
            rel_ids = self.relationships_by_source.get(source_id, [])
        elif target_id:
            rel_ids = self.relationships_by_target.get(target_id, [])
        elif rel_type:
            rel_ids = self.relationships_by_type.get(rel_type, [])
        else:
            rel_ids = list(self.relationships.keys())
        
        for rel_id in rel_ids[:limit]:
            relationship = self.relationships[rel_id]
            
            # Apply filters
            if source_id and relationship.source_id != source_id:
                continue
            if target_id and relationship.target_id != target_id:
                continue
            if rel_type and relationship.type != rel_type:
                continue
            if not self._matches_properties(relationship.properties, properties):
                continue
            
            results.append({
                "id": relationship.id,
                "type": relationship.type,
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "properties": relationship.properties
            })
        
        return results
    
    async def _query_neighbors(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query neighbors of an entity."""
        entity_id = parameters.get("entity_id")
        direction = parameters.get("direction", "outgoing")  # outgoing, incoming, both
        rel_types = parameters.get("relationship_types", [])
        limit = parameters.get("limit", 100)
        
        if not entity_id or entity_id not in self.entities:
            return []
        
        neighbors = []
        
        if direction in ["outgoing", "both"]:
            for rel_id in self.relationships_by_source.get(entity_id, []):
                relationship = self.relationships[rel_id]
                if not rel_types or relationship.type in rel_types:
                    target_entity = self.entities.get(relationship.target_id)
                    if target_entity:
                        neighbors.append({
                            "entity": {
                                "id": target_entity.id,
                                "type": target_entity.type,
                                "properties": target_entity.properties
                            },
                            "relationship": {
                                "id": relationship.id,
                                "type": relationship.type,
                                "properties": relationship.properties
                            },
                            "direction": "outgoing"
                        })
        
        if direction in ["incoming", "both"]:
            for rel_id in self.relationships_by_target.get(entity_id, []):
                relationship = self.relationships[rel_id]
                if not rel_types or relationship.type in rel_types:
                    source_entity = self.entities.get(relationship.source_id)
                    if source_entity:
                        neighbors.append({
                            "entity": {
                                "id": source_entity.id,
                                "type": source_entity.type,
                                "properties": source_entity.properties
                            },
                            "relationship": {
                                "id": relationship.id,
                                "type": relationship.type,
                                "properties": relationship.properties
                            },
                            "direction": "incoming"
                        })
        
        return neighbors[:limit]
    
    async def _query_paths(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query paths between entities."""
        start_id = parameters.get("start_entity_id")
        end_id = parameters.get("end_entity_id")
        max_depth = parameters.get("max_depth", 3)
        
        if not start_id or not end_id:
            return []
        
        paths = await self.find_paths(start_id, end_id, max_depth)
        
        result_paths = []
        for path in paths:
            path_data = []
            for item in path:
                if isinstance(item, Entity):
                    path_data.append({
                        "type": "entity",
                        "id": item.id,
                        "entity_type": item.type,
                        "properties": item.properties
                    })
                elif isinstance(item, Relationship):
                    path_data.append({
                        "type": "relationship",
                        "id": item.id,
                        "relationship_type": item.type,
                        "source_id": item.source_id,
                        "target_id": item.target_id,
                        "properties": item.properties
                    })
            result_paths.append({"path": path_data})
        
        return result_paths
    
    def _matches_properties(self, entity_props: Dict[str, Any], filter_props: Dict[str, Any]) -> bool:
        """Check if entity properties match filter properties."""
        for key, value in filter_props.items():
            if key not in entity_props or entity_props[key] != value:
                return False
        return True
    
    async def store(self, data: Any, metadata: Dict[str, Any] = None) -> str:
        """Store data (base storage interface)."""
        if isinstance(data, Entity):
            return await self.store_entity(data)
        elif isinstance(data, Relationship):
            return await self.store_relationship(data)
        elif isinstance(data, KnowledgeGraph):
            return await self.store_knowledge_graph(data)
        else:
            # Store as a generic entity
            entity = Entity(
                type="Generic",
                properties={"data": data},
                id=str(uuid.uuid4())
            )
            return await self.store_entity(entity)
    
    async def retrieve(self, key: str, **kwargs) -> Optional[Any]:
        """Retrieve data by key."""
        if key in self.entities:
            return self.entities[key]
        elif key in self.relationships:
            return self.relationships[key]
        elif key in self.knowledge_graphs:
            return self.knowledge_graphs[key]
        return None
    
    async def delete(self, key: str, **kwargs) -> bool:
        """Delete data by key."""
        deleted = False
        
        if key in self.entities:
            entity = self.entities[key]
            del self.entities[key]
            self.entity_by_type[entity.type].remove(key)
            deleted = True
        
        if key in self.relationships:
            relationship = self.relationships[key]
            del self.relationships[key]
            self.relationships_by_source[relationship.source_id].remove(key)
            self.relationships_by_target[relationship.target_id].remove(key)
            self.relationships_by_type[relationship.type].remove(key)
            deleted = True
        
        if key in self.knowledge_graphs:
            del self.knowledge_graphs[key]
            deleted = True
        
        return deleted
    
    async def update(self, key: str, data: Any, **kwargs) -> bool:
        """Update data by key."""
        if key in self.entities and isinstance(data, Entity):
            old_entity = self.entities[key]
            data.id = key
            self.entities[key] = data
            
            # Update indices if type changed
            if old_entity.type != data.type:
                self.entity_by_type[old_entity.type].remove(key)
                self.entity_by_type[data.type].append(key)
            
            return True
        
        elif key in self.relationships and isinstance(data, Relationship):
            old_rel = self.relationships[key]
            data.id = key
            self.relationships[key] = data
            
            # Update indices if relationships changed
            if old_rel.source_id != data.source_id:
                self.relationships_by_source[old_rel.source_id].remove(key)
                self.relationships_by_source[data.source_id].append(key)
            
            if old_rel.target_id != data.target_id:
                self.relationships_by_target[old_rel.target_id].remove(key)
                self.relationships_by_target[data.target_id].append(key)
            
            if old_rel.type != data.type:
                self.relationships_by_type[old_rel.type].remove(key)
                self.relationships_by_type[data.type].append(key)
            
            return True
        
        return False
    
    async def list(self, pattern: str = "*", **kwargs) -> List[str]:
        """List all keys matching pattern."""
        all_keys = list(self.entities.keys()) + list(self.relationships.keys()) + list(self.knowledge_graphs.keys())
        
        if pattern == "*":
            return all_keys
        
        # Simple pattern matching (could be enhanced)
        import fnmatch
        return [key for key in all_keys if fnmatch.fnmatch(key, pattern)]
    
    async def clear(self, **kwargs) -> bool:
        """Clear all data."""
        self.entities.clear()
        self.relationships.clear()
        self.knowledge_graphs.clear()
        self.entity_by_type.clear()
        self.relationships_by_source.clear()
        self.relationships_by_target.clear()
        self.relationships_by_type.clear()
        return True
    
    async def search(self, query: Any, **kwargs) -> List[Any]:
        """
        Search for data matching the query.
        
        Args:
            query: The search query (can be a string for text search or dict for structured query)
            **kwargs: Additional search parameters
            
        Returns:
            List of matching data items
        """
        if isinstance(query, str):
            # Text search across entity and relationship properties
            results = []
            query_lower = query.lower()
            
            # Search entities
            for entity in self.entities.values():
                if self._text_matches(entity, query_lower):
                    results.append(entity)
            
            # Search relationships
            for relationship in self.relationships.values():
                if self._text_matches(relationship, query_lower):
                    results.append(relationship)
            
            return results
        
        elif isinstance(query, dict):
            # Structured query using query_graph
            query_type = query.get("type", "get_entities")
            parameters = query.get("parameters", {})
            return await self.query_graph(query_type, parameters)
        
        else:
            return []
    
    def _text_matches(self, item: Union[Entity, Relationship], query: str) -> bool:
        """Check if an item matches a text query."""
        # Check in properties
        for key, value in item.properties.items():
            if isinstance(value, str) and query in value.lower():
                return True
            elif isinstance(value, (int, float)) and query in str(value):
                return True
        
        # Check in type and id
        if query in item.type.lower() or query in item.id.lower():
            return True
        
        return False
    
    async def close(self) -> None:
        """Clean up resources."""
        # For memory storage, just clear the data
        await self.clear() 