"""
Adapter for using the centralized GraphManager with the long-term storage module.

This adapter connects the long-term storage module's interface with the centralized
graph infrastructure, promoting code reuse and consistency.
"""

import uuid
import datetime
from typing import Any, Dict, List, Optional, Union, Set, Tuple

from agent_suite.agents.storage.long_term.base import LongTermMemoryBase
from agent_suite.graph import GraphManager


class GraphitiKnowledgeGraphMemoryAdapter(LongTermMemoryBase):
    """
    Knowledge graph memory implementation using the centralized GraphManager.
    
    This adapter allows the long-term storage module to use the shared graph
    infrastructure while maintaining its existing interface.
    """
    
    def __init__(self, 
                graph_name: str = "agent_memory", 
                persist: bool = True,
                persistence_path: Optional[str] = None):
        """
        Initialize the knowledge graph memory adapter.
        
        Args:
            graph_name: Name of the graph for this agent's memory
            persist: Whether to persist the graph to disk
            persistence_path: Optional path to persistence directory
        """
        # Initialize the centralized graph manager
        self.graph_manager = GraphManager(
            graph_name=graph_name,
            persist=persist,
            persistence_path=persistence_path
        )
        
        # Vector memory is None for this implementation
        self.vector_memory = None
    
    def store_fact(self, subject: str, predicate: str, object_value: Any, 
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a single fact in the knowledge graph.
        
        Args:
            subject: The subject entity of the fact
            predicate: The relationship type
            object_value: The object value or entity
            metadata: Optional metadata about this fact (confidence, source, etc.)
            
        Returns:
            fact_id: A unique identifier for the stored fact
        """
        # Generate a unique ID for the fact
        fact_id = str(uuid.uuid4())
        
        # Process metadata
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add fact ID to metadata
        metadata["fact_id"] = fact_id
        
        # Handle primitive data types vs entity references
        if isinstance(object_value, (str, int, float, bool)):
            # This is a property relationship (entity -> value)
            # Ensure subject entity exists
            if not self.graph_manager.get_entity(subject):
                self.graph_manager.add_entity(entity_id=subject)
                
            # Update entity property 
            self.graph_manager.update_entity(
                entity_id=subject,
                properties={
                    predicate: object_value,
                    f"{predicate}__metadata": metadata
                }
            )
        else:
            # This is an entity-to-entity relationship
            # Ensure both entities exist
            if not self.graph_manager.get_entity(subject):
                self.graph_manager.add_entity(entity_id=subject)
                
            if not self.graph_manager.get_entity(object_value):
                self.graph_manager.add_entity(entity_id=object_value)
                
            # Create relationship
            self.graph_manager.add_relation(
                source_id=subject,
                relation_type=predicate,
                target_id=object_value,
                relation_id=fact_id,
                properties=metadata
            )
        
        return fact_id
    
    def store_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store unstructured text.
        
        Args:
            text: The text content to store
            metadata: Optional metadata about this text
            
        Returns:
            text_id: A unique identifier for the stored text
        """
        # Process metadata
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.datetime.now().isoformat()
            
        # Add text using graph manager
        return self.graph_manager.add_text(text=text, properties=metadata)
    
    def query_facts(self, query: Union[str, Dict[str, Any]], 
                   limit: int = 10) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for facts.
        
        Args:
            query: Either a natural language query or a structured query dict
            limit: Maximum number of results to return
            
        Returns:
            List of matching facts with their metadata
        """
        results = []
        
        if isinstance(query, str):
            # Natural language query - use keyword search
            search_results = self.graph_manager.keyword_search(query=query, limit=limit)
            
            # Convert to uniform fact format
            for result in search_results:
                entity_id = result["id"]
                entity_type = result.get("entity_type", result.get("type", "unknown"))
                confidence = result.get("score", 0.8)  # Default confidence if score not present
                
                if entity_type == "text":
                    # Add text as a fact
                    results.append({
                        "subject": entity_id,
                        "predicate": "content",
                        "object": result.get("content", result.get("properties", {}).get("content", "")),
                        "confidence": confidence,
                        "source": "text_entity",
                        "metadata": result.get("properties", {})
                    })
                elif result.get("match_type") == "relation":
                    # Handle relation match
                    results.append({
                        "subject": result.get("source_id", ""),
                        "predicate": result.get("relation_type", ""),
                        "object": result.get("target_id", ""),
                        "confidence": confidence,
                        "source": "relation",
                        "metadata": result.get("properties", {})
                    })
                else:
                    # Add entity properties as facts
                    for prop, value in result.get("properties", {}).items():
                        # Skip metadata fields and special properties
                        if prop.endswith("__metadata") or prop in ["type", "created_at", "updated_at"]:
                            continue
                            
                        results.append({
                            "subject": entity_id,
                            "predicate": prop,
                            "object": value,
                            "confidence": confidence,
                            "source": "entity_property",
                            "metadata": result.get("properties", {}).get(f"{prop}__metadata", {})
                        })
                        
                    # Also check for relations involving this entity
                    relations = self.graph_manager.query_relations(
                        source_id=entity_id,
                        limit=5
                    )
                    
                    for relation in relations:
                        results.append({
                            "subject": relation["source_id"],
                            "predicate": relation["relation_type"],
                            "object": relation["target_id"],
                            "confidence": 0.8,  # Fixed confidence for explicit relations
                            "source": "relation",
                            "metadata": relation["properties"]
                        })
        else:
            # Structured query
            # Handle specific query formats
            if "subject" in query:
                # Query by subject
                subject = query["subject"]
                predicate = query.get("predicate")
                
                # Get entity properties
                entity = self.graph_manager.get_entity(subject)
                if entity:
                    properties = entity["properties"]
                    
                    # Filter by predicate if specified
                    if predicate:
                        if predicate in properties:
                            results.append({
                                "subject": subject,
                                "predicate": predicate,
                                "object": properties[predicate],
                                "confidence": 1.0,
                                "source": "direct_property",
                                "metadata": properties.get(f"{predicate}__metadata", {})
                            })
                    else:
                        # Include all properties
                        for prop, value in properties.items():
                            # Skip metadata fields and special properties
                            if prop.endswith("__metadata") or prop in ["type", "created_at", "updated_at"]:
                                continue
                                
                            results.append({
                                "subject": subject,
                                "predicate": prop,
                                "object": value,
                                "confidence": 1.0,
                                "source": "direct_property",
                                "metadata": properties.get(f"{prop}__metadata", {})
                            })
                    
                    # Get relations
                    relations = self.graph_manager.query_relations(
                        source_id=subject,
                        relation_type=predicate
                    )
                    
                    for relation in relations:
                        results.append({
                            "subject": relation["source_id"],
                            "predicate": relation["relation_type"],
                            "object": relation["target_id"],
                            "confidence": 1.0,
                            "source": "direct_relation",
                            "metadata": relation["properties"]
                        })
            elif "predicate" in query:
                # Query by relation type
                relations = self.graph_manager.query_relations(
                    relation_type=query["predicate"],
                    limit=limit
                )
                
                for relation in relations:
                    results.append({
                        "subject": relation["source_id"],
                        "predicate": relation["relation_type"],
                        "object": relation["target_id"],
                        "confidence": 1.0,
                        "source": "relation_query",
                        "metadata": relation["properties"]
                    })
        
        # Apply limit
        return results[:limit]
    
    def query_text(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query for text entries.
        
        Args:
            query: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching text entries
        """
        # Get text entities using keyword search
        results = self.graph_manager.keyword_search(query, limit=limit*2)  # Get more results to filter down
        
        # Filter to keep only text entities
        text_results = []
        for result in results:
            entity_type = result.get("entity_type", result.get("type", "unknown"))
            if entity_type == "text":
                content = ""
                # Try to extract content from various possible locations
                if "content" in result:
                    content = result["content"]
                elif "properties" in result and "content" in result["properties"]:
                    content = result["properties"]["content"]
                
                if content:  # Only add if we found content
                    text_results.append({
                        "id": result["id"],
                        "content": content,
                        "score": result.get("score", 0.8),  # Default confidence
                        "metadata": {k: v for k, v in result.get("properties", {}).items() 
                                  if k != "content"}
                    })
                
        # Also try to query text entities directly
        if len(text_results) < limit:
            text_entities = self.graph_manager.query_entities(entity_type="text", limit=limit)
            for entity in text_entities:
                if "content" in entity["properties"]:
                    # Check if the content contains the query term
                    content = entity["properties"]["content"]
                    if query.lower() in content.lower():
                        # Check if this entity is already in results
                        if not any(r["id"] == entity["id"] for r in text_results):
                            text_results.append({
                                "id": entity["id"],
                                "content": content,
                                "score": 0.7,  # Slightly lower default confidence for direct queries
                                "metadata": {k: v for k, v in entity["properties"].items() 
                                          if k != "content"}
                            })
        
        # Apply limit
        return text_results[:limit]
    
    def get_entities(self, entity_type: Optional[str] = None) -> List[str]:
        """
        Get all entity IDs, optionally filtered by type.
        
        Args:
            entity_type: Optional type to filter by
            
        Returns:
            List of entity IDs
        """
        entities = self.graph_manager.query_entities(entity_type=entity_type)
        return [entity["id"] for entity in entities]
    
    def get_entity_facts(self, entity: str) -> List[Dict[str, Any]]:
        """
        Get all facts related to an entity.
        
        Args:
            entity: Entity ID to get facts for
            
        Returns:
            List of facts
        """
        facts = []
        
        # Get entity properties
        entity_data = self.graph_manager.get_entity(entity)
        if entity_data:
            properties = entity_data["properties"]
            
            # Add properties as facts
            for prop, value in properties.items():
                # Skip metadata fields and special properties
                if prop.endswith("__metadata") or prop in ["type", "created_at", "updated_at"]:
                    continue
                    
                facts.append({
                    "subject": entity,
                    "predicate": prop,
                    "object": value,
                    "confidence": 1.0,
                    "source": "property",
                    "metadata": properties.get(f"{prop}__metadata", {})
                })
        
        # Get outgoing relations (entity is subject)
        out_relations = self.graph_manager.query_relations(source_id=entity)
        for relation in out_relations:
            facts.append({
                "subject": entity,
                "predicate": relation["relation_type"],
                "object": relation["target_id"],
                "confidence": 1.0,
                "source": "outgoing_relation",
                "metadata": relation["properties"]
            })
            
        # Get incoming relations (entity is object)
        in_relations = self.graph_manager.query_relations(target_id=entity)
        for relation in in_relations:
            facts.append({
                "subject": relation["source_id"],
                "predicate": relation["relation_type"],
                "object": entity,
                "confidence": 1.0,
                "source": "incoming_relation",
                "metadata": relation["properties"]
            })
            
        return facts
    
    def delete_fact(self, fact_id: str) -> bool:
        """
        Delete a fact from memory.
        
        Args:
            fact_id: ID of the fact to delete
            
        Returns:
            True if deleted, False otherwise
        """
        # Try as relation first
        if self.graph_manager.delete_relation(fact_id):
            return True
            
        # Try to find and delete as property
        # This is more complex as we need to search for the fact ID in metadata
        entities = self.graph_manager.query_entities()
        for entity in entities:
            entity_id = entity["id"]
            properties = entity["properties"]
            
            property_to_delete = None
            for prop, value in properties.items():
                meta_prop = f"{prop}__metadata"
                if meta_prop in properties:
                    metadata = properties[meta_prop]
                    if isinstance(metadata, dict) and metadata.get("fact_id") == fact_id:
                        property_to_delete = prop
                        break
                        
            if property_to_delete:
                # Update entity to remove the property
                update_props = {}
                for prop in [property_to_delete, f"{property_to_delete}__metadata"]:
                    update_props[prop] = None
                    
                self.graph_manager.update_entity(entity_id, update_props)
                return True
                
        return False
    
    def delete_entity(self, entity: str) -> int:
        """
        Delete an entity and all its connected facts.
        
        Args:
            entity: Entity ID to delete
            
        Returns:
            Number of deleted facts
        """
        # Get all facts first
        facts = self.get_entity_facts(entity)
        count = len(facts)
        
        # Delete the entity
        self.graph_manager.delete_entity(entity)
        
        return count
    
    def delete_text(self, text_id: str) -> bool:
        """
        Delete a text entry.
        
        Args:
            text_id: ID of the text to delete
            
        Returns:
            True if deleted, False otherwise
        """
        return self.graph_manager.delete_entity(text_id)
    
    def clear(self) -> None:
        """
        Clear all memory.
        
        This will delete all entities and relations.
        """
        # Get all entities
        entities = self.graph_manager.query_entities(limit=1000000)
        
        # Delete them all
        for entity in entities:
            self.graph_manager.delete_entity(entity["id"])
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory.
        
        Returns:
            Dictionary with statistics
        """
        return self.graph_manager.get_stats() 