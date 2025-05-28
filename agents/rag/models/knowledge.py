"""Knowledge models for the RAG system.

This module provides the knowledge models for storing and retrieving structured
knowledge in the RAG system, including entities and relationships.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class Entity:
    """Entity in a knowledge graph."""
    type: str = "entity"
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Relationship:
    """Relationship between entities in a knowledge graph."""
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class KnowledgeGraph:
    """A collection of entities and relationships."""
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4())) 