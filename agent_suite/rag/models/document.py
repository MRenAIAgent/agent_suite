"""Document models for the RAG system.

This module provides the document models for storing and retrieving unstructured
text data in the RAG system.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid


@dataclass
class DocumentChunk:
    """A chunk of a document for more granular storage and retrieval."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: Optional[str] = None
    index: int = 0


@dataclass
class Document:
    """Base class for documents in the RAG system."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Optional chunking information
    chunks: List[DocumentChunk] = field(default_factory=list) 