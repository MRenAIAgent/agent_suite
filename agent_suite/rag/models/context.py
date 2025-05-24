"""Context models for the RAG system.

This module provides the context models for storing and retrieving contextual
information in the RAG system.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import uuid
import time


@dataclass
class ContextItem:
    """Key-value pair with metadata for contextual information."""
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time) 