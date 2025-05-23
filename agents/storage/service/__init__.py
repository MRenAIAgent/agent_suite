"""Storage service module for managing storage backends.

This module provides a unified service interface for interacting
with different storage backends through a single API.
"""

from agents.storage.service.storage_service import StorageService

__all__ = ["StorageService"] 