"""
Centralized Graph Database module for Agent Suite.

This module provides a shared interface for graph operations that can be used
by both the knowledge module and long-term storage features.
"""

from agent_suite.graph.graph_manager import GraphManager

__all__ = ["GraphManager"] 