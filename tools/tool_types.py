"""
Core type definitions for the enhanced tool system.

This module defines the core types used throughout the enhanced tool system
including categories, domains, capabilities, metadata, and statistics.
"""
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Categories of tools for organization and selection."""
    SEARCH = "search"
    CODING = "coding"
    DATA_ANALYSIS = "data-analysis"
    CONTENT_GENERATION = "content-generation"
    WEB_BROWSING = "web-browsing"
    DOCUMENT_PROCESSING = "document-processing"
    DATABASE = "database"
    FILE_SYSTEM = "file-system"
    COMMUNICATION = "communication"
    MEMORY = "memory"
    UTILITY = "utility"
    CUSTOM = "custom"
    
    
class ToolDomain(str, Enum):
    """Domain areas that tools may operate in."""
    GENERAL = "general"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    PROGRAMMING = "programming"
    MARKETING = "marketing"
    SCIENCE = "science"
    EDUCATION = "education"
    LAW = "law"
    ENTERTAINMENT = "entertainment"
    BUSINESS = "business"
    CUSTOM = "custom"


class ToolCapability(str, Enum):
    """Capabilities that tools may provide."""
    READ = "read"
    WRITE = "write"
    COMPUTE = "compute"
    FETCH = "fetch"
    TRANSFORM = "transform"
    GENERATE = "generate"
    ANALYZE = "analyze"
    SEARCH = "search"
    STORE = "store"
    COMMUNICATE = "communicate"
    CUSTOM = "custom"


class ToolSource(str, Enum):
    """Sources of tools for tracking and management."""
    CUSTOM = "custom"
    LANGCHAIN = "langchain"
    MCP = "mcp"
    SYSTEM = "system"
    PLUGIN = "plugin"


class ToolMetadata(BaseModel):
    """Enhanced tool metadata for better tool selection and organization."""
    
    # Basic identification
    name: str = Field(
        default="",
        description="Unique name of the tool"
    )
    display_name: str = Field(
        default="",
        description="Human-readable display name for the tool"
    )
    description: str = Field(
        default="",
        description="Short description of what the tool does"
    )
    
    # Basic categorization
    categories: List[Union[ToolCategory, str]] = Field(
        default_factory=list,
        description="Categories this tool belongs to"
    )
    domains: List[Union[ToolDomain, str]] = Field(
        default_factory=list,
        description="Domain areas this tool is relevant for"
    )
    capabilities: List[Union[ToolCapability, str]] = Field(
        default_factory=list,
        description="Capabilities this tool provides"
    )
    
    # Role & task matching
    applicable_roles: List[str] = Field(
        default_factory=list,
        description="Roles this tool is especially useful for"
    )
    task_patterns: List[str] = Field(
        default_factory=list,
        description="Task patterns where this tool is likely useful"
    )
    
    # Usage guidance
    examples: List[str] = Field(
        default_factory=list,
        description="Example usages of this tool"
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Required resources or setup for this tool"
    )
    
    # Performance characteristics
    avg_execution_time_ms: Optional[int] = Field(
        default=None,
        description="Average execution time in milliseconds"
    )
    is_cacheable: bool = Field(
        default=False,
        description="Whether results from this tool can be cached"
    )
    is_streaming: bool = Field(
        default=False,
        description="Whether tool supports streaming results"
    )
    
    # Semantic matching
    semantic_description: Optional[str] = Field(
        default=None,
        description="Longer description for semantic matching"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Additional keywords for matching"
    )


class Role(BaseModel):
    """Role definition for role-based tool selection."""
    name: str = Field(..., description="Name of the role")
    description: str = Field(..., description="Description of the role")
    responsibilities: List[str] = Field(
        default_factory=list,
        description="Key responsibilities of this role"
    )
    required_capabilities: List[Union[ToolCapability, str]] = Field(
        default_factory=list,
        description="Capabilities required by this role"
    )
    domains: List[Union[ToolDomain, str]] = Field(
        default_factory=list,
        description="Domains this role operates in"
    )
    preferred_tools: Optional[List[str]] = Field(
        default=None,
        description="Specific tools preferred for this role"
    )


class Task(BaseModel):
    """Task definition for task-based tool selection."""
    description: str = Field(..., description="Description of the task")
    goal: str = Field(..., description="Goal of the task")
    type: str = Field(..., description="Type of task (e.g., research, analysis)")
    domain: Union[ToolDomain, str] = Field(..., description="Domain of the task")
    required_capabilities: List[Union[ToolCapability, str]] = Field(
        default_factory=list,
        description="Capabilities required for this task"
    )
    expected_outputs: List[str] = Field(
        default_factory=list,
        description="Expected outputs from this task"
    )


class ToolUsageStats(BaseModel):
    """Track tool usage statistics for better tool selection."""
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[str] = None
    avg_execution_time: Optional[float] = None
    
    def update(self, success: bool, execution_time: Optional[float] = None) -> None:
        """Update statistics with new usage data."""
        from datetime import datetime
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
            
        self.last_used = datetime.now().isoformat()
        
        if execution_time is not None:
            if self.avg_execution_time is None:
                self.avg_execution_time = execution_time
            else:
                # Rolling average
                total = (self.avg_execution_time * (self.success_count + self.failure_count - 1) 
                         + execution_time)
                self.avg_execution_time = total / (self.success_count + self.failure_count) 