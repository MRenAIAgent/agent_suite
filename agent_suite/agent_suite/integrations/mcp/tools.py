"""
MCP Tool Wrapper

This module provides adapters for MCP (Model Completion Protocol) tools to work with
the agent_suite enhanced tool system. It converts MCP tool formats to the
EnhancedTool format and provides registration capabilities.
"""
from typing import Dict, Any, List, Optional, ClassVar, Union
import logging
import json

from tools.base import EnhancedTool
from tools.tool_types import (
    ToolMetadata, 
    ToolCategory, 
    ToolCapability, 
    ToolDomain,
    ToolSource
)
from tools.registry import registry
from pydantic import Field

# Setup logging
logger = logging.getLogger(__name__)

# Common stop words to filter out when generating keywords
COMMON_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "will", "can", "are", 
    "not", "has", "have", "been", "was", "were", "they", "their", "them", "which",
    "what", "when", "where", "who", "why", "how", "your", "you", "may", "any",
    "all", "some", "such", "more", "other", "these", "those", "its", "his", "her",
    "our", "get", "got", "use", "used", "using"
}

# Define MCP tool categories mapping
MCP_CATEGORY_MAP = {
    "search": ToolCategory.SEARCH,
    "utility": ToolCategory.UTILITY,
    "data_analysis": ToolCategory.DATA_ANALYSIS,
    "content_generation": ToolCategory.CONTENT_GENERATION,
    "web": ToolCategory.WEB_BROWSING,
    "communication": ToolCategory.COMMUNICATION,
    "document": ToolCategory.DOCUMENT_PROCESSING,
    # Add more mappings as needed
    # Default to UTILITY for unknown categories
}

# Define MCP tool capabilities mapping
MCP_CAPABILITY_MAP = {
    "search": ToolCapability.SEARCH,
    "read": ToolCapability.READ,
    "write": ToolCapability.WRITE,
    "calculate": ToolCapability.COMPUTE,
    "fetch": ToolCapability.FETCH,
    "transform": ToolCapability.TRANSFORM,
    "generate": ToolCapability.GENERATE,
    "analyze": ToolCapability.ANALYZE,
    "store": ToolCapability.STORE,
    "communicate": ToolCapability.COMMUNICATE,
    # Add more mappings as needed
}

# Define MCP domain mapping
MCP_DOMAIN_MAP = {
    "general": ToolDomain.GENERAL,
    "finance": ToolDomain.FINANCE,
    "healthcare": ToolDomain.HEALTHCARE,
    "programming": ToolDomain.PROGRAMMING,
    "marketing": ToolDomain.MARKETING,
    "science": ToolDomain.SCIENCE,
    "education": ToolDomain.EDUCATION,
    "law": ToolDomain.LAW,
    "entertainment": ToolDomain.ENTERTAINMENT,
    "business": ToolDomain.BUSINESS,
    # Add more mappings as needed
    # Default to GENERAL for unknown domains
}


class MCPToolWrapper(EnhancedTool):
    """Wrapper for MCP tools to work with the agent_suite enhanced tool system."""
    
    # Store MCP tool information as regular Pydantic fields
    mcp_tool_name: str = Field(..., description="Name of the MCP tool")
    mcp_namespace: str = Field(..., description="Namespace of the MCP tool")
    mcp_tool_data: Dict[str, Any] = Field(..., description="MCP tool data")
    
    # Optional MCP tool instance
    mcp_tool_instance: Optional[Any] = Field(default=None, description="MCP tool instance")
    
    def __init__(
        self, 
        mcp_tool_name: str,
        mcp_namespace: str,
        mcp_tool_data: Dict[str, Any],
        mcp_tool_instance: Any = None
    ):
        """Initialize the MCP tool wrapper.
        
        Args:
            mcp_tool_name: The name of the MCP tool
            mcp_namespace: The namespace of the MCP tool
            mcp_tool_data: The tool data including parameters, description, etc.
            mcp_tool_instance: Optional actual MCP tool instance
        """
        # Create metadata from MCP tool data
        metadata = self._create_metadata_from_mcp(
            mcp_tool_name=mcp_tool_name,
            mcp_namespace=mcp_namespace,
            mcp_tool_data=mcp_tool_data
        )
        
        # Initialize the parent class with all fields
        super().__init__(
            metadata=metadata,
            mcp_tool_name=mcp_tool_name,
            mcp_namespace=mcp_namespace,
            mcp_tool_data=mcp_tool_data,
            mcp_tool_instance=mcp_tool_instance
        )
    
    def _create_metadata_from_mcp(
        self,
        mcp_tool_name: str,
        mcp_namespace: str,
        mcp_tool_data: Dict[str, Any]
    ) -> ToolMetadata:
        """Create ToolMetadata from MCP tool data.
        
        Returns:
            ToolMetadata instance with appropriate values
        """
        # Extract relevant information from MCP tool data
        description = mcp_tool_data.get("description", "")
        display_name = mcp_tool_data.get("display_name", mcp_tool_name)
        
        # Extract or determine categories, capabilities, and domains
        categories = self._extract_categories(mcp_tool_data, mcp_tool_name)
        capabilities = self._extract_capabilities(mcp_tool_data, mcp_tool_name)
        domains = self._extract_domains(mcp_tool_data, mcp_tool_name, mcp_namespace)
        
        # Extract examples if available
        examples = mcp_tool_data.get("examples", [])
        if isinstance(examples, str):
            examples = [examples]
        
        # Construct keywords from MCP tool data
        keywords = self._extract_keywords(mcp_tool_data, mcp_tool_name)
        
        # Create and return metadata
        return ToolMetadata(
            name=mcp_tool_name,
            display_name=display_name,
            description=description,
            categories=categories,
            capabilities=capabilities,
            domains=domains,
            source=ToolSource.MCP,
            keywords=keywords,
            examples=examples,
            applicable_roles=self._extract_applicable_roles(mcp_tool_data)
        )
    
    def _extract_categories(self, mcp_tool_data: Dict[str, Any], mcp_tool_name: str) -> List[ToolCategory]:
        """Extract tool categories from MCP tool data.
        
        Returns:
            List of ToolCategory values
        """
        # Start with a default category
        categories = [ToolCategory.UTILITY]
        
        # Try to extract from MCP data
        mcp_categories = mcp_tool_data.get("categories", [])
        if isinstance(mcp_categories, str):
            mcp_categories = [mcp_categories]
        
        # Map MCP categories to our categories
        if mcp_categories:
            categories = []
            for cat in mcp_categories:
                if cat.lower() in MCP_CATEGORY_MAP:
                    categories.append(MCP_CATEGORY_MAP[cat.lower()])
                else:
                    # Try to find by value
                    try:
                        categories.append(ToolCategory(cat))
                    except ValueError:
                        # Use default if not found
                        categories.append(ToolCategory.UTILITY)
        
        # Infer category from name/description if none found
        if not categories:
            if "search" in mcp_tool_name.lower() or "find" in mcp_tool_name.lower():
                categories.append(ToolCategory.SEARCH)
            elif "analyze" in mcp_tool_name.lower() or "data" in mcp_tool_name.lower():
                categories.append(ToolCategory.DATA_ANALYSIS)
        
        return categories
    
    def _extract_capabilities(self, mcp_tool_data: Dict[str, Any], mcp_tool_name: str) -> List[ToolCapability]:
        """Extract tool capabilities from MCP tool data.
        
        Returns:
            List of ToolCapability values
        """
        # Start with a default capability
        capabilities = [ToolCapability.READ]
        
        # Try to extract from MCP data
        mcp_capabilities = mcp_tool_data.get("capabilities", [])
        if isinstance(mcp_capabilities, str):
            mcp_capabilities = [mcp_capabilities]
        
        # Map MCP capabilities to our capabilities
        if mcp_capabilities:
            capabilities = []
            for cap in mcp_capabilities:
                if cap.lower() in MCP_CAPABILITY_MAP:
                    capabilities.append(MCP_CAPABILITY_MAP[cap.lower()])
                else:
                    # Try to find by value
                    try:
                        capabilities.append(ToolCapability(cap))
                    except ValueError:
                        # Use default if not found
                        capabilities.append(ToolCapability.READ)
        
        # Infer capability from name/description
        name_lower = mcp_tool_name.lower()
        desc_lower = mcp_tool_data.get("description", "").lower()
        
        if "search" in name_lower or "find" in name_lower:
            capabilities.append(ToolCapability.SEARCH)
        if "calculate" in name_lower or "compute" in name_lower:
            capabilities.append(ToolCapability.COMPUTE)
        if "create" in name_lower or "generate" in name_lower:
            capabilities.append(ToolCapability.GENERATE)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_capabilities = []
        for cap in capabilities:
            if cap not in seen:
                seen.add(cap)
                unique_capabilities.append(cap)
        
        return unique_capabilities
    
    def _extract_domains(self, mcp_tool_data: Dict[str, Any], mcp_tool_name: str, mcp_namespace: str) -> List[ToolDomain]:
        """Extract tool domains from MCP tool data.
        
        Returns:
            List of ToolDomain values
        """
        # Always include GENERAL domain
        domains = [ToolDomain.GENERAL]
        
        # Try to extract from MCP data
        mcp_domains = mcp_tool_data.get("domains", [])
        if isinstance(mcp_domains, str):
            mcp_domains = [mcp_domains]
        
        # Map MCP domains to our domains
        if mcp_domains:
            domains = []
            for dom in mcp_domains:
                if dom.lower() in MCP_DOMAIN_MAP:
                    domains.append(MCP_DOMAIN_MAP[dom.lower()])
                else:
                    # Try to find by value
                    try:
                        domains.append(ToolDomain(dom))
                    except ValueError:
                        # Use default if not found
                        domains.append(ToolDomain.GENERAL)
        
        # Infer domain from name/namespace
        name_lower = mcp_tool_name.lower()
        namespace_lower = mcp_namespace.lower()
        
        # Add finance domain for finance-related tools
        if any(x in name_lower or x in namespace_lower for x in ["finance", "stock", "market", "invest"]):
            domains.append(ToolDomain.FINANCE)
        
        # Add programming domain for code-related tools
        if any(x in name_lower or x in namespace_lower for x in ["code", "program", "git", "github"]):
            domains.append(ToolDomain.PROGRAMMING)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_domains = []
        for dom in domains:
            if dom not in seen:
                seen.add(dom)
                unique_domains.append(dom)
        
        return unique_domains
    
    def _extract_keywords(self, mcp_tool_data: Dict[str, Any], mcp_tool_name: str) -> List[str]:
        """Extract keywords from MCP tool data.
        
        Returns:
            List of keywords extracted or inferred from the MCP tool data
        """
        # Start with a default set of keywords
        keywords = ["mcp"]
        
        # Try to extract from MCP data
        mcp_keywords = mcp_tool_data.get("keywords", [])
        if isinstance(mcp_keywords, str):
            mcp_keywords = mcp_keywords.split(",")
        
        if mcp_keywords:
            keywords.extend([k.strip() for k in mcp_keywords])
        
        # Add name and namespace as keywords
        keywords.append(mcp_tool_name)
        if hasattr(self, 'mcp_namespace'):
            keywords.append(self.mcp_namespace)
        
        # Add additional keywords based on name and description
        name_parts = mcp_tool_name.replace("_", " ").replace("-", " ").split()
        keywords.extend(name_parts)
        
        # Add keywords based on tool description
        description = mcp_tool_data.get("description", "")
        if description:
            # Extract key terms from description
            desc_words = description.lower().split()
            important_words = [
                word for word in desc_words 
                if len(word) > 3 and word not in COMMON_STOP_WORDS
            ]
            keywords.extend(important_words[:5])  # Limit to 5 words from description
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword and keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def _extract_applicable_roles(self, mcp_tool_data: Dict[str, Any]) -> List[str]:
        """Extract applicable roles from MCP tool data.
        
        Returns:
            List of applicable role strings
        """
        # Start with empty roles
        roles = []
        
        # Try to extract from MCP data
        mcp_roles = mcp_tool_data.get("applicable_roles", [])
        if isinstance(mcp_roles, str):
            mcp_roles = mcp_roles.split(",")
        
        # Use MCP roles if available
        if mcp_roles:
            roles.extend([r.strip() for r in mcp_roles])
        
        # Add default roles based on categories and capabilities
        categories = self._extract_categories(mcp_tool_data, "")
        capabilities = self._extract_capabilities(mcp_tool_data, "")
        
        if ToolCategory.SEARCH in categories:
            roles.append("researcher")
        if ToolCategory.DATA_ANALYSIS in categories:
            roles.append("analyst")
        if ToolCategory.CONTENT_GENERATION in categories:
            roles.append("writer")
        if ToolCapability.COMPUTE in capabilities:
            roles.append("calculator")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_roles = []
        for role in roles:
            role_lower = role.lower()
            if role_lower not in seen and role_lower:
                seen.add(role_lower)
                unique_roles.append(role)
        
        return unique_roles
    
    async def arun(self, **kwargs) -> str:
        """Run the MCP tool asynchronously.
        
        Args:
            **kwargs: Parameters for the MCP tool
            
        Returns:
            Result from the MCP tool
        """
        if self.mcp_tool_instance is None:
            return f"Error: MCP tool '{self.mcp_tool_name}' is not initialized"
        
        try:
            # Check if the MCP tool has an async run method
            if hasattr(self.mcp_tool_instance, "arun"):
                return await self.mcp_tool_instance.arun(**kwargs)
            elif hasattr(self.mcp_tool_instance, "run"):
                # Fall back to synchronous run method
                return self.mcp_tool_instance.run(**kwargs)
            else:
                # Try calling directly
                return self.mcp_tool_instance(**kwargs)
        except Exception as e:
            logger.error(f"Error running MCP tool '{self.mcp_tool_name}': {e}")
            return f"Error running tool: {str(e)}"


def register_mcp_tool(
    mcp_tool_name: str,
    mcp_namespace: str,
    mcp_tool_data: Dict[str, Any],
    mcp_tool_instance: Any,
    registry_namespace: Optional[str] = None
) -> MCPToolWrapper:
    """Register an MCP tool with the tool registry.
    
    Args:
        mcp_tool_name: Name of the MCP tool
        mcp_namespace: Namespace of the MCP tool
        mcp_tool_data: Tool data including parameters, description, etc.
        mcp_tool_instance: The actual MCP tool instance
        registry_namespace: Optional namespace to register the tool under
                           (defaults to mcp_namespace)
                           
    Returns:
        The created MCPToolWrapper instance
    """
    # Create the tool wrapper
    wrapper = MCPToolWrapper(
        mcp_tool_name=mcp_tool_name,
        mcp_namespace=mcp_namespace,
        mcp_tool_data=mcp_tool_data,
        mcp_tool_instance=mcp_tool_instance
    )
    
    # Use the provided namespace or the MCP namespace
    if registry_namespace is None:
        registry_namespace = mcp_namespace
    
    # Register the tool with the registry
    registry.register_tools_with_namespace(registry_namespace, [wrapper])
    
    logger.info(f"Registered MCP tool '{mcp_tool_name}' from namespace '{mcp_namespace}' to registry namespace '{registry_namespace}'")
    
    return wrapper


def register_mcp_tools(
    mcp_tools: Dict[str, Dict[str, Any]],
    registry_namespace_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, MCPToolWrapper]:
    """Register multiple MCP tools with the tool registry.
    
    Args:
        mcp_tools: Dictionary of MCP tools, with format:
                  {
                    "namespace.tool_name": {
                      "instance": tool_instance,
                      "data": {
                        "description": "Tool description",
                        "parameters": {...},
                        ...
                      },
                      "namespace": "tool_namespace"
                    },
                    ...
                  }
        registry_namespace_mapping: Optional mapping from MCP namespaces to
                                   registry namespaces.
                                   
    Returns:
        Dictionary of tool_name to MCPToolWrapper instances
    """
    wrappers = {}
    
    for full_name, tool_info in mcp_tools.items():
        # Extract namespace and name
        if "." in full_name:
            namespace, name = full_name.split(".", 1)
        else:
            namespace = tool_info.get("namespace", "mcp")
            name = full_name
        
        # Get tool data and instance
        tool_data = tool_info.get("data", {})
        tool_instance = tool_info.get("instance")
        
        # Determine registry namespace
        if registry_namespace_mapping and namespace in registry_namespace_mapping:
            registry_ns = registry_namespace_mapping[namespace]
        else:
            registry_ns = namespace
        
        # Register the tool
        wrapper = register_mcp_tool(
            mcp_tool_name=name,
            mcp_namespace=namespace,
            mcp_tool_data=tool_data,
            mcp_tool_instance=tool_instance,
            registry_namespace=registry_ns
        )
        
        wrappers[full_name] = wrapper
    
    return wrappers 