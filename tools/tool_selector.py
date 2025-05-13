"""
Tool Selector

This module provides tool selection capabilities to intelligently choose
the most appropriate tools based on user queries, context, and tool metadata.
"""
import re
from typing import List, Dict, Any, Optional, Set
import logging
from enum import Enum

from tools.registry import registry
from tools.base import EnhancedTool
from tools.tool_types import ToolCategory, ToolCapability, ToolDomain, ToolMetadata

# Setup logging
logger = logging.getLogger(__name__)


class SelectionMethod(str, Enum):
    """Methods used for tool selection."""
    KEYWORD = "keyword"       # Simple keyword matching
    CATEGORY = "category"     # Category-based selection
    CAPABILITY = "capability" # Capability-based selection
    DOMAIN = "domain"         # Domain-based selection
    ROLE = "role"             # Role-based selection
    HYBRID = "hybrid"         # Combination of methods


class ToolSelector:
    """Selects appropriate tools based on user queries and metadata."""
    
    def __init__(self, registry_instance=None):
        """Initialize the ToolSelector.
        
        Args:
            registry_instance: Tool registry instance to use. Defaults to global registry.
        """
        self.registry = registry_instance or registry
    
    def select_tools(
        self, 
        query: str, 
        method: SelectionMethod = SelectionMethod.HYBRID,
        role: Optional[str] = None,
        task: Optional[str] = None,
        max_tools: int = 5
    ) -> List[EnhancedTool]:
        """Select appropriate tools based on the query and selection method.
        
        Args:
            query: User query to analyze
            method: Tool selection method to use
            role: Optional role context for selection
            task: Optional task context for selection
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        logger.info(f"Selecting tools using {method.value} method for query: {query}")
        
        if method == SelectionMethod.KEYWORD:
            return self._select_by_keyword(query, max_tools)
        elif method == SelectionMethod.CATEGORY:
            return self._select_by_category(query, max_tools)
        elif method == SelectionMethod.CAPABILITY:
            return self._select_by_capability(query, max_tools)
        elif method == SelectionMethod.DOMAIN:
            return self._select_by_domain(query, max_tools)
        elif method == SelectionMethod.ROLE:
            if not role:
                logger.warning("Role-based selection requires a role parameter")
                return []
            return self._select_by_role(role, max_tools)
        elif method == SelectionMethod.HYBRID:
            return self._select_hybrid(query, role, task, max_tools)
        else:
            logger.warning(f"Unknown selection method: {method}")
            return []
    
    def _select_by_keyword(self, query: str, max_tools: int) -> List[EnhancedTool]:
        """Select tools based on keyword matching.
        
        Args:
            query: User query to analyze
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        query = query.lower()
        all_tools = self.registry.get_all_tools()
        
        # Define keyword patterns for different tool types
        keyword_patterns = {
            "weather": r"\b(weather|temperature|forecast|rain|snow|cloudy|sunny|humidity|degrees|cold|hot|warm|climate|meteorological|precipitation|conditions in)\b",
            "calculator": r"\b(calculate|add|subtract|multiply|divide|sum|difference|product|quotient|math|equation|formula|computation|solve|arithmetic)\b",
            "stock": r"\b(stock|price|market|ticker|share|invest|trading|financial|company|equity|securities|exchange|nasdaq|nyse|dow|quote|earnings|dividend|portfolio|etf)\b",
            "search": r"\b(search|find|look up|information|about|what is|who is|where is|details on|tell me about)\b"
        }
        
        # Score tools based on keyword matches - use id() as key instead of tool object
        tool_scores = {}
        for tool in all_tools:
            score = 0
            tool_name = tool.metadata.name.lower()
            tool_description = tool.metadata.description.lower()
            tool_display_name = tool.metadata.display_name.lower()
            
            # Check if the tool name or description directly matches parts of the query
            if tool_name in query:
                score += 5
            
            # Check if display name appears in query
            if any(word in query for word in tool_display_name.split()):
                score += 4
            
            # Score based on description words matching query
            for keyword in tool_description.split():
                if keyword.lower() in query and len(keyword) > 3:  # Only count meaningful words
                    score += 1
            
            # Check for pattern matches
            for category, pattern in keyword_patterns.items():
                # If this tool matches this category (by name or keywords)
                if category in tool_name or any(category in kw.lower() for kw in tool.metadata.keywords):
                    # And the query contains words from this category's pattern
                    if re.search(pattern, query):
                        score += 3
            
            # Check if any examples match the query
            for example in tool.metadata.examples:
                # Higher score for more similar examples
                similarity = sum(1 for word in example.lower().split() if word in query.split())
                if similarity > 0:
                    score += similarity
            
            if score > 0:
                # Use id() as key instead of the tool object
                tool_scores[id(tool)] = (tool, score)
        
        # Sort tools by score and return the top N
        sorted_tools = sorted(tool_scores.values(), key=lambda x: x[1], reverse=True)
        
        # Debug log for top scoring tools
        for i, (tool, score) in enumerate(sorted_tools[:max_tools]):
            logger.debug(f"Tool '{tool.metadata.display_name}' scored {score} for query: '{query}'")
            
        return [tool for tool, score in sorted_tools[:max_tools]]
    
    def _select_by_category(self, query: str, max_tools: int) -> List[EnhancedTool]:
        """Select tools based on categories derived from the query.
        
        Args:
            query: User query to analyze
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        query = query.lower()
        
        # Simple category mapping based on keywords
        category_patterns = {
            ToolCategory.SEARCH: r"\b(search|find|look up|information|about|what is|who is)\b",
            ToolCategory.UTILITY: r"\b(calculate|compute|convert|tool|utility)\b",
            ToolCategory.DATA_ANALYSIS: r"\b(analyze|data|statistics|chart|graph|plot)\b",
            ToolCategory.CONTENT_GENERATION: r"\b(generate|create|write|compose|draft)\b",
            ToolCategory.WEB_BROWSING: r"\b(browse|visit|website|web page|internet|url)\b",
            ToolCategory.COMMUNICATION: r"\b(email|message|send|contact|communicate)\b",
            ToolCategory.DOCUMENT_PROCESSING: r"\b(document|pdf|word|text|file|read)\b"
        }
        
        # Determine likely categories based on the query
        likely_categories = []
        for category, pattern in category_patterns.items():
            if re.search(pattern, query):
                likely_categories.append(category)
        
        # If no categories matched, default to search
        if not likely_categories:
            likely_categories = [ToolCategory.SEARCH]
        
        # Get tools in the identified categories
        selected_tools = []
        for category in likely_categories:
            # Get tools for this category
            category_tools = [
                tool for tool in self.registry.get_all_tools() 
                if category in tool.metadata.categories
            ]
            selected_tools.extend(category_tools)
        
        # De-duplicate using tool IDs and limit to max_tools
        unique_tools = []
        seen_ids = set()
        for tool in selected_tools:
            tool_id = id(tool)
            if tool_id not in seen_ids:
                seen_ids.add(tool_id)
                unique_tools.append(tool)
                
                # Stop when we reach max_tools
                if len(unique_tools) >= max_tools:
                    break
        
        return unique_tools
    
    def _select_by_capability(self, query: str, max_tools: int) -> List[EnhancedTool]:
        """Select tools based on capabilities needed for the query.
        
        Args:
            query: User query to analyze
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        query = query.lower()
        
        # Capability mapping based on query intention
        capability_patterns = {
            ToolCapability.READ: r"\b(read|view|show|display|get|find)\b",
            ToolCapability.WRITE: r"\b(write|create|save|edit|modify|update)\b",
            ToolCapability.COMPUTE: r"\b(calculate|compute|solve|process|determine)\b",
            ToolCapability.FETCH: r"\b(fetch|retrieve|get|download|access)\b",
            ToolCapability.TRANSFORM: r"\b(transform|convert|change|modify|format)\b",
            ToolCapability.GENERATE: r"\b(generate|create|make|produce|build)\b",
            ToolCapability.ANALYZE: r"\b(analyze|examine|study|investigate|evaluate)\b",
            ToolCapability.SEARCH: r"\b(search|find|look up|query|seek)\b",
            ToolCapability.STORE: r"\b(store|save|keep|record|archive)\b",
            ToolCapability.COMMUNICATE: r"\b(communicate|send|share|transmit|message)\b"
        }
        
        # Determine likely capabilities based on the query
        likely_capabilities = []
        for capability, pattern in capability_patterns.items():
            if re.search(pattern, query):
                likely_capabilities.append(capability)
        
        # If no capabilities matched, default to search and read
        if not likely_capabilities:
            likely_capabilities = [ToolCapability.SEARCH, ToolCapability.READ]
        
        # Get tools with the identified capabilities
        selected_tools = []
        for capability in likely_capabilities:
            capability_tools = [
                tool for tool in self.registry.get_all_tools()
                if capability in tool.metadata.capabilities
            ]
            selected_tools.extend(capability_tools)
        
        # De-duplicate using tool IDs and limit to max_tools
        unique_tools = []
        seen_ids = set()
        for tool in selected_tools:
            tool_id = id(tool)
            if tool_id not in seen_ids:
                seen_ids.add(tool_id)
                unique_tools.append(tool)
                
                # Stop when we reach max_tools
                if len(unique_tools) >= max_tools:
                    break
        
        return unique_tools
    
    def _select_by_domain(self, query: str, max_tools: int) -> List[EnhancedTool]:
        """Select tools based on the domain of the query.
        
        Args:
            query: User query to analyze
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        query = query.lower()
        
        # Domain mapping based on query content
        domain_patterns = {
            ToolDomain.GENERAL: r".*",  # Matches anything - fallback
            ToolDomain.FINANCE: r"\b(money|finance|stock|market|invest|price|dividend|financial|economy)\b",
            ToolDomain.HEALTHCARE: r"\b(health|medical|doctor|patient|disease|treatment|symptom|hospital)\b",
            ToolDomain.PROGRAMMING: r"\b(code|program|function|algorithm|variable|class|bug|developer)\b",
            ToolDomain.MARKETING: r"\b(marketing|advertise|campaign|brand|customer|product|market|sales)\b",
            ToolDomain.SCIENCE: r"\b(science|scientific|experiment|research|hypothesis|theory|data|physics|chemistry|biology)\b",
            ToolDomain.EDUCATION: r"\b(education|school|student|teach|learn|course|study|academic|knowledge)\b",
            ToolDomain.LAW: r"\b(legal|law|lawyer|attorney|court|judge|case|contract|rights|regulation)\b",
            ToolDomain.ENTERTAINMENT: r"\b(entertainment|movie|music|game|play|video|sport|fun|relax)\b",
            ToolDomain.BUSINESS: r"\b(business|company|corporate|management|strategy|executive|ceo|industry)\b"
        }
        
        # Determine likely domains based on the query
        likely_domains = []
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, query):
                likely_domains.append(domain)
        
        # If only GENERAL domain matched or none matched, add GENERAL domain
        if len(likely_domains) <= 1:
            likely_domains.append(ToolDomain.GENERAL)
        
        # Get tools in the identified domains
        selected_tools = []
        for domain in likely_domains:
            domain_tools = [
                tool for tool in self.registry.get_all_tools()
                if domain in tool.metadata.domains
            ]
            selected_tools.extend(domain_tools)
        
        # De-duplicate using tool IDs and limit to max_tools
        unique_tools = []
        seen_ids = set()
        for tool in selected_tools:
            tool_id = id(tool)
            if tool_id not in seen_ids:
                seen_ids.add(tool_id)
                unique_tools.append(tool)
                
                # Stop when we reach max_tools
                if len(unique_tools) >= max_tools:
                    break
        
        return unique_tools
    
    def _select_by_role(self, role: str, max_tools: int) -> List[EnhancedTool]:
        """Select tools based on the agent's role.
        
        Args:
            role: Agent's role description
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        role = role.lower()
        all_tools = self.registry.get_all_tools()
        
        # Define role patterns to match tools
        role_patterns = {
            "analyst": r"\b(analyst|analyze|examine|data|insight|report|evaluate)\b",
            "researcher": r"\b(research|investigate|study|explore|discover|find|inquiry)\b",
            "writer": r"\b(write|author|create|compose|draft|content|article)\b",
            "developer": r"\b(develop|code|program|engineer|build|software|application)\b",
            "assistant": r"\b(assist|help|support|aid|service|facilitate)\b",
            "tutor": r"\b(teach|educate|instruct|tutor|lesson|learn|student)\b",
            "financial": r"\b(finance|financial|money|invest|stock|market|economic)\b"
        }
        
        # Score tools based on role relevance - use id() as key instead of tool object
        tool_scores = {}
        for tool in all_tools:
            score = 0
            
            # Check if any applicable roles are mentioned
            for role_name in tool.metadata.applicable_roles:
                if role_name.lower() in role:
                    score += 3
            
            # Check for pattern matches in the role
            for role_type, pattern in role_patterns.items():
                if re.search(pattern, role):
                    # If the tool's categories or domains match this role type
                    if any(role_type in cat.value.lower() for cat in tool.metadata.categories):
                        score += 2
                    if any(role_type in dom.value.lower() for dom in tool.metadata.domains):
                        score += 2
            
            # Include all tools with a positive score
            if score > 0:
                tool_scores[id(tool)] = (tool, score)
        
        # Sort tools by score and return the top N
        sorted_tools = sorted(tool_scores.values(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, score in sorted_tools[:max_tools]]
        
        # If no tools were selected, return a default set of tools
        if not selected_tools:
            # Return some general purpose tools
            return [tool for tool in all_tools if ToolDomain.GENERAL in tool.metadata.domains][:max_tools]
        
        return selected_tools
    
    def _select_hybrid(
        self, 
        query: str, 
        role: Optional[str], 
        task: Optional[str], 
        max_tools: int
    ) -> List[EnhancedTool]:
        """Select tools using a hybrid approach that combines multiple methods.
        
        Args:
            query: User query to analyze
            role: Optional role context for selection
            task: Optional task context for selection
            max_tools: Maximum number of tools to return
            
        Returns:
            List of selected tools
        """
        # Collect tools from various methods with their scores - use id() as key
        tool_scores = {}
        
        # Get tools by keyword with their scores
        keyword_tools = self._select_by_keyword(query, max_tools * 2)
        for i, tool in enumerate(keyword_tools):
            tool_scores[id(tool)] = (tool, tool_scores.get(id(tool), (tool, 0))[1] + (max_tools * 2 - i))
        
        # Get tools by category
        category_tools = self._select_by_category(query, max_tools * 2)
        for i, tool in enumerate(category_tools):
            tool_scores[id(tool)] = (tool, tool_scores.get(id(tool), (tool, 0))[1] + (max_tools - i // 2))
        
        # Get tools by capability
        capability_tools = self._select_by_capability(query, max_tools * 2)
        for i, tool in enumerate(capability_tools):
            tool_scores[id(tool)] = (tool, tool_scores.get(id(tool), (tool, 0))[1] + (max_tools - i // 2))
        
        # Get tools by domain
        domain_tools = self._select_by_domain(query, max_tools * 2)
        for i, tool in enumerate(domain_tools):
            tool_scores[id(tool)] = (tool, tool_scores.get(id(tool), (tool, 0))[1] + (max_tools - i // 2))
        
        # If role is provided, include role-based selection
        if role:
            role_tools = self._select_by_role(role, max_tools * 2)
            for i, tool in enumerate(role_tools):
                tool_scores[id(tool)] = (tool, tool_scores.get(id(tool), (tool, 0))[1] + (max_tools - i // 2))
        
        # Sort tools by their combined scores
        sorted_tools = sorted(tool_scores.values(), key=lambda x: x[1], reverse=True)
        selected_tools = [tool for tool, score in sorted_tools[:max_tools]]
        
        logger.info(f"Hybrid selection found {len(selected_tools)} tools for query: {query}")
        for i, tool in enumerate(selected_tools):
            logger.debug(f"  {i+1}. {tool.metadata.name} (Score: {tool_scores[id(tool)][1]})")
        
        return selected_tools 