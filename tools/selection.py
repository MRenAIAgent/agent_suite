"""
Tool Selection Service for intelligently selecting tools based on roles and tasks.

This module provides services for selecting the most appropriate tools
for a given role or task, using both explicit metadata matching and
semantic matching.
"""
import logging
from typing import Any, Dict, List, Optional, Set, Union

from tools.tool_types import ToolCategory, ToolDomain, ToolCapability, Task, Role
from tools.base import EnhancedTool
from tools.registry import registry

# Configure logging
logger = logging.getLogger(__name__)


class ToolSelectionService:
    """
    Service for selecting appropriate tools based on roles and tasks.
    
    This service provides methods for:
    1. Selecting tools based on roles
    2. Selecting tools based on tasks
    3. Semantic matching for more nuanced selection
    4. Context-aware tool ranking
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the tool selection service.
        
        Args:
            embedding_model: Optional embedding model for semantic matching
        """
        self.registry = registry
        self.embedding_model = embedding_model
        self._task_cache = {}  # Cache semantic matches by task description
    
    def select_tools_for_role(self, role: Role, max_tools: int = None) -> List[EnhancedTool]:
        """
        Select tools based on the given role.
        
        Args:
            role: Role to select tools for
            max_tools: Maximum number of tools to return
            
        Returns:
            List of tool instances suitable for the role
        """
        all_tools = self.registry.get_all_tools()
        
        # Check if role specifies preferred tools
        if role.preferred_tools:
            # Get tools by name if they exist
            preferred_tools = [
                self.registry.get_tool(name) 
                for name in role.preferred_tools
                if self.registry.get_tool(name) is not None
            ]
            
            # If we have enough preferred tools, return them
            if preferred_tools and (max_tools is None or len(preferred_tools) >= max_tools):
                return preferred_tools[:max_tools] if max_tools else preferred_tools
        
        # Score tools based on role matching
        scored_tools = []
        for tool in all_tools:
            score = 0
            
            # Check if tool explicitly supports this role
            if role.name in tool.metadata.applicable_roles:
                score += 3
            
            # Check capability match
            for capability in role.required_capabilities:
                if any(cap == capability for cap in tool.metadata.capabilities):
                    score += 2
            
            # Check domain match
            for domain in role.domains:
                if any(dom == domain for dom in tool.metadata.domains):
                    score += 1
            
            # Only include tools with a positive score
            if score > 0:
                scored_tools.append((tool, score))
        
        # Sort by score (descending) and return the tools
        tools = [tool for tool, _ in sorted(scored_tools, key=lambda x: x[1], reverse=True)]
        
        # Limit the number of tools if needed
        return tools[:max_tools] if max_tools else tools
    
    def select_tools_for_task(self, task: Task, max_tools: int = None) -> List[EnhancedTool]:
        """
        Select tools based on the given task.
        
        Args:
            task: Task to select tools for
            max_tools: Maximum number of tools to return
            
        Returns:
            List of tool instances suitable for the task
        """
        all_tools = self.registry.get_all_tools()
        
        # Score tools based on task matching
        scored_tools = []
        for tool in all_tools:
            score = 0
            
            # Check for task pattern matches
            for pattern in tool.metadata.task_patterns:
                if pattern.lower() in task.description.lower():
                    score += 3
                    break
            
            # Check capability match
            for capability in task.required_capabilities:
                if any(cap == capability for cap in tool.metadata.capabilities):
                    score += 2
            
            # Check domain match
            if any(dom == task.domain for dom in tool.metadata.domains):
                score += 1
            
            # Only include tools with a positive score
            if score > 0:
                scored_tools.append((tool, score))
        
        # Sort by score (descending) and return the tools
        tools = [tool for tool, _ in sorted(scored_tools, key=lambda x: x[1], reverse=True)]
        
        # If no matches were found, try semantic matching
        if not tools and self.embedding_model:
            return self.select_tools_semantic_match(task.description, max_tools)
        
        # Limit the number of tools if needed
        return tools[:max_tools] if max_tools else tools
    
    async def select_tools_semantic_match(
        self, 
        task_description: str, 
        max_tools: int = None
    ) -> List[EnhancedTool]:
        """
        Select tools based on semantic matching with task description.
        
        Args:
            task_description: Description of the task
            max_tools: Maximum number of tools to return
            
        Returns:
            List of tool instances semantically relevant to the task
        """
        # Check cache first
        if task_description in self._task_cache:
            tools = self._task_cache[task_description]
            return tools[:max_tools] if max_tools else tools
        
        # If no embedding model available, return empty list
        if not self.embedding_model:
            logger.warning("No embedding model available for semantic matching")
            return []
        
        all_tools = self.registry.get_all_tools()
        
        try:
            # Get task embedding
            task_embedding = await self.embedding_model.get_embedding(task_description)
            
            # Get tool embeddings (using semantic description if available)
            tool_embeddings = []
            for tool in all_tools:
                description = (
                    tool.metadata.semantic_description or 
                    tool.__doc__ or 
                    f"{tool.__class__.__name__}: {', '.join(tool.metadata.keywords)}"
                )
                embedding = await self.embedding_model.get_embedding(description)
                tool_embeddings.append(embedding)
            
            # Calculate similarity scores
            tools_with_scores = []
            for i, tool in enumerate(all_tools):
                similarity = self._calculate_cosine_similarity(task_embedding, tool_embeddings[i])
                tools_with_scores.append((tool, similarity))
            
            # Sort by similarity (descending) and apply threshold
            threshold = 0.7  # Minimum similarity score
            tools = [
                tool for tool, score in 
                sorted(tools_with_scores, key=lambda x: x[1], reverse=True) 
                if score > threshold
            ]
            
            # Cache the results
            self._task_cache[task_description] = tools
            
            # Limit the number of tools if needed
            return tools[:max_tools] if max_tools else tools
        
        except Exception as e:
            logger.error(f"Error performing semantic matching: {e}")
            return []
    
    def _calculate_cosine_similarity(self, vec1, vec2) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        # Convert to numpy arrays if they aren't already
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)
    
    async def analyze_task(self, task_description: str, llm=None) -> Task:
        """
        Analyze a task description to extract structured information.
        
        Args:
            task_description: Natural language task description
            llm: Optional LLM for task analysis
            
        Returns:
            Task object with structured information
        """
        if not llm:
            # If no LLM is provided, create a basic task object
            return Task(
                description=task_description,
                goal="Complete the task successfully",
                type="general",
                domain=ToolDomain.GENERAL,
                required_capabilities=[],
                expected_outputs=[]
            )
        
        try:
            # Use the LLM to extract structured information
            prompt = f"""
            Analyze the following task and extract structured information:
            
            Task: "{task_description}"
            
            Extract the following:
            1. Overall goal
            2. Task type (e.g., research, analysis, creation)
            3. Task domain (one of: general, finance, healthcare, programming, marketing, 
               science, education, law, entertainment, business)
            4. Required capabilities (from this list: read, write, compute, fetch, transform, 
               generate, analyze, search, store, communicate)
            5. Expected outputs
            
            Format as JSON.
            """
            
            analysis_result = await llm.generate(prompt)
            
            # Parse the JSON response
            import json
            
            # Try to extract JSON from the response (it might be wrapped in markdown or other text)
            json_start = analysis_result.find('{')
            json_end = analysis_result.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = analysis_result[json_start:json_end]
                
                try:
                    analysis_data = json.loads(json_str)
                    
                    # Convert string capabilities to enum values if possible
                    capabilities = []
                    for cap in analysis_data.get("required_capabilities", []):
                        try:
                            capabilities.append(ToolCapability(cap.lower()))
                        except ValueError:
                            capabilities.append(cap)
                    
                    # Convert domain string to enum value if possible
                    domain = analysis_data.get("domain", "general")
                    try:
                        domain = ToolDomain(domain.lower())
                    except ValueError:
                        pass
                    
                    return Task(
                        description=task_description,
                        goal=analysis_data.get("goal", "Complete the task successfully"),
                        type=analysis_data.get("type", "general"),
                        domain=domain,
                        required_capabilities=capabilities,
                        expected_outputs=analysis_data.get("expected_outputs", [])
                    )
                
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from LLM response: {analysis_result}")
            
            # Fall back to a basic task if parsing fails
            return Task(
                description=task_description,
                goal="Complete the task successfully",
                type="general",
                domain=ToolDomain.GENERAL,
                required_capabilities=[],
                expected_outputs=[]
            )
        
        except Exception as e:
            logger.error(f"Error analyzing task: {e}")
            
            # Return a basic task object in case of error
            return Task(
                description=task_description,
                goal="Complete the task successfully",
                type="general",
                domain=ToolDomain.GENERAL,
                required_capabilities=[],
                expected_outputs=[]
            )


# Create the global tool selection service instance
tool_selector = ToolSelectionService() 