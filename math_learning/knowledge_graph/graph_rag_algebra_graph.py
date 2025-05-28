"""Graph-first RAG-based algebra knowledge graph.

This module implements an algebra knowledge graph that uses the RAG system's
graph storage as the primary storage mechanism, with vector and key-value
storage providing additional capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

from agents.rag.api.rag_service import RagService
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph
from agents.rag.middleware.storage_router import StorageType
from math_learning.config.rag_config import RagConfig, create_configured_rag_service

logger = logging.getLogger(__name__)


@dataclass
class AlgebraConcept:
    """Represents an algebra concept with metadata."""
    id: str
    name: str
    description: str
    category: str
    difficulty_level: int  # 1-10 scale
    grade_level: str
    examples: List[str]
    common_mistakes: List[str]
    teaching_strategies: List[str]
    
    def to_entity(self) -> Entity:
        """Convert to RAG Entity format."""
        return Entity(
            type="algebra_concept",
            properties={
                "name": self.name,
                "description": self.description,
                "category": self.category,
                "difficulty_level": self.difficulty_level,
                "grade_level": self.grade_level,
                "examples": self.examples,
                "common_mistakes": self.common_mistakes,
                "teaching_strategies": self.teaching_strategies
            },
            id=self.id
        )


class GraphRagAlgebraGraph:
    """
    Algebra knowledge graph using RAG system's graph storage as primary storage.
    
    This implementation uses the RAG system's graph storage for all core
    graph operations, with vector storage for semantic search and key-value
    storage for metadata and caching.
    """
    
    def __init__(self, rag_service: RagService, config: RagConfig):
        """
        Initialize the graph-first RAG algebra graph.
        
        Args:
            rag_service: Configured RAG service with graph storage
            config: RAG configuration
        """
        self.rag_service = rag_service
        self.config = config
        self.graph_name = "K-12 Algebra Knowledge Graph"
        self._initialized = False
        
        # Verify that graph storage is available
        if not self.rag_service.has_storage_adaptor(StorageType.GRAPH):
            raise ValueError("RAG service must have graph storage configured")
    
    async def initialize(self) -> None:
        """Initialize the algebra knowledge graph with concepts and relationships."""
        if self._initialized:
            return
        
        logger.info("Initializing algebra knowledge graph...")
        
        # Create the knowledge graph in RAG storage
        knowledge_graph = KnowledgeGraph(
            entities=[],
            relationships=[],
            metadata={
                "name": self.graph_name,
                "description": "Comprehensive K-12 algebra knowledge graph"
            },
            id="algebra_kg"
        )
        
        # Store the knowledge graph
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        await graph_storage.store_knowledge_graph(knowledge_graph)
        
        # Add all algebra concepts
        await self._add_algebra_concepts()
        
        # Add prerequisite relationships
        await self._add_prerequisite_relationships()
        
        self._initialized = True
        logger.info("Algebra knowledge graph initialized successfully")
    
    async def _add_algebra_concepts(self) -> None:
        """Add all algebra concepts to the graph."""
        concepts = self._get_algebra_concepts()
        
        for concept in concepts:
            entity = concept.to_entity()
            await self.add_concept(entity)
    
    async def _add_prerequisite_relationships(self) -> None:
        """Add prerequisite relationships between concepts."""
        prerequisites = self._get_prerequisite_relationships()
        
        for prerequisite_id, concept_id in prerequisites:
            await self.add_prerequisite(prerequisite_id, concept_id)
    
    async def add_concept(self, entity: Entity) -> None:
        """
        Add a concept to the knowledge graph.
        
        Args:
            entity: The concept entity to add
        """
        # Store in graph storage (primary)
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        await graph_storage.store_entity(entity)
        
        # Store in vector storage for semantic search (if available)
        if self.rag_service.has_storage_adaptor(StorageType.VECTOR):
            # Create a document for vector search
            content = f"{entity.name}: {entity.properties.get('description', '')}"
            await self.rag_service.store_knowledge(entity, content)
        
        # Store metadata in key-value storage (if available)
        if self.rag_service.has_storage_adaptor(StorageType.KEY_VALUE):
            kv_storage = self.rag_service.get_storage_adaptor(StorageType.KEY_VALUE)
            await kv_storage.store(f"concept:{entity.id}", {
                "name": entity.name,
                "category": entity.properties.get("category"),
                "difficulty_level": entity.properties.get("difficulty_level"),
                "grade_level": entity.properties.get("grade_level")
            })
    
    async def add_prerequisite(self, prerequisite_id: str, concept_id: str) -> None:
        """
        Add a prerequisite relationship between concepts.
        
        Args:
            prerequisite_id: ID of the prerequisite concept
            concept_id: ID of the concept that requires the prerequisite
        """
        relationship = Relationship(
            source_id=prerequisite_id,
            target_id=concept_id,
            type="prerequisite_for",
            properties={
                "relationship_type": "prerequisite",
                "strength": 1.0,
                "description": f"{prerequisite_id} is a prerequisite for {concept_id}"
            },
            id=f"{prerequisite_id}_prerequisite_for_{concept_id}"
        )
        
        # Store in graph storage
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        await graph_storage.store_relationship(relationship)
    
    async def get_concept(self, concept_id: str) -> Optional[Entity]:
        """
        Get a concept by ID.
        
        Args:
            concept_id: The concept ID
            
        Returns:
            The concept entity or None if not found
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        return await graph_storage.get_entity(concept_id)
    
    async def get_prerequisites(self, concept_id: str) -> List[Entity]:
        """
        Get all prerequisites for a concept.
        
        Args:
            concept_id: The concept ID
            
        Returns:
            List of prerequisite concept entities
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Get incoming prerequisite relationships
        relationships = await graph_storage.get_relationships(
            target_id=concept_id,
            relationship_type="prerequisite_for"
        )
        
        # Get the prerequisite entities
        prerequisites = []
        for rel in relationships:
            prerequisite = await graph_storage.get_entity(rel.source_id)
            if prerequisite:
                prerequisites.append(prerequisite)
        
        return prerequisites
    
    async def get_dependents(self, concept_id: str) -> List[Entity]:
        """
        Get all concepts that depend on this concept.
        
        Args:
            concept_id: The concept ID
            
        Returns:
            List of dependent concept entities
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Get outgoing prerequisite relationships
        relationships = await graph_storage.get_relationships(
            source_id=concept_id,
            relationship_type="prerequisite_for"
        )
        
        # Get the dependent entities
        dependents = []
        for rel in relationships:
            dependent = await graph_storage.get_entity(rel.target_id)
            if dependent:
                dependents.append(dependent)
        
        return dependents
    
    async def find_learning_path(self, start_concept_id: str, target_concept_id: str) -> List[Entity]:
        """
        Find a learning path between two concepts.
        
        Args:
            start_concept_id: Starting concept ID
            target_concept_id: Target concept ID
            
        Returns:
            List of concepts in the learning path
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Use graph storage's path finding capabilities
        path_ids = await graph_storage.find_path(
            start_concept_id,
            target_concept_id,
            relationship_type="prerequisite_for"
        )
        
        # Convert IDs to entities
        path = []
        for concept_id in path_ids:
            concept = await graph_storage.get_entity(concept_id)
            if concept:
                path.append(concept)
        
        return path
    
    async def search_concepts(self, query: str, limit: int = 10) -> List[Entity]:
        """
        Search for concepts using semantic search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concept entities
        """
        if not self.rag_service.has_storage_adaptor(StorageType.VECTOR):
            # Fallback to graph-based search
            return await self._graph_based_search(query, limit)
        
        # Use vector search for semantic matching
        results = await self.rag_service.search_knowledge(query, limit=limit)
        
        # Convert results to entities
        concepts = []
        for result in results:
            if hasattr(result, 'entity') and result.entity:
                concepts.append(result.entity)
        
        return concepts
    
    async def _graph_based_search(self, query: str, limit: int) -> List[Entity]:
        """Fallback search using graph storage only."""
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        
        # Get all entities and filter by name/description
        all_entities = await graph_storage.get_all_entities()
        
        query_lower = query.lower()
        matches = []
        
        for entity in all_entities:
            if entity.type == "algebra_concept":
                name_match = query_lower in entity.name.lower()
                desc_match = query_lower in entity.properties.get("description", "").lower()
                
                if name_match or desc_match:
                    matches.append(entity)
                    
                if len(matches) >= limit:
                    break
        
        return matches
    
    async def get_concepts_by_category(self, category: str) -> List[Entity]:
        """
        Get all concepts in a specific category.
        
        Args:
            category: The category name
            
        Returns:
            List of concept entities in the category
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        all_entities = await graph_storage.get_all_entities()
        
        return [
            entity for entity in all_entities
            if (entity.type == "algebra_concept" and 
                entity.properties.get("category") == category)
        ]
    
    async def get_concepts_by_difficulty(self, min_level: int, max_level: int) -> List[Entity]:
        """
        Get concepts within a difficulty range.
        
        Args:
            min_level: Minimum difficulty level
            max_level: Maximum difficulty level
            
        Returns:
            List of concept entities within the difficulty range
        """
        graph_storage = self.rag_service.get_storage_adaptor(StorageType.GRAPH)
        all_entities = await graph_storage.get_all_entities()
        
        return [
            entity for entity in all_entities
            if (entity.type == "algebra_concept" and 
                min_level <= entity.properties.get("difficulty_level", 0) <= max_level)
        ]
    
    def _get_algebra_concepts(self) -> List[AlgebraConcept]:
        """Get the complete list of algebra concepts."""
        return [
            # Number Sense and Operations
            AlgebraConcept(
                id="integers",
                name="Integers",
                description="Positive and negative whole numbers including zero",
                category="Number Sense",
                difficulty_level=2,
                grade_level="6-7",
                examples=["-3, -2, -1, 0, 1, 2, 3", "Temperature readings", "Bank account balances"],
                common_mistakes=["Confusing negative signs with subtraction", "Ordering negative numbers incorrectly"],
                teaching_strategies=["Use number line visualization", "Real-world contexts like temperature"]
            ),
            AlgebraConcept(
                id="rational_numbers",
                name="Rational Numbers",
                description="Numbers that can be expressed as fractions a/b where b ≠ 0",
                category="Number Sense",
                difficulty_level=3,
                grade_level="7-8",
                examples=["1/2, -3/4, 0.25, 0.333...", "Fractions and decimals", "Mixed numbers"],
                common_mistakes=["Thinking all decimals are rational", "Confusion with irrational numbers"],
                teaching_strategies=["Fraction-decimal conversion practice", "Visual fraction models"]
            ),
            AlgebraConcept(
                id="real_numbers",
                name="Real Numbers",
                description="All rational and irrational numbers",
                category="Number Sense",
                difficulty_level=4,
                grade_level="8-9",
                examples=["√2, π, e", "All previous number types", "Square roots of non-perfect squares"],
                common_mistakes=["Thinking all numbers are rational", "Confusion about √(-1)"],
                teaching_strategies=["Number line representation", "Calculator exploration of irrationals"]
            ),
            
            # Variables and Expressions
            AlgebraConcept(
                id="variables",
                name="Variables",
                description="Symbols (usually letters) that represent unknown or changing quantities",
                category="Algebraic Thinking",
                difficulty_level=2,
                grade_level="6-7",
                examples=["x, y, n", "Let x = number of apples", "Temperature T varies with time"],
                common_mistakes=["Thinking variables always equal 1", "Confusion between variable and coefficient"],
                teaching_strategies=["Use concrete examples", "Start with word problems"]
            ),
            AlgebraConcept(
                id="algebraic_expressions",
                name="Algebraic Expressions",
                description="Mathematical phrases containing variables, numbers, and operations",
                category="Algebraic Thinking",
                difficulty_level=3,
                grade_level="7-8",
                examples=["3x + 5", "2a - 7b", "x² + 4x - 1"],
                common_mistakes=["Combining unlike terms", "Incorrect order of operations"],
                teaching_strategies=["Color-code like terms", "Use algebra tiles"]
            ),
            AlgebraConcept(
                id="evaluating_expressions",
                name="Evaluating Expressions",
                description="Finding the value of an expression by substituting values for variables",
                category="Algebraic Thinking",
                difficulty_level=3,
                grade_level="7-8",
                examples=["If x=3, then 2x+1 = 7", "Substitute and calculate", "Function evaluation"],
                common_mistakes=["Order of operations errors", "Sign errors with negative substitutions"],
                teaching_strategies=["Step-by-step substitution", "Use parentheses for clarity"]
            ),
            
            # Equations and Solving
            AlgebraConcept(
                id="linear_equations_one_variable",
                name="Linear Equations in One Variable",
                description="Equations of the form ax + b = c where a ≠ 0",
                category="Equations",
                difficulty_level=4,
                grade_level="8-9",
                examples=["2x + 3 = 11", "5 - 3x = 14", "x/4 + 7 = 10"],
                common_mistakes=["Not applying operations to both sides", "Sign errors"],
                teaching_strategies=["Balance method", "Work backwards from solution"]
            ),
            AlgebraConcept(
                id="systems_linear_equations",
                name="Systems of Linear Equations",
                description="Two or more linear equations with the same variables",
                category="Equations",
                difficulty_level=6,
                grade_level="9-10",
                examples=["x + y = 5, 2x - y = 1", "Three equations, three unknowns"],
                common_mistakes=["Arithmetic errors in elimination", "Forgetting to check solutions"],
                teaching_strategies=["Graphical interpretation", "Substitution method first"]
            ),
            
            # Functions
            AlgebraConcept(
                id="functions",
                name="Functions",
                description="Relations where each input has exactly one output",
                category="Functions",
                difficulty_level=5,
                grade_level="9-10",
                examples=["f(x) = 2x + 1", "Area as function of radius", "Vertical line test"],
                common_mistakes=["Confusing function notation", "Not understanding domain/range"],
                teaching_strategies=["Input-output tables", "Real-world function examples"]
            ),
            AlgebraConcept(
                id="linear_functions",
                name="Linear Functions",
                description="Functions of the form f(x) = mx + b",
                category="Functions",
                difficulty_level=5,
                grade_level="9-10",
                examples=["f(x) = 3x - 2", "Slope-intercept form", "Direct variation"],
                common_mistakes=["Confusing slope and y-intercept", "Graphing errors"],
                teaching_strategies=["Slope as rate of change", "Graphing practice"]
            ),
            AlgebraConcept(
                id="quadratic_functions",
                name="Quadratic Functions",
                description="Functions of the form f(x) = ax² + bx + c where a ≠ 0",
                category="Functions",
                difficulty_level=7,
                grade_level="10-11",
                examples=["f(x) = x² - 4x + 3", "Parabolas", "Vertex form"],
                common_mistakes=["Sign errors in vertex formula", "Incomplete factoring"],
                teaching_strategies=["Graphing transformations", "Complete the square method"]
            ),
        ]
    
    def _get_prerequisite_relationships(self) -> List[tuple]:
        """Get prerequisite relationships as (prerequisite_id, concept_id) pairs."""
        return [
            # Number system progression
            ("integers", "rational_numbers"),
            ("rational_numbers", "real_numbers"),
            
            # Algebraic thinking progression
            ("integers", "variables"),
            ("variables", "algebraic_expressions"),
            ("algebraic_expressions", "evaluating_expressions"),
            
            # Equation solving progression
            ("algebraic_expressions", "linear_equations_one_variable"),
            ("linear_equations_one_variable", "systems_linear_equations"),
            
            # Function progression
            ("evaluating_expressions", "functions"),
            ("linear_equations_one_variable", "linear_functions"),
            ("functions", "linear_functions"),
            ("linear_functions", "quadratic_functions"),
            ("real_numbers", "quadratic_functions"),
        ]


async def create_graph_rag_algebra_graph(config: RagConfig = None) -> GraphRagAlgebraGraph:
    """
    Create and initialize a graph-first RAG algebra graph.
    
    Args:
        config: RAG configuration, defaults to memory-based config
        
    Returns:
        Initialized GraphRagAlgebraGraph instance
    """
    if config is None:
        from math_learning.config.rag_config import get_memory_config
        config = get_memory_config()
    
    # Create RAG service with graph storage as primary
    rag_service = await create_configured_rag_service(config)
    
    # Create and initialize the graph
    graph = GraphRagAlgebraGraph(rag_service, config)
    await graph.initialize()
    
    return graph


# Example usage
async def main():
    """Example usage of the graph-first RAG algebra graph."""
    # Create with memory-based configuration
    graph = await create_graph_rag_algebra_graph()
    
    # Search for concepts
    results = await graph.search_concepts("linear equations")
    print(f"Found {len(results)} concepts related to 'linear equations'")
    
    # Get prerequisites for quadratic functions
    prereqs = await graph.get_prerequisites("quadratic_functions")
    print(f"Prerequisites for quadratic functions: {[p.name for p in prereqs]}")
    
    # Find learning path
    path = await graph.find_learning_path("integers", "quadratic_functions")
    print(f"Learning path: {[c.name for c in path]}")


if __name__ == "__main__":
    asyncio.run(main()) 