"""
RAG-Enhanced Algebra Knowledge Graph for K-12 Education.

This module builds a comprehensive knowledge graph for algebra topics
using the repository's RAG system for enhanced storage and retrieval capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from agents.rag.api.rag_service import RagService
from agents.rag.models.knowledge import Entity, Relationship, KnowledgeGraph as RagKnowledgeGraph
from agents.rag.models.document import Document

from .concept import Concept
from .graph import KnowledgeGraph


@dataclass
class MathConcept:
    """Enhanced math concept with RAG integration."""
    name: str
    description: str
    difficulty: int
    time_to_master: int
    category: str
    concept_id: str
    examples: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    common_misconceptions: List[str] = field(default_factory=list)
    teaching_strategies: List[str] = field(default_factory=list)


class RagEnhancedAlgebraGraph:
    """RAG-enhanced algebra knowledge graph with intelligent retrieval capabilities."""
    
    def __init__(self):
        """Initialize the RAG-enhanced algebra graph."""
        self.rag_service = RagService()
        self.traditional_graph = KnowledgeGraph(name="K-12 Algebra")
        self.concepts: Dict[str, MathConcept] = {}
        self._setup_storage()
    
    def _setup_storage(self):
        """Setup RAG storage adaptors."""
        # Use memory storage by default for testing and development
        print("Using memory storage for RAG system")
        # No additional storage setup needed - RagService uses memory by default
    
    async def add_concept_with_rag(self, concept: MathConcept) -> str:
        """
        Add a math concept to both traditional graph and RAG system.
        
        Args:
            concept: The math concept to add
            
        Returns:
            The concept ID
        """
        # Add to traditional graph
        traditional_concept = Concept(
            name=concept.name,
            description=concept.description,
            difficulty=concept.difficulty,
            time_to_master=concept.time_to_master,
            category=concept.category,
            concept_id=concept.concept_id,
            examples=concept.examples
        )
        self.traditional_graph.add_concept(traditional_concept)
        self.concepts[concept.concept_id] = concept
        
        # Create RAG entity for the concept
        entity = Entity(
            type="math_concept",
            properties={
                "name": concept.name,
                "description": concept.description,
                "difficulty": concept.difficulty,
                "time_to_master": concept.time_to_master,
                "category": concept.category,
                "concept_id": concept.concept_id,
                "examples": concept.examples,
                "learning_objectives": concept.learning_objectives,
                "common_misconceptions": concept.common_misconceptions,
                "teaching_strategies": concept.teaching_strategies
            },
            id=concept.concept_id
        )
        
        # Store in RAG system
        await self.rag_service.store_knowledge(entity)
        
        # Create document for full-text search
        document_content = f"""
        Concept: {concept.name}
        Category: {concept.category}
        Description: {concept.description}
        Difficulty Level: {concept.difficulty}/5
        Time to Master: {concept.time_to_master} minutes
        
        Examples:
        {chr(10).join(f"- {example}" for example in concept.examples)}
        
        Learning Objectives:
        {chr(10).join(f"- {obj}" for obj in concept.learning_objectives)}
        
        Common Misconceptions:
        {chr(10).join(f"- {misc}" for misc in concept.common_misconceptions)}
        
        Teaching Strategies:
        {chr(10).join(f"- {strategy}" for strategy in concept.teaching_strategies)}
        """
        
        document = Document(
            content=document_content,
            metadata={
                "concept_id": concept.concept_id,
                "category": concept.category,
                "difficulty": concept.difficulty,
                "type": "math_concept"
            },
            id=f"doc_{concept.concept_id}"
        )
        
        await self.rag_service.store_document(document)
        
        return concept.concept_id
    
    async def add_prerequisite_relationship(self, prerequisite_id: str, concept_id: str) -> str:
        """
        Add a prerequisite relationship between concepts.
        
        Args:
            prerequisite_id: ID of the prerequisite concept
            concept_id: ID of the concept that requires the prerequisite
            
        Returns:
            The relationship ID
        """
        # Add to traditional graph
        self.traditional_graph.add_prerequisite(prerequisite_id, concept_id)
        
        # Create RAG relationship
        relationship = Relationship(
            source_id=prerequisite_id,
            target_id=concept_id,
            type="prerequisite",
            properties={
                "description": f"{prerequisite_id} is a prerequisite for {concept_id}",
                "strength": 1.0,
                "type": "learning_dependency"
            }
        )
        
        await self.rag_service.store_knowledge(relationship)
        
        return relationship.id
    
    async def find_similar_concepts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find concepts similar to the query using RAG semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of similar concepts with relevance scores
        """
        results = await self.rag_service.retrieve_knowledge(
            query=query,
            top_k=top_k,
            filter_criteria={"type": "math_concept"}
        )
        
        return [
            {
                "concept_id": result.content.id if hasattr(result.content, 'id') else None,
                "content": result.content,
                "relevance_score": result.relevance_score,
                "metadata": result.metadata
            }
            for result in results
        ]
    
    async def get_learning_path_suggestions(self, current_concept_id: str, 
                                          learning_goal: str) -> List[str]:
        """
        Get AI-powered learning path suggestions using RAG.
        
        Args:
            current_concept_id: Current concept the student is working on
            learning_goal: Description of what the student wants to learn
            
        Returns:
            List of suggested concept IDs for the learning path
        """
        # Query for concepts related to the learning goal
        query = f"learning path from {current_concept_id} to achieve {learning_goal}"
        
        results = await self.rag_service.retrieve(
            query=query,
            top_k=10,
            filter_criteria={"type": "math_concept"}
        )
        
        # Extract concept IDs and use traditional graph for path finding
        suggested_concepts = []
        for result in results:
            if hasattr(result.content, 'properties') and 'concept_id' in result.content.properties:
                concept_id = result.content.properties['concept_id']
                if concept_id not in suggested_concepts:
                    suggested_concepts.append(concept_id)
        
        return suggested_concepts[:5]  # Return top 5 suggestions
    
    async def get_teaching_recommendations(self, concept_id: str, 
                                         student_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get personalized teaching recommendations using RAG.
        
        Args:
            concept_id: The concept to teach
            student_profile: Student's learning profile and preferences
            
        Returns:
            Teaching recommendations including strategies, examples, and resources
        """
        # Build query based on student profile
        learning_style = student_profile.get('learning_style', 'visual')
        difficulty_preference = student_profile.get('difficulty_preference', 'moderate')
        interests = student_profile.get('interests', [])
        
        query = f"""
        teaching strategies for {concept_id} 
        learning style: {learning_style}
        difficulty: {difficulty_preference}
        interests: {', '.join(interests)}
        """
        
        # Retrieve relevant teaching content
        results = await self.rag_service.retrieve(
            query=query,
            top_k=5,
            filter_criteria={"concept_id": concept_id}
        )
        
        recommendations = {
            "concept_id": concept_id,
            "teaching_strategies": [],
            "examples": [],
            "common_misconceptions": [],
            "personalized_tips": []
        }
        
        for result in results:
            if hasattr(result.content, 'properties'):
                props = result.content.properties
                recommendations["teaching_strategies"].extend(
                    props.get("teaching_strategies", [])
                )
                recommendations["examples"].extend(
                    props.get("examples", [])
                )
                recommendations["common_misconceptions"].extend(
                    props.get("common_misconceptions", [])
                )
        
        # Add personalized tips based on learning style
        if learning_style == "visual":
            recommendations["personalized_tips"].append(
                "Use visual aids, diagrams, and graphical representations"
            )
        elif learning_style == "auditory":
            recommendations["personalized_tips"].append(
                "Incorporate verbal explanations and discussions"
            )
        elif learning_style == "kinesthetic":
            recommendations["personalized_tips"].append(
                "Include hands-on activities and manipulatives"
            )
        
        return recommendations
    
    async def build_enhanced_algebra_graph(self) -> "RagEnhancedAlgebraGraph":
        """
        Build the complete RAG-enhanced algebra knowledge graph.
        
        Returns:
            The populated algebra graph
        """
        # Define enhanced concepts with additional RAG-specific data
        concepts_data = [
            MathConcept(
                name="Counting Up/Down",
                description="Count forward or backward by 1s within 1,000.",
                difficulty=1,
                time_to_master=30,
                category="Number Sense",
                concept_id="NS-01",
                examples=[
                    "Count from 345 up to 356.",
                    "Count backward from 120 down to 107."
                ],
                learning_objectives=[
                    "Develop number sequence understanding",
                    "Build foundation for addition and subtraction"
                ],
                common_misconceptions=[
                    "Confusing counting up vs counting down",
                    "Skipping numbers in sequence"
                ],
                teaching_strategies=[
                    "Use number lines for visual support",
                    "Practice with physical objects",
                    "Start with smaller ranges and gradually increase"
                ]
            ),
            MathConcept(
                name="Place Value to 1,000,000",
                description="Identify the value of each digit up to the millions place.",
                difficulty=1,
                time_to_master=60,
                category="Number Sense",
                concept_id="NS-02",
                examples=[
                    "What is the value of 7 in 275,412?",
                    "Write 603,000 in expanded form."
                ],
                prerequisites=["NS-01"],
                learning_objectives=[
                    "Understand positional notation",
                    "Read and write large numbers correctly"
                ],
                common_misconceptions=[
                    "Confusing place value with digit value",
                    "Misreading zeros in large numbers"
                ],
                teaching_strategies=[
                    "Use place value charts and manipulatives",
                    "Practice with real-world examples like population numbers",
                    "Break down numbers into expanded form"
                ]
            ),
            MathConcept(
                name="Variable Expressions",
                description="Write and evaluate algebraic expressions with variables.",
                difficulty=3,
                time_to_master=120,
                category="Pre-Algebra",
                concept_id="AT-17",
                examples=[
                    "Evaluate 3x + 5 when x = 4",
                    "Write an expression for 'twice a number plus 7'"
                ],
                prerequisites=["NS-02"],
                learning_objectives=[
                    "Understand variables as placeholders",
                    "Translate between words and algebraic expressions"
                ],
                common_misconceptions=[
                    "Thinking variables always equal 1",
                    "Confusing coefficients with exponents"
                ],
                teaching_strategies=[
                    "Start with concrete examples using boxes or blanks",
                    "Use real-world contexts for variables",
                    "Practice translation exercises regularly"
                ]
            )
        ]
        
        # Add all concepts to the graph
        for concept in concepts_data:
            await self.add_concept_with_rag(concept)
        
        # Add prerequisite relationships
        await self.add_prerequisite_relationship("NS-01", "NS-02")
        await self.add_prerequisite_relationship("NS-02", "AT-17")
        
        return self
    
    async def close(self):
        """Close the RAG service and cleanup resources."""
        await self.rag_service.close()


# Async factory function to create and populate the graph
async def create_rag_enhanced_algebra_graph() -> RagEnhancedAlgebraGraph:
    """
    Create and populate a RAG-enhanced algebra knowledge graph.
    
    Returns:
        Populated RagEnhancedAlgebraGraph instance
    """
    graph = RagEnhancedAlgebraGraph()
    await graph.build_enhanced_algebra_graph()
    return graph


# Example usage
async def main():
    """Example usage of the RAG-enhanced algebra graph."""
    # Create the enhanced graph
    graph = await create_rag_enhanced_algebra_graph()
    
    try:
        # Example 1: Find similar concepts
        print("=== Finding Similar Concepts ===")
        similar = await graph.find_similar_concepts("number operations", top_k=3)
        for result in similar:
            print(f"Concept: {result['concept_id']}, Score: {result['relevance_score']:.3f}")
        
        # Example 2: Get learning path suggestions
        print("\n=== Learning Path Suggestions ===")
        path = await graph.get_learning_path_suggestions(
            "NS-01", 
            "learn basic algebra"
        )
        print(f"Suggested path: {' -> '.join(path)}")
        
        # Example 3: Get teaching recommendations
        print("\n=== Teaching Recommendations ===")
        student_profile = {
            "learning_style": "visual",
            "difficulty_preference": "moderate",
            "interests": ["sports", "music"]
        }
        recommendations = await graph.get_teaching_recommendations("AT-17", student_profile)
        print(f"Teaching strategies: {recommendations['teaching_strategies'][:2]}")
        print(f"Personalized tips: {recommendations['personalized_tips']}")
        
    finally:
        await graph.close()


if __name__ == "__main__":
    asyncio.run(main()) 