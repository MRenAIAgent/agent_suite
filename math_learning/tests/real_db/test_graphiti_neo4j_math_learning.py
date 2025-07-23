#!/usr/bin/env python3
"""
Test for Math Learning using Graphiti + Neo4j Integration

This test demonstrates how to use Graphiti with Neo4j as the backend
for building knowledge graphs and learning graphs in math education.

Graphiti provides intelligent memory management and relationship inference,
while Neo4j provides robust graph storage and querying capabilities.
"""

import asyncio
import pytest
import sys
import os
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add project root to path for imports
# From math_learning/tests/real_db/test_graphiti_neo4j_math_learning.py
# Go up 4 levels: real_db -> tests -> math_learning -> agent_suite
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from math_learning.scripts.utils.database_usage_checker import DatabaseUsageChecker

# Try to import required modules
try:
    import graphiti
    GRAPHITI_AVAILABLE = True
except ImportError:
    GRAPHITI_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


class MathLearningGraphitiNeo4j:
    """
    Math Learning system using Graphiti + Neo4j integration.
    
    This class demonstrates how to build knowledge graphs and learning graphs
    for math education using Graphiti's intelligent memory management
    with Neo4j as the persistent storage backend.
    """
    
    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", 
                 neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        """Initialize the math learning system with Graphiti + Neo4j."""
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.namespace = f"math_learning_{uuid.uuid4().hex[:8]}"
        
        # Initialize Graphiti with Neo4j backend
        self.graph = None
        self.neo4j_driver = None
        
    async def initialize(self):
        """Initialize the Graphiti graph with Neo4j backend."""
        if not GRAPHITI_AVAILABLE:
            raise ImportError("Graphiti is required. Install with: pip install graphiti")
        if not NEO4J_AVAILABLE:
            raise ImportError("Neo4j is required. Install with: pip install neo4j")
        
        # Initialize Neo4j driver
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        # Initialize Graphiti with Neo4j backend
        # Note: This is a conceptual implementation - actual Graphiti Neo4j integration
        # may require specific configuration depending on Graphiti version
        self.graph = graphiti.Graph(
            backend="neo4j",
            uri=self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
            namespace=self.namespace
        )
        
        await self._setup_math_schema()
    
    async def _setup_math_schema(self):
        """Set up the schema for math learning concepts."""
        # Define node types for math learning
        node_types = {
            "MathConcept": {
                "name": str,
                "description": str,
                "difficulty_level": float,
                "grade_level": int,
                "concept_type": str,  # "arithmetic", "algebra", "geometry", etc.
                "mastery_criteria": str
            },
            "Student": {
                "student_id": str,
                "name": str,
                "grade_level": int,
                "learning_style": str
            },
            "LearningSession": {
                "session_id": str,
                "student_id": str,
                "concept_id": str,
                "timestamp": str,
                "performance_score": float,
                "time_spent": int,  # minutes
                "mastery_achieved": bool
            },
            "LearningPath": {
                "path_id": str,
                "name": str,
                "description": str,
                "target_grade": int
            }
        }
        
        # Define relationship types
        relationship_types = {
            "PREREQUISITE": {
                "strength": float,  # How essential the prerequisite is (0.0-1.0)
                "description": str
            },
            "BUILDS_ON": {
                "connection_type": str,  # "extends", "applies", "combines"
                "strength": float
            },
            "STUDIED": {
                "session_count": int,
                "total_time": int,
                "average_score": float,
                "mastery_level": float
            },
            "FOLLOWS": {
                "sequence_order": int,
                "recommended_gap": int  # days between concepts
            },
            "INCLUDES": {
                "order": int,
                "is_optional": bool
            }
        }
        
        # Create node and relationship types in Graphiti
        for node_type, schema in node_types.items():
            await self.graph.create_node_type(node_type, schema)
        
        for rel_type, schema in relationship_types.items():
            await self.graph.create_relationship_type(rel_type, schema)
    
    async def build_knowledge_graph(self):
        """Build a comprehensive knowledge graph of math concepts."""
        print("🧠 Building Math Knowledge Graph...")
        
        # Define math concepts with their properties
        concepts = [
            {
                "name": "Counting",
                "description": "Basic counting from 1 to 100",
                "difficulty_level": 0.1,
                "grade_level": 1,
                "concept_type": "arithmetic",
                "mastery_criteria": "Count to 100 without errors"
            },
            {
                "name": "Addition",
                "description": "Adding two or more numbers",
                "difficulty_level": 0.2,
                "grade_level": 1,
                "concept_type": "arithmetic",
                "mastery_criteria": "Solve 10 addition problems with 90% accuracy"
            },
            {
                "name": "Subtraction",
                "description": "Subtracting one number from another",
                "difficulty_level": 0.25,
                "grade_level": 2,
                "concept_type": "arithmetic",
                "mastery_criteria": "Solve 10 subtraction problems with 90% accuracy"
            },
            {
                "name": "Multiplication",
                "description": "Multiplying numbers using times tables",
                "difficulty_level": 0.4,
                "grade_level": 3,
                "concept_type": "arithmetic",
                "mastery_criteria": "Know multiplication tables 1-12"
            },
            {
                "name": "Division",
                "description": "Dividing numbers with and without remainders",
                "difficulty_level": 0.45,
                "grade_level": 4,
                "concept_type": "arithmetic",
                "mastery_criteria": "Solve division problems with 85% accuracy"
            },
            {
                "name": "Fractions",
                "description": "Understanding and working with fractions",
                "difficulty_level": 0.6,
                "grade_level": 4,
                "concept_type": "arithmetic",
                "mastery_criteria": "Add, subtract, multiply, and divide fractions"
            },
            {
                "name": "Decimals",
                "description": "Working with decimal numbers",
                "difficulty_level": 0.55,
                "grade_level": 5,
                "concept_type": "arithmetic",
                "mastery_criteria": "Convert between fractions and decimals"
            },
            {
                "name": "Percentages",
                "description": "Understanding and calculating percentages",
                "difficulty_level": 0.65,
                "grade_level": 6,
                "concept_type": "arithmetic",
                "mastery_criteria": "Calculate percentages and percentage changes"
            },
            {
                "name": "Basic Algebra",
                "description": "Introduction to variables and simple equations",
                "difficulty_level": 0.7,
                "grade_level": 7,
                "concept_type": "algebra",
                "mastery_criteria": "Solve linear equations with one variable"
            },
            {
                "name": "Linear Equations",
                "description": "Solving linear equations and graphing lines",
                "difficulty_level": 0.8,
                "grade_level": 8,
                "concept_type": "algebra",
                "mastery_criteria": "Graph linear equations and find slope"
            }
        ]
        
        # Create concept nodes
        concept_ids = {}
        for concept in concepts:
            node_id = await self.graph.add_node("MathConcept", concept)
            concept_ids[concept["name"]] = node_id
            print(f"  ✅ Created concept: {concept['name']}")
        
        # Define prerequisite relationships
        prerequisites = [
            ("Addition", "Counting", 0.9, "Must understand counting before addition"),
            ("Subtraction", "Addition", 0.8, "Addition helps understand subtraction"),
            ("Multiplication", "Addition", 0.9, "Multiplication is repeated addition"),
            ("Division", "Multiplication", 0.9, "Division is inverse of multiplication"),
            ("Fractions", "Division", 0.7, "Division helps understand fractions"),
            ("Fractions", "Multiplication", 0.6, "Multiplication needed for fraction operations"),
            ("Decimals", "Fractions", 0.8, "Decimals are another way to express fractions"),
            ("Percentages", "Decimals", 0.7, "Percentages use decimal concepts"),
            ("Percentages", "Fractions", 0.6, "Percentages can be expressed as fractions"),
            ("Basic Algebra", "Addition", 0.9, "Algebra uses arithmetic operations"),
            ("Basic Algebra", "Subtraction", 0.9, "Algebra uses arithmetic operations"),
            ("Basic Algebra", "Multiplication", 0.8, "Algebra uses multiplication"),
            ("Basic Algebra", "Division", 0.7, "Algebra uses division"),
            ("Linear Equations", "Basic Algebra", 0.95, "Linear equations build on basic algebra")
        ]
        
        # Create prerequisite relationships
        for concept, prereq, strength, description in prerequisites:
            if concept in concept_ids and prereq in concept_ids:
                await self.graph.add_relationship(
                    "PREREQUISITE",
                    concept_ids[prereq],  # prerequisite -> concept
                    concept_ids[concept],
                    {"strength": strength, "description": description}
                )
                print(f"  🔗 {prereq} → {concept} (strength: {strength})")
        
        return concept_ids
    
    async def build_learning_graph(self, concept_ids: Dict[str, str]):
        """Build learning graphs that track student progress."""
        print("\n📚 Building Learning Graph...")
        
        # Create sample students
        students = [
            {
                "student_id": "student_001",
                "name": "Alice Johnson",
                "grade_level": 3,
                "learning_style": "visual"
            },
            {
                "student_id": "student_002", 
                "name": "Bob Smith",
                "grade_level": 4,
                "learning_style": "kinesthetic"
            },
            {
                "student_id": "student_003",
                "name": "Carol Davis",
                "grade_level": 5,
                "learning_style": "auditory"
            }
        ]
        
        student_ids = {}
        for student in students:
            node_id = await self.graph.add_node("Student", student)
            student_ids[student["student_id"]] = node_id
            print(f"  👤 Created student: {student['name']}")
        
        # Create learning sessions (student progress)
        learning_sessions = [
            # Alice's progress (Grade 3)
            ("student_001", "Counting", 0.95, 15, True),
            ("student_001", "Addition", 0.88, 25, True),
            ("student_001", "Subtraction", 0.82, 30, True),
            ("student_001", "Multiplication", 0.75, 45, False),  # Still learning
            
            # Bob's progress (Grade 4)
            ("student_002", "Counting", 1.0, 10, True),
            ("student_002", "Addition", 0.92, 20, True),
            ("student_002", "Subtraction", 0.89, 22, True),
            ("student_002", "Multiplication", 0.85, 40, True),
            ("student_002", "Division", 0.72, 50, False),  # Still learning
            
            # Carol's progress (Grade 5)
            ("student_003", "Counting", 1.0, 8, True),
            ("student_003", "Addition", 0.95, 15, True),
            ("student_003", "Subtraction", 0.93, 18, True),
            ("student_003", "Multiplication", 0.90, 35, True),
            ("student_003", "Division", 0.87, 42, True),
            ("student_003", "Fractions", 0.78, 60, False),  # Still learning
            ("student_003", "Decimals", 0.65, 45, False),  # Just started
        ]
        
        # Create learning session nodes and relationships
        for student_id, concept_name, score, time_spent, mastery in learning_sessions:
            if student_id in student_ids and concept_name in concept_ids:
                session_data = {
                    "session_id": f"session_{uuid.uuid4().hex[:8]}",
                    "student_id": student_id,
                    "concept_id": concept_ids[concept_name],
                    "timestamp": datetime.now().isoformat(),
                    "performance_score": score,
                    "time_spent": time_spent,
                    "mastery_achieved": mastery
                }
                
                session_id = await self.graph.add_node("LearningSession", session_data)
                
                # Create STUDIED relationship between student and concept
                await self.graph.add_relationship(
                    "STUDIED",
                    student_ids[student_id],
                    concept_ids[concept_name],
                    {
                        "session_count": 1,
                        "total_time": time_spent,
                        "average_score": score,
                        "mastery_level": 1.0 if mastery else score
                    }
                )
                
                mastery_status = "✅ Mastered" if mastery else "📖 Learning"
                print(f"  📊 {student_id} studied {concept_name}: {score:.2f} ({mastery_status})")
        
        # Create learning paths
        learning_paths = [
            {
                "path_id": "elementary_arithmetic",
                "name": "Elementary Arithmetic Path",
                "description": "Basic arithmetic skills for grades 1-4",
                "target_grade": 4,
                "concepts": ["Counting", "Addition", "Subtraction", "Multiplication", "Division"]
            },
            {
                "path_id": "fraction_mastery",
                "name": "Fraction Mastery Path", 
                "description": "Understanding fractions and decimals",
                "target_grade": 5,
                "concepts": ["Fractions", "Decimals", "Percentages"]
            },
            {
                "path_id": "algebra_prep",
                "name": "Algebra Preparation Path",
                "description": "Preparing for algebra with strong arithmetic foundation",
                "target_grade": 7,
                "concepts": ["Basic Algebra", "Linear Equations"]
            }
        ]
        
        # Create learning path nodes and relationships
        for path in learning_paths:
            path_data = {
                "path_id": path["path_id"],
                "name": path["name"],
                "description": path["description"],
                "target_grade": path["target_grade"]
            }
            
            path_id = await self.graph.add_node("LearningPath", path_data)
            
            # Link concepts to the learning path
            for i, concept_name in enumerate(path["concepts"]):
                if concept_name in concept_ids:
                    await self.graph.add_relationship(
                        "INCLUDES",
                        path_id,
                        concept_ids[concept_name],
                        {"order": i + 1, "is_optional": False}
                    )
            
            print(f"  🛤️  Created learning path: {path['name']}")
        
        return student_ids
    
    async def demonstrate_graph_queries(self, concept_ids: Dict[str, str], student_ids: Dict[str, str]):
        """Demonstrate various graph queries using Graphiti + Neo4j."""
        print("\n🔍 Demonstrating Graph Queries...")
        
        # Query 1: Find all prerequisites for a concept
        print("\n1. Finding prerequisites for 'Linear Equations':")
        if "Linear Equations" in concept_ids:
            prereq_query = """
            MATCH (prereq:MathConcept)-[:PREREQUISITE*]->(concept:MathConcept {name: 'Linear Equations'})
            RETURN prereq.name as prerequisite, prereq.difficulty_level as difficulty
            ORDER BY prereq.difficulty_level
            """
            results = await self.graph.query(prereq_query)
            for result in results:
                print(f"   📋 {result['prerequisite']} (difficulty: {result['difficulty']})")
        
        # Query 2: Find learning path for a student
        print("\n2. Finding recommended next concepts for Alice:")
        alice_query = """
        MATCH (student:Student {name: 'Alice Johnson'})-[:STUDIED]->(mastered:MathConcept)
        MATCH (next:MathConcept)-[:PREREQUISITE]->(mastered)
        WHERE NOT (student)-[:STUDIED]->(next)
        RETURN DISTINCT next.name as next_concept, next.difficulty_level as difficulty
        ORDER BY next.difficulty_level
        """
        results = await self.graph.query(alice_query)
        for result in results:
            print(f"   🎯 {result['next_concept']} (difficulty: {result['difficulty']})")
        
        # Query 3: Find students struggling with a concept
        print("\n3. Finding students who need help with multiplication:")
        struggling_query = """
        MATCH (student:Student)-[studied:STUDIED]->(concept:MathConcept {name: 'Multiplication'})
        WHERE studied.mastery_level < 0.8
        RETURN student.name as student_name, studied.average_score as score
        ORDER BY studied.average_score
        """
        results = await self.graph.query(struggling_query)
        for result in results:
            print(f"   🆘 {result['student_name']} (score: {result['score']:.2f})")
        
        # Query 4: Find optimal learning sequence
        print("\n4. Finding optimal learning sequence from Addition to Algebra:")
        sequence_query = """
        MATCH path = shortestPath((start:MathConcept {name: 'Addition'})-[:PREREQUISITE*]->(end:MathConcept {name: 'Basic Algebra'}))
        RETURN [node in nodes(path) | node.name] as learning_sequence
        """
        results = await self.graph.query(sequence_query)
        for result in results:
            sequence = " → ".join(result['learning_sequence'])
            print(f"   🛤️  {sequence}")
    
    async def demonstrate_graphiti_features(self):
        """Demonstrate Graphiti's intelligent features."""
        print("\n🤖 Demonstrating Graphiti Intelligence Features...")
        
        # Simulate adding educational content and let Graphiti extract relationships
        educational_content = [
            "Addition is the foundation of all arithmetic operations and must be mastered before multiplication.",
            "Students who struggle with fractions often lack a solid understanding of division concepts.",
            "Visual learners benefit from using manipulatives when learning multiplication tables.",
            "Algebra requires strong arithmetic skills, especially in addition, subtraction, and multiplication.",
            "Decimals and fractions are closely related - students should learn both concepts together."
        ]
        
        for content in educational_content:
            # Add content to Graphiti and let it extract entities and relationships
            await self.graph.add_message({
                "role": "educator",
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "type": "educational_insight"
            })
            print(f"   📝 Added insight: {content[:50]}...")
        
        # Query for automatically extracted relationships
        print("\n   🔍 Graphiti automatically extracted these relationships:")
        auto_relationships = await self.graph.get_relationships(limit=10)
        for rel in auto_relationships[:5]:  # Show first 5
            print(f"   🔗 {rel.get('type', 'RELATED')} relationship detected")
    
    async def cleanup(self):
        """Clean up test data."""
        print(f"\n🧹 Cleaning up test data for namespace: {self.namespace}")
        
        if self.graph:
            # Delete all nodes and relationships in our namespace
            cleanup_query = f"""
            MATCH (n) WHERE n.namespace = '{self.namespace}'
            DETACH DELETE n
            """
            await self.graph.query(cleanup_query)
        
        if self.neo4j_driver:
            self.neo4j_driver.close()


# Test fixtures and functions
@pytest.fixture(scope="session")
async def usage_checker():
    """Fixture to provide database usage checking."""
    return DatabaseUsageChecker()


@pytest.fixture(scope="session", autouse=True)
async def check_database_usage(usage_checker):
    """Check database usage before and after all tests."""
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE BEFORE GRAPHITI + NEO4J TESTS")
    print("="*80)
    
    # Check usage before tests
    usage_before = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_before, "Database Usage BEFORE Graphiti + Neo4j Tests")
    
    yield  # Run all tests
    
    print("\n" + "="*80)
    print("🔍 CHECKING DATABASE USAGE AFTER GRAPHITI + NEO4J TESTS")
    print("="*80)
    
    # Check usage after tests
    usage_after = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_after, "Database Usage AFTER Graphiti + Neo4j Tests")
    
    # Show the difference
    print("\n" + "="*80)
    print("📊 DATABASE USAGE CHANGES")
    print("="*80)
    usage_checker.print_usage_comparison(usage_before, usage_after)


@pytest.mark.asyncio
@pytest.mark.skipif(not GRAPHITI_AVAILABLE, reason="Graphiti is not available")
@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j is not available")
async def test_graphiti_neo4j_math_learning():
    """
    Test the complete Graphiti + Neo4j integration for math learning.
    
    This test demonstrates:
    1. Setting up Graphiti with Neo4j backend
    2. Building knowledge graphs for math concepts
    3. Building learning graphs for student progress
    4. Performing complex graph queries
    5. Using Graphiti's intelligent features
    """
    print("\n" + "="*80)
    print("🚀 TESTING GRAPHITI + NEO4J MATH LEARNING INTEGRATION")
    print("="*80)
    
    # Initialize the system
    math_system = MathLearningGraphitiNeo4j()
    
    try:
        # Initialize Graphiti + Neo4j
        await math_system.initialize()
        print("✅ Graphiti + Neo4j system initialized successfully")
        
        # Build knowledge graph
        concept_ids = await math_system.build_knowledge_graph()
        assert len(concept_ids) > 0, "Should create math concepts"
        print(f"✅ Knowledge graph built with {len(concept_ids)} concepts")
        
        # Build learning graph
        student_ids = await math_system.build_learning_graph(concept_ids)
        assert len(student_ids) > 0, "Should create students"
        print(f"✅ Learning graph built with {len(student_ids)} students")
        
        # Demonstrate graph queries
        await math_system.demonstrate_graph_queries(concept_ids, student_ids)
        print("✅ Graph queries demonstrated successfully")
        
        # Demonstrate Graphiti features
        await math_system.demonstrate_graphiti_features()
        print("✅ Graphiti intelligence features demonstrated")
        
        print("\n" + "="*80)
        print("🎉 GRAPHITI + NEO4J MATH LEARNING TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        raise
    finally:
        # Clean up
        await math_system.cleanup()


if __name__ == "__main__":
    """Run the test directly."""
    asyncio.run(test_graphiti_neo4j_math_learning()) 