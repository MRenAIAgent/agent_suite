#!/usr/bin/env python3
"""
Database Usage Checker and Mock Graphiti + Neo4j Math Learning Test

This script checks database usage and demonstrates the concept of using
Graphiti with Neo4j for building knowledge graphs and learning graphs
in math education.
"""

import asyncio
import sys
import os
import uuid
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the math_learning directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'math_learning'))

from database_usage_checker import DatabaseUsageChecker

# Mock Graphiti functionality for demonstration
class MockGraphitiNeo4j:
    """
    Mock implementation demonstrating Graphiti + Neo4j integration concepts
    for math learning knowledge graphs and learning graphs.
    """
    
    def __init__(self):
        self.namespace = f"math_learning_{uuid.uuid4().hex[:8]}"
        self.nodes = {}
        self.relationships = []
        
    async def initialize(self):
        """Initialize the mock system."""
        print("🔧 Initializing Mock Graphiti + Neo4j System...")
        print(f"   📝 Namespace: {self.namespace}")
        print("   ✅ System ready for knowledge graph building")
        
    async def create_math_concepts(self):
        """Create math concept nodes in the knowledge graph."""
        print("\n🧠 Building Math Knowledge Graph...")
        
        concepts = [
            {"name": "Counting", "difficulty": 0.1, "grade": 1, "type": "arithmetic"},
            {"name": "Addition", "difficulty": 0.2, "grade": 1, "type": "arithmetic"},
            {"name": "Subtraction", "difficulty": 0.25, "grade": 2, "type": "arithmetic"},
            {"name": "Multiplication", "difficulty": 0.4, "grade": 3, "type": "arithmetic"},
            {"name": "Division", "difficulty": 0.45, "grade": 4, "type": "arithmetic"},
            {"name": "Fractions", "difficulty": 0.6, "grade": 4, "type": "arithmetic"},
            {"name": "Decimals", "difficulty": 0.55, "grade": 5, "type": "arithmetic"},
            {"name": "Percentages", "difficulty": 0.65, "grade": 6, "type": "arithmetic"},
            {"name": "Basic Algebra", "difficulty": 0.7, "grade": 7, "type": "algebra"},
            {"name": "Linear Equations", "difficulty": 0.8, "grade": 8, "type": "algebra"},
        ]
        
        for concept in concepts:
            concept_id = f"concept_{uuid.uuid4().hex[:8]}"
            self.nodes[concept_id] = {
                "type": "MathConcept",
                "properties": concept
            }
            print(f"   ✅ Created: {concept['name']} (Grade {concept['grade']}, Difficulty: {concept['difficulty']})")
        
        return len(concepts)
    
    async def create_prerequisite_relationships(self):
        """Create prerequisite relationships between concepts."""
        print("\n🔗 Building Prerequisite Relationships...")
        
        prerequisites = [
            ("Addition", "Counting", 0.9),
            ("Subtraction", "Addition", 0.8),
            ("Multiplication", "Addition", 0.9),
            ("Division", "Multiplication", 0.9),
            ("Fractions", "Division", 0.7),
            ("Decimals", "Fractions", 0.8),
            ("Percentages", "Decimals", 0.7),
            ("Basic Algebra", "Addition", 0.9),
            ("Linear Equations", "Basic Algebra", 0.95),
        ]
        
        for concept, prereq, strength in prerequisites:
            relationship = {
                "type": "PREREQUISITE",
                "from": prereq,
                "to": concept,
                "strength": strength
            }
            self.relationships.append(relationship)
            print(f"   🔗 {prereq} → {concept} (strength: {strength})")
        
        return len(prerequisites)
    
    async def create_student_progress(self):
        """Create student learning progress data."""
        print("\n📚 Building Student Learning Graph...")
        
        students = [
            {"id": "student_001", "name": "Alice Johnson", "grade": 3},
            {"id": "student_002", "name": "Bob Smith", "grade": 4},
            {"id": "student_003", "name": "Carol Davis", "grade": 5},
        ]
        
        # Create student nodes
        for student in students:
            student_id = f"student_{uuid.uuid4().hex[:8]}"
            self.nodes[student_id] = {
                "type": "Student",
                "properties": student
            }
            print(f"   👤 Created student: {student['name']} (Grade {student['grade']})")
        
        # Create learning progress
        progress_data = [
            ("Alice Johnson", "Counting", 0.95, True),
            ("Alice Johnson", "Addition", 0.88, True),
            ("Alice Johnson", "Multiplication", 0.75, False),
            ("Bob Smith", "Addition", 0.92, True),
            ("Bob Smith", "Multiplication", 0.85, True),
            ("Bob Smith", "Division", 0.72, False),
            ("Carol Davis", "Fractions", 0.78, False),
            ("Carol Davis", "Decimals", 0.65, False),
        ]
        
        for student_name, concept, score, mastery in progress_data:
            progress = {
                "type": "STUDIED",
                "student": student_name,
                "concept": concept,
                "score": score,
                "mastery": mastery
            }
            self.relationships.append(progress)
            status = "✅ Mastered" if mastery else "📖 Learning"
            print(f"   📊 {student_name} studied {concept}: {score:.2f} ({status})")
        
        return len(students)
    
    async def demonstrate_graph_queries(self):
        """Demonstrate the types of queries possible with Graphiti + Neo4j."""
        print("\n🔍 Demonstrating Graph Query Capabilities...")
        
        print("\n1. 📋 Prerequisites for Linear Equations:")
        print("   Query: MATCH (p)-[:PREREQUISITE*]->(c {name: 'Linear Equations'}) RETURN p.name")
        print("   Results: Basic Algebra → Addition → Counting")
        
        print("\n2. 🎯 Next concepts for Alice (Grade 3):")
        print("   Query: Find concepts Alice hasn't mastered but has prerequisites for")
        print("   Results: Division (needs to master Multiplication first)")
        
        print("\n3. 🆘 Students struggling with concepts:")
        print("   Query: MATCH (s)-[r:STUDIED]->(c) WHERE r.mastery = false RETURN s.name, c.name")
        print("   Results:")
        print("     - Alice: Multiplication (75%)")
        print("     - Bob: Division (72%)")
        print("     - Carol: Fractions (78%), Decimals (65%)")
        
        print("\n4. 🛤️ Learning path optimization:")
        print("   Query: shortestPath((start)-[:PREREQUISITE*]->(end))")
        print("   Results: Counting → Addition → Multiplication → Division → Fractions")
        
        print("\n5. 🤖 Graphiti Intelligence Features:")
        print("   - Automatic relationship inference from educational content")
        print("   - Learning pattern recognition across students")
        print("   - Adaptive prerequisite strength adjustment")
        print("   - Personalized learning path recommendations")
    
    async def get_statistics(self):
        """Get statistics about the knowledge graph."""
        concept_count = len([n for n in self.nodes.values() if n["type"] == "MathConcept"])
        student_count = len([n for n in self.nodes.values() if n["type"] == "Student"])
        relationship_count = len(self.relationships)
        
        return {
            "concepts": concept_count,
            "students": student_count,
            "relationships": relationship_count,
            "total_nodes": len(self.nodes)
        }


async def main():
    """Main function to run the database usage check and mock test."""
    print("🚀 Database Usage Check + Mock Graphiti + Neo4j Math Learning Test")
    print("=" * 80)
    
    # Initialize usage checker
    usage_checker = DatabaseUsageChecker()
    
    # Check usage before test
    print("\n📊 CHECKING DATABASE USAGE BEFORE TEST")
    print("-" * 50)
    usage_before = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_before, "Before Mock Test")
    
    # Run the mock Graphiti + Neo4j test
    print("\n🧪 RUNNING MOCK GRAPHITI + NEO4J MATH LEARNING TEST")
    print("-" * 50)
    
    try:
        # Initialize mock system
        mock_system = MockGraphitiNeo4j()
        await mock_system.initialize()
        
        # Build knowledge graph
        concept_count = await mock_system.create_math_concepts()
        print(f"   ✅ Created {concept_count} math concepts")
        
        # Build prerequisite relationships
        prereq_count = await mock_system.create_prerequisite_relationships()
        print(f"   ✅ Created {prereq_count} prerequisite relationships")
        
        # Build learning graph
        student_count = await mock_system.create_student_progress()
        print(f"   ✅ Created {student_count} students with learning progress")
        
        # Demonstrate queries
        await mock_system.demonstrate_graph_queries()
        
        # Get final statistics
        stats = await mock_system.get_statistics()
        print(f"\n📈 FINAL GRAPH STATISTICS:")
        print(f"   📚 Math Concepts: {stats['concepts']}")
        print(f"   👥 Students: {stats['students']}")
        print(f"   🔗 Relationships: {stats['relationships']}")
        print(f"   📊 Total Nodes: {stats['total_nodes']}")
        
        print("\n✅ Mock test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return
    
    # Check usage after test
    print("\n📊 CHECKING DATABASE USAGE AFTER TEST")
    print("-" * 50)
    usage_after = await usage_checker.check_both_databases()
    usage_checker.print_usage_report(usage_after, "After Mock Test")
    
    # Show the difference
    print("\n📈 DATABASE USAGE CHANGES")
    print("-" * 50)
    
    # Simple comparison
    neo4j_before = usage_before.get('neo4j', {})
    neo4j_after = usage_after.get('neo4j', {})
    qdrant_before = usage_before.get('qdrant', {})
    qdrant_after = usage_after.get('qdrant', {})
    
    print("🔗 Neo4j Changes:")
    if neo4j_before.get('status') == 'connected' and neo4j_after.get('status') == 'connected':
        node_diff = neo4j_after.get('node_count', 0) - neo4j_before.get('node_count', 0)
        rel_diff = neo4j_after.get('relationship_count', 0) - neo4j_before.get('relationship_count', 0)
        print(f"   📊 Nodes: {node_diff:+d}")
        print(f"   🔗 Relationships: {rel_diff:+d}")
    else:
        print("   ❌ Could not compare (connection issues)")
    
    print("\n🔍 Qdrant Changes:")
    if qdrant_before.get('status') == 'connected' and qdrant_after.get('status') == 'connected':
        collection_diff = qdrant_after.get('total_collections', 0) - qdrant_before.get('total_collections', 0)
        vector_diff = qdrant_after.get('total_vectors', 0) - qdrant_before.get('total_vectors', 0)
        print(f"   📦 Collections: {collection_diff:+d}")
        print(f"   🔢 Vectors: {vector_diff:+d}")
    else:
        print("   ❌ Could not compare (connection issues)")
    
    print("\n" + "=" * 80)
    print("🎉 GRAPHITI + NEO4J CONCEPT DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print("\n📝 WHAT THIS DEMONSTRATES:")
    print("   🧠 Knowledge Graph: Math concepts with prerequisite relationships")
    print("   📚 Learning Graph: Student progress and mastery tracking")
    print("   🔍 Graph Queries: Complex relationship traversal and analysis")
    print("   🤖 Graphiti Features: Intelligent memory and relationship inference")
    print("   💾 Neo4j Storage: Persistent, scalable graph database backend")
    print("\n🔧 TO USE REAL GRAPHITI + NEO4J:")
    print("   1. Install: pip install graphiti-core neo4j")
    print("   2. Configure Graphiti with Neo4j backend")
    print("   3. Replace mock implementation with actual Graphiti calls")
    print("   4. Use Neo4j Cypher queries for complex graph operations")


if __name__ == "__main__":
    asyncio.run(main()) 