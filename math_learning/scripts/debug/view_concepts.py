"""
View all concepts in the algebra knowledge graph.
"""

import os
import sys

# Ensure proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph.algebra_graph import build_algebra_knowledge_graph

def main():
    # Build the graph
    graph = build_algebra_knowledge_graph()
    
    # Print total number of concepts
    print(f'Total concepts: {len(graph.concepts)}')
    
    # Elementary School Concepts (Difficulty 1-2)
    print("\nELEMENTARY SCHOOL ALGEBRA CONCEPTS (K-5):")
    print("=" * 50)
    for concept_id, concept in graph.concepts.items():
        if concept.difficulty <= 2:
            print(f"- {concept.name} (Difficulty: {concept.difficulty}/5, Category: {concept.category})")
            print(f"  Description: {concept.description}")
    
    # Middle School Concepts (Difficulty 3)
    print("\nMIDDLE SCHOOL ALGEBRA CONCEPTS (6-8):")
    print("=" * 50)
    for concept_id, concept in graph.concepts.items():
        if concept.difficulty == 3:
            print(f"- {concept.name} (Difficulty: {concept.difficulty}/5, Category: {concept.category})")
            print(f"  Description: {concept.description}")
    
    # High School Concepts (Difficulty 4-5)
    print("\nHIGH SCHOOL ALGEBRA CONCEPTS (9-12):")
    print("=" * 50)
    for concept_id, concept in graph.concepts.items():
        if concept.difficulty >= 4:
            print(f"- {concept.name} (Difficulty: {concept.difficulty}/5, Category: {concept.category})")
            print(f"  Description: {concept.description}")
    
    # Print most central concepts
    central_concepts = graph.get_central_concepts(10)
    print("\nMOST CENTRAL CONCEPTS:")
    print("=" * 50)
    for concept in central_concepts:
        print(f"- {concept.name} (Centrality: {graph.calculate_centrality(concept.id):.2f})")

if __name__ == "__main__":
    main() 