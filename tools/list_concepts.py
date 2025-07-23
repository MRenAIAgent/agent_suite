"""
List all concepts in the algebra knowledge graph.
"""

from knowledge_graph.algebra_graph import build_algebra_knowledge_graph

def main():
    # Build the graph
    graph = build_algebra_knowledge_graph()
    
    # Print total number of concepts
    print(f'Total concepts: {len(graph.concepts)}')
    
    # Group concepts by difficulty
    concepts_by_difficulty = {1: [], 2: [], 3: [], 4: [], 5: []}
    for concept_id, concept in graph.concepts.items():
        concepts_by_difficulty[concept.difficulty].append(concept)
    
    # Print concepts grouped by difficulty
    print("\nAll concepts by difficulty level:")
    for difficulty in range(1, 6):
        concepts = concepts_by_difficulty[difficulty]
        if concepts:
            print(f"\nDifficulty Level {difficulty}/5:")
            print("-" * 30)
            for concept in sorted(concepts, key=lambda x: x.name):
                print(f"- {concept.name} (Category: {concept.category})")
                print(f"  Description: {concept.description}")
                if concept.prerequisites:
                    prereqs = [graph.concepts[p].name for p in concept.prerequisites if p in graph.concepts]
                    print(f"  Prerequisites: {', '.join(prereqs)}")
    
    # Print most central concepts
    central_concepts = graph.get_central_concepts(10)
    print("\nMost central concepts:")
    for concept in central_concepts:
        print(f"- {concept.name} (Centrality: {graph.calculate_centrality(concept.id):.2f})")

if __name__ == "__main__":
    main() 