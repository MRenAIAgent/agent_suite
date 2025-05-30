import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'math_learning'))

from knowledge_graph.concept import Concept
from knowledge_graph.graph import KnowledgeGraph
from knowledge_graph.algebra_graph import build_algebra_knowledge_graph
import networkx as nx

# Build the graph
graph = build_algebra_knowledge_graph()

print('=== KNOWLEDGE GRAPH ANALYSIS ===')
print(f'Total concepts: {len(graph.concepts)}')
print(f'Total edges: {graph.graph.number_of_edges()}')
print(f'Graph density: {nx.density(graph.graph):.3f}')

# Analyze edge types
prerequisite_edges = [(u,v) for u,v,d in graph.graph.edges(data=True) if d.get('type') == 'prerequisite']
related_edges = [(u,v) for u,v,d in graph.graph.edges(data=True) if d.get('type') == 'related']

print(f'Prerequisite edges: {len(prerequisite_edges)}')
print(f'Related edges: {len(related_edges)}')

# Category distribution
categories = {}
for concept in graph.concepts.values():
    cat = concept.category
    categories[cat] = categories.get(cat, 0) + 1

print('\n=== CATEGORY DISTRIBUTION ===')
for cat, count in sorted(categories.items()):
    print(f'{cat}: {count} concepts')

# Difficulty distribution
difficulties = {}
for concept in graph.concepts.values():
    diff = concept.difficulty
    difficulties[diff] = difficulties.get(diff, 0) + 1

print('\n=== DIFFICULTY DISTRIBUTION ===')
for diff in sorted(difficulties.keys()):
    print(f'Level {diff}: {difficulties[diff]} concepts')

# Most connected concepts
print('\n=== MOST CONNECTED CONCEPTS ===')
centrality = nx.degree_centrality(graph.graph)
top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
for concept_id, cent in top_concepts:
    concept = graph.concepts[concept_id]
    print(f'{concept.name} ({concept_id}): {cent:.3f} centrality')

# Check for isolated nodes
isolated = list(nx.isolates(graph.graph))
if isolated:
    print(f'\n=== ISOLATED CONCEPTS ===')
    for concept_id in isolated:
        concept = graph.concepts[concept_id]
        print(f'{concept.name} ({concept_id})')
else:
    print('\n=== NO ISOLATED CONCEPTS ===')

# Check graph connectivity
if nx.is_weakly_connected(graph.graph):
    print('\n=== GRAPH IS WEAKLY CONNECTED ===')
else:
    print('\n=== GRAPH HAS DISCONNECTED COMPONENTS ===')
    components = list(nx.weakly_connected_components(graph.graph))
    print(f'Number of components: {len(components)}')
    for i, component in enumerate(components):
        print(f'Component {i+1}: {len(component)} nodes')

# Analyze prerequisite chains
print('\n=== PREREQUISITE CHAIN ANALYSIS ===')
# Find concepts with no prerequisites (starting points)
starting_concepts = [c for c in graph.concepts.values() if len(c.prerequisites) == 0]
print(f'Starting concepts (no prerequisites): {len(starting_concepts)}')
for concept in starting_concepts[:5]:  # Show first 5
    print(f'  - {concept.name} ({concept.id})')

# Find concepts with no dependents (endpoints)
ending_concepts = [c for c in graph.concepts.values() if len(c.dependents) == 0]
print(f'Ending concepts (no dependents): {len(ending_concepts)}')
for concept in ending_concepts[:5]:  # Show first 5
    print(f'  - {concept.name} ({concept.id})')

# Find longest prerequisite chains
def find_longest_path_from(start_id, visited=None):
    if visited is None:
        visited = set()
    if start_id in visited:
        return 0
    visited.add(start_id)
    
    concept = graph.concepts[start_id]
    if not concept.dependents:
        return 1
    
    max_length = 0
    for dependent_id in concept.dependents:
        length = find_longest_path_from(dependent_id, visited.copy())
        max_length = max(max_length, length)
    
    return 1 + max_length

print('\n=== LONGEST PREREQUISITE CHAINS ===')
chain_lengths = {}
for concept_id in graph.concepts:
    if len(graph.concepts[concept_id].prerequisites) == 0:  # Starting concept
        length = find_longest_path_from(concept_id)
        chain_lengths[concept_id] = length

sorted_chains = sorted(chain_lengths.items(), key=lambda x: x[1], reverse=True)
for concept_id, length in sorted_chains[:5]:
    concept = graph.concepts[concept_id]
    print(f'{concept.name} ({concept_id}): {length} steps') 