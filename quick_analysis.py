import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'math_learning'))

from knowledge_graph.concept import Concept
from knowledge_graph.graph import KnowledgeGraph
import networkx as nx

# Import the build function directly
exec(open('math_learning/knowledge_graph/algebra_graph.py').read())

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

# Check connectivity
if nx.is_weakly_connected(graph.graph):
    print('Graph is weakly connected')
else:
    components = list(nx.weakly_connected_components(graph.graph))
    print(f'Graph has {len(components)} disconnected components')
    for i, comp in enumerate(components):
        print(f'  Component {i+1}: {len(comp)} nodes')

# Category distribution
categories = {}
for concept in graph.concepts.values():
    cat = concept.category
    categories[cat] = categories.get(cat, 0) + 1

print('\nCategory distribution:')
for cat, count in sorted(categories.items()):
    print(f'  {cat}: {count} concepts')

# Find isolated nodes
isolated = list(nx.isolates(graph.graph))
if isolated:
    print(f'\nIsolated concepts: {len(isolated)}')
    for concept_id in isolated[:5]:
        concept = graph.concepts[concept_id]
        print(f'  - {concept.name} ({concept_id})')

# Find concepts with no prerequisites (starting points)
no_prereqs = [c for c in graph.concepts.values() if len(c.prerequisites) == 0]
print(f'\nStarting concepts (no prerequisites): {len(no_prereqs)}')
for concept in no_prereqs[:5]:
    print(f'  - {concept.name} ({concept.id})')

# Find concepts with no dependents (endpoints)
no_dependents = [c for c in graph.concepts.values() if len(c.dependents) == 0]
print(f'\nEndpoint concepts (no dependents): {len(no_dependents)}')
for concept in no_dependents[:5]:
    print(f'  - {concept.name} ({concept.id})') 