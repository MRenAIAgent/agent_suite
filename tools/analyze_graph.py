import sys
sys.path.append('/Users/minren/code/agent_suite')

from math_learning.knowledge_graph.algebra_graph import build_algebra_knowledge_graph
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