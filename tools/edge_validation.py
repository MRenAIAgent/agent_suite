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

print('=== EDGE VALIDATION ANALYSIS ===')
print(f'Total concepts: {len(graph.concepts)}')
print(f'Total edges: {graph.graph.number_of_edges()}')

# Analyze all edges
prerequisite_edges = []
related_edges = []

for u, v, data in graph.graph.edges(data=True):
    edge_type = data.get('type', 'unknown')
    if edge_type == 'prerequisite':
        prerequisite_edges.append((u, v, data))
    elif edge_type == 'related':
        related_edges.append((u, v, data))

print(f'\nPrerequisite edges: {len(prerequisite_edges)}')
print(f'Related edges: {len(related_edges)}')

# Check prerequisite relationships for logical issues
print('\n=== PREREQUISITE RELATIONSHIP VALIDATION ===')
problematic_prereqs = []

for u, v, data in prerequisite_edges:
    prereq_concept = graph.concepts[u]
    dependent_concept = graph.concepts[v]
    
    # Check if prerequisite has higher difficulty than dependent
    if prereq_concept.difficulty > dependent_concept.difficulty:
        problematic_prereqs.append({
            'prereq': f"{prereq_concept.name} ({u})",
            'dependent': f"{dependent_concept.name} ({v})",
            'prereq_difficulty': prereq_concept.difficulty,
            'dependent_difficulty': dependent_concept.difficulty,
            'issue': 'Prerequisite has higher difficulty'
        })
    
    # Check if prerequisite takes longer to master than dependent
    if prereq_concept.time_to_master > dependent_concept.time_to_master:
        problematic_prereqs.append({
            'prereq': f"{prereq_concept.name} ({u})",
            'dependent': f"{dependent_concept.name} ({v})",
            'prereq_time': prereq_concept.time_to_master,
            'dependent_time': dependent_concept.time_to_master,
            'issue': 'Prerequisite takes longer to master'
        })

if problematic_prereqs:
    print(f"Found {len(problematic_prereqs)} potentially problematic prerequisite relationships:")
    for issue in problematic_prereqs[:10]:  # Show first 10
        print(f"  - {issue['prereq']} → {issue['dependent']}")
        print(f"    Issue: {issue['issue']}")
        if 'prereq_difficulty' in issue:
            print(f"    Difficulty: {issue['prereq_difficulty']} → {issue['dependent_difficulty']}")
        if 'prereq_time' in issue:
            print(f"    Time to master: {issue['prereq_time']} → {issue['dependent_time']}")
        print()
else:
    print("No difficulty/time issues found in prerequisite relationships.")

# Check for conceptual logic in prerequisites
print('\n=== CONCEPTUAL LOGIC VALIDATION ===')
conceptual_issues = []

# Define logical prerequisite patterns
logical_patterns = {
    # Basic arithmetic should come before algebra
    'arithmetic_before_algebra': [
        ('NS-06', 'AT-19'),  # multiplication_facts → distributive_property
        ('NS-07', 'NS-14'),  # division_facts → integer_multiplication_division
    ],
    # Coordinate plane before graphing
    'coordinates_before_graphing': [
        ('AT-28', 'LF-31'),  # coordinate_plane → slope_intercept_form
    ],
    # Basic equations before systems
    'equations_before_systems': [
        ('AT-24', 'LF-37'),  # two_step_equations → system_graphing
    ],
    # Exponent rules before polynomial operations
    'exponents_before_polynomials': [
        ('EXP-41', 'POL-50'),  # exponent_product_rule → multiply_monomial_polynomial
    ]
}

# Check each prerequisite edge for conceptual sense
for u, v, data in prerequisite_edges:
    prereq_concept = graph.concepts[u]
    dependent_concept = graph.concepts[v]
    
    # Print all prerequisite relationships for manual review
    print(f"{prereq_concept.name} ({u}) → {dependent_concept.name} ({v})")
    print(f"  Categories: {prereq_concept.category} → {dependent_concept.category}")
    print(f"  Difficulty: {prereq_concept.difficulty} → {dependent_concept.difficulty}")
    print()

print('\n=== RELATED RELATIONSHIP VALIDATION ===')
print("Checking related relationships for logical connections...")

for u, v, data in related_edges:
    concept1 = graph.concepts[u]
    concept2 = graph.concepts[v]
    
    print(f"{concept1.name} ({u}) ↔ {concept2.name} ({v})")
    print(f"  Categories: {concept1.category} ↔ {concept2.category}")
    print(f"  Difficulty: {concept1.difficulty} ↔ {concept2.difficulty}")
    print() 