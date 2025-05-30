#!/usr/bin/env python3
"""
Direct analysis of the knowledge graph without circular imports.
"""

import sys
import os
import networkx as nx

# Add the math_learning directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'math_learning'))

# Import directly from the modules
from knowledge_graph.concept import Concept
from knowledge_graph.graph import KnowledgeGraph

# Manually build a simplified version for analysis
def analyze_knowledge_graph():
    """Analyze the knowledge graph structure."""
    
    # Create the graph instance
    graph = KnowledgeGraph(name="K-12 Algebra")
    
    # Category prefixes
    NS = "NS"   # Number Sense
    AT = "AT"   # Algebraic Thinking  
    LF = "LF"   # Linear Functions
    EXP = "EXP" # Exponential
    POL = "POL" # Polynomials
    Q = "Q"     # Quadratics
    FN = "FN"   # Function Tools
    
    # Sample concepts for analysis (subset of the full graph)
    concepts = [
        # Number Sense
        Concept("Counting Up/Down", "Count forward or backward by 1s within 1,000.", 1, 30, "Number Sense", f"{NS}-01"),
        Concept("Place Value to 1,000,000", "Identify the value of each digit up to the millions place.", 1, 60, "Number Sense", f"{NS}-02"),
        Concept("Order of Operations", "Apply PEMDAS to simplify numeric expressions.", 3, 90, "Number Sense", f"{NS}-16"),
        
        # Algebraic Thinking
        Concept("Translate Expressions", "Convert verbal phrases to algebraic expressions.", 3, 90, "Pre-Algebra", f"{AT}-17"),
        Concept("Two-Step Linear Equations", "Solve ax+b=c.", 3, 120, "Pre-Algebra", f"{AT}-24"),
        Concept("Multi-Step Linear Equations", "Equations requiring distribution, combining terms, etc.", 4, 150, "Pre-Algebra", f"{AT}-25"),
        
        # Linear Functions
        Concept("Slope from Two Points", "Compute m = (y₂−y₁)/(x₂−x₁).", 3, 90, "Linear", f"{LF}-30"),
        Concept("Slope-Intercept Form", "Write/graph y=mx+b.", 3, 120, "Linear", f"{LF}-31"),
        Concept("System – Graphing", "Solve a linear system by intersecting lines.", 3, 120, "Linear", f"{LF}-37"),
        
        # Exponential
        Concept("Exponent Product Rule", "a^m · a^n = a^{m+n}.", 2, 90, "Exponential", f"{EXP}-41"),
        Concept("Negative Integer Exponents", "a^{−n}=1/a^n.", 2, 90, "Exponential", f"{EXP}-45"),
        
        # Polynomials
        Concept("Add/Subtract Polynomials", "Combine like terms across polynomials.", 2, 90, "Polynomials", f"{POL}-49"),
        Concept("Factor Trinomials a=1", "Factor x²+bx+c.", 3, 120, "Polynomials", f"{POL}-53"),
        
        # Quadratics
        Concept("Quadratic Formula", "Solve ax²+bx+c=0 using −b±√(b²−4ac)/2a.", 3, 120, "Quadratic", f"{Q}-59"),
        
        # Function Tools
        Concept("Function Notation", "Use f(x) and evaluate.", 2, 90, "Function Tools", f"{FN}-62"),
    ]
    
    # Add concepts to graph
    for concept in concepts:
        graph.add_concept(concept)
    
    # Add some sample prerequisite relationships
    graph.add_prerequisite(f"{NS}-01", f"{NS}-02")    # counting → place_value
    graph.add_prerequisite(f"{NS}-16", f"{AT}-24")    # order_of_operations → two_step_equations
    graph.add_prerequisite(f"{AT}-24", f"{AT}-25")    # two_step_equations → multi_step_equations
    graph.add_prerequisite(f"{AT}-25", f"{LF}-30")    # multi_step_equations → slope_two_points
    graph.add_prerequisite(f"{LF}-30", f"{LF}-31")    # slope_two_points → slope_intercept_form
    graph.add_prerequisite(f"{LF}-31", f"{LF}-37")    # slope_intercept_form → system_graphing
    graph.add_prerequisite(f"{EXP}-41", f"{EXP}-45")  # exponent_product_rule → negative_exponents
    graph.add_prerequisite(f"{POL}-49", f"{POL}-53")  # add_subtract_polynomials → factor_trinomials_a1
    graph.add_prerequisite(f"{LF}-31", f"{FN}-62")    # slope_intercept_form → function_notation
    
    # Add some related relationships
    graph.add_related(f"{LF}-30", f"{LF}-31")         # slope concepts are related
    graph.add_related(f"{POL}-53", f"{Q}-59")         # factoring and quadratic formula are related
    
    return graph

def print_analysis(graph):
    """Print detailed analysis of the knowledge graph."""
    
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
    top_concepts = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
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
    starting_concepts = [c for c in graph.concepts.values() if len(c.prerequisites) == 0]
    ending_concepts = [c for c in graph.concepts.values() if len(c.dependents) == 0]
    
    print(f'Starting concepts (no prerequisites): {len(starting_concepts)}')
    for concept in starting_concepts[:3]:
        print(f'  - {concept.name} ({concept.id})')
    
    print(f'Ending concepts (no dependents): {len(ending_concepts)}')
    for concept in ending_concepts[:3]:
        print(f'  - {concept.name} ({concept.id})')

if __name__ == "__main__":
    graph = analyze_knowledge_graph()
    print_analysis(graph) 