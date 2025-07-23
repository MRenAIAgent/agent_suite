"""
Simple graph operations tests for math_learning system.

Tests basic graph functionality without complex dependencies.
"""

import pytest
import sys
import os
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, deque

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)


class SimpleGraphNode:
    """Simple graph node for testing."""
    
    def __init__(self, node_id: str, name: str, data: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.name = name
        self.data = data or {}
        self.connections = set()
    
    def add_connection(self, other_node_id: str):
        """Add a connection to another node."""
        self.connections.add(other_node_id)
    
    def remove_connection(self, other_node_id: str):
        """Remove a connection to another node."""
        self.connections.discard(other_node_id)
    
    def get_connections(self) -> Set[str]:
        """Get all connections."""
        return self.connections.copy()


class SimpleGraph:
    """Simple graph implementation for testing."""
    
    def __init__(self):
        self.nodes: Dict[str, SimpleGraphNode] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}  # edge_id -> edge_data
    
    def add_node(self, node_id: str, name: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Add a node to the graph."""
        if node_id in self.nodes:
            return False
        
        self.nodes[node_id] = SimpleGraphNode(node_id, name, data)
        return True
    
    def add_edge(self, from_node: str, to_node: str, edge_type: str = "related", 
                 weight: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add an edge between two nodes."""
        if from_node not in self.nodes or to_node not in self.nodes:
            return False
        
        edge_id = f"{from_node}->{to_node}"
        self.edges[edge_id] = {
            "from": from_node,
            "to": to_node,
            "type": edge_type,
            "weight": weight,
            "metadata": metadata or {}
        }
        
        # Update node connections
        self.nodes[from_node].add_connection(to_node)
        
        return True
    
    def get_node(self, node_id: str) -> Optional[SimpleGraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[str]:
        """Get all neighbors of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return list(node.connections)
    
    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find a path between two nodes using BFS."""
        if start not in self.nodes or end not in self.nodes:
            return None
        
        if start == end:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            for neighbor in self.get_neighbors(current):
                if neighbor == end:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_all_nodes(self) -> List[SimpleGraphNode]:
        """Get all nodes in the graph."""
        return list(self.nodes.values())
    
    def get_edges_from(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all edges from a specific node."""
        return [edge for edge in self.edges.values() if edge["from"] == node_id]
    
    def get_edges_to(self, node_id: str) -> List[Dict[str, Any]]:
        """Get all edges to a specific node."""
        return [edge for edge in self.edges.values() if edge["to"] == node_id]
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all its connections."""
        if node_id not in self.nodes:
            return False
        
        # Remove all edges involving this node
        edges_to_remove = [
            edge_id for edge_id, edge in self.edges.items()
            if edge["from"] == node_id or edge["to"] == node_id
        ]
        
        for edge_id in edges_to_remove:
            del self.edges[edge_id]
        
        # Remove from other nodes' connections
        for node in self.nodes.values():
            node.remove_connection(node_id)
        
        # Remove the node itself
        del self.nodes[node_id]
        return True


class MathConceptGraph(SimpleGraph):
    """Graph specifically for math concepts."""
    
    def add_math_concept(self, concept_id: str, name: str, difficulty: int = 1,
                        prerequisites: Optional[List[str]] = None) -> bool:
        """Add a math concept to the graph."""
        data = {
            "difficulty": difficulty,
            "prerequisites": prerequisites or [],
            "type": "math_concept"
        }
        
        success = self.add_node(concept_id, name, data)
        
        # Add prerequisite relationships
        if success and prerequisites:
            for prereq in prerequisites:
                if prereq in self.nodes:
                    self.add_edge(prereq, concept_id, "prerequisite", weight=0.8)
        
        return success
    
    def get_prerequisites(self, concept_id: str) -> List[str]:
        """Get all prerequisites for a concept."""
        prereq_edges = [
            edge for edge in self.edges.values()
            if edge["to"] == concept_id and edge["type"] == "prerequisite"
        ]
        return [edge["from"] for edge in prereq_edges]
    
    def get_dependents(self, concept_id: str) -> List[str]:
        """Get all concepts that depend on this one."""
        dependent_edges = [
            edge for edge in self.edges.values()
            if edge["from"] == concept_id and edge["type"] == "prerequisite"
        ]
        return [edge["to"] for edge in dependent_edges]
    
    def get_learning_path(self, start_concept: str, end_concept: str) -> Optional[List[str]]:
        """Get a learning path between two concepts."""
        return self.find_path(start_concept, end_concept)
    
    def get_concepts_by_difficulty(self, difficulty: int) -> List[str]:
        """Get all concepts at a specific difficulty level."""
        return [
            node.node_id for node in self.nodes.values()
            if node.data.get("difficulty") == difficulty
        ]


class TestSimpleGraphOperations:
    """Test simple graph operations."""
    
    def test_graph_creation(self):
        """Test creating a simple graph."""
        graph = SimpleGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
    
    def test_add_nodes(self):
        """Test adding nodes to the graph."""
        graph = SimpleGraph()
        
        # Add a node
        result = graph.add_node("node1", "First Node", {"type": "test"})
        assert result is True
        assert len(graph.nodes) == 1
        
        node = graph.get_node("node1")
        assert node is not None
        assert node.name == "First Node"
        assert node.data["type"] == "test"
        
        # Try to add duplicate
        result = graph.add_node("node1", "Duplicate", {})
        assert result is False
        assert len(graph.nodes) == 1
    
    def test_add_edges(self):
        """Test adding edges between nodes."""
        graph = SimpleGraph()
        
        # Add nodes first
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")
        
        # Add edge
        result = graph.add_edge("A", "B", "connects", weight=0.5)
        assert result is True
        assert len(graph.edges) == 1
        
        # Check connections
        neighbors = graph.get_neighbors("A")
        assert "B" in neighbors
        
        # Try to add edge to non-existent node
        result = graph.add_edge("A", "C", "invalid")
        assert result is False
        assert len(graph.edges) == 1
    
    def test_path_finding(self):
        """Test finding paths between nodes."""
        graph = SimpleGraph()
        
        # Create a simple path: A -> B -> C
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")
        graph.add_node("C", "Node C")
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        # Find path
        path = graph.find_path("A", "C")
        assert path is not None
        assert path == ["A", "B", "C"]
        
        # Test direct path
        path = graph.find_path("A", "B")
        assert path == ["A", "B"]
        
        # Test same node
        path = graph.find_path("A", "A")
        assert path == ["A"]
        
        # Test no path
        graph.add_node("D", "Node D")  # Isolated node
        path = graph.find_path("A", "D")
        assert path is None
    
    def test_node_removal(self):
        """Test removing nodes from the graph."""
        graph = SimpleGraph()
        
        # Create graph: A -> B -> C
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")
        graph.add_node("C", "Node C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        
        # Remove middle node
        result = graph.remove_node("B")
        assert result is True
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 0  # Both edges removed
        
        # Verify path is broken
        path = graph.find_path("A", "C")
        assert path is None
        
        # Try to remove non-existent node
        result = graph.remove_node("X")
        assert result is False
    
    def test_math_concept_graph(self):
        """Test the math concept graph functionality."""
        graph = MathConceptGraph()
        
        # Add basic math concepts
        graph.add_math_concept("arithmetic", "Basic Arithmetic", difficulty=1)
        graph.add_math_concept("algebra", "Algebra", difficulty=2, prerequisites=["arithmetic"])
        graph.add_math_concept("calculus", "Calculus", difficulty=4, prerequisites=["algebra"])
        
        assert len(graph.nodes) == 3
        
        # Test prerequisites
        prereqs = graph.get_prerequisites("algebra")
        assert "arithmetic" in prereqs
        
        prereqs = graph.get_prerequisites("calculus")
        assert "algebra" in prereqs
        
        # Test dependents
        dependents = graph.get_dependents("arithmetic")
        assert "algebra" in dependents
    
    def test_learning_path_generation(self):
        """Test generating learning paths."""
        graph = MathConceptGraph()
        
        # Create a learning sequence
        concepts = [
            ("numbers", "Number Sense", 1, []),
            ("addition", "Addition", 1, ["numbers"]),
            ("multiplication", "Multiplication", 2, ["addition"]),
            ("fractions", "Fractions", 3, ["multiplication"]),
            ("algebra", "Basic Algebra", 4, ["fractions"])
        ]
        
        for concept_id, name, difficulty, prereqs in concepts:
            graph.add_math_concept(concept_id, name, difficulty, prereqs)
        
        # Test learning path
        path = graph.get_learning_path("numbers", "algebra")
        assert path is not None
        assert len(path) == 5
        assert path[0] == "numbers"
        assert path[-1] == "algebra"
        
        # Test intermediate path
        path = graph.get_learning_path("addition", "fractions")
        assert path is not None
        assert "multiplication" in path
    
    def test_difficulty_queries(self):
        """Test querying concepts by difficulty."""
        graph = MathConceptGraph()
        
        # Add concepts with different difficulties
        graph.add_math_concept("basic1", "Basic Concept 1", difficulty=1)
        graph.add_math_concept("basic2", "Basic Concept 2", difficulty=1)
        graph.add_math_concept("intermediate", "Intermediate Concept", difficulty=2)
        graph.add_math_concept("advanced", "Advanced Concept", difficulty=3)
        
        # Test difficulty queries
        basic_concepts = graph.get_concepts_by_difficulty(1)
        assert len(basic_concepts) == 2
        assert "basic1" in basic_concepts
        assert "basic2" in basic_concepts
        
        intermediate_concepts = graph.get_concepts_by_difficulty(2)
        assert len(intermediate_concepts) == 1
        assert "intermediate" in intermediate_concepts
        
        expert_concepts = graph.get_concepts_by_difficulty(5)
        assert len(expert_concepts) == 0
    
    def test_graph_analysis(self):
        """Test analyzing graph structure."""
        graph = MathConceptGraph()
        
        # Create a more complex graph
        concepts = [
            ("A", "Concept A", 1, []),
            ("B", "Concept B", 1, []),
            ("C", "Concept C", 2, ["A", "B"]),
            ("D", "Concept D", 2, ["A"]),
            ("E", "Concept E", 3, ["C", "D"])
        ]
        
        for concept_id, name, difficulty, prereqs in concepts:
            graph.add_math_concept(concept_id, name, difficulty, prereqs)
        
        # Test getting all nodes
        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == 5
        
        # Test edge queries
        edges_from_a = graph.get_edges_from("A")
        assert len(edges_from_a) == 2  # A -> C, A -> D
        
        edges_to_c = graph.get_edges_to("C")
        assert len(edges_to_c) == 2  # A -> C, B -> C
        
        # Test multiple prerequisites
        prereqs_e = graph.get_prerequisites("E")
        assert len(prereqs_e) == 2
        assert "C" in prereqs_e
        assert "D" in prereqs_e
    
    def test_graph_modification(self):
        """Test modifying the graph after creation."""
        graph = MathConceptGraph()
        
        # Initial setup
        graph.add_math_concept("basic", "Basic Math", difficulty=1)
        graph.add_math_concept("advanced", "Advanced Math", difficulty=3, prerequisites=["basic"])
        
        # Add intermediate concept
        graph.add_math_concept("intermediate", "Intermediate Math", difficulty=2)
        
        # Manually add edges to insert intermediate concept
        graph.add_edge("basic", "intermediate", "prerequisite", weight=0.8)
        graph.add_edge("intermediate", "advanced", "prerequisite", weight=0.8)
        
        # Test new path
        path = graph.get_learning_path("basic", "advanced")
        assert path is not None
        # Note: BFS might find the direct path or the longer path depending on graph structure
        assert "basic" in path
        assert "advanced" in path
    
    def test_graph_performance(self):
        """Test graph performance with larger datasets."""
        graph = MathConceptGraph()
        
        # Add many concepts
        num_concepts = 50
        for i in range(num_concepts):
            prereqs = [f"concept_{j}" for j in range(max(0, i-2), i) if j >= 0]
            graph.add_math_concept(f"concept_{i}", f"Concept {i}", 
                                 difficulty=(i // 10) + 1, prerequisites=prereqs)
        
        assert len(graph.nodes) == num_concepts
        
        # Test path finding
        path = graph.get_learning_path("concept_0", "concept_49")
        assert path is not None
        assert len(path) >= 2
        
        # Test queries
        all_nodes = graph.get_all_nodes()
        assert len(all_nodes) == num_concepts
        
        difficulty_1_concepts = graph.get_concepts_by_difficulty(1)
        assert len(difficulty_1_concepts) == 10  # concepts 0-9


class TestGraphMemoryManagement:
    """Test memory management in graph operations."""
    
    def test_graph_cleanup(self):
        """Test that graph data is properly cleaned up."""
        graph = SimpleGraph()
        
        # Add many nodes and edges
        for i in range(20):
            graph.add_node(f"node_{i}", f"Node {i}")
            if i > 0:
                graph.add_edge(f"node_{i-1}", f"node_{i}")
        
        initial_count = len(graph.nodes)
        assert initial_count == 20
        
        # Remove half the nodes
        for i in range(0, 20, 2):
            graph.remove_node(f"node_{i}")
        
        assert len(graph.nodes) == 10
        
        # Verify edges were cleaned up too
        remaining_edges = len(graph.edges)
        # Should be fewer edges since we removed nodes
        assert remaining_edges < initial_count - 1
    
    def test_circular_reference_handling(self):
        """Test handling of circular references."""
        graph = SimpleGraph()
        
        # Create circular references
        graph.add_node("A", "Node A")
        graph.add_node("B", "Node B")
        graph.add_node("C", "Node C")
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")  # Creates cycle
        
        # Should still be able to find paths
        path = graph.find_path("A", "C")
        assert path is not None
        assert len(path) == 3  # A -> B -> C
        
        # Remove a node to break the cycle
        graph.remove_node("B")
        
        # Path should now be broken
        path = graph.find_path("A", "C")
        assert path is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 