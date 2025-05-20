#!/usr/bin/env python
"""
End-to-end tests for the centralized graph infrastructure.

This script tests:
1. Direct GraphManager operations
2. Knowledge module adapter
3. Long-term storage adapter
4. Cross-module integration
5. Persistence and data integrity

Run with: python test_graph_e2e.py
"""

import os
import uuid
import shutil
import unittest
from typing import Dict, Any, List

# Import the components to test
from agent_suite.graph import GraphManager
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter
from agent_suite.agents.storage.long_term.graphiti import GraphitiKnowledgeGraphMemoryAdapter


class GraphE2ETests(unittest.TestCase):
    """End-to-end tests for the graph infrastructure."""

    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = os.path.join(os.getcwd(), "test_graph_data")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a unique test ID to avoid conflicts
        self.test_id = uuid.uuid4().hex[:8]
        
        # Initialize components with unique names
        self.graph_manager = GraphManager(
            graph_name=f"test_graph_{self.test_id}",
            persist=True,
            persistence_path=self.test_dir
        )
        
        self.knowledge_adapter = GraphitiKnowledgeGraphAdapter(
            graph_name=f"test_knowledge_{self.test_id}",
            persist=True,
            persistence_path=self.test_dir
        )
        
        self.memory_adapter = GraphitiKnowledgeGraphMemoryAdapter(
            graph_name=f"test_memory_{self.test_id}",
            persist=True,
            persistence_path=self.test_dir
        )
        
        # Test data
        self.test_data = {
            "entities": [
                {
                    "type": "person",
                    "properties": {
                        "name": "Jane Smith",
                        "age": 35,
                        "occupation": "Software Architect"
                    }
                },
                {
                    "type": "project",
                    "properties": {
                        "name": "Graph Database",
                        "status": "Active",
                        "priority": "High"
                    }
                },
                {
                    "type": "concept",
                    "properties": {
                        "name": "Graph Theory",
                        "category": "Computer Science"
                    }
                }
            ],
            "relations": [
                {
                    "source_type": "person",
                    "relation": "WORKS_ON",
                    "target_type": "project",
                    "properties": {
                        "role": "Lead Developer",
                        "since": "2023-01-01"
                    }
                },
                {
                    "source_type": "person",
                    "relation": "KNOWS_ABOUT",
                    "target_type": "concept",
                    "properties": {
                        "expertise_level": "Expert",
                        "years_experience": 8
                    }
                },
                {
                    "source_type": "project",
                    "relation": "USES",
                    "target_type": "concept",
                    "properties": {
                        "importance": "Critical"
                    }
                }
            ],
            "texts": [
                {
                    "content": "Graph databases represent and store data in the form of nodes and edges, offering efficient ways to query connected data.",
                    "properties": {
                        "title": "Graph Database Overview",
                        "category": "Technical"
                    }
                },
                {
                    "content": "Graph theory is the study of graphs, which are mathematical structures used to model pairwise relations between objects.",
                    "properties": {
                        "title": "Graph Theory Introduction",
                        "category": "Academic"
                    }
                }
            ]
        }

    def tearDown(self):
        """Clean up after tests."""
        # Clean up test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_1_direct_graph_manager(self):
        """Test basic GraphManager operations."""
        print("\n--- Testing direct GraphManager operations ---")
        
        # Add entities
        entity_ids = {}
        for entity_data in self.test_data["entities"]:
            entity_id = self.graph_manager.add_entity(
                entity_type=entity_data["type"],
                properties=entity_data["properties"]
            )
            entity_ids[entity_data["type"]] = entity_id
            print(f"Added {entity_data['type']}: {entity_id}")
        
        # Verify entities were added
        for entity_type, entity_id in entity_ids.items():
            entity = self.graph_manager.get_entity(entity_id)
            self.assertIsNotNone(entity, f"Entity {entity_id} not found")
            self.assertEqual(entity["type"], entity_type, f"Entity type mismatch for {entity_id}")
            print(f"Verified {entity_type}: {entity_id}")
        
        # Add relations
        relation_ids = []
        for relation_data in self.test_data["relations"]:
            source_id = entity_ids[relation_data["source_type"]]
            target_id = entity_ids[relation_data["target_type"]]
            
            relation_id = self.graph_manager.add_relation(
                source_id=source_id,
                relation_type=relation_data["relation"],
                target_id=target_id,
                properties=relation_data["properties"]
            )
            relation_ids.append(relation_id)
            print(f"Added relation {relation_data['relation']}: {relation_id}")
        
        # Add texts
        text_ids = []
        for text_data in self.test_data["texts"]:
            text_id = self.graph_manager.add_text(
                text=text_data["content"],
                properties=text_data["properties"]
            )
            text_ids.append(text_id)
            print(f"Added text: {text_id}")
        
        # Query and verify
        # 1. Query by entity type
        persons = self.graph_manager.query_entities(entity_type="person")
        self.assertEqual(len(persons), 1, "Should find exactly one person")
        
        # 2. Query by relation
        works_on_relations = self.graph_manager.query_relations(relation_type="WORKS_ON")
        self.assertEqual(len(works_on_relations), 1, "Should find exactly one WORKS_ON relation")
        
        # 3. Keyword search
        graph_results = self.graph_manager.keyword_search("graph")
        self.assertGreaterEqual(len(graph_results), 2, "Should find at least 2 items related to 'graph'")
        
        # 4. Get neighbors
        person_id = entity_ids["person"]
        neighbors = self.graph_manager.get_neighbors(person_id)
        self.assertEqual(len(neighbors), 2, "Person should have exactly 2 neighbors")
        
        # 5. Graph traversal
        traversal = self.graph_manager.graph_traversal(person_id, max_depth=2)
        self.assertEqual(len(traversal["nodes"]), 3, "Traversal should find 3 nodes")
        self.assertEqual(len(traversal["edges"]), 3, "Traversal should find 3 edges")
        
        # Get stats
        stats = self.graph_manager.get_stats()
        print(f"Graph stats: {len(stats['entity_counts'])} entity types, {stats['total_entities']} entities, {stats['total_relations']} relations")

    def test_2_knowledge_adapter(self):
        """Test knowledge module adapter."""
        print("\n--- Testing knowledge module adapter ---")
        
        # Add entities
        entity_ids = {}
        for entity_data in self.test_data["entities"]:
            entity_id = str(uuid.uuid4())
            self.knowledge_adapter.add_entity(
                entity_id=entity_id,
                entity_type=entity_data["type"],
                properties=entity_data["properties"]
            )
            entity_ids[entity_data["type"]] = entity_id
            print(f"Added {entity_data['type']}: {entity_id}")
        
        # Add relations
        for relation_data in self.test_data["relations"]:
            source_id = entity_ids[relation_data["source_type"]]
            target_id = entity_ids[relation_data["target_type"]]
            
            relation_id = self.knowledge_adapter.add_relation(
                source_id=source_id,
                relation_type=relation_data["relation"],
                target_id=target_id,
                properties=relation_data["properties"]
            )
            print(f"Added relation {relation_data['relation']}: {relation_id}")
        
        # Add texts
        text_ids = []
        for text_data in self.test_data["texts"]:
            text_id = self.knowledge_adapter.add_text(
                text=text_data["content"],
                properties=text_data["properties"]
            )
            text_ids.append(text_id)
            print(f"Added text: {text_id}")
        
        # Verify entities were added
        for entity_type, entity_id in entity_ids.items():
            entity = self.knowledge_adapter.get_entity(entity_id)
            self.assertIsNotNone(entity, f"Entity {entity_id} not found")
            self.assertEqual(entity["type"], entity_type, f"Entity type mismatch for {entity_id}")
        
        # Test semantic search
        search_results = self.knowledge_adapter.semantic_search("graph database")
        self.assertGreaterEqual(len(search_results), 1, "Should find at least one result for 'graph database'")
        print(f"Found {len(search_results)} results for 'graph database'")
        
        # Test knowledge queries
        person_entity = self.knowledge_adapter.query_entities(entity_type="person")[0]
        self.assertEqual(person_entity["properties"]["name"], "Jane Smith", "Name should match")
        
        # Test relation queries
        works_on_relations = self.knowledge_adapter.query_relations(relation_type="WORKS_ON")
        self.assertEqual(len(works_on_relations), 1, "Should find exactly one WORKS_ON relation")
        
        # Get stats
        stats = self.knowledge_adapter.get_stats()
        print(f"Knowledge graph stats: {len(stats['entity_counts'])} entity types, {stats['total_entities']} entities, {stats['total_relations']} relations")

    def test_3_memory_adapter(self):
        """Test long-term storage memory adapter."""
        print("\n--- Testing long-term memory adapter ---")
        
        # Store facts
        print("Storing facts...")
        fact_ids = []
        
        # Store properties as facts
        fact_id = self.memory_adapter.store_fact(
            subject="person:jane",
            predicate="name",
            object_value="Jane Smith",
            metadata={"confidence": 0.95, "source": "user_profile"}
        )
        fact_ids.append(fact_id)
        
        fact_id = self.memory_adapter.store_fact(
            subject="person:jane",
            predicate="occupation",
            object_value="Software Architect",
            metadata={"confidence": 0.9, "source": "conversation"}
        )
        fact_ids.append(fact_id)
        
        # Store entity-to-entity relations
        fact_id = self.memory_adapter.store_fact(
            subject="person:jane",
            predicate="WORKS_ON",
            object_value="project:graphdb",
            metadata={"confidence": 0.95, "role": "Lead Developer"}
        )
        fact_ids.append(fact_id)
        
        fact_id = self.memory_adapter.store_fact(
            subject="project:graphdb",
            predicate="status",
            object_value="Active",
            metadata={"confidence": 1.0, "source": "project_management"}
        )
        fact_ids.append(fact_id)
        
        # Store texts
        text_ids = []
        for text_data in self.test_data["texts"]:
            text_id = self.memory_adapter.store_text(
                text=text_data["content"],
                metadata=text_data["properties"]
            )
            text_ids.append(text_id)
            print(f"Added text: {text_id}")
        
        # Query facts - natural language
        print("\nQuerying facts with natural language...")
        fact_results = self.memory_adapter.query_facts("Jane Smith", limit=3)
        self.assertGreaterEqual(len(fact_results), 1, "Should find at least one fact about Jane Smith")
        
        for i, fact in enumerate(fact_results):
            print(f"Fact {i+1}: {fact['subject']} - {fact['predicate']} - {fact['object']}")
        
        # Query facts - structured
        print("\nQuerying facts with structured query...")
        jane_facts = self.memory_adapter.query_facts({"subject": "person:jane"})
        self.assertGreaterEqual(len(jane_facts), 2, "Should find at least 2 facts about person:jane")
        
        # Query text
        print("\nQuerying text...")
        text_results = self.memory_adapter.query_text("graph", limit=2)
        self.assertGreaterEqual(len(text_results), 1, "Should find at least one text about 'graph'")
        
        # Get all entities
        entities = self.memory_adapter.get_entities()
        print(f"Found {len(entities)} entities")
        
        # Get entity facts
        jane_facts = self.memory_adapter.get_entity_facts("person:jane")
        self.assertGreaterEqual(len(jane_facts), 2, "Should find at least 2 facts about person:jane")
        
        # Test deleting a fact
        self.memory_adapter.delete_fact(fact_ids[0])
        
        # Get stats
        stats = self.memory_adapter.get_stats()
        print(f"Memory graph stats: {len(stats.get('entity_counts', {}))} entity types, {stats['total_entities']} entities, {stats['total_relations']} relations")

    def test_4_persistence(self):
        """Test persistence and loading from disk."""
        print("\n--- Testing persistence ---")
        
        # Add data
        entity_id = self.graph_manager.add_entity(
            entity_type="test_entity",
            properties={"name": "Test Entity", "test_case": "persistence"}
        )
        
        # Create a new GraphManager that should load from disk
        graph_manager2 = GraphManager(
            graph_name=f"test_graph_{self.test_id}",
            persist=True,
            persistence_path=self.test_dir
        )
        
        # Verify data was loaded
        entity = graph_manager2.get_entity(entity_id)
        self.assertIsNotNone(entity, "Entity should be loaded from disk")
        self.assertEqual(entity["properties"]["name"], "Test Entity", "Entity properties should be preserved")
        self.assertEqual(entity["properties"]["test_case"], "persistence", "Entity properties should be preserved")
        
        print(f"Successfully loaded entity from disk: {entity_id}")

    def test_5_cross_module_integration(self):
        """Test integration between knowledge and memory modules."""
        print("\n--- Testing cross-module integration ---")
        
        # Create shared entities in knowledge
        print("Creating entities in knowledge module...")
        person_id = self.knowledge_adapter.add_entity(
            entity_id="shared:person:jane",
            entity_type="person",
            properties={
                "name": "Jane Smith",
                "age": 35,
                "occupation": "Software Architect"
            }
        )
        
        project_id = self.knowledge_adapter.add_entity(
            entity_id="shared:project:graphdb",
            entity_type="project",
            properties={
                "name": "Graph Database",
                "status": "Active",
                "priority": "High"
            }
        )
        
        # Add relation in knowledge
        self.knowledge_adapter.add_relation(
            source_id=person_id,
            relation_type="WORKS_ON",
            target_id=project_id,
            properties={"role": "Lead Developer"}
        )
        
        # Reference same entities in memory
        print("Referencing same entities in memory module...")
        self.memory_adapter.store_fact(
            subject="shared:person:jane",
            predicate="last_conversation",
            object_value="2023-06-15",
            metadata={"confidence": 1.0, "source": "system"}
        )
        
        self.memory_adapter.store_fact(
            subject="shared:person:jane",
            predicate="INTERESTED_IN",
            object_value="shared:project:graphdb",
            metadata={"confidence": 0.9, "source": "conversation"}
        )
        
        # Query from both modules
        k_person = self.knowledge_adapter.get_entity("shared:person:jane")
        m_facts = self.memory_adapter.get_entity_facts("shared:person:jane")
        
        self.assertIsNotNone(k_person, "Person should exist in knowledge module")
        self.assertGreaterEqual(len(m_facts), 2, "Should find at least 2 facts about person in memory module")
        
        print(f"Successfully verified cross-module entity: {k_person['properties']['name']}")
        print(f"Found {len(m_facts)} facts about the entity in memory module")

    def test_6_complex_graph_operations(self):
        """Test complex graph operations like traversal and analysis."""
        print("\n--- Testing complex graph operations ---")
        
        # Create a connected graph
        print("Creating a connected graph...")
        entities = {}
        
        # Add entities
        for i in range(5):
            entity_id = self.graph_manager.add_entity(
                entity_type="node",
                properties={"name": f"Node {i}", "value": i * 10}
            )
            entities[f"node_{i}"] = entity_id
        
        # Create a connected structure: 0-(1,2)-(3,4)
        self.graph_manager.add_relation(
            source_id=entities["node_0"],
            relation_type="CONNECTS",
            target_id=entities["node_1"]
        )
        
        self.graph_manager.add_relation(
            source_id=entities["node_0"],
            relation_type="CONNECTS",
            target_id=entities["node_2"]
        )
        
        self.graph_manager.add_relation(
            source_id=entities["node_1"],
            relation_type="CONNECTS",
            target_id=entities["node_3"]
        )
        
        self.graph_manager.add_relation(
            source_id=entities["node_2"],
            relation_type="CONNECTS",
            target_id=entities["node_4"]
        )
        
        # Test traversal with different depths
        # Depth 1 should give node_0, node_1, node_2
        depth1 = self.graph_manager.graph_traversal(entities["node_0"], max_depth=1)
        self.assertEqual(len(depth1["nodes"]), 3, "Depth 1 traversal should find 3 nodes")
        self.assertEqual(len(depth1["edges"]), 2, "Depth 1 traversal should find 2 edges")
        
        # Depth 2 should give all 5 nodes
        depth2 = self.graph_manager.graph_traversal(entities["node_0"], max_depth=2)
        self.assertEqual(len(depth2["nodes"]), 5, "Depth 2 traversal should find all 5 nodes")
        self.assertEqual(len(depth2["edges"]), 4, "Depth 2 traversal should find all 4 edges")
        
        print(f"Successfully verified graph traversal at depths 1 ({len(depth1['nodes'])} nodes) and 2 ({len(depth2['nodes'])} nodes)")
        
        # Test neighbor analysis
        neighbors = self.graph_manager.get_neighbors(entities["node_0"])
        self.assertEqual(len(neighbors), 2, "Node 0 should have 2 neighbors")
        
        # Test direction-specific neighbors
        out_neighbors = self.graph_manager.get_neighbors(entities["node_0"], direction="out")
        self.assertEqual(len(out_neighbors), 2, "Node 0 should have 2 outgoing neighbors")
        
        in_neighbors = self.graph_manager.get_neighbors(entities["node_0"], direction="in")
        self.assertEqual(len(in_neighbors), 0, "Node 0 should have 0 incoming neighbors")
        
        print(f"Successfully verified neighbor analysis: {len(neighbors)} total, {len(out_neighbors)} outgoing, {len(in_neighbors)} incoming")
        
        # Test update and delete
        # Update a node
        self.graph_manager.update_entity(
            entity_id=entities["node_0"],
            properties={"name": "Updated Node 0", "value": 100}
        )
        
        updated_node = self.graph_manager.get_entity(entities["node_0"])
        self.assertEqual(updated_node["properties"]["name"], "Updated Node 0", "Node name should be updated")
        self.assertEqual(updated_node["properties"]["value"], 100, "Node value should be updated")
        
        # Delete a node (this should remove connected edges too)
        self.graph_manager.delete_entity(entities["node_1"])
        
        # Check if node was deleted
        deleted_node = self.graph_manager.get_entity(entities["node_1"])
        self.assertIsNone(deleted_node, "Node 1 should be deleted")
        
        # Check if edges were removed
        edges_to_node1 = self.graph_manager.query_relations(target_id=entities["node_1"])
        self.assertEqual(len(edges_to_node1), 0, "No edges should point to deleted node")
        
        print("Successfully verified entity update and deletion")

if __name__ == "__main__":
    unittest.main() 