#!/usr/bin/env python
"""
Hybrid Knowledge Example: Vector DB + Graph DB Integration

This example demonstrates how to:
1. Process a document using both vector and graph databases
2. Use an LLM to extract structured entities and relationships from text
3. Store unstructured text in vector database for semantic search
4. Store structured data in graph database for relationship queries
5. Perform hybrid queries across both knowledge sources

Usage:
    python hybrid_knowledge_example.py
"""

import os
import json
import uuid
import datetime
from typing import Dict, List, Any, Optional
from pprint import pprint

# Import knowledge components
from agent_suite.graph import GraphManager
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter
from knowledge.vector import QdrantVectorDatabase

# For LLM integration
import openai


def print_header(title):
    """Print a section header with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_subheader(title):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


class HybridKnowledgeManager:
    """
    Manager that combines vector and graph knowledge storage.
    
    This class demonstrates how to use both storage types together
    for a more complete knowledge management solution.
    """
    
    def __init__(
        self,
        graph_name: str = "hybrid_knowledge_graph",
        vector_collection: str = "hybrid_knowledge_vectors",
        persist: bool = True,
        persistence_path: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize the hybrid knowledge manager.
        
        Args:
            graph_name: Name for the graph database
            vector_collection: Name for the vector collection
            persist: Whether to persist data
            persistence_path: Path for persistent storage
            openai_api_key: API key for OpenAI
        """
        # Set up storage path
        if persist and persistence_path is None:
            self.persistence_path = os.path.join(os.getcwd(), "hybrid_knowledge_data")
            os.makedirs(self.persistence_path, exist_ok=True)
        else:
            self.persistence_path = persistence_path
        
        # Initialize graph database
        self.graph_db = GraphitiKnowledgeGraphAdapter(
            graph_name=graph_name,
            persist=persist,
            persistence_path=self.persistence_path
        )
        
        # Initialize vector database
        self.vector_db = QdrantVectorDatabase(
            collection_name=vector_collection,
            persist=persist,
            persistence_path=self.persistence_path
        )
        
        # Set up OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.llm_available = True
        else:
            print("WARNING: No OpenAI API key provided. LLM extraction will be simulated.")
            self.llm_available = False
    
    def process_document(self, document_text: str, document_metadata: Dict[str, Any]) -> str:
        """
        Process a document and store its content in both knowledge stores.
        
        Args:
            document_text: The text content of the document
            document_metadata: Metadata about the document
            
        Returns:
            The document ID
        """
        # 1. Store the full text in the vector database for semantic search
        doc_id = self.vector_db.add_text(document_text, document_metadata)
        
        # 2. Extract entities and relationships using LLM
        if self.llm_available:
            extracted_data = self.extract_knowledge_with_llm(document_text)
        else:
            # Simulate LLM extraction with predefined data
            extracted_data = self.simulate_knowledge_extraction(document_text)
        
        # 3. Store structured data in the graph database
        self.store_structured_data(extracted_data, doc_id)
        
        # 4. Link the document to related entities in the graph
        self.link_document_to_entities(doc_id, extracted_data)
        
        return doc_id
    
    def extract_knowledge_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Use LLM to extract structured knowledge from text.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        # Create a prompt for the LLM
        prompt = f"""
        Extract structured knowledge from the following text. Identify:
        1. Entities (people, organizations, projects, concepts, etc.)
        2. Relationships between entities
        3. Key attributes of entities

        Return the extraction as JSON with the following format:
        {{
            "entities": [
                {{
                    "id": "unique_id",
                    "type": "person|organization|project|concept",
                    "name": "Entity Name",
                    "properties": {{
                        "attribute1": "value1",
                        "attribute2": "value2"
                    }}
                }}
            ],
            "relations": [
                {{
                    "source": "source_entity_id",
                    "relation_type": "WORKS_FOR|PART_OF|MANAGES|RELATED_TO",
                    "target": "target_entity_id",
                    "properties": {{
                        "attribute1": "value1"
                    }}
                }}
            ]
        }}

        TEXT:
        {text}
        """
        
        try:
            # Create OpenAI client
            client = openai.OpenAI(api_key=self.openai_api_key)
            
            # Make API call to OpenAI using the new client
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a knowledge extraction assistant that identifies entities and relationships in text and formats them as structured data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Extract JSON from response
            result_text = response.choices[0].message.content
            
            # Extract JSON portion
            json_start = result_text.find("{")
            json_end = result_text.rfind("}")
            
            if json_start >= 0 and json_end >= 0:
                json_str = result_text[json_start:json_end+1]
                extracted_data = json.loads(json_str)
                return extracted_data
            else:
                print("Error: Could not find JSON in LLM response")
                return {"entities": [], "relations": []}
                
        except Exception as e:
            print(f"Error extracting knowledge with LLM: {e}")
            return {"entities": [], "relations": []}
    
    def simulate_knowledge_extraction(self, text: str) -> Dict[str, Any]:
        """
        Simulate knowledge extraction when LLM is not available.
        
        Args:
            text: The document text
            
        Returns:
            Dictionary with extracted entities and relationships
        """
        # Simple keyword-based extraction
        sample_entities = []
        sample_relations = []
        
        # Create entity IDs based on possible entities in the text
        entity_keywords = {
            "project": ["project", "initiative", "program"],
            "person": ["team member", "manager", "director", "lead"],
            "organization": ["department", "company", "team", "division"],
            "technology": ["technology", "software", "system", "platform"]
        }
        
        # Extract some basic entities
        entity_id_counter = 1
        entities_added = set()
        
        for entity_type, keywords in entity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Find a name for this entity (simplistic approach)
                    start_idx = text.lower().find(keyword.lower())
                    name_context = text[max(0, start_idx-20):min(len(text), start_idx+50)]
                    
                    # Generate a simple name
                    entity_name = f"{keyword.title()} {entity_id_counter}"
                    
                    # Ensure we don't add duplicates
                    if entity_name in entities_added:
                        continue
                        
                    entities_added.add(entity_name)
                    
                    # Create the entity
                    entity_id = f"{entity_type}_{entity_id_counter}"
                    sample_entities.append({
                        "id": entity_id,
                        "type": entity_type,
                        "name": entity_name,
                        "properties": {
                            "context": name_context,
                            "confidence": 0.7
                        }
                    })
                    
                    entity_id_counter += 1
        
        # Create some sample relationships
        if len(sample_entities) >= 2:
            relation_types = ["RELATED_TO", "PART_OF", "WORKS_WITH"]
            
            for i in range(min(5, len(sample_entities)-1)):
                source_idx = i
                target_idx = (i + 1) % len(sample_entities)
                
                relation_type = relation_types[i % len(relation_types)]
                
                sample_relations.append({
                    "source": sample_entities[source_idx]["id"],
                    "relation_type": relation_type,
                    "target": sample_entities[target_idx]["id"],
                    "properties": {
                        "confidence": 0.6,
                        "extracted_from": "text"
                    }
                })
        
        return {
            "entities": sample_entities,
            "relations": sample_relations
        }
    
    def store_structured_data(self, extracted_data: Dict[str, Any], doc_id: str) -> None:
        """
        Store extracted structured data in graph database.
        
        Args:
            extracted_data: The extracted entities and relationships
            doc_id: ID of the source document
        """
        # Store entities
        for entity in extracted_data.get("entities", []):
            entity_id = entity["id"]
            entity_type = entity["type"]
            
            # Add extracted source to properties
            properties = entity.get("properties", {}).copy()
            properties["name"] = entity.get("name", f"Unnamed {entity_type}")
            properties["extracted_from"] = doc_id
            
            # Store in graph
            self.graph_db.add_entity(entity_id, entity_type, properties)
        
        # Store relations
        for relation in extracted_data.get("relations", []):
            source_id = relation["source"]
            relation_type = relation["relation_type"]
            target_id = relation["target"]
            
            # Add extracted source to properties
            properties = relation.get("properties", {}).copy()
            properties["extracted_from"] = doc_id
            
            # Store in graph
            self.graph_db.add_relation(source_id, relation_type, target_id, properties)
    
    def link_document_to_entities(self, doc_id: str, extracted_data: Dict[str, Any]) -> None:
        """
        Create links between the document and extracted entities in the graph.
        
        Args:
            doc_id: ID of the document
            extracted_data: Extracted structured data
        """
        # Get document from vector DB to add as a text node in graph
        doc = self.vector_db.get_entity(doc_id)
        
        if not doc:
            return
        
        # Create text node in graph
        graph_doc_id = self.graph_db.add_text(
            text=doc.get("content", ""),
            properties=doc.get("properties", {})
        )
        
        # Link document to each entity
        for entity in extracted_data.get("entities", []):
            entity_id = entity["id"]
            self.graph_db.add_relation(
                graph_doc_id, 
                "MENTIONS", 
                entity_id,
                {"confidence": entity.get("properties", {}).get("confidence", 0.8)}
            )
    
    def hybrid_search(self, query: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform hybrid search across both knowledge stores.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            Combined results from both stores
        """
        # 1. Semantic search in vector database
        vector_results = self.vector_db.semantic_search(query, limit=limit)
        
        # 2. Keyword search in graph database
        graph_results = self.graph_db.semantic_search(query, limit=limit)
        
        # 3. Combine results
        return {
            "vector_results": vector_results,
            "graph_results": graph_results
        }
    
    def knowledge_query(self, query_type: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform specialized knowledge queries.
        
        Args:
            query_type: Type of query to perform
            params: Parameters for the query
            
        Returns:
            Query results
        """
        if query_type == "related_entities":
            # Find entities related to the given entity
            entity_id = params.get("entity_id")
            if not entity_id:
                return []
                
            # Use graph database for relationship queries
            neighbors = self.graph_db.graph_manager.get_neighbors(entity_id)
            return neighbors
            
        elif query_type == "document_entities":
            # Find entities mentioned in a document
            doc_id = params.get("doc_id")
            if not doc_id:
                return []
                
            # Get the graph document node
            doc_relations = self.graph_db.query_relations(
                source_id=doc_id,
                relation_type="MENTIONS"
            )
            
            # Get the target entities
            entities = []
            for relation in doc_relations:
                entity_id = relation["target_id"]
                entity = self.graph_db.get_entity(entity_id)
                if entity:
                    entities.append(entity)
                    
            return entities
            
        elif query_type == "entity_documents":
            # Find documents that mention an entity
            entity_id = params.get("entity_id")
            if not entity_id:
                return []
                
            # Get relations where entity is mentioned
            doc_relations = self.graph_db.query_relations(
                relation_type="MENTIONS",
                target_id=entity_id
            )
            
            # Get the source documents
            documents = []
            for relation in doc_relations:
                doc_id = relation["source_id"]
                doc = self.graph_db.get_entity(doc_id)
                if doc and doc.get("type") == "text":
                    documents.append(doc)
                    
            return documents
        
        return []


def run_example():
    """Run the hybrid knowledge example."""
    print_header("Hybrid Knowledge Example")
    
    # Initialize the hybrid knowledge manager
    knowledge_manager = HybridKnowledgeManager(
        persist=True,
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Sample documents
    documents = [
        {
            "title": "Project Phoenix Overview",
            "content": """
            Project Phoenix is our strategic initiative to modernize our digital infrastructure.
            Led by Sarah Johnson, the project aims to replace legacy systems with a cloud-native
            architecture. The development team, which is part of the Technology Department, has
            been working on this initiative since January 2023. 
            
            Key technologies include:
            - Kubernetes for container orchestration
            - React for frontend development
            - GraphQL for API development
            
            Michael Chen from the Data Analytics team is collaborating with the Phoenix team
            to ensure proper data migration. The project is sponsored by TechCorp's CTO office
            and has a timeline of 18 months.
            """,
            "metadata": {
                "category": "Project Documentation",
                "author": "Project Management Office",
                "created_date": "2023-03-15"
            }
        },
        {
            "title": "Technology Department Structure",
            "content": """
            The Technology Department at TechCorp consists of several specialized teams:
            
            1. Software Development - Led by Director Lisa Park
               - Frontend Team: 8 engineers focused on React and Angular
               - Backend Team: 12 engineers working on Java, Python and Node.js
               
            2. Data Engineering - Led by Director James Wilson
               - Data Analytics: 6 data scientists including Michael Chen
               - Data Infrastructure: 4 engineers maintaining our Hadoop and Spark clusters
               
            3. Project Management Office
               - Sarah Johnson manages our strategic projects including Project Phoenix
               
            The department reports to CTO David Miller who joined TechCorp in 2020 after
            serving as VP of Engineering at InnovateInc. The Technology Department
            collaborates closely with the Product Department on all major initiatives.
            """,
            "metadata": {
                "category": "Organization",
                "author": "HR Department",
                "created_date": "2023-01-10"
            }
        }
    ]
    
    # Process documents
    doc_ids = []
    print_subheader("Processing Documents")
    
    for doc in documents:
        print(f"Processing document: {doc['title']}")
        doc_id = knowledge_manager.process_document(
            document_text=doc["content"],
            document_metadata={
                "title": doc["title"],
                **doc["metadata"]
            }
        )
        doc_ids.append(doc_id)
        print(f"Document processed with ID: {doc_id}")
        
    print("\nAll documents have been processed and stored in both vector and graph databases.")
    
    # Semantic search
    print_subheader("Hybrid Semantic Search")
    
    search_queries = [
        "cloud architecture",
        "Sarah Johnson projects",
        "data analytics team"
    ]
    
    for query in search_queries:
        print(f"\nQuery: '{query}'")
        results = knowledge_manager.hybrid_search(query)
        
        # Show vector results
        print("\nVector Database Results:")
        for i, result in enumerate(results["vector_results"][:3]):
            score = result.get("score", 0)
            title = result.get("properties", {}).get("title", "Unnamed document")
            content = result.get("content", "")[:100] + "..."
            print(f"{i+1}. [{score:.2f}] {title}\n   {content}")
        
        # Show graph results
        print("\nGraph Database Results:")
        for i, result in enumerate(results["graph_results"][:3]):
            if result.get("entity_type") == "text":
                title = result.get("properties", {}).get("title", "Unnamed document")
                content = result.get("properties", {}).get("content", "")[:100] + "..."
                print(f"{i+1}. [TEXT] {title}\n   {content}")
            else:
                entity_type = result.get("entity_type", "unknown")
                name = result.get("properties", {}).get("name", "Unnamed entity")
                print(f"{i+1}. [{entity_type.upper()}] {name}")
    
    # Specialized knowledge queries
    print_subheader("Graph-Based Knowledge Queries")
    
    # 1. Find entities in the first document
    print("\nEntities mentioned in document: " + documents[0]["title"])
    doc_entities = knowledge_manager.knowledge_query(
        query_type="document_entities",
        params={"doc_id": doc_ids[0]}
    )
    
    for i, entity in enumerate(doc_entities):
        entity_type = entity.get("type", "unknown")
        name = entity.get("properties", {}).get("name", "Unnamed entity")
        print(f"{i+1}. [{entity_type.upper()}] {name}")
    
    # 2. Find entity relationships
    if doc_entities:
        # Take the first entity as an example
        example_entity = doc_entities[0]
        entity_id = example_entity["id"]
        name = example_entity.get("properties", {}).get("name", "Unnamed entity")
        
        print(f"\nRelationships for entity: {name}")
        relations = knowledge_manager.knowledge_query(
            query_type="related_entities",
            params={"entity_id": entity_id}
        )
        
        for i, relation in enumerate(relations):
            direction = relation.get("direction", "unknown")
            relation_type = relation.get("relation_type", "unknown")
            entity_name = relation.get("entity", {}).get("properties", {}).get("name", "Unnamed entity")
            
            if direction == "in":
                print(f"{i+1}. {entity_name} --[{relation_type}]--> {name}")
            else:
                print(f"{i+1}. {name} --[{relation_type}]--> {entity_name}")
    
    # 3. Find documents mentioning an entity
    print_subheader("Cross-Referenced Knowledge")
    
    # Get a person entity
    person_entities = knowledge_manager.graph_db.query_entities(entity_type="person")
    
    if person_entities:
        person = person_entities[0]
        person_id = person["id"]
        person_name = person["properties"].get("name", "Unnamed person")
        
        print(f"\nDocuments mentioning: {person_name}")
        documents = knowledge_manager.knowledge_query(
            query_type="entity_documents",
            params={"entity_id": person_id}
        )
        
        for i, doc in enumerate(documents):
            title = doc.get("properties", {}).get("title", "Unnamed document")
            print(f"{i+1}. {title}")
    
    # Get statistics
    print_subheader("Knowledge Statistics")
    
    vector_stats = knowledge_manager.vector_db.get_stats()
    graph_stats = knowledge_manager.graph_db.get_stats()
    
    print("\nVector Database:")
    print(f"- Total documents: {vector_stats.get('total_documents', 'N/A')}")
    print(f"- Vector dimension: {vector_stats.get('vector_size', 'N/A')}")
    
    print("\nGraph Database:")
    print(f"- Total entities: {graph_stats.get('total_entities', 'N/A')}")
    print(f"- Total relations: {graph_stats.get('total_relations', 'N/A')}")
    
    print("\nEntity types:")
    for entity_type, count in graph_stats.get("entity_counts", {}).items():
        print(f"- {entity_type}: {count}")
    
    print("\nRelation types:")
    for relation_type, count in graph_stats.get("relation_counts", {}).items():
        print(f"- {relation_type}: {count}")


if __name__ == "__main__":
    run_example() 