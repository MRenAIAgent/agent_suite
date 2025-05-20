"""
Data Generators for Benchmark Suite

This module provides functions to generate synthetic test data
for benchmarking knowledge graph and vector database systems.
"""

import os
import random
import uuid
import json
import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

# Set fixed random seed for reproducibility
SEED = 42
random.seed(SEED)


class DataGenerator:
    """Generator for benchmark test data."""
    
    # Word lists for generating coherent text
    SUBJECTS = [
        "Project", "Team", "Customer", "Product", "Service", "Department", 
        "Initiative", "Application", "System", "Platform", "Framework", 
        "API", "Database", "Server", "Client", "User", "Administrator"
    ]
    
    ADJECTIVES = [
        "Advanced", "Strategic", "Innovative", "Scalable", "Robust", "Agile",
        "Modern", "Efficient", "Reliable", "Secure", "Flexible", "Integrated",
        "Cloud-based", "On-premise", "Hybrid", "Real-time", "Distributed"
    ]
    
    TECHNOLOGIES = [
        "Python", "Java", "JavaScript", "React", "Angular", "Vue", "Node.js",
        "Django", "Flask", "Spring", "Kubernetes", "Docker", "AWS", "Azure",
        "GCP", "GraphQL", "REST", "SQL", "NoSQL", "MongoDB", "Redis", "Kafka"
    ]
    
    RELATION_TYPES = [
        "PART_OF", "WORKS_WITH", "DEPENDS_ON", "IMPLEMENTS", "USES", 
        "MANAGES", "DEVELOPED_BY", "HOSTED_ON", "INTEGRATED_WITH",
        "RELATED_TO", "SUCCESSOR_OF", "PREDECESSOR_OF", "CREATED_BY"
    ]
    
    ENTITY_TYPES = [
        "project", "person", "organization", "technology", "concept",
        "document", "service", "product", "team", "event"
    ]
    
    def __init__(
        self, 
        num_entities: int = 1000,
        num_relations: int = 5000,
        num_documents: int = 100,
        complexity: str = "medium",
        output_dir: str = "benchmark_suite/datasets"
    ):
        """
        Initialize the data generator.
        
        Args:
            num_entities: Number of entities to generate
            num_relations: Number of relations to generate
            num_documents: Number of documents to generate
            complexity: Complexity of generated data (simple, medium, complex)
            output_dir: Directory to save generated data
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_documents = num_documents
        self.complexity = complexity
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for generated data
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, Any]] = []
        self.documents: List[Dict[str, Any]] = []
        
        # Adjust complexity factors
        if complexity == "simple":
            self.avg_properties = 3
            self.max_text_length = 200
            self.max_relations_per_entity = 5
        elif complexity == "complex":
            self.avg_properties = 10
            self.max_text_length = 2000
            self.max_relations_per_entity = 20
        else:  # medium
            self.avg_properties = 5
            self.max_text_length = 500
            self.max_relations_per_entity = 10
    
    def generate_random_entity(self, entity_type: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a random entity.
        
        Args:
            entity_type: Optional specific entity type
            
        Returns:
            Tuple of entity ID and entity data
        """
        # Generate random entity ID
        entity_id = str(uuid.uuid4())
        
        # Select or generate random entity type
        if entity_type is None:
            entity_type = random.choice(self.ENTITY_TYPES)
        
        # Generate entity name based on type
        adj = random.choice(self.ADJECTIVES)
        if entity_type == "person":
            first_names = ["Alex", "Jamie", "Jordan", "Taylor", "Casey", "Morgan", "Riley", "Quinn"]
            last_names = ["Smith", "Johnson", "Brown", "Davis", "Wilson", "Miller", "Jones", "Garcia"]
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
        elif entity_type == "technology":
            name = random.choice(self.TECHNOLOGIES)
        else:
            subj = random.choice(self.SUBJECTS)
            name = f"{adj} {subj} {random.randint(1, 100)}"
        
        # Generate basic properties
        properties = {
            "name": name,
            "type": entity_type,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        # Generate additional properties based on complexity
        num_properties = random.randint(1, self.avg_properties * 2)
        
        property_keys = [
            "description", "status", "priority", "version", "category",
            "location", "department", "email", "phone", "website",
            "start_date", "end_date", "budget", "tags", "owner"
        ]
        
        for _ in range(num_properties):
            key = random.choice(property_keys)
            if key not in properties:
                if key == "description":
                    properties[key] = self.generate_sentence(3, 10)
                elif key == "status":
                    properties[key] = random.choice(["active", "inactive", "pending", "completed"])
                elif key == "priority":
                    properties[key] = random.choice(["low", "medium", "high"])
                elif key == "tags":
                    num_tags = random.randint(1, 5)
                    properties[key] = [random.choice(self.TECHNOLOGIES) for _ in range(num_tags)]
                else:
                    properties[key] = f"Value-{random.randint(1, 1000)}"
        
        return entity_id, properties
    
    def generate_random_relation(self) -> Dict[str, Any]:
        """
        Generate a random relation between entities.
        
        Returns:
            Relation data dictionary
        """
        # Select random source and target entities
        source_id = random.choice(list(self.entities.keys()))
        target_id = random.choice(list(self.entities.keys()))
        
        # Choose random relation type
        relation_type = random.choice(self.RELATION_TYPES)
        
        # Generate relation properties
        properties = {
            "created_at": datetime.datetime.now().isoformat(),
            "confidence": random.uniform(0.5, 1.0),
            "source": "generated_data"
        }
        
        # Add more properties for complex relations
        if self.complexity == "complex":
            properties["description"] = self.generate_sentence(2, 8)
            if random.random() > 0.7:
                properties["weight"] = random.uniform(0.1, 10.0)
            if random.random() > 0.7:
                properties["bidirectional"] = random.choice([True, False])
        
        return {
            "id": str(uuid.uuid4()),
            "source_id": source_id,
            "relation_type": relation_type,
            "target_id": target_id,
            "properties": properties
        }
    
    def generate_random_document(self) -> Dict[str, Any]:
        """
        Generate a random document.
        
        Returns:
            Document data dictionary
        """
        # Generate document title
        adj = random.choice(self.ADJECTIVES)
        subj = random.choice(self.SUBJECTS)
        title = f"{adj} {subj} Documentation"
        
        # Generate document content
        paragraphs = []
        num_paragraphs = random.randint(1, 5)
        
        for _ in range(num_paragraphs):
            sentences = []
            num_sentences = random.randint(3, 10)
            
            for _ in range(num_sentences):
                sentences.append(self.generate_sentence(5, 15))
            
            paragraphs.append(" ".join(sentences))
        
        content = "\n\n".join(paragraphs)
        
        # Ensure content is within length limits
        if len(content) > self.max_text_length:
            content = content[:self.max_text_length]
        
        # Generate metadata
        metadata = {
            "title": title,
            "created_at": datetime.datetime.now().isoformat(),
            "author": f"User-{random.randint(1, 100)}",
            "category": random.choice(["Documentation", "Report", "Memo", "Guide", "Specification"]),
            "tags": [random.choice(self.TECHNOLOGIES) for _ in range(random.randint(1, 3))]
        }
        
        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "properties": metadata
        }
    
    def generate_sentence(self, min_words: int = 5, max_words: int = 15) -> str:
        """
        Generate a random sentence.
        
        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words
            
        Returns:
            Generated sentence
        """
        word_lists = [
            self.SUBJECTS, self.ADJECTIVES, self.TECHNOLOGIES,
            ["is", "was", "will be", "can be", "should be", "could be"],
            ["using", "with", "without", "through", "via", "by", "for"],
            ["the", "a", "an", "some", "many", "few"]
        ]
        
        num_words = random.randint(min_words, max_words)
        words = []
        
        for _ in range(num_words):
            word_list = random.choice(word_lists)
            words.append(random.choice(word_list))
        
        # Simple capitalization and period
        sentence = " ".join(words)
        sentence = sentence[0].upper() + sentence[1:] + "."
        
        return sentence
    
    def generate_entities(self, num_entities: int) -> Dict[str, Dict[str, Any]]:
        """
        Generate a specified number of entities.
        
        Args:
            num_entities: Number of entities to generate
            
        Returns:
            Dictionary of entity IDs to entity data
        """
        entities = {}
        
        for _ in range(num_entities):
            entity_id, entity_data = self.generate_random_entity()
            entities[entity_id] = entity_data
            
        self.entities = entities
        return entities
    
    def generate_relations(self, entities: Dict[str, Dict[str, Any]], num_relations: int) -> List[Dict[str, Any]]:
        """
        Generate a specified number of relations between entities.
        
        Args:
            entities: Dictionary of entity IDs to entity data
            num_relations: Number of relations to generate
            
        Returns:
            List of relation dictionaries
        """
        if not entities:
            raise ValueError("No entities provided for relation generation")
            
        self.entities = entities
        relations = []
        
        for _ in range(num_relations):
            relation = self.generate_random_relation()
            relations.append(relation)
            
        self.relations = relations
        return relations
    
    def generate_documents(self, entities: Dict[str, Dict[str, Any]], num_documents: int) -> List[Dict[str, Any]]:
        """
        Generate a specified number of documents.
        
        Args:
            entities: Dictionary of entity IDs to entity data
            num_documents: Number of documents to generate
            
        Returns:
            List of document dictionaries
        """
        if not entities:
            raise ValueError("No entities provided for document generation")
            
        self.entities = entities
        documents = []
        
        for _ in range(num_documents):
            document = self.generate_random_document()
            documents.append(document)
            
        self.documents = documents
        return documents
    
    def generate_data(self) -> Dict[str, Any]:
        """
        Generate the full dataset.
        
        Returns:
            Dictionary with generated entities, relations, and documents
        """
        print(f"Generating {self.num_entities} entities...")
        
        # Generate entities
        for _ in range(self.num_entities):
            entity_id, properties = self.generate_random_entity()
            self.entities[entity_id] = properties
        
        print(f"Generating {self.num_relations} relations...")
        
        # Generate relations
        for _ in range(self.num_relations):
            relation = self.generate_random_relation()
            self.relations.append(relation)
        
        print(f"Generating {self.num_documents} documents...")
        
        # Generate documents
        for _ in range(self.num_documents):
            document = self.generate_random_document()
            self.documents.append(document)
        
        # Return the generated data
        return {
            "entities": self.entities,
            "relations": self.relations,
            "documents": self.documents
        }
    
    def save_data(self, filename: Optional[str] = None) -> str:
        """
        Save generated data to files.
        
        Args:
            filename: Optional base filename (without extension)
            
        Returns:
            Path to saved dataset directory
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_data_{self.complexity}_{timestamp}"
        
        # Create dataset directory
        dataset_dir = os.path.join(self.output_dir, filename)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save entities
        with open(os.path.join(dataset_dir, "entities.json"), 'w') as f:
            json.dump(self.entities, f, indent=2)
        
        # Save relations
        with open(os.path.join(dataset_dir, "relations.json"), 'w') as f:
            json.dump(self.relations, f, indent=2)
        
        # Save documents
        with open(os.path.join(dataset_dir, "documents.json"), 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        # Save metadata
        metadata = {
            "num_entities": len(self.entities),
            "num_relations": len(self.relations),
            "num_documents": len(self.documents),
            "complexity": self.complexity,
            "generated_at": datetime.datetime.now().isoformat(),
            "entity_types": list(set(entity["type"] for entity in self.entities.values())),
            "relation_types": list(set(relation["relation_type"] for relation in self.relations))
        }
        
        with open(os.path.join(dataset_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {dataset_dir}")
        return dataset_dir
    
    def load_data(self, dataset_dir: str) -> Dict[str, Any]:
        """
        Load a previously generated dataset.
        
        Args:
            dataset_dir: Path to dataset directory
            
        Returns:
            Dictionary with loaded entities, relations, and documents
        """
        # Load entities
        with open(os.path.join(dataset_dir, "entities.json"), 'r') as f:
            self.entities = json.load(f)
        
        # Load relations
        with open(os.path.join(dataset_dir, "relations.json"), 'r') as f:
            self.relations = json.load(f)
        
        # Load documents
        with open(os.path.join(dataset_dir, "documents.json"), 'r') as f:
            self.documents = json.load(f)
        
        print(f"Loaded dataset from {dataset_dir}")
        return {
            "entities": self.entities,
            "relations": self.relations,
            "documents": self.documents
        }


def generate_dataset(config: Dict[str, Any]) -> str:
    """
    Generate a dataset based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to the generated dataset
    """
    generator = DataGenerator(
        num_entities=config.get("num_entities", 1000),
        num_relations=config.get("num_relations", 5000),
        num_documents=config.get("num_documents", 100),
        complexity=config.get("dataset_complexity", "medium"),
        output_dir=config.get("output_dir", "benchmark_suite/datasets")
    )
    
    generator.generate_data()
    return generator.save_data(f"benchmark_data_{config.get('dataset_size', 'medium')}") 