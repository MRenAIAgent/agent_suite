#!/usr/bin/env python
"""
Example script demonstrating how to integrate the knowledge base with agents.

This script shows how an agent can:
1. Store knowledge from interactions
2. Retrieve relevant information during conversations
3. Use knowledge to improve responses
"""

import os
import json
from typing import Dict, Any, List, Optional

from knowledge.factory import KnowledgeBaseFactory


class KnowledgeAwareAgent:
    """A simple agent that uses a knowledge base to improve responses."""
    
    def __init__(self, knowledge_type: str = "combined", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent with a knowledge base.
        
        Args:
            knowledge_type: Type of knowledge base to use ('graph', 'vector', or 'combined')
            config: Optional configuration for the knowledge base
        """
        if config is None:
            # Default config with persistence
            knowledge_dir = os.path.join(os.getcwd(), "knowledge_data", "agent")
            os.makedirs(knowledge_dir, exist_ok=True)
            
            config = {
                "kg_graph_name": "agent_knowledge",
                "vector_collection_name": "agent_memory",
                "persist": True,
                "persistence_path": knowledge_dir
            }
        
        # Initialize the knowledge base
        self.knowledge_base = KnowledgeBaseFactory.create_knowledge_base(knowledge_type, config)
        
        # Track conversation history
        self.conversation_history = []
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text (simplified version).
        
        In a real implementation, this would use NER models or similar.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        # This is a simplified mock implementation
        # In a real system, use NLP libraries for entity extraction
        entities = []
        
        # Very simple person detection
        if "my name is" in text.lower():
            name = text.lower().split("my name is")[1].strip().split()[0].capitalize()
            entities.append({
                "entity_id": f"person-{len(self.conversation_history)}",
                "entity_type": "person",
                "properties": {"name": name}
            })
        
        # Simple topic detection
        topics = {
            "python": "programming language",
            "javascript": "programming language",
            "machine learning": "ai technology",
            "deep learning": "ai technology",
            "knowledge graph": "data structure"
        }
        
        for topic, description in topics.items():
            if topic.lower() in text.lower():
                entities.append({
                    "entity_id": f"topic-{topic.replace(' ', '_')}",
                    "entity_type": "topic",
                    "properties": {
                        "name": topic,
                        "description": description
                    }
                })
        
        return entities
    
    def _extract_relations(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relations between entities (simplified version).
        
        Args:
            text: Input text
            entities: Extracted entities
            
        Returns:
            List of relations
        """
        # This is a simplified mock implementation
        # In a real system, use more sophisticated NLP for relation extraction
        relations = []
        
        # Only try to extract relations if we have at least 2 entities
        if len(entities) >= 2:
            # Simple relation: person knows about topic
            person_entities = [e for e in entities if e["entity_type"] == "person"]
            topic_entities = [e for e in entities if e["entity_type"] == "topic"]
            
            if person_entities and topic_entities and "know" in text.lower():
                for person in person_entities:
                    for topic in topic_entities:
                        relations.append({
                            "source_id": person["entity_id"],
                            "relation_type": "knows_about",
                            "target_id": topic["entity_id"],
                            "properties": {
                                "confidence": 0.8,
                                "extracted_from": f"message_{len(self.conversation_history)}"
                            }
                        })
        
        return relations
    
    def process_message(self, message: str) -> str:
        """
        Process an incoming message and generate a response.
        
        Args:
            message: The message to process
            
        Returns:
            The agent's response
        """
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Extract knowledge from the message
        entities = self._extract_entities(message)
        
        # Store extracted entities
        for entity in entities:
            # Check if entity already exists
            existing = self.knowledge_base.query_entities(
                entity_type=entity["entity_type"],
                properties={"name": entity["properties"]["name"]}
            )
            
            if not existing:
                # Add new entity
                self.knowledge_base.add_entity(
                    entity_id=entity["entity_id"],
                    entity_type=entity["entity_type"],
                    properties=entity["properties"]
                )
        
        # Extract and store relations
        relations = self._extract_relations(message, entities)
        for relation in relations:
            try:
                self.knowledge_base.add_relation(
                    source_id=relation["source_id"],
                    relation_type=relation["relation_type"],
                    target_id=relation["target_id"],
                    properties=relation["properties"]
                )
            except ValueError:
                # Skip if entities don't exist
                pass
        
        # Store the message text for future reference
        text_id = self.knowledge_base.add_text(
            text=message,
            properties={
                "type": "user_message",
                "message_index": len(self.conversation_history) - 1
            }
        )
        
        # Retrieve relevant information from knowledge base
        semantic_results = self.knowledge_base.semantic_search(message, limit=3)
        
        # Generate a response using the retrieved knowledge
        response = self._generate_response(message, semantic_results)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Store the response
        self.knowledge_base.add_text(
            text=response,
            properties={
                "type": "assistant_message",
                "message_index": len(self.conversation_history) - 1
            }
        )
        
        return response
    
    def _generate_response(self, message: str, knowledge_results: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the message and retrieved knowledge.
        
        In a real implementation, this would use an LLM or similar.
        
        Args:
            message: The user's message
            knowledge_results: Retrieved knowledge from the knowledge base
            
        Returns:
            The generated response
        """
        # This is a simplified implementation
        # In a real system, use an LLM with the retrieved knowledge as context
        
        # Check for greetings
        if any(greeting in message.lower() for greeting in ["hello", "hi", "hey"]):
            return "Hello! How can I help you today?"
        
        # Check if we have relevant knowledge
        if knowledge_results:
            # Use the most relevant result to inform the response
            top_result = knowledge_results[0]
            
            if top_result["type"] == "text":
                response = f"I recall some information that might be relevant: {top_result['content'][:100]}..."
            elif top_result["type"] == "entity":
                entity_type = top_result.get("entity_type", "item")
                props = top_result.get("properties", {})
                name = props.get("name", "this")
                description = props.get("description", f"a {entity_type}")
                response = f"I know about {name}, which is {description}."
            else:
                response = "I have some information that might be helpful, but I'm not sure how to describe it."
                
            return response
        
        # Default response
        return "I'm still learning about that. Could you tell me more?"
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the agent's knowledge.
        
        Returns:
            Dictionary with knowledge statistics
        """
        return self.knowledge_base.get_stats()


def run_demo():
    """Run a simple interactive demo of the knowledge-aware agent."""
    print("Knowledge-Aware Agent Demo")
    print("==========================")
    print("This agent stores knowledge from your conversation and uses it to improve responses.")
    print("Type 'exit' to end the conversation and see knowledge statistics.\n")
    
    # Create the agent
    agent = KnowledgeAwareAgent()
    
    # Main interaction loop
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
            
        response = agent.process_message(user_input)
        print(f"Agent: {response}\n")
    
    # Show knowledge stats at the end
    print("\nKnowledge Base Statistics:")
    stats = agent.get_knowledge_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    run_demo() 