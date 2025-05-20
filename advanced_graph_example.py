#!/usr/bin/env python
"""
Advanced example demonstrating the centralized graph infrastructure.

This example creates a sophisticated knowledge graph representing an organization
with multiple teams, projects, skills, and their interconnections. It shows how to:

1. Create a complex knowledge graph with multiple entity types and relations
2. Perform advanced queries to extract insights
3. Use traversal and neighbor operations to explore connected data
4. Answer complex questions using graph operations

Usage:
    python advanced_graph_example.py
"""

import os
import uuid
from pathlib import Path
from pprint import pprint
import datetime

# Import the centralized graph manager
from agent_suite.graph import GraphManager
from knowledge.graphiti import GraphitiKnowledgeGraphAdapter

def print_header(title):
    """Print a section header with title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")

def print_subheader(title):
    """Print a subsection header."""
    print(f"\n--- {title} ---")

def print_result(item, indent=0):
    """Pretty print a result item."""
    spaces = " " * indent
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, dict):
                print(f"{spaces}{key}:")
                print_result(value, indent + 2)
            elif isinstance(value, list):
                print(f"{spaces}{key}:")
                for i, v in enumerate(value):
                    print(f"{spaces}  [{i}]:")
                    print_result(v, indent + 4)
            else:
                print(f"{spaces}{key}: {value}")
    else:
        print(f"{spaces}{item}")

def main():
    """Run the advanced graph example."""
    print_header("Advanced Knowledge Graph Example")
    
    # Create a persistent graph
    graph_dir = "./knowledge_data"
    os.makedirs(graph_dir, exist_ok=True)
    
    # Use the knowledge graph adapter to access the graph
    knowledge = GraphitiKnowledgeGraphAdapter(
        graph_name="organization_knowledge",
        persist=True,
        persistence_path=graph_dir
    )
    
    # Create the organization knowledge graph
    print_subheader("Building Organization Knowledge Graph")
    
    # Track entity IDs
    entities = {}
    
    # ===== Add Departments =====
    departments = {
        "engineering": {"name": "Engineering Department", "budget": 2000000},
        "product": {"name": "Product Department", "budget": 1500000},
        "marketing": {"name": "Marketing Department", "budget": 1000000},
        "hr": {"name": "Human Resources", "budget": 500000},
    }
    
    for dept_id, props in departments.items():
        entity_id = f"dept:{dept_id}"
        knowledge.add_entity(entity_id, "department", props)
        entities[entity_id] = props["name"]
        print(f"Added department: {props['name']}")
    
    # ===== Add Teams =====
    teams = {
        "backend": {
            "name": "Backend Team",
            "department": "dept:engineering",
            "size": 12
        },
        "frontend": {
            "name": "Frontend Team",
            "department": "dept:engineering",
            "size": 10
        },
        "data": {
            "name": "Data Science Team",
            "department": "dept:engineering",
            "size": 8
        },
        "design": {
            "name": "Design Team",
            "department": "dept:product",
            "size": 6
        },
        "ux": {
            "name": "UX Research Team",
            "department": "dept:product",
            "size": 4
        },
        "content": {
            "name": "Content Team",
            "department": "dept:marketing",
            "size": 7
        },
        "acquisition": {
            "name": "Acquisition Team",
            "department": "dept:marketing",
            "size": 5
        },
        "recruiting": {
            "name": "Recruiting Team",
            "department": "dept:hr",
            "size": 4
        }
    }
    
    for team_id, props in teams.items():
        entity_id = f"team:{team_id}"
        department = props.pop("department")
        knowledge.add_entity(entity_id, "team", props)
        entities[entity_id] = props["name"]
        
        # Connect team to department
        knowledge.add_relation(entity_id, "BELONGS_TO", department, {
            "established": f"2020-{(hash(team_id) % 12) + 1:02d}-01"
        })
        
        print(f"Added team: {props['name']}")
    
    # ===== Add Projects =====
    projects = {
        "atlas": {
            "name": "Project Atlas",
            "description": "Core backend infrastructure upgrade",
            "priority": "High",
            "start_date": "2023-01-15",
            "status": "In Progress"
        },
        "phoenix": {
            "name": "Project Phoenix",
            "description": "Frontend redesign and modernization",
            "priority": "Medium",
            "start_date": "2023-02-01",
            "status": "In Progress"
        },
        "insight": {
            "name": "Project Insight",
            "description": "Customer analytics platform",
            "priority": "High",
            "start_date": "2023-03-10",
            "status": "Planning"
        },
        "pulse": {
            "name": "Project Pulse",
            "description": "Real-time monitoring dashboard",
            "priority": "Medium",
            "start_date": "2023-04-05",
            "status": "In Progress"
        },
        "horizon": {
            "name": "Project Horizon",
            "description": "New market expansion campaign",
            "priority": "High",
            "start_date": "2023-03-15",
            "status": "Planning"
        },
        "talent": {
            "name": "Project Talent",
            "description": "Employee development program",
            "priority": "Medium",
            "start_date": "2023-02-15",
            "status": "In Progress"
        }
    }
    
    # Project team assignments (project -> list of teams)
    project_teams = {
        "atlas": ["backend", "data"],
        "phoenix": ["frontend", "design", "ux"],
        "insight": ["data", "backend", "ux"],
        "pulse": ["frontend", "backend", "design"],
        "horizon": ["content", "acquisition"],
        "talent": ["recruiting"]
    }
    
    for proj_id, props in projects.items():
        entity_id = f"project:{proj_id}"
        knowledge.add_entity(entity_id, "project", props)
        entities[entity_id] = props["name"]
        
        # Connect project to teams
        for team_id in project_teams[proj_id]:
            knowledge.add_relation(f"team:{team_id}", "WORKS_ON", entity_id, {
                "assigned": props["start_date"],
                "role": "Primary" if team_id == project_teams[proj_id][0] else "Supporting"
            })
        
        print(f"Added project: {props['name']}")
    
    # ===== Add Skills =====
    skills = [
        {"id": "python", "name": "Python", "category": "Programming"},
        {"id": "java", "name": "Java", "category": "Programming"},
        {"id": "javascript", "name": "JavaScript", "category": "Programming"},
        {"id": "react", "name": "React", "category": "Frontend"},
        {"id": "angular", "name": "Angular", "category": "Frontend"},
        {"id": "node", "name": "Node.js", "category": "Backend"},
        {"id": "sql", "name": "SQL", "category": "Database"},
        {"id": "nosql", "name": "NoSQL", "category": "Database"},
        {"id": "ml", "name": "Machine Learning", "category": "Data Science"},
        {"id": "stats", "name": "Statistics", "category": "Data Science"},
        {"id": "ui", "name": "UI Design", "category": "Design"},
        {"id": "ux", "name": "UX Design", "category": "Design"},
        {"id": "content", "name": "Content Creation", "category": "Marketing"},
        {"id": "seo", "name": "SEO", "category": "Marketing"},
        {"id": "recruiting", "name": "Recruiting", "category": "HR"}
    ]
    
    for skill in skills:
        entity_id = f"skill:{skill['id']}"
        knowledge.add_entity(entity_id, "skill", {
            "name": skill["name"],
            "category": skill["category"]
        })
        entities[entity_id] = skill["name"]
        print(f"Added skill: {skill['name']}")
    
    # Team skill mappings (team -> list of skills)
    team_skills = {
        "backend": ["python", "java", "node", "sql", "nosql"],
        "frontend": ["javascript", "react", "angular"],
        "data": ["python", "ml", "stats", "sql"],
        "design": ["ui", "ux"],
        "ux": ["ux"],
        "content": ["content", "seo"],
        "acquisition": ["seo"],
        "recruiting": ["recruiting"]
    }
    
    # Connect teams to skills
    for team_id, skill_list in team_skills.items():
        for skill_id in skill_list:
            knowledge.add_relation(f"team:{team_id}", "HAS_SKILL", f"skill:{skill_id}", {
                "proficiency": "High" if skill_id == skill_list[0] else "Medium"
            })
    
    # ===== Add People =====
    people = [
        {"id": "alice", "name": "Alice Johnson", "title": "Engineering Director", "department": "dept:engineering"},
        {"id": "bob", "name": "Bob Smith", "title": "Backend Lead", "department": "dept:engineering", "team": "team:backend"},
        {"id": "carol", "name": "Carol Lee", "title": "Frontend Developer", "department": "dept:engineering", "team": "team:frontend"},
        {"id": "dave", "name": "Dave Chen", "title": "Data Scientist", "department": "dept:engineering", "team": "team:data"},
        {"id": "eve", "name": "Eve Wilson", "title": "Product Manager", "department": "dept:product", "team": "team:design"},
        {"id": "frank", "name": "Frank Miller", "title": "UX Researcher", "department": "dept:product", "team": "team:ux"},
        {"id": "grace", "name": "Grace Kim", "title": "Marketing Lead", "department": "dept:marketing", "team": "team:content"},
        {"id": "henry", "name": "Henry Garcia", "title": "HR Director", "department": "dept:hr", "team": "team:recruiting"}
    ]
    
    # Person skill mappings
    person_skills = {
        "alice": ["python", "java", "ml"],
        "bob": ["python", "java", "node", "sql"],
        "carol": ["javascript", "react", "angular"],
        "dave": ["python", "ml", "stats"],
        "eve": ["ux", "ui"],
        "frank": ["ux"],
        "grace": ["content", "seo"],
        "henry": ["recruiting"]
    }
    
    # Person project mappings
    person_projects = {
        "alice": ["atlas", "phoenix"],
        "bob": ["atlas"],
        "carol": ["phoenix", "pulse"],
        "dave": ["insight"],
        "eve": ["phoenix", "pulse"],
        "frank": ["insight"],
        "grace": ["horizon"],
        "henry": ["talent"]
    }
    
    for person in people:
        entity_id = f"person:{person['id']}"
        
        # Extract and remove non-property fields
        department = person.pop("department", None)
        team = person.pop("team", None)
        
        # Add the person entity
        knowledge.add_entity(entity_id, "person", person)
        entities[entity_id] = person["name"]
        
        # Connect person to department
        if department:
            knowledge.add_relation(entity_id, "WORKS_IN", department, {
                "joined": f"2020-{(hash(person['id']) % 12) + 1:02d}-01"
            })
        
        # Connect person to team
        if team:
            knowledge.add_relation(entity_id, "MEMBER_OF", team, {
                "joined": f"2021-{(hash(person['id']) % 12) + 1:02d}-01"
            })
        
        # Connect person to skills
        for skill_id in person_skills.get(person["id"], []):
            knowledge.add_relation(entity_id, "KNOWS", f"skill:{skill_id}", {
                "level": "Expert" if skill_id == person_skills[person["id"]][0] else "Proficient",
                "years": hash(person["id"] + skill_id) % 10 + 1
            })
        
        # Connect person to projects
        for project_id in person_projects.get(person["id"], []):
            knowledge.add_relation(entity_id, "ASSIGNED_TO", f"project:{project_id}", {
                "role": "Lead" if project_id == person_projects[person["id"]][0] else "Member",
                "assigned_date": projects[project_id]["start_date"]
            })
        
        print(f"Added person: {person['name']}")
    
    # Add relationships between people
    relationships = [
        ("alice", "bob", "MANAGES"),
        ("bob", "carol", "MANAGES"),
        ("alice", "dave", "MANAGES"),
        ("eve", "frank", "MANAGES"),
        ("grace", "henry", "COLLABORATES_WITH"),
        ("bob", "dave", "COLLABORATES_WITH"),
        ("carol", "eve", "COLLABORATES_WITH"),
        ("frank", "dave", "COLLABORATES_WITH")
    ]
    
    for source, target, relation in relationships:
        knowledge.add_relation(f"person:{source}", relation, f"person:{target}")
        print(f"Added relationship: {source} {relation} {target}")
    
    # Add documents
    documents = [
        {
            "title": "Project Atlas Technical Specification",
            "content": "This document outlines the technical architecture for Project Atlas, including database schema design, API endpoints, and scaling considerations.",
            "related_project": "project:atlas"
        },
        {
            "title": "Project Phoenix UI Guidelines",
            "content": "This document provides the design system guidelines for Project Phoenix, including color schemes, component specifications, and responsive design patterns.",
            "related_project": "project:phoenix"
        },
        {
            "title": "Data Science Strategy 2023",
            "content": "This document outlines our data science strategy for 2023, focusing on machine learning applications, data infrastructure, and analytics capabilities.",
            "related_team": "team:data"
        },
        {
            "title": "Marketing Plan Q2 2023",
            "content": "This document details the marketing strategy for Q2 2023, including campaign planning, content calendar, and performance metrics.",
            "related_project": "project:horizon"
        }
    ]
    
    for doc in documents:
        # Extract related entity
        related_project = doc.pop("related_project", None)
        related_team = doc.pop("related_team", None)
        
        # Add document text
        doc_id = knowledge.add_text(doc["content"], {
            "title": doc["title"],
            "created_at": datetime.datetime.now().isoformat()
        })
        
        # Connect document to project or team
        if related_project:
            knowledge.add_relation(related_project, "HAS_DOCUMENTATION", doc_id)
        
        if related_team:
            knowledge.add_relation(related_team, "HAS_DOCUMENTATION", doc_id)
        
        print(f"Added document: {doc['title']}")
    
    # ===== Example Queries =====
    print_header("Example Knowledge Queries")
    
    # Query 1: Who works on Project Atlas?
    print_subheader("Who works on Project Atlas?")
    
    # First find teams working on Atlas
    teams_on_atlas = knowledge.query_relations(
        relation_type="WORKS_ON",
        target_id="project:atlas"
    )
    
    team_ids = [relation["source_id"] for relation in teams_on_atlas]
    
    # Then find people in those teams
    atlas_people = []
    for team_id in team_ids:
        people_in_team = knowledge.query_relations(
            relation_type="MEMBER_OF",
            target_id=team_id
        )
        atlas_people.extend([relation["source_id"] for relation in people_in_team])
    
    # Also find people directly assigned to project
    directly_assigned = knowledge.query_relations(
        relation_type="ASSIGNED_TO",
        target_id="project:atlas"
    )
    atlas_people.extend([relation["source_id"] for relation in directly_assigned])
    
    # Remove duplicates
    atlas_people = list(set(atlas_people))
    
    # Get details about these people
    for person_id in atlas_people:
        person = knowledge.get_entity(person_id)
        if person:
            role = "Unknown"
            for relation in directly_assigned:
                if relation["source_id"] == person_id:
                    role = relation["properties"].get("role", "Unknown")
            
            print(f"- {person['properties']['name']} ({role})")
    
    # Query 2: What skills are needed for Project Phoenix?
    print_subheader("What skills are needed for Project Phoenix?")
    
    # Get teams working on Phoenix
    phoenix_teams = knowledge.query_relations(
        relation_type="WORKS_ON",
        target_id="project:phoenix"
    )
    
    # Get skills for those teams
    phoenix_skills = set()
    for relation in phoenix_teams:
        team_id = relation["source_id"]
        
        skill_relations = knowledge.query_relations(
            source_id=team_id,
            relation_type="HAS_SKILL"
        )
        
        for skill_relation in skill_relations:
            skill_id = skill_relation["target_id"]
            skill = knowledge.get_entity(skill_id)
            if skill:
                phoenix_skills.add((skill_id, skill["properties"]["name"], skill["properties"]["category"]))
    
    # Group skills by category
    skill_categories = {}
    for skill_id, skill_name, category in phoenix_skills:
        if category not in skill_categories:
            skill_categories[category] = []
        skill_categories[category].append(skill_name)
    
    # Print results
    for category, skills in skill_categories.items():
        print(f"- {category}: {', '.join(skills)}")
    
    # Query 3: Find path between people (traversal)
    print_subheader("Connection path between Bob and Frank")
    
    # Use graph traversal to find connections
    graph = knowledge.graph_manager
    
    # Start from Bob and explore the graph
    traversal = graph.graph_traversal("person:bob", max_depth=3)
    
    # Find if Frank is in the traversal
    frank_node = None
    for node in traversal["nodes"]:
        if node["id"] == "person:frank":
            frank_node = node
            break
    
    if frank_node:
        print("Found connection path:")
        
        # Find the path by exploring the edges in traversal
        # This is a simplified path finding - a real system would use proper path finding algorithms
        
        # Find all edges from bob
        bob_edges = [edge for edge in traversal["edges"] if edge["source"] == "person:bob"]
        
        # Check if there's a direct connection
        direct_edge = None
        for edge in bob_edges:
            if edge["target"] == "person:frank":
                direct_edge = edge
                break
        
        if direct_edge:
            print(f"- Bob {direct_edge['type']} Frank directly")
        else:
            # Look for indirect connections (this is simplified)
            indirect_paths = []
            
            # Check each of Bob's connections
            for bob_edge in bob_edges:
                intermediate_id = bob_edge["target"]
                
                # Check if this intermediate node connects to Frank
                for edge in traversal["edges"]:
                    if edge["source"] == intermediate_id and edge["target"] == "person:frank":
                        # Found a two-step path
                        intermediate = knowledge.get_entity(intermediate_id)
                        intermediate_name = intermediate["properties"].get("name", intermediate_id)
                        
                        path = {
                            "intermediate_id": intermediate_id,
                            "intermediate_name": intermediate_name,
                            "bob_to_intermediate": bob_edge["type"],
                            "intermediate_to_frank": edge["type"]
                        }
                        indirect_paths.append(path)
            
            if indirect_paths:
                for path in indirect_paths:
                    print(f"- Bob {path['bob_to_intermediate']} {path['intermediate_name']} who {path['intermediate_to_frank']} Frank")
            else:
                # Try three-step paths (simplified)
                print("- More distant connection found (3+ steps)")
    else:
        print("No connection found within 3 steps")
    
    # Query 4: Search for relevant information about machine learning
    print_subheader("Information about Machine Learning")
    
    # Keyword search for ML
    ml_results = knowledge.semantic_search("machine learning", limit=10)
    
    for result in ml_results:
        if result.get("match_type") == "entity" and result.get("entity_type") == "skill":
            # ML skill entity
            print(f"- Skill: {result['properties'].get('name')}")
            
            # Find people who know this skill
            skill_id = result["id"]
            people_with_skill = knowledge.query_relations(
                relation_type="KNOWS",
                target_id=skill_id
            )
            
            if people_with_skill:
                print("  Known by:")
                for relation in people_with_skill:
                    person_id = relation["source_id"]
                    person = knowledge.get_entity(person_id)
                    if person:
                        level = relation["properties"].get("level", "unknown")
                        name = person["properties"].get("name", person_id)
                        print(f"  - {name} (Level: {level})")
        
        elif result.get("match_type") == "text" or result.get("entity_type") == "text":
            # Text content
            content = result.get("content", result.get("properties", {}).get("content", ""))
            title = result.get("properties", {}).get("title", "Untitled")
            
            if content and "machine learning" in content.lower():
                print(f"- Document: {title}")
                print(f"  Excerpt: {content[:100]}...")
    
    # Print Graph Statistics
    print_header("Knowledge Graph Statistics")
    
    stats = knowledge.get_stats()
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relations: {stats['total_relations']}")
    
    print("\nEntity types:")
    for entity_type, count in stats["entity_counts"].items():
        print(f"- {entity_type}: {count}")
    
    print("\nRelation types:")
    for relation_type, count in stats["relation_counts"].items():
        print(f"- {relation_type}: {count}")

if __name__ == "__main__":
    main() 