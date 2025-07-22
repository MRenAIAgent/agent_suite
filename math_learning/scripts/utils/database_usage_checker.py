#!/usr/bin/env python3
"""
Database Usage Checker for Neo4j and Qdrant

This utility provides functions to check storage usage, node/relationship counts,
collection sizes, and other metrics for both databases.
"""

import asyncio
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DatabaseUsageChecker:
    """Utility class to check usage statistics for Neo4j and Qdrant databases."""
    
    def __init__(self):
        self.neo4j_driver = None
        self.qdrant_client = None
    
    async def check_neo4j_usage(self, uri: str = "bolt://localhost:7687", 
                               username: str = "neo4j", password: str = "password") -> Dict[str, Any]:
        """Check Neo4j database usage statistics."""
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with driver.session() as session:
                # Get node count
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # Get relationship count
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                # Get node labels
                labels_result = session.run("CALL db.labels()")
                labels = [record["label"] for record in labels_result]
                
                # Get relationship types
                types_result = session.run("CALL db.relationshipTypes()")
                rel_types = [record["relationshipType"] for record in types_result]
                
                # Get database info
                try:
                    db_info_result = session.run("CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes') YIELD attributes")
                    db_info = list(db_info_result)
                except Exception:
                    db_info = []
                
                # Get some sample nodes by label
                label_counts = {}
                for label in labels:
                    try:
                        count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                        label_counts[label] = count_result.single()["count"]
                    except Exception:
                        label_counts[label] = 0
            
            driver.close()
            
            return {
                "status": "connected",
                "node_count": node_count,
                "relationship_count": rel_count,
                "labels": labels,
                "relationship_types": rel_types,
                "label_counts": label_counts,
                "database_info": db_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_qdrant_usage(self, host: str = "localhost", port: int = 6333) -> Dict[str, Any]:
        """Check Qdrant database usage statistics."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse
            
            client = QdrantClient(host=host, port=port)
            
            # Get cluster info
            try:
                cluster_info = client.get_cluster_info()
            except Exception:
                cluster_info = None
            
            # Get collections
            collections_response = client.get_collections()
            collections = collections_response.collections
            
            collection_stats = {}
            total_vectors = 0
            
            for collection in collections:
                collection_name = collection.name
                try:
                    # Get collection info
                    collection_info = client.get_collection(collection_name)
                    
                    # Count vectors in collection
                    count_result = client.count(collection_name)
                    vector_count = count_result.count
                    total_vectors += vector_count
                    
                    collection_stats[collection_name] = {
                        "vector_count": vector_count,
                        "config": {
                            "vector_size": collection_info.config.params.vectors.size if hasattr(collection_info.config.params.vectors, 'size') else "unknown",
                            "distance": collection_info.config.params.vectors.distance.value if hasattr(collection_info.config.params.vectors, 'distance') else "unknown"
                        },
                        "status": collection_info.status.value if hasattr(collection_info, 'status') else "unknown"
                    }
                except Exception as e:
                    collection_stats[collection_name] = {
                        "error": str(e)
                    }
            
            return {
                "status": "connected",
                "total_collections": len(collections),
                "total_vectors": total_vectors,
                "collections": collection_stats,
                "cluster_info": cluster_info,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_both_databases(self) -> Dict[str, Any]:
        """Check usage for both Neo4j and Qdrant databases."""
        neo4j_usage = await self.check_neo4j_usage()
        qdrant_usage = await self.check_qdrant_usage()
        
        return {
            "neo4j": neo4j_usage,
            "qdrant": qdrant_usage,
            "timestamp": datetime.now().isoformat()
        }
    
    def print_usage_report(self, usage_data: Dict[str, Any], title: str = "Database Usage Report"):
        """Print a formatted usage report."""
        print(f"\n{'='*80}")
        print(f"📊 {title}")
        print(f"{'='*80}")
        print(f"⏰ Timestamp: {usage_data.get('timestamp', 'Unknown')}")
        
        # Neo4j Report
        print(f"\n🔗 NEO4J DATABASE")
        print(f"{'─'*40}")
        neo4j_data = usage_data.get('neo4j', {})
        
        if neo4j_data.get('status') == 'connected':
            print(f"✅ Status: Connected")
            print(f"📊 Nodes: {neo4j_data.get('node_count', 0):,}")
            print(f"🔗 Relationships: {neo4j_data.get('relationship_count', 0):,}")
            print(f"🏷️  Labels: {len(neo4j_data.get('labels', []))}")
            print(f"🔀 Relationship Types: {len(neo4j_data.get('relationship_types', []))}")
            
            if neo4j_data.get('labels'):
                print(f"\n   Label Breakdown:")
                for label, count in neo4j_data.get('label_counts', {}).items():
                    print(f"   • {label}: {count:,} nodes")
            
            if neo4j_data.get('relationship_types'):
                print(f"\n   Relationship Types: {', '.join(neo4j_data.get('relationship_types', []))}")
        else:
            print(f"❌ Status: {neo4j_data.get('status', 'Unknown')}")
            if neo4j_data.get('error'):
                print(f"   Error: {neo4j_data['error']}")
        
        # Qdrant Report
        print(f"\n🔍 QDRANT DATABASE")
        print(f"{'─'*40}")
        qdrant_data = usage_data.get('qdrant', {})
        
        if qdrant_data.get('status') == 'connected':
            print(f"✅ Status: Connected")
            print(f"📦 Collections: {qdrant_data.get('total_collections', 0)}")
            print(f"🔢 Total Vectors: {qdrant_data.get('total_vectors', 0):,}")
            
            collections = qdrant_data.get('collections', {})
            if collections:
                print(f"\n   Collection Breakdown:")
                for collection_name, stats in collections.items():
                    if 'error' in stats:
                        print(f"   • {collection_name}: Error - {stats['error']}")
                    else:
                        vector_count = stats.get('vector_count', 0)
                        config = stats.get('config', {})
                        vector_size = config.get('vector_size', 'unknown')
                        distance = config.get('distance', 'unknown')
                        print(f"   • {collection_name}: {vector_count:,} vectors (size: {vector_size}, distance: {distance})")
        else:
            print(f"❌ Status: {qdrant_data.get('status', 'Unknown')}")
            if qdrant_data.get('error'):
                print(f"   Error: {qdrant_data['error']}")
        
        print(f"\n{'='*80}")


async def main():
    """Main function to run the database usage checker."""
    checker = DatabaseUsageChecker()
    
    print("🔍 Checking database usage...")
    usage_data = await checker.check_both_databases()
    checker.print_usage_report(usage_data)


if __name__ == "__main__":
    asyncio.run(main()) 