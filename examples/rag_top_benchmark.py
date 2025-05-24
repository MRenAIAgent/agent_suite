#!/usr/bin/env python3
"""
RAG Top Benchmark

This script runs benchmarks for the top 3 RAG metrics using the top 3 datasets.
Metrics evaluated:
1. Retrieval accuracy/context relevance
2. Answer relevance/quality
3. Factual correctness/hallucination detection

Datasets used:
1. Natural Questions (NQ)
2. MS MARCO
3. HotpotQA
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from agent_suite.rag.models.document import Document
from agent_suite.rag.api.rag_service import RagService
from agent_suite.rag.middleware.storage_router import StorageType
from agent_suite.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor, QDRANT_AVAILABLE
from agent_suite.rag.utils.embedding_provider import SentenceTransformerEmbeddingProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset information
DATASET_INFO = {
    "nq": {
        "name": "Natural Questions",
        "description": "Real user queries from Google with answers from Wikipedia",
        "source": "https://ai.google.com/research/NaturalQuestions",
        "sample_path": "benchmark_suite/datasets/nq_sample.jsonl"
    },
    "msmarco": {
        "name": "MS MARCO",
        "description": "Real search queries from Bing with human-written answers",
        "source": "https://microsoft.github.io/msmarco/",
        "sample_path": "benchmark_suite/datasets/msmarco_sample.jsonl"
    },
    "hotpotqa": {
        "name": "HotpotQA",
        "description": "Multi-hop questions requiring reasoning across documents",
        "source": "https://hotpotqa.github.io/",
        "sample_path": "benchmark_suite/datasets/hotpotqa_sample.jsonl"
    }
}

class RagTopBenchmark:
    """Benchmark the top 3 RAG metrics using the top 3 datasets"""
    
    def __init__(self, 
                config: Dict[str, Any],
                datasets: List[str] = None,
                output_dir: str = "benchmark_results"):
        """
        Initialize the RAG benchmarking framework.
        
        Args:
            config: Configuration dictionary with settings for the RAG service
            datasets: List of dataset IDs to benchmark (default: all)
            output_dir: Directory to save benchmark results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use all datasets if none specified
        self.datasets = datasets or list(DATASET_INFO.keys())
        
        # Metrics to evaluate
        self.metrics = {
            "context_relevance": [],
            "answer_relevance": [],
            "factual_correctness": [],
            "latency": {
                "retrieval": [],
                "generation": [],
                "total": []
            }
        }
        
        # Load test sets
        self.test_sets = self._load_test_sets()
        
    def _load_test_sets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load the benchmark test sets.
        
        Returns:
            Dictionary of dataset_id -> test cases
        """
        test_sets = {}
        
        for dataset_id in self.datasets:
            if dataset_id not in DATASET_INFO:
                logger.warning(f"Unknown dataset: {dataset_id}, skipping")
                continue
                
            dataset_path = DATASET_INFO[dataset_id]["sample_path"]
            
            # Check if sample dataset exists
            if not Path(dataset_path).exists():
                logger.warning(f"Dataset file not found: {dataset_path}")
                # Create mock data if real dataset doesn't exist
                test_sets[dataset_id] = self._create_mock_dataset(dataset_id)
                continue
                
            # Load dataset
            logger.info(f"Loading dataset: {DATASET_INFO[dataset_id]['name']}")
            
            # Load JSONL file
            data = []
            with open(dataset_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    data.append(item)
            
            test_sets[dataset_id] = data
            logger.info(f"Loaded {len(data)} examples from {dataset_id}")
        
        return test_sets
            
    def _create_mock_dataset(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Create a mock dataset if the real one is not available.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            List of mock test cases
        """
        logger.warning(f"Creating mock dataset for {dataset_id}")
        
        if dataset_id == "nq":
            return [
                {
                    "question": "What is the capital of France?",
                    "answer": "Paris",
                    "context": ["Paris is the capital and most populous city of France."]
                },
                {
                    "question": "Who invented the telephone?",
                    "answer": "Alexander Graham Bell",
                    "context": ["Alexander Graham Bell was credited with inventing the first practical telephone."]
                },
                {
                    "question": "What year did World War II end?",
                    "answer": "1945",
                    "context": ["World War II ended in 1945 with the victory of the Allies over the Axis powers."]
                }
            ]
        elif dataset_id == "msmarco":
            return [
                {
                    "query": "what is a corporation?",
                    "passages": [
                        "A corporation is a legal entity that is separate and distinct from its owners.",
                        "Corporations enjoy most of the rights and responsibilities that individuals possess."
                    ],
                    "answer": "A corporation is a legal entity that is separate and distinct from its owners."
                },
                {
                    "query": "how to make pancakes?",
                    "passages": [
                        "Mix flour, baking powder, salt, and sugar in a bowl.",
                        "Add milk, eggs, and butter, then stir until smooth.",
                        "Heat a lightly oiled griddle and pour batter to form pancakes."
                    ],
                    "answer": "To make pancakes, mix dry ingredients, add wet ingredients, then cook on a griddle."
                }
            ]
        elif dataset_id == "hotpotqa":
            return [
                {
                    "question": "The actor who played Darth Vader in the original Star Wars, was born in what country?",
                    "supporting_facts": [
                        "David Prowse played Darth Vader in the original Star Wars trilogy.",
                        "David Prowse was born in Bristol, England."
                    ],
                    "answer": "England"
                },
                {
                    "question": "Which novel featuring the character Hannibal Lecter came first?",
                    "supporting_facts": [
                        "Hannibal Lecter is a fictional character in a series of novels by Thomas Harris.",
                        "Red Dragon (1981) was the first novel to feature the character."
                    ],
                    "answer": "Red Dragon"
                }
            ]
            
        return []
    
    async def setup_rag_service(self) -> RagService:
        """
        Set up the RAG service for benchmarking.
        
        Returns:
            Configured RagService
        """
        logger.info("Setting up RAG service for benchmarking...")
        
        # Create RAG service
        service = RagService()
        
        # Check if Qdrant is available
        if not QDRANT_AVAILABLE:
            logger.error("Qdrant client not available. Please install with 'pip install qdrant-client'")
            raise ImportError("Qdrant client not available")
        
        # Get config settings
        collection_name = self.config.get("collection_name", "benchmark_collection")
        vector_size = self.config.get("vector_size", 384)
        host = self.config.get("qdrant_host", "localhost")
        port = self.config.get("qdrant_port", 6333)
        embedding_model = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        
        try:
            # Try to use SentenceTransformer for embeddings
            embedding_provider = SentenceTransformerEmbeddingProvider(model_name=embedding_model)
            logger.info(f"Using SentenceTransformer '{embedding_model}' for embeddings")
        except (ImportError, Exception) as e:
            logger.warning(f"Could not create SentenceTransformer provider: {e}")
            raise ImportError("SentenceTransformer is required for this benchmark")
        
        # Create Qdrant adaptor
        qdrant_adaptor = QdrantStorageAdaptor(
            collection_name=collection_name,
            vector_size=vector_size,
            embedding_provider=embedding_provider,
            host=host,
            port=port
        )
        
        # Register vector storage adaptor
        service.register_storage_adaptor(StorageType.VECTOR, qdrant_adaptor)
        logger.info(f"Registered Qdrant adaptor with collection: {collection_name}")
        
        return service
    
    async def load_dataset_documents(self, service: RagService, dataset_id: str) -> List[str]:
        """
        Load documents from a dataset into the RAG service.
        
        Args:
            service: The RAG service to load documents into
            dataset_id: Dataset identifier
            
        Returns:
            List of document IDs
        """
        logger.info(f"Loading documents from {DATASET_INFO[dataset_id]['name']}...")
        
        test_set = self.test_sets[dataset_id]
        doc_ids = []
        
        # Each dataset has a different structure
        if dataset_id == "nq":
            # For NQ, we use the contexts as documents
            for item in test_set:
                for context in item.get("context", []):
                    doc = Document(
                        content=context,
                        metadata={"source": "nq", "question": item.get("question", "")}
                    )
                    doc_id = await service.store_document(doc)
                    doc_ids.append(doc_id)
        
        elif dataset_id == "msmarco":
            # For MS MARCO, we use passages as documents
            for item in test_set:
                for passage in item.get("passages", []):
                    doc = Document(
                        content=passage,
                        metadata={"source": "msmarco", "query": item.get("query", "")}
                    )
                    doc_id = await service.store_document(doc)
                    doc_ids.append(doc_id)
                    
        elif dataset_id == "hotpotqa":
            # For HotpotQA, we use supporting facts as documents
            for item in test_set:
                for fact in item.get("supporting_facts", []):
                    doc = Document(
                        content=fact,
                        metadata={"source": "hotpotqa", "question": item.get("question", "")}
                    )
                    doc_id = await service.store_document(doc)
                    doc_ids.append(doc_id)
        
        logger.info(f"Loaded {len(doc_ids)} documents from {dataset_id}")
        return doc_ids
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the RAG benchmark on all datasets.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting RAG top benchmark...")
        all_results = {}
        
        for dataset_id in self.datasets:
            logger.info(f"\n=== Benchmarking on {DATASET_INFO[dataset_id]['name']} ===")
            
            # Set up RAG service
            service = await self.setup_rag_service()
            
            # Load dataset documents
            await self.load_dataset_documents(service, dataset_id)
            
            # Small delay to ensure indexing
            await asyncio.sleep(2)
            
            # Run tests on this dataset
            dataset_results = await self._run_dataset_benchmark(service, dataset_id)
            all_results[dataset_id] = dataset_results
            
            # Clean up
            await service.cleanup()
        
        # Calculate aggregate metrics
        benchmark_results = self._calculate_aggregate_metrics(all_results)
        
        # Save results
        self._save_results(all_results, benchmark_results)
        
        return benchmark_results
    
    async def _run_dataset_benchmark(self, service: RagService, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Run benchmark tests on a specific dataset.
        
        Args:
            service: Configured RAG service
            dataset_id: Dataset identifier
            
        Returns:
            List of test results
        """
        test_set = self.test_sets[dataset_id]
        results = []
        
        for item in tqdm(test_set, desc=f"Testing {dataset_id}"):
            # Get question/query based on dataset format
            if dataset_id == "nq" or dataset_id == "hotpotqa":
                query = item.get("question", "")
            else:  # msmarco
                query = item.get("query", "")
                
            # Skip empty queries
            if not query:
                continue
                
            # Run test case
            start_time = time.time()
            
            # Retrieve documents
            retrieval_start = time.time()
            retrieved_docs = await service.retrieve_documents(query, top_k=5)
            retrieval_time = time.time() - retrieval_start
            
            # Get reference answer
            reference_answer = item.get("answer", "")
            
            # Evaluate retrieved context
            context_texts = [doc.item.content for doc in retrieved_docs]
            context_relevance = self._evaluate_context_relevance(query, context_texts)
            
            # In a real benchmark, we'd generate an answer using an LLM here
            # For this example, we'll simulate the generation
            generation_start = time.time()
            answer = self._simulate_answer_generation(query, context_texts)
            generation_time = time.time() - generation_start
            
            # Calculate metrics
            answer_relevance = self._evaluate_answer_relevance(query, answer)
            factual_correctness = self._evaluate_factual_correctness(answer, reference_answer, context_texts)
            
            # Total time
            total_time = time.time() - start_time
            
            # Store result
            result = {
                "dataset": dataset_id,
                "query": query,
                "reference_answer": reference_answer,
                "answer": answer,
                "retrieved_context": context_texts,
                "metrics": {
                    "context_relevance": context_relevance,
                    "answer_relevance": answer_relevance,
                    "factual_correctness": factual_correctness,
                    "retrieval_latency": retrieval_time,
                    "generation_latency": generation_time,
                    "total_latency": total_time
                }
            }
            
            results.append(result)
            
        return results
    
    def _simulate_answer_generation(self, query: str, context: List[str]) -> str:
        """
        Simulate answer generation (in a real system, this would use an LLM).
        
        Args:
            query: User query
            context: Retrieved context passages
            
        Returns:
            Generated answer
        """
        # This is a simple simulation - in a real system this would call an LLM
        if not context:
            return "I don't have enough information to answer this question."
            
        # Simple heuristic: use the first sentence of the most relevant context
        if query.lower().startswith("what is") or query.lower().startswith("who is"):
            return context[0].split(".")[0] + "."
            
        # For other questions, combine first sentences from top contexts
        answer_parts = []
        for ctx in context[:2]:  # Use top 2 contexts
            sentences = ctx.split(".")
            if sentences:
                answer_parts.append(sentences[0])
                
        return ". ".join(answer_parts) + "."
    
    def _evaluate_context_relevance(self, query: str, context: List[str]) -> float:
        """
        Evaluate the relevance of retrieved context to the query.
        
        Args:
            query: User query
            context: Retrieved context passages
            
        Returns:
            Relevance score (0-1)
        """
        # In a real benchmark, this would use a more sophisticated approach
        # For example, using a cross-encoder model to evaluate relevance
        
        # Simple keyword matching as a placeholder
        query_words = set(query.lower().split())
        relevant_contexts = 0
        
        for ctx in context:
            ctx_words = set(ctx.lower().split())
            # If context shares at least 2 non-stopwords with the query, consider it relevant
            if len(query_words.intersection(ctx_words)) >= 2:
                relevant_contexts += 1
                
        # Return the proportion of relevant contexts
        return relevant_contexts / max(1, len(context))
    
    def _evaluate_answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate the relevance of the generated answer to the query.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Relevance score (0-1)
        """
        # In a real benchmark, this would use a model to evaluate answer quality
        # For example, using a supervised model trained on human judgments
        
        # Simple keyword matching as a placeholder
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(answer_words))
        
        # Score based on overlap and length
        score = min(1.0, overlap / max(1, len(query_words)))
        
        return score
    
    def _evaluate_factual_correctness(self, answer: str, reference: str, context: List[str]) -> float:
        """
        Evaluate the factual correctness of the generated answer.
        
        Args:
            answer: Generated answer
            reference: Reference (gold) answer
            context: Retrieved context passages
            
        Returns:
            Correctness score (0-1)
        """
        # In a real benchmark, this would use NLI or a factual consistency model
        # For example, checking if answer is entailed by context
        
        # Simple text overlap with reference answer as a placeholder
        if not reference:
            # If we don't have a reference, check if answer is supported by context
            score = 0.0
            answer_words = set(answer.lower().split())
            
            for ctx in context:
                ctx_words = set(ctx.lower().split())
                overlap = len(answer_words.intersection(ctx_words))
                score = max(score, overlap / max(1, len(answer_words)))
            
            return score
        
        # Compare with reference answer
        answer_words = set(answer.lower().split())
        reference_words = set(reference.lower().split())
        
        # Calculate word overlap
        overlap = len(answer_words.intersection(reference_words))
        
        # Score based on overlap with reference
        score = overlap / max(1, len(reference_words))
        
        return min(1.0, score)
    
    def _calculate_aggregate_metrics(self, results_by_dataset: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics across all datasets.
        
        Args:
            results_by_dataset: Results organized by dataset
            
        Returns:
            Dictionary of aggregate metrics
        """
        # Flatten all results
        all_results = []
        for dataset_results in results_by_dataset.values():
            all_results.extend(dataset_results)
            
        # Aggregate metrics
        context_relevance = np.mean([r["metrics"]["context_relevance"] for r in all_results])
        answer_relevance = np.mean([r["metrics"]["answer_relevance"] for r in all_results])
        factual_correctness = np.mean([r["metrics"]["factual_correctness"] for r in all_results])
        
        # Latency metrics
        retrieval_latency = [r["metrics"]["retrieval_latency"] for r in all_results]
        generation_latency = [r["metrics"]["generation_latency"] for r in all_results]
        total_latency = [r["metrics"]["total_latency"] for r in all_results]
        
        # Calculate percentiles
        retrieval_metrics = {
            "mean": np.mean(retrieval_latency),
            "p50": np.percentile(retrieval_latency, 50),
            "p95": np.percentile(retrieval_latency, 95),
            "p99": np.percentile(retrieval_latency, 99)
        }
        
        generation_metrics = {
            "mean": np.mean(generation_latency),
            "p50": np.percentile(generation_latency, 50),
            "p95": np.percentile(generation_latency, 95),
            "p99": np.percentile(generation_latency, 99)
        }
        
        total_metrics = {
            "mean": np.mean(total_latency),
            "p50": np.percentile(total_latency, 50),
            "p95": np.percentile(total_latency, 95),
            "p99": np.percentile(total_latency, 99)
        }
        
        # Calculate per-dataset metrics
        dataset_metrics = {}
        for dataset_id, dataset_results in results_by_dataset.items():
            if not dataset_results:
                continue
                
            dataset_metrics[dataset_id] = {
                "context_relevance": np.mean([r["metrics"]["context_relevance"] for r in dataset_results]),
                "answer_relevance": np.mean([r["metrics"]["answer_relevance"] for r in dataset_results]),
                "factual_correctness": np.mean([r["metrics"]["factual_correctness"] for r in dataset_results]),
                "total_latency": np.mean([r["metrics"]["total_latency"] for r in dataset_results])
            }
        
        # Create summary score (weighted combination)
        weights = {
            "context_relevance": 0.3,
            "answer_relevance": 0.4,
            "factual_correctness": 0.3
        }
        
        summary_score = (
            weights["context_relevance"] * context_relevance +
            weights["answer_relevance"] * answer_relevance +
            weights["factual_correctness"] * factual_correctness
        )
        
        return {
            "overall": {
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance,
                "factual_correctness": factual_correctness,
                "summary_score": summary_score
            },
            "latency": {
                "retrieval": retrieval_metrics,
                "generation": generation_metrics,
                "total": total_metrics
            },
            "per_dataset": dataset_metrics
        }
    
    def _save_results(self, results_by_dataset: Dict[str, List[Dict[str, Any]]], metrics: Dict[str, Any]) -> None:
        """
        Save benchmark results to files.
        
        Args:
            results_by_dataset: Results organized by dataset
            metrics: Calculated metrics
        """
        # Create output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_dir = self.output_dir / f"rag_top_benchmark_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        for dataset_id, results in results_by_dataset.items():
            result_file = result_dir / f"{dataset_id}_results.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Save metrics summary
        metrics_file = result_dir / "metrics_summary.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        # Save CSV summary for easy import into other tools
        csv_file = result_dir / "metrics_summary.csv"
        rows = []
        
        # Overall metrics
        for metric, value in metrics["overall"].items():
            rows.append({"dataset": "overall", "metric": metric, "value": value})
            
        # Per-dataset metrics
        for dataset_id, dataset_metrics in metrics["per_dataset"].items():
            for metric, value in dataset_metrics.items():
                rows.append({"dataset": dataset_id, "metric": metric, "value": value})
        
        # Write CSV
        pd.DataFrame(rows).to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {result_dir}")


async def main():
    """Main function to run the benchmark"""
    parser = argparse.ArgumentParser(description="Run RAG benchmark with top metrics and datasets")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_INFO.keys()),
                      help="Datasets to benchmark (default: all)")
    parser.add_argument("--output-dir", default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--qdrant-host", default="localhost",
                      help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333,
                      help="Qdrant port (default: 6333)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                      help="SentenceTransformer model to use for embeddings")
    args = parser.parse_args()
    
    # Prepare config
    config = {
        "collection_name": f"rag_top_benchmark_{int(time.time())}",
        "qdrant_host": args.qdrant_host,
        "qdrant_port": args.qdrant_port,
        "embedding_model": args.embedding_model
    }
    
    # Create datasets directory if it doesn't exist
    os.makedirs("benchmark_suite/datasets", exist_ok=True)
    
    # Run benchmark
    benchmark = RagTopBenchmark(
        config=config,
        datasets=args.datasets,
        output_dir=args.output_dir
    )
    
    await benchmark.run_benchmark()


if __name__ == "__main__":
    asyncio.run(main()) 