#!/usr/bin/env python3
"""
RAG Benchmarking Framework

This script provides a comprehensive benchmarking framework for the RAG system,
evaluating retrieval accuracy, answer relevance, factual correctness, latency,
and other key metrics.
"""

import asyncio
import logging
import time
import json
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from tqdm import tqdm

from agents.rag.models.document import Document
from agents.rag.api.rag_service import RagService
from agents.rag.middleware.storage_router import StorageType
from agents.rag.storage.vector.qdrant_storage import QdrantStorageAdaptor, QDRANT_AVAILABLE
from agents.rag.utils.embedding_provider import SentenceTransformerEmbeddingProvider, DummyEmbeddingProvider
from agents.rag.storage.vector.semantic_search import SemanticSearchLayer, relevance_reranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RagBenchmark:
    """RAG Benchmarking Framework for evaluating RAG systems"""
    
    def __init__(self, 
                config: Dict[str, Any],
                benchmark_dataset: str = None,
                output_dir: str = "benchmark_results"):
        """
        Initialize the RAG benchmarking framework.
        
        Args:
            config: Configuration dictionary with settings for the RAG service
            benchmark_dataset: Path to the benchmark dataset (if not provided, will use default test set)
            output_dir: Directory to save benchmark results
        """
        self.config = config
        self.benchmark_dataset = benchmark_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics to evaluate
        self.metrics = {
            "retrieval_accuracy": [],
            "answer_relevance": [],
            "factual_correctness": [],
            "latency": {
                "retrieval": [],
                "generation": [],
                "total": []
            },
            "token_usage": [],
            "context_relevance": []
        }
        
        # Load or create test set
        self.test_set = self._load_test_set()
        
    def _load_test_set(self) -> List[Dict[str, Any]]:
        """
        Load the benchmark test set or create a default one if not provided.
        
        Returns:
            List of test cases with questions, reference answers, and metadata
        """
        if self.benchmark_dataset and Path(self.benchmark_dataset).exists():
            logger.info(f"Loading benchmark dataset from {self.benchmark_dataset}")
            with open(self.benchmark_dataset, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No benchmark dataset provided or file not found. Using default test set.")
            return self._create_default_test_set()
            
    def _create_default_test_set(self) -> List[Dict[str, Any]]:
        """
        Create a default test set for benchmarking.
        
        Returns:
            List of test cases with questions, reference answers, and metadata
        """
        # Simple default test set with a few questions
        return [
            {
                "question": "What are vector databases and how do they work?",
                "reference_answer": "Vector databases store and retrieve data as high-dimensional vectors, which represent semantic meaning. They work by using efficient similarity search algorithms like Approximate Nearest Neighbors (ANN) to find the most similar vectors to a query vector.",
                "metadata": {"category": "technical", "complexity": "medium"}
            },
            {
                "question": "How does retrieval augmented generation improve LLM accuracy?",
                "reference_answer": "Retrieval Augmented Generation (RAG) improves LLM accuracy by retrieving relevant context from external sources before generation. This provides the LLM with up-to-date and factual information, reducing hallucinations and providing more accurate responses.",
                "metadata": {"category": "technical", "complexity": "high"}
            },
            {
                "question": "What are the key components of a RAG system?",
                "reference_answer": "The key components of a RAG system include: a retriever that searches for relevant information, a knowledge base or vector database to store documents, an embedding model to convert text to vectors, and a language model to generate responses based on the retrieved context.",
                "metadata": {"category": "technical", "complexity": "medium"}
            }
        ]
    
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
            logger.info("Using DummyEmbeddingProvider as fallback")
            embedding_provider = DummyEmbeddingProvider(vector_size=vector_size)
        
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
    
    async def load_test_documents(self, service: RagService) -> None:
        """
        Load test documents into the RAG service for benchmarking.
        
        Args:
            service: The RAG service to load documents into
        """
        logger.info("Loading test documents...")
        
        # Sample documents to test with
        docs = [
            Document(
                content="Vector databases are specialized database systems designed to store and search vector embeddings efficiently. "
                       "They are optimized for similarity search operations using techniques like approximate nearest neighbors (ANN) algorithms. "
                       "Vector databases index high-dimensional vectors for fast retrieval, enabling semantic search capabilities. "
                       "Common vector databases include Qdrant, Pinecone, Milvus, and Weaviate.",
                metadata={"source": "benchmark_data", "category": "databases", "tags": ["vectors", "embeddings", "similarity search"]}
            ),
            Document(
                content="Retrieval-Augmented Generation (RAG) is an AI framework that combines retrieval-based and generation-based approaches. "
                       "It first retrieves relevant documents from a knowledge base and then uses them to generate more accurate and contextually relevant responses. "
                       "RAG helps address hallucinations in LLMs by grounding the generation in factual information. "
                       "The key components include a retriever, a knowledge base, and a generator.",
                metadata={"source": "benchmark_data", "category": "artificial intelligence", "tags": ["RAG", "retrieval", "NLP"]}
            ),
            Document(
                content="The RAG architecture consists of several key components working together: "
                       "1. A vector database to store document embeddings, "
                       "2. An embedding model to convert text into vector representations, "
                       "3. A retrieval system to find relevant documents, "
                       "4. A reranker to improve retrieval quality (optional), "
                       "5. A language model to generate responses based on the retrieved context, "
                       "6. A prompt template to structure the input to the language model.",
                metadata={"source": "benchmark_data", "category": "architecture", "tags": ["RAG", "components", "system design"]}
            ),
            Document(
                content="Reranking is a critical component in advanced retrieval systems. After initial retrieval, "
                       "rerankers take the top-k candidates and apply more sophisticated models to improve ranking quality. "
                       "Cross-encoders are commonly used as rerankers because they process query and document pairs jointly, "
                       "producing more accurate relevance scores than initial bi-encoders. "
                       "Reranking significantly improves retrieval precision at the cost of some additional computation.",
                metadata={"source": "benchmark_data", "category": "information retrieval", "tags": ["reranking", "cross-encoders", "search"]}
            ),
            Document(
                content="Hybrid search combines multiple search approaches, typically vector search and keyword search, "
                       "to leverage the strengths of both. Vector search excels at semantic understanding but may miss exact matches, "
                       "while keyword search is precise for explicit terms but lacks semantic awareness. "
                       "Hybrid search methods include: 1. Pipeline approaches that run both search types and merge results, "
                       "2. Single-pass approaches that combine sparse and dense representations in one index.",
                metadata={"source": "benchmark_data", "category": "search", "tags": ["hybrid search", "vector search", "keyword search"]}
            )
        ]
        
        # Store documents
        for doc in docs:
            doc_id = await service.store_document(doc)
            logger.info(f"Stored document: {doc.content[:50]}... (ID: {doc_id})")
        
        # Add a small delay to ensure documents are indexed
        logger.info("Waiting for documents to be indexed...")
        await asyncio.sleep(2)
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the RAG benchmark.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("Starting RAG benchmark...")
        
        # Set up RAG service
        service = await self.setup_rag_service()
        
        # Load test documents
        await self.load_test_documents(service)
        
        # Run tests
        results = []
        for test_case in tqdm(self.test_set, desc="Running benchmark tests"):
            result = await self._run_test_case(service, test_case)
            results.append(result)
        
        # Calculate metrics
        benchmark_results = self._calculate_metrics(results)
        
        # Save results
        self._save_results(results, benchmark_results)
        
        # Clean up
        await service.close()
        
        logger.info("Benchmark completed!")
        return benchmark_results
    
    async def _run_test_case(self, service: RagService, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            service: The RAG service to test
            test_case: Test case with question and reference answer
            
        Returns:
            Test case result with metrics
        """
        question = test_case["question"]
        reference_answer = test_case["reference_answer"]
        
        # Measure retrieval latency
        retrieval_start = time.time()
        filter_criteria = test_case.get("filter_criteria", None)
        top_k = self.config.get("top_k", 3)
        
        # Get retrieval results
        retrieval_results = await service.retrieve_documents(
            question, 
            top_k=top_k,
            filter_criteria=filter_criteria
        )
        retrieval_end = time.time()
        retrieval_latency = retrieval_end - retrieval_start
        
        # Measure generation latency
        generation_start = time.time()
        answer = await service.answer_question(question, context=retrieval_results)
        generation_end = time.time()
        generation_latency = generation_end - generation_start
        
        # Calculate total latency
        total_latency = retrieval_latency + generation_latency
        
        # Get context from retrieval results
        retrieved_context = [result.item.content for result in retrieval_results]
        
        # Estimate token usage (simple approximation)
        token_usage = {
            "prompt": len(question.split()) + sum(len(ctx.split()) for ctx in retrieved_context),
            "completion": len(answer.split()),
            "total": len(question.split()) + sum(len(ctx.split()) for ctx in retrieved_context) + len(answer.split())
        }
        
        # Simulate evaluation of metrics (in a real implementation, you would use actual metrics)
        # For this example, we'll use simple heuristics
        context_relevance = self._evaluate_context_relevance(question, retrieved_context)
        answer_relevance = self._evaluate_answer_relevance(question, answer)
        factual_correctness = self._evaluate_factual_correctness(answer, reference_answer, retrieved_context)
        
        # Store metrics
        self.metrics["retrieval_accuracy"].append(context_relevance)
        self.metrics["answer_relevance"].append(answer_relevance)
        self.metrics["factual_correctness"].append(factual_correctness)
        self.metrics["latency"]["retrieval"].append(retrieval_latency)
        self.metrics["latency"]["generation"].append(generation_latency)
        self.metrics["latency"]["total"].append(total_latency)
        self.metrics["token_usage"].append(token_usage["total"])
        self.metrics["context_relevance"].append(context_relevance)
        
        return {
            "question": question,
            "reference_answer": reference_answer,
            "retrieved_context": retrieved_context,
            "answer": answer,
            "metrics": {
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance,
                "factual_correctness": factual_correctness,
                "retrieval_latency": retrieval_latency,
                "generation_latency": generation_latency,
                "total_latency": total_latency,
                "token_usage": token_usage
            }
        }
    
    def _evaluate_context_relevance(self, question: str, context: List[str]) -> float:
        """
        Evaluate relevance of retrieved context to the question.
        
        Args:
            question: The query
            context: List of retrieved context passages
            
        Returns:
            Context relevance score between 0 and 1
        """
        # In a real implementation, you would use a more sophisticated evaluation
        # such as an LLM-based evaluator or semantic similarity metrics
        
        # Simple keyword-based heuristic for demo purposes
        question_keywords = set(word.lower() for word in question.split() if len(word) > 3)
        relevant_contexts = 0
        
        for passage in context:
            passage_words = set(word.lower() for word in passage.split() if len(word) > 3)
            overlap = len(question_keywords.intersection(passage_words))
            if overlap >= 2:  # Consider relevant if at least 2 keywords match
                relevant_contexts += 1
        
        return relevant_contexts / max(1, len(context))
    
    def _evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate relevance of the answer to the question.
        
        Args:
            question: The query
            answer: Generated answer
            
        Returns:
            Answer relevance score between 0 and 1
        """
        # In a real implementation, you would use an LLM-based evaluator
        # or a trained model to assess answer relevance
        
        # Simple keyword-based heuristic for demo purposes
        question_keywords = set(word.lower() for word in question.split() if len(word) > 3)
        answer_words = set(word.lower() for word in answer.split() if len(word) > 3)
        
        overlap = len(question_keywords.intersection(answer_words))
        
        return min(1.0, overlap / max(1, len(question_keywords)))
    
    def _evaluate_factual_correctness(self, answer: str, reference: str, context: List[str]) -> float:
        """
        Evaluate factual correctness of the answer based on reference and context.
        
        Args:
            answer: Generated answer
            reference: Reference answer
            context: Retrieved context passages
            
        Returns:
            Factual correctness score between 0 and 1
        """
        # In a real implementation, you would use an LLM-based evaluator
        # or advanced NLP techniques to assess factual correctness
        
        # Simple word overlap heuristic for demo purposes
        reference_words = set(word.lower() for word in reference.split() if len(word) > 3)
        answer_words = set(word.lower() for word in answer.split() if len(word) > 3)
        
        # Check overlap with reference answer
        reference_overlap = len(reference_words.intersection(answer_words))
        reference_score = reference_overlap / max(1, len(reference_words))
        
        # Also check if answer is grounded in context
        context_words = set()
        for passage in context:
            context_words.update(word.lower() for word in passage.split() if len(word) > 3)
        
        context_overlap = len(context_words.intersection(answer_words))
        context_score = context_overlap / max(1, len(answer_words))
        
        # Combine scores with more weight on reference
        return 0.7 * reference_score + 0.3 * context_score
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from test results.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with aggregate metrics
        """
        metrics = {
            "retrieval_accuracy": np.mean(self.metrics["retrieval_accuracy"]),
            "answer_relevance": np.mean(self.metrics["answer_relevance"]),
            "factual_correctness": np.mean(self.metrics["factual_correctness"]),
            "latency": {
                "retrieval": {
                    "mean": np.mean(self.metrics["latency"]["retrieval"]),
                    "p50": np.percentile(self.metrics["latency"]["retrieval"], 50),
                    "p95": np.percentile(self.metrics["latency"]["retrieval"], 95),
                    "p99": np.percentile(self.metrics["latency"]["retrieval"], 99),
                },
                "generation": {
                    "mean": np.mean(self.metrics["latency"]["generation"]),
                    "p50": np.percentile(self.metrics["latency"]["generation"], 50),
                    "p95": np.percentile(self.metrics["latency"]["generation"], 95),
                    "p99": np.percentile(self.metrics["latency"]["generation"], 99),
                },
                "total": {
                    "mean": np.mean(self.metrics["latency"]["total"]),
                    "p50": np.percentile(self.metrics["latency"]["total"], 50),
                    "p95": np.percentile(self.metrics["latency"]["total"], 95),
                    "p99": np.percentile(self.metrics["latency"]["total"], 99),
                }
            },
            "token_usage": {
                "mean": np.mean(self.metrics["token_usage"]),
                "total": np.sum(self.metrics["token_usage"])
            },
            "context_relevance": np.mean(self.metrics["context_relevance"]),
            "summary_score": (
                np.mean(self.metrics["retrieval_accuracy"]) * 0.25 +
                np.mean(self.metrics["answer_relevance"]) * 0.25 +
                np.mean(self.metrics["factual_correctness"]) * 0.5
            )
        }
        
        return metrics
    
    def _save_results(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
        """
        Save benchmark results to files.
        
        Args:
            results: List of test results
            metrics: Aggregate metrics
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        with open(self.output_dir / f"rag_benchmark_results_{timestamp}.json", 'w') as f:
            json.dump({
                "config": self.config,
                "metrics": metrics,
                "results": results
            }, f, indent=2)
        
        # Save summary metrics
        with open(self.output_dir / f"rag_benchmark_summary_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save CSV for easy analysis
        with open(self.output_dir / f"rag_benchmark_results_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Question", 
                "Context Relevance",
                "Answer Relevance",
                "Factual Correctness",
                "Retrieval Latency (s)",
                "Generation Latency (s)",
                "Total Latency (s)",
                "Token Usage"
            ])
            
            for result in results:
                writer.writerow([
                    result["question"],
                    result["metrics"]["context_relevance"],
                    result["metrics"]["answer_relevance"],
                    result["metrics"]["factual_correctness"],
                    result["metrics"]["retrieval_latency"],
                    result["metrics"]["generation_latency"],
                    result["metrics"]["total_latency"],
                    result["metrics"]["token_usage"]["total"]
                ])
        
        logger.info(f"Benchmark results saved to {self.output_dir}")

async def main():
    """Run the RAG benchmark."""
    parser = argparse.ArgumentParser(description="RAG Benchmarking Framework")
    parser.add_argument("--dataset", type=str, help="Path to benchmark dataset")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save results")
    parser.add_argument("--collection", type=str, default="benchmark_collection", help="Qdrant collection name")
    parser.add_argument("--host", type=str, default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Embedding model name")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    args = parser.parse_args()
    
    config = {
        "collection_name": args.collection,
        "qdrant_host": args.host,
        "qdrant_port": args.port,
        "embedding_model": args.embedding_model,
        "top_k": args.top_k
    }
    
    benchmark = RagBenchmark(
        config=config,
        benchmark_dataset=args.dataset,
        output_dir=args.output_dir
    )
    
    results = await benchmark.run_benchmark()
    
    print("\n=== RAG Benchmark Results ===")
    print(f"Retrieval Accuracy: {results['retrieval_accuracy']:.4f}")
    print(f"Answer Relevance: {results['answer_relevance']:.4f}")
    print(f"Factual Correctness: {results['factual_correctness']:.4f}")
    print(f"Context Relevance: {results['context_relevance']:.4f}")
    print(f"Avg. Total Latency: {results['latency']['total']['mean']:.4f}s")
    print(f"Avg. Token Usage: {results['token_usage']['mean']:.1f}")
    print(f"Summary Score: {results['summary_score']:.4f}")
    print(f"Full results saved to {args.output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 