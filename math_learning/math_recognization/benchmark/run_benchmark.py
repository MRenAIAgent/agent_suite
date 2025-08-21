#!/usr/bin/env python3
"""
🧮 Math OCR Benchmark Suite
============================

A unified benchmark runner for testing OCR solutions on mathematical content.
Supports multiple datasets and OCR configurations with detailed analytics.

Usage:
    python run_benchmark.py --help
    python run_benchmark.py --list-datasets
    python run_benchmark.py --list-solutions
    python run_benchmark.py --dataset custom_images --solution gpt-5
    python run_benchmark.py --dataset all --solution mathpix_gpt5_hybrid
"""

import asyncio
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.")

from improved_benchmark_runner import ImprovedBenchmarkRunner
from ocr_config import get_config_manager

class UnifiedBenchmarkRunner:
    """
    🎯 Unified benchmark runner for all OCR solutions and datasets.
    
    Features:
    - Multiple dataset support (Custom Images, PGDP5K, Expanded Dataset)
    - Multiple OCR solutions (GPT-5, Mathpix+GPT-5, Geometry Specialist, etc.)
    - Detailed performance analytics
    - Easy-to-use command line interface
    """
    
    def __init__(self):
        self.config_manager = get_config_manager()
        self.benchmark_runner = ImprovedBenchmarkRunner()
        
        # 📊 Available datasets with metadata
        self.datasets = {
            "custom_images": {
                "name": "Custom Test Images",
                "description": "14 hand-crafted math problems (algebra, geometry, mixed)",
                "path": "./test_images",
                "samples": 14,
                "types": ["algebra", "geometry", "calculus", "statistics"],
                "difficulty": "varied",
                "best_for": "Quick testing and validation"
            },
            "pgdp5k": {
                "name": "PGDP5K Dataset", 
                "description": "Large-scale mathematical document dataset",
                "path": "./datasets/pgdp5k",
                "samples": "5000+",
                "types": ["equations", "formulas", "expressions"],
                "difficulty": "academic",
                "best_for": "Comprehensive evaluation"
            },
            "expanded_dataset": {
                "name": "Expanded Mixed Dataset",
                "description": "Extended collection with diverse math content",
                "path": "./datasets/expanded",
                "samples": "300+",
                "types": ["mixed_content", "complex_expressions"],
                "difficulty": "challenging",
                "best_for": "Stress testing"
            }
        }
        
        # 🤖 Available OCR solutions with metadata
        self.solutions = {}
        self._load_solutions()
    
    def _load_solutions(self):
        """Load and categorize available OCR solutions."""
        configs = self.config_manager.list_configs()
        
        for config_name, description in configs.items():
            config = self.config_manager.get_config(config_name)
            
            # Categorize solutions
            if "gpt-5" in config_name.lower() or config.primary_ocr.model_name == "gpt-5":
                category = "🤖 AI Vision Models"
                speed = "Medium" if config.fallback_ocr else "Fast"
                cost = "High" if config.fallback_ocr else "Medium"
            elif "mathpix" in config_name.lower():
                category = "🔄 Hybrid Solutions"
                speed = "Medium"
                cost = "Medium"
            elif "geometry" in config_name.lower() or "got_ocr" in str(config.primary_ocr.provider).lower():
                category = "🔺 Specialized OCR"
                speed = "Slow"
                cost = "Low"
            elif "expression" in config_name.lower() or "unimer" in str(config.primary_ocr.provider).lower():
                category = "📐 Math Specialists"
                speed = "Slow"
                cost = "Low"
            else:
                category = "🔧 Other Solutions"
                speed = "Unknown"
                cost = "Unknown"
            
            self.solutions[config_name] = {
                "name": config.name,
                "description": description,
                "category": category,
                "primary_ocr": str(config.primary_ocr.provider.value),
                "fallback_ocr": str(config.fallback_ocr.provider.value) if config.fallback_ocr else None,
                "strategy": str(config.processing_strategy.value),
                "speed": speed,
                "cost": cost,
                "best_for": self._get_best_use_case(config_name, config)
            }
    
    def _get_best_use_case(self, config_name: str, config) -> str:
        """Determine the best use case for each solution."""
        if "gpt-5" in config_name:
            return "General math problems, mixed content"
        elif "geometry" in config_name:
            return "Geometric diagrams, coordinate extraction"
        elif "expression" in config_name:
            return "Complex mathematical expressions"
        elif "mathpix" in config_name:
            return "Production use, reliable results"
        elif "parallel" in config_name:
            return "Maximum accuracy, research"
        elif "cost" in config_name:
            return "Budget-conscious applications"
        else:
            return "Specialized use cases"
    
    def list_datasets(self):
        """📊 Display all available datasets with details."""
        print("\n📊 AVAILABLE DATASETS")
        print("=" * 60)
        
        for dataset_id, info in self.datasets.items():
            print(f"\n🗂️  {dataset_id}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Samples: {info['samples']}")
            print(f"   Types: {', '.join(info['types'])}")
            print(f"   Difficulty: {info['difficulty']}")
            print(f"   Best for: {info['best_for']}")
            
            # Check if dataset exists
            if Path(info['path']).exists():
                print(f"   Status: ✅ Available")
            else:
                print(f"   Status: ❌ Not found at {info['path']}")
    
    def list_solutions(self):
        """🤖 Display all available OCR solutions with details."""
        print("\n🤖 AVAILABLE OCR SOLUTIONS")
        print("=" * 60)
        
        # Group by category
        categories = {}
        for solution_id, info in self.solutions.items():
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((solution_id, info))
        
        for category, solutions in categories.items():
            print(f"\n{category}")
            print("-" * 40)
            
            for solution_id, info in solutions:
                print(f"\n🔧 {solution_id}")
                print(f"   Name: {info['name']}")
                print(f"   Primary OCR: {info['primary_ocr']}")
                if info['fallback_ocr']:
                    print(f"   Fallback OCR: {info['fallback_ocr']}")
                print(f"   Strategy: {info['strategy']}")
                print(f"   Speed: {info['speed']} | Cost: {info['cost']}")
                print(f"   Best for: {info['best_for']}")
    
    def show_recommendations(self, dataset: str = None):
        """💡 Show recommended solution-dataset combinations."""
        print("\n💡 RECOMMENDED COMBINATIONS")
        print("=" * 50)
        
        recommendations = [
            {
                "dataset": "custom_images",
                "solution": "gpt-5", 
                "reason": "Fast testing with high accuracy for mixed content",
                "expected_accuracy": "75-85%",
                "time_estimate": "3-5 minutes"
            },
            {
                "dataset": "custom_images",
                "solution": "geometry_specialist",
                "reason": "Best for geometric problems in the test set",
                "expected_accuracy": "85-95%", 
                "time_estimate": "4-6 minutes"
            },
            {
                "dataset": "pgdp5k",
                "solution": "mathpix_gpt5_hybrid",
                "reason": "Reliable for large-scale academic content",
                "expected_accuracy": "80-90%",
                "time_estimate": "2-4 hours"
            },
            {
                "dataset": "expanded_dataset",
                "solution": "comprehensive_parallel",
                "reason": "Maximum accuracy for challenging content",
                "expected_accuracy": "85-95%",
                "time_estimate": "30-60 minutes"
            }
        ]
        
        filtered_recs = recommendations
        if dataset:
            filtered_recs = [r for r in recommendations if r['dataset'] == dataset]
        
        for rec in filtered_recs:
            print(f"\n🎯 {rec['dataset']} + {rec['solution']}")
            print(f"   Reason: {rec['reason']}")
            print(f"   Expected Accuracy: {rec['expected_accuracy']}")
            print(f"   Time Estimate: {rec['time_estimate']}")
    
    async def run_benchmark(self, 
                          dataset: str, 
                          solution: str, 
                          max_samples: Optional[int] = None,
                          save_results: bool = True) -> Dict[str, Any]:
        """
        🚀 Run benchmark on specified dataset with specified solution.
        
        Args:
            dataset: Dataset identifier (e.g., 'custom_images', 'pgdp5k')
            solution: Solution identifier (e.g., 'gpt-5', 'mathpix_gpt5_hybrid')
            max_samples: Limit number of samples (optional)
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing benchmark results
        """
        
        # Validate inputs
        if dataset not in self.datasets and dataset != "all":
            raise ValueError(f"Unknown dataset '{dataset}'. Use --list-datasets to see options.")
        
        if solution not in self.solutions:
            raise ValueError(f"Unknown solution '{solution}'. Use --list-solutions to see options.")
        
        print(f"\n🚀 STARTING BENCHMARK")
        print("=" * 50)
        print(f"📊 Dataset: {dataset}")
        print(f"🤖 Solution: {solution}")
        if max_samples:
            print(f"🔢 Max Samples: {max_samples}")
        print(f"💾 Save Results: {save_results}")
        print("=" * 50)
        
        start_time = time.time()
        results = {}
        
        if dataset == "all":
            # Run on all datasets
            for dataset_id in self.datasets.keys():
                if Path(self.datasets[dataset_id]['path']).exists():
                    print(f"\n📊 Processing dataset: {dataset_id}")
                    dataset_result = await self._run_single_benchmark(
                        dataset_id, solution, max_samples
                    )
                    results[dataset_id] = dataset_result
                else:
                    print(f"⚠️  Skipping {dataset_id}: Dataset not found")
        else:
            # Run on single dataset
            results[dataset] = await self._run_single_benchmark(
                dataset, solution, max_samples
            )
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(results, solution, total_time)
        
        if save_results:
            self._save_results(summary, dataset, solution)
        
        self._display_results(summary)
        
        return summary
    
    async def _run_single_benchmark(self, dataset: str, solution: str, max_samples: Optional[int]) -> Dict[str, Any]:
        """Run benchmark on a single dataset."""
        dataset_info = self.datasets[dataset]
        dataset_path = dataset_info['path']
        
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Run the benchmark
        session = await self.benchmark_runner.run_benchmark(
            config_name=solution,
            dataset_path=dataset_path,
            max_samples=max_samples
        )
        
        return {
            "session": session,
            "dataset_info": dataset_info,
            "dataset_path": dataset_path
        }
    
    def _generate_summary(self, results: Dict[str, Any], solution: str, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        summary = {
            "benchmark_info": {
                "solution": solution,
                "solution_details": self.solutions[solution],
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time
            },
            "datasets": {},
            "overall_metrics": {}
        }
        
        total_samples = 0
        total_accuracy = 0
        total_processing_time = 0
        
        for dataset_id, result in results.items():
            session = result['session']
            dataset_info = result['dataset_info']
            
            if session and hasattr(session, 'results'):
                # Handle both dict and list results formats
                if isinstance(session.results, dict) and session.results:
                    dataset_results = list(session.results.values())[0]
                    if dataset_results and len(dataset_results) > 0:
                        sample_result = dataset_results[0]
                        accuracy = sample_result.get('overall_accuracy', 0)
                        samples = len(dataset_results)
                        processing_time = sample_result.get('processing_time', 0)
                    else:
                        accuracy = 0
                        samples = 0
                        processing_time = 0
                elif hasattr(session, 'session_id'):
                    # Fallback: extract from session attributes if available
                    accuracy = getattr(session, 'overall_accuracy', 0)
                    samples = getattr(session, 'total_samples', 0)
                    processing_time = getattr(session, 'processing_time', 0)
                else:
                    accuracy = 0
                    samples = 0
                    processing_time = 0
                
                summary['datasets'][dataset_id] = {
                    "name": dataset_info['name'],
                    "samples_processed": samples,
                    "accuracy": accuracy,
                    "processing_time": processing_time,
                    "session_id": session.session_id if hasattr(session, 'session_id') else None
                }
                
                total_samples += samples
                total_accuracy += accuracy * samples
                total_processing_time += processing_time
        
        # Calculate overall metrics
        if total_samples > 0:
            summary['overall_metrics'] = {
                "total_samples": total_samples,
                "weighted_accuracy": total_accuracy / total_samples,
                "total_processing_time": total_processing_time,
                "avg_time_per_sample": total_processing_time / total_samples
            }
        
        return summary
    
    def _save_results(self, summary: Dict[str, Any], dataset: str, solution: str):
        """Save benchmark results to file."""
        results_dir = Path("./results/unified_benchmarks")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{dataset}_{solution}_{timestamp}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    def _display_results(self, summary: Dict[str, Any]):
        """Display benchmark results in a formatted way."""
        print(f"\n🎯 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        solution_info = summary['benchmark_info']['solution_details']
        print(f"🤖 Solution: {solution_info['name']}")
        print(f"⏱️  Total Time: {summary['benchmark_info']['total_time']:.1f}s")
        
        if summary['overall_metrics']:
            metrics = summary['overall_metrics']
            print(f"📊 Overall Accuracy: {metrics['weighted_accuracy']:.1f}%")
            print(f"📈 Total Samples: {metrics['total_samples']}")
            print(f"⚡ Avg Time/Sample: {metrics['avg_time_per_sample']:.2f}s")
        
        print(f"\n📊 DATASET BREAKDOWN:")
        print("-" * 40)
        
        for dataset_id, dataset_result in summary['datasets'].items():
            print(f"\n🗂️  {dataset_result['name']}")
            print(f"   Samples: {dataset_result['samples_processed']}")
            print(f"   Accuracy: {dataset_result['accuracy']:.1f}%")
            print(f"   Time: {dataset_result['processing_time']:.1f}s")

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="🧮 Math OCR Benchmark Suite - Test OCR solutions on mathematical content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --list-datasets                    # Show available datasets
  python run_benchmark.py --list-solutions                   # Show available OCR solutions  
  python run_benchmark.py --recommendations                  # Show recommended combinations
  python run_benchmark.py -d custom_images -s gpt-5         # Quick test with GPT-5
  python run_benchmark.py -d pgdp5k -s mathpix_gpt5_hybrid  # Production test
  python run_benchmark.py -d all -s gpt-5 --max-samples 10  # Test all datasets (limited)
        """
    )
    
    # Information commands
    parser.add_argument('--list-datasets', action='store_true', 
                       help='List all available datasets')
    parser.add_argument('--list-solutions', action='store_true',
                       help='List all available OCR solutions')
    parser.add_argument('--recommendations', action='store_true',
                       help='Show recommended dataset-solution combinations')
    
    # Benchmark commands
    parser.add_argument('-d', '--dataset', type=str,
                       help='Dataset to benchmark (use --list-datasets to see options, or "all")')
    parser.add_argument('-s', '--solution', type=str,
                       help='OCR solution to test (use --list-solutions to see options)')
    parser.add_argument('--max-samples', type=int,
                       help='Limit number of samples to process (optional)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    
    args = parser.parse_args()
    
    runner = UnifiedBenchmarkRunner()
    
    # Handle information commands
    if args.list_datasets:
        runner.list_datasets()
        return
    
    if args.list_solutions:
        runner.list_solutions()
        return
    
    if args.recommendations:
        runner.show_recommendations()
        return
    
    # Handle benchmark commands
    if not args.dataset or not args.solution:
        print("❌ Error: Both --dataset and --solution are required for benchmarking")
        print("Use --help for usage information")
        return
    
    try:
        # Run the benchmark
        asyncio.run(runner.run_benchmark(
            dataset=args.dataset,
            solution=args.solution,
            max_samples=args.max_samples,
            save_results=not args.no_save
        ))
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
