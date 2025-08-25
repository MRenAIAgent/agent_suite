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

from scripts.improved_benchmark_runner import ImprovedBenchmarkRunner
from scripts.ocr_config import get_config_manager

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
                "path": "./real_datasets/PGDP5K/images",
                "samples": "30",
                "types": ["equations", "formulas", "expressions"],
                "difficulty": "academic",
                "best_for": "Comprehensive evaluation"
            },
            "complex_geometry": {
                "name": "Complex Geometry Dataset",
                "description": "Specialized geometric problems and diagrams",
                "path": "./real_datasets/ComplexGeometry/images",
                "samples": 5,
                "types": ["geometry", "diagrams", "coordinate_systems"],
                "difficulty": "advanced",
                "best_for": "Geometry-focused evaluation"
            },
            "hme100k_sample": {
                "name": "HME100K Sample",
                "description": "Random sample from HME100K handwritten math expressions",
                "path": "./real_datasets/HME100K/samples/current/images",
                "samples": "Variable (50-100 typical)",
                "types": ["handwritten_math", "expressions", "equations"],
                "difficulty": "K-12",
                "best_for": "Real handwritten math evaluation with LaTeX ground truth"
            },
            "mlhme38k_sample": {
                "name": "MLHME-38K Multi-line Sample",
                "description": "Multi-line handwritten mathematical expressions (homework-like problems)",
                "path": "./real_datasets/MLHME38K/samples/current/images",
                "samples": "Variable (50-200 typical)",
                "types": ["multiline_math", "homework_problems", "complex_expressions"],
                "difficulty": "K-12 to University",
                "best_for": "Real homework-like multi-line math problems with LaTeX ground truth"
            },
            "mathwriting_sample": {
                "name": "MathWriting Human Sample",
                "description": "Large-scale human-written handwritten mathematical expressions",
                "path": "./real_datasets/MathWriting/samples/current/current/images",
                "samples": "Variable (50-1000 typical)",
                "types": ["handwritten_math", "diverse_expressions", "real_handwriting"],
                "difficulty": "Elementary to University",
                "best_for": "Largest available handwritten math dataset with LaTeX ground truth"
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
                "expected_accuracy": "75-95%",
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
                "solution": "gpt-5",
                "reason": "Reliable for academic mathematical content",
                "expected_accuracy": "90-100%",
                "time_estimate": "8-12 minutes"
            },
            {
                "dataset": "complex_geometry",
                "solution": "geometry_specialist",
                "reason": "Specialized for geometric diagrams and coordinate systems",
                "expected_accuracy": "85-95%",
                "time_estimate": "2-3 minutes"
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
                          save_results: bool = True,
                          comprehensive: bool = False,
                          comparison_table: bool = False) -> Dict[str, Any]:
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
        
        # Handle comprehensive mode
        if comprehensive or (dataset == "all" and solution == "all"):
            return await self._run_comprehensive_benchmark(max_samples, save_results, comparison_table)
        
        # Validate inputs
        if dataset not in self.datasets and dataset != "all":
            raise ValueError(f"Unknown dataset '{dataset}'. Use --list-datasets to see options.")
        
        if solution not in self.solutions and solution != "all":
            raise ValueError(f"Unknown solution '{solution}'. Use --list-solutions to see options.")
        
        # Handle "all" for single parameter
        if solution == "all":
            return await self._run_all_solutions_single_dataset(dataset, max_samples, save_results, comparison_table)
        elif dataset == "all":
            return await self._run_single_solution_all_datasets(solution, max_samples, save_results, comparison_table)
        
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
    
    async def _run_comprehensive_benchmark(self, max_samples: Optional[int], save_results: bool, comparison_table: bool) -> Dict[str, Any]:
        """Run all datasets with all solutions."""
        print(f"\n🚀 COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        print(f"📊 Datasets: {len(self.datasets)} ({', '.join(self.datasets.keys())})")
        print(f"🤖 Solutions: {len(self.solutions)} ({', '.join(self.solutions.keys())})")
        print(f"🔢 Total Combinations: {len(self.datasets) * len(self.solutions)}")
        if max_samples:
            print(f"📝 Max Samples per Dataset: {max_samples}")
        print("=" * 60)
        
        all_results = {}
        combination_count = 0
        total_combinations = len(self.datasets) * len(self.solutions)
        
        for dataset_id in self.datasets.keys():
            all_results[dataset_id] = {}
            for solution_id in self.solutions.keys():
                combination_count += 1
                print(f"\n🎯 Combination {combination_count}/{total_combinations}: {dataset_id} + {solution_id}")
                
                try:
                    result = await self.run_benchmark(dataset_id, solution_id, max_samples, save_results, False, False)
                    all_results[dataset_id][solution_id] = result
                    accuracy = result.get('overall_metrics', {}).get('weighted_accuracy', 0)
                    print(f"✅ Completed: {accuracy:.1f}% accuracy")
                except Exception as e:
                    print(f"❌ Failed: {str(e)}")
                    all_results[dataset_id][solution_id] = {"error": str(e), "success": False}
        
        # Generate comprehensive summary
        comprehensive_summary = self._generate_comprehensive_summary(all_results)
        
        if comparison_table:
            self._generate_comparison_table(comprehensive_summary)
        
        self._display_comprehensive_summary(comprehensive_summary)
        
        return comprehensive_summary
    
    async def _run_all_solutions_single_dataset(self, dataset: str, max_samples: Optional[int], save_results: bool, comparison_table: bool) -> Dict[str, Any]:
        """Run single dataset with all solutions."""
        print(f"\n🚀 TESTING ALL SOLUTIONS ON {dataset.upper()}")
        print("=" * 60)
        
        results = {}
        for i, solution_id in enumerate(self.solutions.keys(), 1):
            print(f"\n🎯 Solution {i}/{len(self.solutions)}: {solution_id}")
            try:
                result = await self.run_benchmark(dataset, solution_id, max_samples, save_results, False, False)
                results[solution_id] = result
                accuracy = result.get('overall_metrics', {}).get('weighted_accuracy', 0)
                print(f"✅ Completed: {accuracy:.1f}% accuracy")
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                results[solution_id] = {"error": str(e), "success": False}
        
        summary = self._generate_multi_solution_summary(dataset, results)
        
        if comparison_table:
            self._generate_comparison_table(summary)
        
        self._display_multi_solution_summary(summary)
        
        return summary
    
    async def _run_single_solution_all_datasets(self, solution: str, max_samples: Optional[int], save_results: bool, comparison_table: bool) -> Dict[str, Any]:
        """Run single solution on all datasets."""
        print(f"\n🚀 TESTING {solution.upper()} ON ALL DATASETS")
        print("=" * 60)
        
        results = {}
        for i, dataset_id in enumerate(self.datasets.keys(), 1):
            print(f"\n🎯 Dataset {i}/{len(self.datasets)}: {dataset_id}")
            try:
                result = await self.run_benchmark(dataset_id, solution, max_samples, save_results, False, False)
                results[dataset_id] = result
                accuracy = result.get('overall_metrics', {}).get('weighted_accuracy', 0)
                print(f"✅ Completed: {accuracy:.1f}% accuracy")
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
                results[dataset_id] = {"error": str(e), "success": False}
        
        summary = self._generate_multi_dataset_summary(solution, results)
        
        if comparison_table:
            self._generate_comparison_table(summary)
        
        self._display_multi_dataset_summary(summary)
        
        return summary
    
    def _generate_comprehensive_summary(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary for all combinations."""
        try:
            from tabulate import tabulate
        except ImportError:
            print("⚠️  Warning: tabulate not available. Install with: pip install tabulate")
            tabulate = None
        
        summary = {
            "type": "comprehensive",
            "results_matrix": all_results,
            "rankings": {"by_accuracy": [], "by_speed": []},
            "recommendations": {}
        }
        
        # Calculate rankings
        solution_stats = {}
        for dataset_id, solutions in all_results.items():
            for solution_id, result in solutions.items():
                if solution_id not in solution_stats:
                    solution_stats[solution_id] = {"accuracies": [], "times": [], "samples": []}
                
                if result.get("success", True) and "overall_metrics" in result:
                    metrics = result["overall_metrics"]
                    solution_stats[solution_id]["accuracies"].append(metrics.get("weighted_accuracy", 0))
                    solution_stats[solution_id]["times"].append(metrics.get("total_processing_time", 0))
                    solution_stats[solution_id]["samples"].append(metrics.get("total_samples", 0))
        
        # Rank by accuracy
        accuracy_ranking = []
        for solution_id, stats in solution_stats.items():
            if stats["accuracies"]:
                avg_accuracy = sum(stats["accuracies"]) / len(stats["accuracies"])
                accuracy_ranking.append({"solution": solution_id, "avg_accuracy": avg_accuracy})
        
        summary["rankings"]["by_accuracy"] = sorted(accuracy_ranking, key=lambda x: x["avg_accuracy"], reverse=True)
        
        # Rank by speed
        speed_ranking = []
        for solution_id, stats in solution_stats.items():
            if stats["times"] and stats["samples"]:
                total_samples = sum(stats["samples"])
                total_time = sum(stats["times"])
                if total_time > 0:
                    speed = total_samples / total_time
                    speed_ranking.append({"solution": solution_id, "samples_per_second": speed})
        
        summary["rankings"]["by_speed"] = sorted(speed_ranking, key=lambda x: x["samples_per_second"], reverse=True)
        
        # Generate recommendations
        if summary["rankings"]["by_accuracy"]:
            summary["recommendations"]["most_accurate"] = summary["rankings"]["by_accuracy"][0]["solution"]
        if summary["rankings"]["by_speed"]:
            summary["recommendations"]["fastest"] = summary["rankings"]["by_speed"][0]["solution"]
        
        return summary
    
    def _generate_multi_solution_summary(self, dataset: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for multiple solutions on single dataset."""
        summary = {
            "type": "multi_solution",
            "dataset": dataset,
            "results": results,
            "rankings": []
        }
        
        # Rank solutions by accuracy
        rankings = []
        for solution_id, result in results.items():
            if result.get("success", True) and "overall_metrics" in result:
                accuracy = result["overall_metrics"].get("weighted_accuracy", 0)
                time = result["overall_metrics"].get("total_processing_time", 0)
                rankings.append({
                    "solution": solution_id,
                    "accuracy": accuracy,
                    "time": time
                })
        
        summary["rankings"] = sorted(rankings, key=lambda x: x["accuracy"], reverse=True)
        return summary
    
    def _generate_multi_dataset_summary(self, solution: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary for single solution on multiple datasets."""
        summary = {
            "type": "multi_dataset", 
            "solution": solution,
            "results": results,
            "rankings": []
        }
        
        # Rank datasets by accuracy
        rankings = []
        for dataset_id, result in results.items():
            if result.get("success", True) and "overall_metrics" in result:
                accuracy = result["overall_metrics"].get("weighted_accuracy", 0)
                time = result["overall_metrics"].get("total_processing_time", 0)
                rankings.append({
                    "dataset": dataset_id,
                    "accuracy": accuracy,
                    "time": time
                })
        
        summary["rankings"] = sorted(rankings, key=lambda x: x["accuracy"], reverse=True)
        return summary
    
    def _generate_comparison_table(self, summary: Dict[str, Any]):
        """Generate CSV and HTML comparison tables."""
        try:
            import pandas as pd
        except ImportError:
            print("⚠️  Warning: pandas not available for CSV generation. Install with: pip install pandas")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("./results/comparison_tables")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        if summary["type"] == "comprehensive":
            # Create comprehensive comparison table
            rows = []
            for dataset_id, solutions in summary["results_matrix"].items():
                for solution_id, result in solutions.items():
                    if result.get("success", True) and "overall_metrics" in result:
                        metrics = result["overall_metrics"]
                        rows.append({
                            "Dataset": dataset_id,
                            "Solution": solution_id,
                            "Accuracy (%)": f"{metrics.get('weighted_accuracy', 0):.1f}",
                            "Samples": metrics.get('total_samples', 0),
                            "Time (s)": f"{metrics.get('total_processing_time', 0):.1f}",
                            "Time/Sample (s)": f"{metrics.get('avg_time_per_sample', 0):.2f}",
                            "Status": "Success"
                        })
                    else:
                        rows.append({
                            "Dataset": dataset_id,
                            "Solution": solution_id,
                            "Accuracy (%)": "0.0",
                            "Samples": 0,
                            "Time (s)": "0.0", 
                            "Time/Sample (s)": "0.00",
                            "Status": "Failed"
                        })
            
            df = pd.DataFrame(rows)
            csv_file = results_dir / f"comprehensive_comparison_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            print(f"\n📊 Comparison table saved: {csv_file}")
    
    def _display_comprehensive_summary(self, summary: Dict[str, Any]):
        """Display comprehensive benchmark summary."""
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        
        print(f"\n🎯 COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 80)
        
        if tabulate:
            # Create results matrix table
            headers = ["Dataset", "Solution", "Accuracy (%)", "Samples", "Time (s)", "Status"]
            rows = []
            
            for dataset_id, solutions in summary["results_matrix"].items():
                for solution_id, result in solutions.items():
                    if result.get("success", True) and "overall_metrics" in result:
                        metrics = result["overall_metrics"]
                        rows.append([
                            dataset_id,
                            solution_id,
                            f"{metrics.get('weighted_accuracy', 0):.1f}",
                            metrics.get('total_samples', 0),
                            f"{metrics.get('total_processing_time', 0):.1f}",
                            "✅ Success"
                        ])
                    else:
                        rows.append([
                            dataset_id,
                            solution_id,
                            "0.0",
                            0,
                            "0.0",
                            "❌ Failed"
                        ])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        # Display rankings
        if summary["rankings"]["by_accuracy"]:
            print(f"\n🏆 TOP SOLUTIONS BY ACCURACY")
            print("-" * 40)
            for i, rank in enumerate(summary["rankings"]["by_accuracy"][:5], 1):
                print(f"{i}. {rank['solution']}: {rank['avg_accuracy']:.1f}%")
        
        if summary["rankings"]["by_speed"]:
            print(f"\n⚡ TOP SOLUTIONS BY SPEED")
            print("-" * 40)
            for i, rank in enumerate(summary["rankings"]["by_speed"][:5], 1):
                print(f"{i}. {rank['solution']}: {rank['samples_per_second']:.2f} samples/sec")
        
        # Display recommendations
        recs = summary.get("recommendations", {})
        if recs:
            print(f"\n💡 RECOMMENDATIONS")
            print("-" * 40)
            if recs.get("most_accurate"):
                print(f"🎯 Most Accurate: {recs['most_accurate']}")
            if recs.get("fastest"):
                print(f"⚡ Fastest: {recs['fastest']}")
    
    def _display_multi_solution_summary(self, summary: Dict[str, Any]):
        """Display multi-solution summary."""
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        
        print(f"\n🎯 SOLUTION COMPARISON FOR {summary['dataset'].upper()}")
        print("=" * 60)
        
        if tabulate and summary["rankings"]:
            headers = ["Rank", "Solution", "Accuracy (%)", "Time (s)"]
            rows = []
            for i, rank in enumerate(summary["rankings"], 1):
                rows.append([
                    i,
                    rank["solution"],
                    f"{rank['accuracy']:.1f}",
                    f"{rank['time']:.1f}"
                ])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        if summary["rankings"]:
            best = summary["rankings"][0]
            print(f"\n🏆 Best Solution: {best['solution']} ({best['accuracy']:.1f}% accuracy)")
    
    def _display_multi_dataset_summary(self, summary: Dict[str, Any]):
        """Display multi-dataset summary."""
        try:
            from tabulate import tabulate
        except ImportError:
            tabulate = None
        
        print(f"\n🎯 DATASET PERFORMANCE FOR {summary['solution'].upper()}")
        print("=" * 60)
        
        if tabulate and summary["rankings"]:
            headers = ["Rank", "Dataset", "Accuracy (%)", "Time (s)"]
            rows = []
            for i, rank in enumerate(summary["rankings"], 1):
                rows.append([
                    i,
                    rank["dataset"],
                    f"{rank['accuracy']:.1f}",
                    f"{rank['time']:.1f}"
                ])
            
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        
        if summary["rankings"]:
            best = summary["rankings"][0]
            print(f"\n🏆 Best Dataset Performance: {best['dataset']} ({best['accuracy']:.1f}% accuracy)")
    
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
                if isinstance(session.results, list) and session.results:
                    # session.results is a list containing BenchmarkResult objects
                    result_obj = session.results[0]  # Get first (and typically only) result
                    accuracy = getattr(result_obj, 'overall_accuracy', 0) * 100  # Convert to percentage
                    samples = getattr(result_obj, 'total_samples', 0)
                    processing_time = getattr(result_obj, 'processing_time', 0)
                elif isinstance(session.results, dict) and session.results:
                    dataset_results = list(session.results.values())[0]
                    if dataset_results and len(dataset_results) > 0:
                        sample_result = dataset_results[0]
                        accuracy = sample_result.get('overall_accuracy', 0) * 100  # Convert to percentage
                        samples = len(dataset_results)
                        processing_time = sample_result.get('processing_time', 0)
                    else:
                        accuracy = 0
                        samples = 0
                        processing_time = 0
                elif hasattr(session, 'session_id'):
                    # Fallback: extract from session attributes if available
                    accuracy = getattr(session, 'overall_accuracy', 0) * 100  # Convert to percentage
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
  python run_benchmark.py -d custom_images -s gpt-5         # Single dataset + solution
  python run_benchmark.py -d all -s gpt-5 --max-samples 10  # Single solution on all datasets
  python run_benchmark.py -d custom_images -s all           # All solutions on single dataset
  python run_benchmark.py --comprehensive                    # All datasets + all solutions
  python run_benchmark.py --comprehensive --quick            # Comprehensive (quick mode)
  python run_benchmark.py --comprehensive --comparison-table # With CSV comparison table
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
                       help='OCR solution to test (use --list-solutions to see options, or "all")')
    parser.add_argument('--max-samples', type=int,
                       help='Limit number of samples to process (optional)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to file')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive benchmark (all datasets with all solutions)')
    parser.add_argument('--comparison-table', action='store_true',
                       help='Generate comparison table (CSV/HTML) from results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: limit samples for faster execution')
    
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
    if args.comprehensive:
        # Run comprehensive benchmark (all datasets with all solutions)
        try:
            max_samples = args.max_samples
            if args.quick and max_samples is None:
                max_samples = 3  # Quick mode: limit to 3 samples
            
            asyncio.run(runner.run_benchmark(
                dataset="all",
                solution="all", 
                max_samples=max_samples,
                save_results=not args.no_save,
                comprehensive=True,
                comparison_table=args.comparison_table
            ))
        except Exception as e:
            print(f"❌ Comprehensive benchmark failed: {e}")
            sys.exit(1)
        return
    
    if not args.dataset or not args.solution:
        print("❌ Error: Both --dataset and --solution are required for benchmarking")
        print("Use --help for usage information or --comprehensive for all combinations")
        return
    
    try:
        # Set max samples for quick mode
        max_samples = args.max_samples
        if args.quick and max_samples is None:
            max_samples = 3  # Quick mode: limit to 3 samples
        
        # Run the benchmark
        asyncio.run(runner.run_benchmark(
            dataset=args.dataset,
            solution=args.solution,
            max_samples=max_samples,
            save_results=not args.no_save,
            comprehensive=False,
            comparison_table=args.comparison_table
        ))
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
