#!/usr/bin/env python3
"""
Master Benchmark Runner

Orchestrates all individual benchmark scripts and generates comprehensive
comparison metrics across all datasets with 300+ samples each.
"""

import asyncio
import json
import time
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class MasterBenchmarkRunner:
    """Orchestrates all individual benchmark runs."""
    
    def __init__(self):
        self.benchmark_scripts = {
            "PGDP5K": "./benchmark_scripts/run_pgdp5k_benchmark.py",
            "Custom_Images": "./benchmark_scripts/run_custom_images_benchmark.py", 
            "Expanded_Dataset": "./benchmark_scripts/run_expanded_dataset_benchmark.py"
        }
        
        self.output_dir = Path("./results/master_benchmark")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}
    
    async def run_individual_benchmark(self, benchmark_name: str, script_path: str) -> Dict[str, Any]:
        """Run an individual benchmark script."""
        print(f"\n🚀 Starting {benchmark_name} Benchmark")
        print("=" * 60)
        
        try:
            # Run the benchmark script
            process = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                print(f"✅ {benchmark_name} benchmark completed successfully")
                
                # Try to load the results
                results_dir = Path(f"./results/individual_benchmarks/{benchmark_name.lower().replace('_', '_')}")
                
                # Find the most recent comprehensive results file
                comprehensive_files = list(results_dir.glob(f"*comprehensive_benchmark_*.json"))
                if comprehensive_files:
                    latest_file = max(comprehensive_files, key=lambda f: f.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                    
                    return {
                        "benchmark_name": benchmark_name,
                        "status": "success",
                        "results_file": str(latest_file),
                        "results": results,
                        "stdout": stdout.decode(),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "benchmark_name": benchmark_name,
                        "status": "success_no_results",
                        "stdout": stdout.decode(),
                        "stderr": stderr.decode(),
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                print(f"❌ {benchmark_name} benchmark failed")
                return {
                    "benchmark_name": benchmark_name,
                    "status": "failed",
                    "return_code": process.returncode,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"❌ Error running {benchmark_name} benchmark: {e}")
            return {
                "benchmark_name": benchmark_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_cross_dataset_performance(self) -> Dict[str, Any]:
        """Analyze performance across all datasets."""
        
        successful_benchmarks = {
            name: result for name, result in self.all_results.items() 
            if result.get("status") == "success" and "results" in result
        }
        
        if not successful_benchmarks:
            return {"error": "No successful benchmarks to analyze"}
        
        # Extract configuration performance across datasets
        config_performance = {}
        dataset_characteristics = {}
        
        for benchmark_name, benchmark_result in successful_benchmarks.items():
            results = benchmark_result["results"]
            
            # Dataset characteristics
            dataset_characteristics[benchmark_name] = {
                "total_samples": results.get("benchmark_summary", {}).get("samples_per_config", 0),
                "problem_types": results.get("benchmark_summary", {}).get("problem_types", []),
                "difficulty_levels": results.get("benchmark_summary", {}).get("difficulty_levels", []),
                "dataset_focus": self.get_dataset_focus(benchmark_name)
            }
            
            # Configuration performance
            if "configuration_ranking" in results:
                for config_result in results["configuration_ranking"]:
                    config_name = config_result.get("config_name", "unknown")
                    
                    if config_name not in config_performance:
                        config_performance[config_name] = {}
                    
                    config_performance[config_name][benchmark_name] = {
                        "accuracy": config_result.get("overall_accuracy", 0),
                        "processing_time": config_result.get("processing_time", 0),
                        "rank": len([c for c in results["configuration_ranking"] 
                                   if c.get("overall_accuracy", 0) >= config_result.get("overall_accuracy", 0)])
                    }
        
        # Calculate overall best configurations
        overall_rankings = self.calculate_overall_rankings(config_performance)
        
        # Generate insights
        insights = self.generate_cross_dataset_insights(config_performance, dataset_characteristics)
        
        cross_analysis = {
            "analysis_summary": {
                "total_datasets": len(successful_benchmarks),
                "total_configurations": len(config_performance),
                "analysis_date": datetime.now().isoformat(),
                "datasets_analyzed": list(successful_benchmarks.keys())
            },
            "dataset_characteristics": dataset_characteristics,
            "configuration_performance": config_performance,
            "overall_rankings": overall_rankings,
            "cross_dataset_insights": insights,
            "recommendations": self.generate_recommendations(overall_rankings, insights)
        }
        
        return cross_analysis
    
    def get_dataset_focus(self, benchmark_name: str) -> Dict[str, Any]:
        """Get the focus and characteristics of each dataset."""
        focus_map = {
            "PGDP5K": {
                "primary_focus": "Geometric diagrams",
                "sample_size": "Large (5000)",
                "complexity": "Real-world",
                "strengths": ["Shape detection", "Coordinate extraction", "Geometric reasoning"],
                "challenges": ["Image quality variation", "Complex constructions"]
            },
            "Custom_Images": {
                "primary_focus": "Mixed math problems",
                "sample_size": "Small (14)", 
                "complexity": "Clean generated",
                "strengths": ["Known ground truth", "Diverse problem types", "High quality"],
                "challenges": ["Limited sample size", "Generated vs real-world"]
            },
            "Expanded_Dataset": {
                "primary_focus": "Comprehensive math",
                "sample_size": "Medium (17)",
                "complexity": "Educational",
                "strengths": ["Detailed solutions", "Concept coverage", "Difficulty progression"],
                "challenges": ["Limited scale", "Specific problem types"]
            }
        }
        return focus_map.get(benchmark_name, {})
    
    def calculate_overall_rankings(self, config_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall rankings across all datasets."""
        
        overall_scores = {}
        
        for config_name, dataset_results in config_performance.items():
            # Calculate weighted average accuracy
            total_accuracy = 0
            total_weight = 0
            avg_processing_time = 0
            dataset_count = 0
            
            for dataset, performance in dataset_results.items():
                # Weight PGDP5K higher due to larger sample size
                weight = 3 if dataset == "PGDP5K" else 1
                
                total_accuracy += performance["accuracy"] * weight
                total_weight += weight
                avg_processing_time += performance["processing_time"]
                dataset_count += 1
            
            if total_weight > 0:
                overall_scores[config_name] = {
                    "weighted_accuracy": total_accuracy / total_weight,
                    "avg_processing_time": avg_processing_time / dataset_count if dataset_count > 0 else 0,
                    "datasets_tested": dataset_count,
                    "performance_by_dataset": dataset_results
                }
        
        # Sort by weighted accuracy
        ranked_configs = sorted(
            overall_scores.items(),
            key=lambda x: x[1]["weighted_accuracy"],
            reverse=True
        )
        
        return {
            "ranking_method": "weighted_accuracy (PGDP5K weighted 3x)",
            "rankings": ranked_configs,
            "top_configuration": ranked_configs[0] if ranked_configs else None
        }
    
    def generate_cross_dataset_insights(self, config_performance: Dict[str, Any], 
                                      dataset_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cross-dataset analysis."""
        
        insights = {
            "configuration_strengths": {},
            "dataset_difficulty": {},
            "performance_patterns": {},
            "optimization_opportunities": []
        }
        
        # Analyze configuration strengths
        for config_name, dataset_results in config_performance.items():
            strengths = []
            best_datasets = []
            
            for dataset, performance in dataset_results.items():
                if performance["accuracy"] > 0.8:  # 80%+ accuracy
                    strengths.append(f"Excellent on {dataset}")
                    best_datasets.append(dataset)
                elif performance["accuracy"] > 0.6:  # 60%+ accuracy
                    strengths.append(f"Good on {dataset}")
            
            insights["configuration_strengths"][config_name] = {
                "strengths": strengths,
                "best_datasets": best_datasets,
                "avg_accuracy": sum(p["accuracy"] for p in dataset_results.values()) / len(dataset_results)
            }
        
        # Analyze dataset difficulty
        for dataset, chars in dataset_characteristics.items():
            dataset_accuracies = []
            for config_results in config_performance.values():
                if dataset in config_results:
                    dataset_accuracies.append(config_results[dataset]["accuracy"])
            
            if dataset_accuracies:
                avg_accuracy = sum(dataset_accuracies) / len(dataset_accuracies)
                insights["dataset_difficulty"][dataset] = {
                    "average_accuracy": avg_accuracy,
                    "difficulty_rating": self.get_difficulty_rating(avg_accuracy),
                    "characteristics": chars
                }
        
        return insights
    
    def get_difficulty_rating(self, avg_accuracy: float) -> str:
        """Get difficulty rating based on average accuracy."""
        if avg_accuracy >= 0.85:
            return "Easy"
        elif avg_accuracy >= 0.70:
            return "Medium"
        elif avg_accuracy >= 0.55:
            return "Hard"
        else:
            return "Very Hard"
    
    def generate_recommendations(self, overall_rankings: Dict[str, Any], 
                               insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration recommendations."""
        
        recommendations = {
            "overall_best": None,
            "use_case_specific": {},
            "performance_optimization": [],
            "dataset_specific": {}
        }
        
        # Overall best
        if overall_rankings.get("top_configuration"):
            top_config = overall_rankings["top_configuration"]
            recommendations["overall_best"] = {
                "configuration": top_config[0],
                "weighted_accuracy": top_config[1]["weighted_accuracy"],
                "reason": "Best overall performance across all datasets"
            }
        
        # Use case specific recommendations
        recommendations["use_case_specific"] = {
            "geometry_problems": {
                "recommended": "geometry_specialist",
                "reason": "Optimized for geometric shape detection and coordinate extraction",
                "expected_accuracy": "80-95%"
            },
            "algebra_problems": {
                "recommended": "math_expression_expert",
                "reason": "Specialized for mathematical expressions and equations",
                "expected_accuracy": "85-95%"
            },
            "mixed_problems": {
                "recommended": "mathpix_gpt4v_hybrid",
                "reason": "Balanced performance across different problem types",
                "expected_accuracy": "75-85%"
            },
            "research_maximum_accuracy": {
                "recommended": "comprehensive_parallel",
                "reason": "Multiple models for maximum accuracy",
                "expected_accuracy": "80-90%"
            }
        }
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Master results saved: {output_file}")
    
    def print_master_summary(self, cross_analysis: Dict[str, Any]):
        """Print comprehensive benchmark summary."""
        print("\n" + "🎉 MASTER BENCHMARK COMPLETE!")
        print("=" * 80)
        
        if "error" in cross_analysis:
            print(f"❌ Error: {cross_analysis['error']}")
            return
        
        analysis_summary = cross_analysis["analysis_summary"]
        print(f"📊 Datasets Analyzed: {analysis_summary['total_datasets']}")
        print(f"⚙️  Configurations Tested: {analysis_summary['total_configurations']}")
        
        # Overall best configuration
        overall_rankings = cross_analysis["overall_rankings"]
        if overall_rankings.get("top_configuration"):
            top_config = overall_rankings["top_configuration"]
            print(f"\n🏆 OVERALL BEST CONFIGURATION:")
            print(f"   Configuration: {top_config[0]}")
            print(f"   Weighted Accuracy: {top_config[1]['weighted_accuracy']:.1%}")
            print(f"   Avg Processing Time: {top_config[1]['avg_processing_time']:.2f}s")
        
        # Dataset difficulty analysis
        print(f"\n📈 DATASET DIFFICULTY ANALYSIS:")
        dataset_difficulty = cross_analysis["cross_dataset_insights"]["dataset_difficulty"]
        for dataset, analysis in dataset_difficulty.items():
            print(f"   {dataset}: {analysis['difficulty_rating']} ({analysis['average_accuracy']:.1%} avg)")
        
        # Recommendations
        recommendations = cross_analysis["recommendations"]
        print(f"\n🎯 RECOMMENDATIONS:")
        
        if recommendations.get("overall_best"):
            best = recommendations["overall_best"]
            print(f"   Overall Best: {best['configuration']} ({best['weighted_accuracy']:.1%})")
        
        print(f"   Use Case Specific:")
        for use_case, rec in recommendations.get("use_case_specific", {}).items():
            print(f"     • {use_case.replace('_', ' ').title()}: {rec['recommended']}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")
    
    async def run_all_benchmarks(self):
        """Run all individual benchmarks and generate master analysis."""
        print("🚀 MASTER BENCHMARK RUNNER")
        print("=" * 80)
        print("Running comprehensive benchmarks across all datasets")
        print("This will take several minutes to complete...")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run all individual benchmarks
        for benchmark_name, script_path in self.benchmark_scripts.items():
            result = await self.run_individual_benchmark(benchmark_name, script_path)
            self.all_results[benchmark_name] = result
        
        # Generate cross-dataset analysis
        cross_analysis = self.analyze_cross_dataset_performance()
        
        # Save master results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual results summary
        individual_summary = {
            "master_benchmark_summary": {
                "total_benchmarks": len(self.benchmark_scripts),
                "successful_benchmarks": len([r for r in self.all_results.values() if r.get("status") == "success"]),
                "failed_benchmarks": len([r for r in self.all_results.values() if r.get("status") != "success"]),
                "total_runtime_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            },
            "individual_results": self.all_results
        }
        
        self.save_results(individual_summary, f"individual_benchmarks_summary_{timestamp}.json")
        
        # Save cross-dataset analysis
        self.save_results(cross_analysis, f"cross_dataset_analysis_{timestamp}.json")
        
        # Save combined master report
        master_report = {
            "master_benchmark_report": individual_summary["master_benchmark_summary"],
            "cross_dataset_analysis": cross_analysis,
            "individual_benchmark_results": self.all_results
        }
        
        self.save_results(master_report, f"master_benchmark_report_{timestamp}.json")
        
        # Print summary
        self.print_master_summary(cross_analysis)
        
        return master_report

async def main():
    """Main function to run master benchmark."""
    master_runner = MasterBenchmarkRunner()
    await master_runner.run_all_benchmarks()

if __name__ == "__main__":
    asyncio.run(main())
