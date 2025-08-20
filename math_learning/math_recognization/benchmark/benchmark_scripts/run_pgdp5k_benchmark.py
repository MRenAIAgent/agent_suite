#!/usr/bin/env python3
"""
PGDP5K Dataset Benchmark Script

Runs comprehensive benchmarking on the PGDP5K dataset (5,000 plane geometry diagrams)
with 300 samples and generates detailed metrics.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")

from improved_benchmark_runner import ImprovedBenchmarkRunner

class PGDP5KBenchmarkRunner:
    """Specialized benchmark runner for PGDP5K dataset."""
    
    def __init__(self):
        self.dataset_name = "PGDP5K"
        self.dataset_path = "./real_datasets/PGDP5K"
        self.samples = 300
        self.output_dir = Path("./results/individual_benchmarks/pgdp5k")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurations optimized for geometry
        self.configs_to_test = [
            "geometry_specialist",
            "mathpix_gpt4v_hybrid", 
            "gpt4v_only",
            "comprehensive_parallel"
        ]
        
    async def run_benchmark(self, config_name: str) -> Dict[str, Any]:
        """Run benchmark for a specific configuration."""
        print(f"\n🚀 Starting PGDP5K Benchmark: {config_name}")
        print("=" * 60)
        
        runner = ImprovedBenchmarkRunner(output_dir=str(self.output_dir))
        
        try:
            session = await runner.run_benchmark(
                config_name=config_name,
                dataset_path=self.dataset_path,
                max_samples=self.samples
            )
            
            # Extract key metrics
            results = {
                "config_name": config_name,
                "dataset": self.dataset_name,
                "total_samples": self.samples,
                "timestamp": datetime.now().isoformat(),
                "session_id": session.session_id,
                "overall_accuracy": session.summary.get("overall_accuracy", 0.0),
                "processing_time": session.summary.get("total_processing_time", 0.0),
                "detailed_metrics": session.summary.get("detailed_metrics", {}),
                "dataset_results": []
            }
            
            # Process individual results
            for result in session.results:
                dataset_result = {
                    "dataset_type": result.dataset_name,
                    "accuracy": result.overall_accuracy,
                    "samples_processed": result.total_samples,
                    "processing_time": result.processing_time,
                    "metrics": {
                        "shape_detection_accuracy": getattr(result, 'shape_detection_accuracy', 0.0),
                        "text_recognition_accuracy": getattr(result, 'text_recognition_accuracy', 0.0),
                        "coordinate_extraction_accuracy": getattr(result, 'coordinate_extraction_accuracy', 0.0),
                        "overall_parsing_accuracy": getattr(result, 'overall_parsing_accuracy', 0.0)
                    }
                }
                results["dataset_results"].append(dataset_result)
            
            return results
            
        except Exception as e:
            print(f"❌ Error running benchmark for {config_name}: {e}")
            return {
                "config_name": config_name,
                "dataset": self.dataset_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_comprehensive_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics across all configurations."""
        
        if not all_results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in all_results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful benchmark runs"}
        
        # Find best performing configuration
        best_config = max(successful_results, key=lambda x: x.get("overall_accuracy", 0))
        
        # Calculate average metrics
        avg_accuracy = sum(r.get("overall_accuracy", 0) for r in successful_results) / len(successful_results)
        avg_processing_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results)
        
        # Geometry-specific metrics analysis
        geometry_metrics = {}
        for result in successful_results:
            config_name = result["config_name"]
            geometry_metrics[config_name] = {
                "overall_accuracy": result.get("overall_accuracy", 0),
                "processing_time": result.get("processing_time", 0),
                "geometry_performance": {}
            }
            
            # Extract geometry-specific performance
            for dataset_result in result.get("dataset_results", []):
                if "geometry" in dataset_result.get("dataset_type", "").lower():
                    geometry_metrics[config_name]["geometry_performance"] = dataset_result.get("metrics", {})
        
        comprehensive_metrics = {
            "benchmark_summary": {
                "dataset": self.dataset_name,
                "total_configurations_tested": len(successful_results),
                "samples_per_config": self.samples,
                "benchmark_date": datetime.now().isoformat()
            },
            "performance_overview": {
                "best_configuration": {
                    "name": best_config["config_name"],
                    "accuracy": best_config.get("overall_accuracy", 0),
                    "processing_time": best_config.get("processing_time", 0)
                },
                "average_accuracy": avg_accuracy,
                "average_processing_time": avg_processing_time
            },
            "geometry_analysis": geometry_metrics,
            "configuration_ranking": sorted(
                successful_results, 
                key=lambda x: x.get("overall_accuracy", 0), 
                reverse=True
            ),
            "detailed_results": all_results
        }
        
        return comprehensive_metrics
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Results saved: {output_file}")
    
    def print_metrics_explanation(self):
        """Print explanation of what each metric means."""
        print("\n" + "=" * 80)
        print("📊 PGDP5K BENCHMARK METRICS EXPLANATION")
        print("=" * 80)
        
        print("\n🎯 PRIMARY METRICS:")
        print("• Overall Accuracy: Percentage of problems correctly solved end-to-end")
        print("• Processing Time: Average time per image in seconds")
        print("• Samples Processed: Number of geometry diagrams analyzed")
        
        print("\n📐 GEOMETRY-SPECIFIC METRICS:")
        print("• Shape Detection Accuracy: % of geometric shapes correctly identified")
        print("  - Points, lines, circles, triangles, rectangles, polygons")
        print("• Text Recognition Accuracy: % of mathematical text correctly read")
        print("  - Numbers, variables, equations, labels, measurements")
        print("• Coordinate Extraction Accuracy: % of coordinate points correctly parsed")
        print("  - (x,y) coordinates, grid positions, axis values")
        print("• Overall Parsing Accuracy: Combined geometric and textual understanding")
        
        print("\n🏆 CONFIGURATION COMPARISON:")
        print("• geometry_specialist: GOT-OCR2.0 optimized for geometric diagrams")
        print("• mathpix_gpt4v_hybrid: Industry standard with Mathpix + GPT-4V fallback")
        print("• gpt4v_only: Pure GPT-4 Vision for comprehensive analysis")
        print("• comprehensive_parallel: Multiple OCR models running in parallel")
        
        print("\n📈 EXPECTED PERFORMANCE RANGES:")
        print("• Shape Detection: 70-95% (depends on diagram complexity)")
        print("• Text Recognition: 80-95% (mathematical notation challenges)")
        print("• Coordinate Extraction: 60-85% (precision-dependent)")
        print("• Overall Accuracy: 70-85% (PGDP5K is challenging real-world data)")
        
        print("\n🎨 DATASET CHARACTERISTICS:")
        print("• 5,000 real plane geometry diagrams from educational materials")
        print("• 16 different symbol types, 6 text categories")
        print("• Difficulty levels: Easy (basic shapes) → Hard (complex constructions)")
        print("• Real-world noise: Scanning artifacts, varying image quality")
        
        print("=" * 80)
    
    async def run_full_benchmark(self):
        """Run complete benchmark across all configurations."""
        print("🚀 PGDP5K COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"📁 Path: {self.dataset_path}")
        print(f"🔢 Samples per config: {self.samples}")
        print(f"⚙️  Configurations: {len(self.configs_to_test)}")
        print("=" * 60)
        
        self.print_metrics_explanation()
        
        all_results = []
        
        for config in self.configs_to_test:
            result = await self.run_benchmark(config)
            all_results.append(result)
            
            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_file = f"pgdp5k_{config}_{timestamp}.json"
            self.save_results(result, individual_file)
        
        # Calculate comprehensive metrics
        comprehensive_metrics = self.calculate_comprehensive_metrics(all_results)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_file = f"pgdp5k_comprehensive_benchmark_{timestamp}.json"
        self.save_results(comprehensive_metrics, comprehensive_file)
        
        # Print summary
        self.print_benchmark_summary(comprehensive_metrics)
        
        return comprehensive_metrics
    
    def print_benchmark_summary(self, metrics: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "🎉 PGDP5K BENCHMARK COMPLETE!")
        print("=" * 60)
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        best_config = metrics["performance_overview"]["best_configuration"]
        print(f"🏆 Best Configuration: {best_config['name']}")
        print(f"📈 Best Accuracy: {best_config['accuracy']:.1%}")
        print(f"⚡ Processing Time: {best_config['processing_time']:.2f}s")
        
        print(f"\n📊 Average Accuracy: {metrics['performance_overview']['average_accuracy']:.1%}")
        print(f"⏱️  Average Processing Time: {metrics['performance_overview']['average_processing_time']:.2f}s")
        
        print("\n🏅 Configuration Ranking:")
        for i, config in enumerate(metrics["configuration_ranking"], 1):
            if "error" not in config:
                print(f"   {i}. {config['config_name']}: {config.get('overall_accuracy', 0):.1%}")
        
        print(f"\n📁 Results saved to: {self.output_dir}")

async def main():
    """Main function to run PGDP5K benchmark."""
    benchmark = PGDP5KBenchmarkRunner()
    await benchmark.run_full_benchmark()

if __name__ == "__main__":
    asyncio.run(main())
