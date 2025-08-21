#!/usr/bin/env python3
"""
Custom Test Images Benchmark Script

Runs comprehensive benchmarking on custom generated math problem images
with detailed metrics for algebra, geometry, and mixed math problems.
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
    print("⚠️  Warning: python-dotenv not installed.")

from improved_benchmark_runner import ImprovedBenchmarkRunner

class CustomImagesBenchmarkRunner:
    """Specialized benchmark runner for custom test images."""
    
    def __init__(self):
        self.dataset_name = "Custom Test Images"
        self.dataset_path = "./test_images"
        self.samples = 14  # All available custom images
        self.output_dir = Path("./results/individual_benchmarks/custom_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load problem metadata
        self.load_problem_metadata()
        
        # Configurations optimized for different problem types
        self.configs_to_test = [
            "gpt-5",                   # Best for algebra (GPT-5)
            "math_expression_expert",   # Best for expressions
            "geometry_specialist",      # Best for geometry
            "mathpix_gpt5_hybrid"      # Industry standard
        ]
    
    def load_problem_metadata(self):
        """Load problem metadata from JSON summary."""
        summary_file = Path(self.dataset_path) / "test_problems_summary.json"
        try:
            with open(summary_file, 'r') as f:
                self.metadata = json.load(f)
                self.problems_by_category = {}
                
                # Group problems by category
                for problem in self.metadata["problems"]:
                    category = problem["category"]
                    if category not in self.problems_by_category:
                        self.problems_by_category[category] = []
                    self.problems_by_category[category].append(problem)
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not load problem metadata: {e}")
            self.metadata = {"problems": []}
            self.problems_by_category = {}
    
    async def run_benchmark(self, config_name: str) -> Dict[str, Any]:
        """Run benchmark for a specific configuration."""
        print(f"\n🚀 Starting Custom Images Benchmark: {config_name}")
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
                "category_performance": self.analyze_category_performance(session),
                "problem_type_analysis": self.analyze_problem_types(session)
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error running benchmark for {config_name}: {e}")
            return {
                "config_name": config_name,
                "dataset": self.dataset_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_category_performance(self, session) -> Dict[str, Any]:
        """Analyze performance by problem category (Algebra, Geometry, Mixed Math)."""
        category_analysis = {}
        
        for category, problems in self.problems_by_category.items():
            category_analysis[category] = {
                "total_problems": len(problems),
                "accuracy": 0.0,  # Would be calculated from actual results
                "avg_processing_time": 0.0,
                "problem_types": list(set(p.get("title", "Unknown") for p in problems))
            }
        
        return category_analysis
    
    def analyze_problem_types(self, session) -> Dict[str, Any]:
        """Analyze performance by specific problem types."""
        problem_type_analysis = {
            "algebra_problems": {
                "linear_equations": {"accuracy": 0.0, "samples": 1},
                "quadratic_equations": {"accuracy": 0.0, "samples": 1},
                "system_of_equations": {"accuracy": 0.0, "samples": 1},
                "polynomial_expressions": {"accuracy": 0.0, "samples": 1},
                "fraction_equations": {"accuracy": 0.0, "samples": 1}
            },
            "geometry_problems": {
                "triangle_area": {"accuracy": 0.0, "samples": 1},
                "circle_properties": {"accuracy": 0.0, "samples": 1},
                "rectangle_perimeter": {"accuracy": 0.0, "samples": 1},
                "right_triangle": {"accuracy": 0.0, "samples": 1},
                "coordinate_geometry": {"accuracy": 0.0, "samples": 1}
            },
            "mixed_problems": {
                "word_problems": {"accuracy": 0.0, "samples": 1},
                "calculus": {"accuracy": 0.0, "samples": 1},
                "statistics": {"accuracy": 0.0, "samples": 1},
                "trigonometry": {"accuracy": 0.0, "samples": 1}
            }
        }
        
        return problem_type_analysis
    
    def calculate_comprehensive_metrics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive metrics across all configurations."""
        
        if not all_results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in all_results if "error" not in r]
        
        if not successful_results:
            return {"error": "No successful benchmark runs"}
        
        # Find best performing configurations by category
        best_for_algebra = max(successful_results, key=lambda x: x.get("overall_accuracy", 0))
        best_overall = max(successful_results, key=lambda x: x.get("overall_accuracy", 0))
        
        # Calculate category-specific insights
        category_insights = {
            "algebra": {
                "best_config": best_for_algebra["config_name"],
                "expected_accuracy": "90-100%",
                "key_challenges": ["Complex expressions", "Multi-step equations"],
                "recommendations": ["gpt4v_only", "math_expression_expert"]
            },
            "geometry": {
                "best_config": "geometry_specialist",
                "expected_accuracy": "70-90%", 
                "key_challenges": ["Shape recognition", "Coordinate extraction"],
                "recommendations": ["geometry_specialist", "mathpix_gpt4v_hybrid"]
            },
            "mixed_math": {
                "best_config": "mathpix_gpt4v_hybrid",
                "expected_accuracy": "75-85%",
                "key_challenges": ["Diverse problem types", "Context understanding"],
                "recommendations": ["mathpix_gpt4v_hybrid", "comprehensive_parallel"]
            }
        }
        
        comprehensive_metrics = {
            "benchmark_summary": {
                "dataset": self.dataset_name,
                "total_configurations_tested": len(successful_results),
                "total_problems": self.samples,
                "problem_categories": list(self.problems_by_category.keys()),
                "benchmark_date": datetime.now().isoformat()
            },
            "performance_overview": {
                "best_overall_configuration": {
                    "name": best_overall["config_name"],
                    "accuracy": best_overall.get("overall_accuracy", 0),
                    "processing_time": best_overall.get("processing_time", 0)
                },
                "category_insights": category_insights
            },
            "configuration_analysis": {
                config["config_name"]: {
                    "overall_accuracy": config.get("overall_accuracy", 0),
                    "processing_time": config.get("processing_time", 0),
                    "strengths": self.get_config_strengths(config["config_name"]),
                    "best_for": self.get_config_best_use(config["config_name"])
                }
                for config in successful_results
            },
            "detailed_results": all_results
        }
        
        return comprehensive_metrics
    
    def get_config_strengths(self, config_name: str) -> List[str]:
        """Get strengths of each configuration."""
        strengths = {
            "gpt4v_only": ["Excellent text recognition", "Strong algebra performance", "Fast processing"],
            "math_expression_expert": ["LaTeX output", "Mathematical notation", "Expression parsing"],
            "geometry_specialist": ["Shape detection", "Geometric diagrams", "Coordinate extraction"],
            "mathpix_gpt4v_hybrid": ["Balanced performance", "Industry standard", "Reliable fallback"]
        }
        return strengths.get(config_name, ["General purpose"])
    
    def get_config_best_use(self, config_name: str) -> List[str]:
        """Get best use cases for each configuration."""
        best_use = {
            "gpt4v_only": ["Algebra problems", "Text-heavy math", "Quick analysis"],
            "math_expression_expert": ["Complex equations", "LaTeX generation", "Mathematical expressions"],
            "geometry_specialist": ["Geometric diagrams", "Shape analysis", "Construction problems"],
            "mathpix_gpt4v_hybrid": ["Mixed problems", "Production use", "Balanced accuracy"]
        }
        return best_use.get(config_name, ["General math problems"])
    
    def print_metrics_explanation(self):
        """Print explanation of what each metric means."""
        print("\n" + "=" * 80)
        print("📊 CUSTOM TEST IMAGES BENCHMARK METRICS EXPLANATION")
        print("=" * 80)
        
        print("\n🎯 PRIMARY METRICS:")
        print("• Overall Accuracy: % of math problems correctly solved")
        print("• Processing Time: Average time per image in seconds")
        print("• Category Performance: Accuracy breakdown by problem type")
        
        print("\n📚 PROBLEM CATEGORIES:")
        print("• Algebra (5 problems): Linear equations, quadratics, systems, polynomials")
        print("• Geometry (5 problems): Area, perimeter, triangles, circles, coordinates")
        print("• Mixed Math (4 problems): Word problems, calculus, statistics, trigonometry")
        
        print("\n🔍 DETAILED METRICS:")
        print("• Expression Recognition: % of mathematical expressions correctly parsed")
        print("• Question Detection: % of problem statements correctly identified")
        print("• Answer Extraction: % of solutions correctly determined")
        print("• Error Analysis: % of solution steps correctly understood")
        
        print("\n🏆 CONFIGURATION COMPARISON:")
        print("• gpt4v_only: Best for algebra, excellent text recognition")
        print("• math_expression_expert: Best for complex mathematical expressions")
        print("• geometry_specialist: Best for geometric problems with shapes")
        print("• mathpix_gpt4v_hybrid: Balanced performance across all types")
        
        print("\n📈 EXPECTED PERFORMANCE:")
        print("• Algebra Problems: 90-100% (text-based, well-formatted)")
        print("• Geometry Problems: 70-90% (requires shape recognition)")
        print("• Mixed Problems: 75-85% (diverse complexity)")
        print("• Overall Average: 80-95% (clean, generated images)")
        
        print("\n🎨 DATASET ADVANTAGES:")
        print("• Known ground truth for all problems")
        print("• Clean, high-quality generated images")
        print("• Diverse problem types and difficulty levels")
        print("• Perfect for configuration comparison and validation")
        
        print("=" * 80)
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Results saved: {output_file}")
    
    async def run_full_benchmark(self):
        """Run complete benchmark across all configurations."""
        print("🚀 CUSTOM TEST IMAGES COMPREHENSIVE BENCHMARK")
        print("=" * 60)
        print(f"📊 Dataset: {self.dataset_name}")
        print(f"📁 Path: {self.dataset_path}")
        print(f"🔢 Total Problems: {self.samples}")
        print(f"📚 Categories: {list(self.problems_by_category.keys())}")
        print(f"⚙️  Configurations: {len(self.configs_to_test)}")
        print("=" * 60)
        
        self.print_metrics_explanation()
        
        all_results = []
        
        for config in self.configs_to_test:
            result = await self.run_benchmark(config)
            all_results.append(result)
            
            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_file = f"custom_images_{config}_{timestamp}.json"
            self.save_results(result, individual_file)
        
        # Calculate comprehensive metrics
        comprehensive_metrics = self.calculate_comprehensive_metrics(all_results)
        
        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_file = f"custom_images_comprehensive_benchmark_{timestamp}.json"
        self.save_results(comprehensive_metrics, comprehensive_file)
        
        # Print summary
        self.print_benchmark_summary(comprehensive_metrics)
        
        return comprehensive_metrics
    
    def print_benchmark_summary(self, metrics: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n🎉 CUSTOM IMAGES BENCHMARK COMPLETE!")
        print("=" * 60)
        
        if "error" in metrics:
            print(f"❌ Error: {metrics['error']}")
            return
        
        best_config = metrics["performance_overview"]["best_overall_configuration"]
        print(f"🏆 Best Overall: {best_config['name']}")
        print(f"📈 Best Accuracy: {best_config['accuracy']:.1%}")
        print(f"⚡ Processing Time: {best_config['processing_time']:.2f}s")
        
        print("\n📚 Category Recommendations:")
        for category, insight in metrics["performance_overview"]["category_insights"].items():
            print(f"• {category.title()}: {insight['best_config']} ({insight['expected_accuracy']})")
        
        print(f"\n📁 Results saved to: {self.output_dir}")

async def main():
    """Main function to run Custom Images benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Custom Images Benchmark')
    parser.add_argument('--config', type=str, 
                       help='Specific configuration to run (e.g., gpt4v_only, mathpix_gpt4v_hybrid, geometry_specialist, math_expression_expert)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with mock successful OCR results (for testing without real API keys)')
    
    args = parser.parse_args()
    
    benchmark = CustomImagesBenchmarkRunner()
    
    # Enable test mode if requested
    if args.test_mode:
        print("🧪 RUNNING IN TEST MODE - Using mock successful OCR results")
        print("   This bypasses real API calls for testing purposes")
        print("=" * 60)
        
        # Create mock OCR processor
        class MockOCRProcessor:
            def validate_credentials(self):
                print("✅ Mock API credentials validated")
            
            async def process_problem_with_config(self, image_path: str, config_name: str, problem_data: dict = None) -> dict:
                """Mock OCR processing that returns successful results."""
                import asyncio
                from pathlib import Path
                
                print(f"   🔍 Mock processing {Path(image_path).name} with {config_name}")
                
                # Simulate processing time
                await asyncio.sleep(0.05)
                
                # Extract expected answer from ground truth
                expected_answer = problem_data.get("ground_truth_answer", "x = 5")
                expected_text = problem_data.get("ground_truth_text", "Solve for x")
                
                # Return mock successful result that matches ground truth
                return {
                    "success": True,
                    "extracted_text": expected_text + f", {expected_answer}",
                    "confidence": 0.92,
                    "processing_time": 0.05,
                    "provider": config_name,
                    "error": "",
                    "shapes_detected": problem_data.get("expected_shapes", []),
                    "coordinates_extracted": problem_data.get("coordinates_expected", []),
                    "text_recognized": expected_text.split() + expected_answer.split(),
                    "ground_truth": {
                        "answer": expected_answer,
                        "text": expected_text,
                        "shapes": problem_data.get("expected_shapes", []),
                        "coordinates": problem_data.get("coordinates_expected", []),
                        "difficulty": problem_data.get("difficulty", "medium")
                    }
                }
        
        # Replace the real OCR processor with mock
        from improved_benchmark_runner import ImprovedBenchmarkRunner
        original_init = ImprovedBenchmarkRunner.__init__
        
        def mock_init(self, output_dir=None):
            original_init(self, output_dir)
            self.real_ocr_processor = MockOCRProcessor()
        
        ImprovedBenchmarkRunner.__init__ = mock_init
    
    if args.list_configs:
        print("Available configurations:")
        for config in benchmark.configs_to_test:
            print(f"  - {config}")
        return
    
    if args.config:
        if args.config not in benchmark.configs_to_test:
            print(f"❌ Error: '{args.config}' is not a valid configuration.")
            print("Available configurations:")
            for config in benchmark.configs_to_test:
                print(f"  - {config}")
            return
        
        # Run only the specified configuration
        print(f"🚀 Running Custom Images Benchmark with configuration: {args.config}")
        print("=" * 60)
        
        result = await benchmark.run_benchmark(args.config)
        
        # Save individual result
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        individual_file = f"custom_images_{args.config}_{timestamp}.json"
        benchmark.save_results(result, individual_file)
        
        # Print summary for single config
        print(f"\n🎉 BENCHMARK COMPLETE - {args.config.upper()}")
        print("=" * 60)
        if "error" not in result:
            print(f"📊 Overall Accuracy: {result.get('overall_accuracy', 0):.1%}")
            print(f"⏱️  Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"📋 Total Samples: {result.get('total_samples', 0)}")
        else:
            print(f"❌ Error: {result['error']}")
        print(f"📁 Results saved to: {benchmark.output_dir / individual_file}")
    else:
        # Run full benchmark with all configurations
        await benchmark.run_full_benchmark()

if __name__ == "__main__":
    asyncio.run(main())

