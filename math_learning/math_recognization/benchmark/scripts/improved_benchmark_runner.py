#!/usr/bin/env python3
"""
Improved Benchmark Runner for Math Recognition System

This module provides a comprehensive benchmarking system that uses the OCR
configuration system to evaluate different models and strategies.
"""

import asyncio
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Environment variables from .env file won't be loaded.")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr_config import get_config_manager, BenchmarkConfig, OCRProvider, ProcessingStrategy
from scripts.evaluation_metrics import BenchmarkEvaluator, BenchmarkResult
from real_ocr_processor import RealOCRProcessor

@dataclass
class BenchmarkSession:
    """Represents a complete benchmark session."""
    session_id: str
    config_name: str
    config: BenchmarkConfig
    start_time: datetime
    end_time: Optional[datetime] = None
    results: List[BenchmarkResult] = None
    summary: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.results is None:
            self.results = []

class ImprovedBenchmarkRunner:
    """Improved benchmark runner with OCR configuration support."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the benchmark runner."""
        self.config_manager = get_config_manager()
        self.evaluator = BenchmarkEvaluator()
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent / "results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize real OCR processor
        self.real_ocr_processor = RealOCRProcessor()
        
        # Load math recognition modules
        self._load_modules()
    
    def _load_modules(self):
        """Load required math recognition modules."""
        try:
            # Import the main modules
            from math_learning.math_recognization import (
                CompleteTestAnalyzer, HybridImageProcessor, 
                GPT4VisionClient, MathpixOCRClient
            )
            from math_learning.math_recognization.advanced_geometry_ocr import (
                GeometryOCRProcessor, GOTOCRClient, UniMERNetClient
            )
            
            self.modules = {
                'CompleteTestAnalyzer': CompleteTestAnalyzer,
                'HybridImageProcessor': HybridImageProcessor,
                'GPT4VisionClient': GPT4VisionClient,
                'MathpixOCRClient': MathpixOCRClient,
                'GeometryOCRProcessor': GeometryOCRProcessor,
                'GOTOCRClient': GOTOCRClient,
                'UniMERNetClient': UniMERNetClient
            }
            print("✅ Successfully loaded math recognition modules")
            
        except ImportError as e:
            print(f"⚠️  Warning: Could not load some modules: {e}")
            self.modules = {}
    
    async def run_benchmark(self, 
                          config_name: str,
                          dataset_path: str = None,
                          max_samples: int = None,
                          save_results: bool = True) -> BenchmarkSession:
        """Run a complete benchmark using the specified configuration."""
        
        # Get configuration
        config = self.config_manager.get_config(config_name)
        
        # Validate configuration
        validation = self.config_manager.validate_config(config)
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration: {validation['errors']}")
        
        if validation["warnings"]:
            print("⚠️  Configuration warnings:")
            for warning in validation["warnings"]:
                print(f"   - {warning}")
        
        # Create benchmark session
        session_id = f"{config_name}_{int(time.time())}"
        session = BenchmarkSession(
            session_id=session_id,
            config_name=config_name,
            config=config,
            start_time=datetime.now()
        )
        
        print(f"\n🚀 Starting Benchmark Session: {session_id}")
        print(f"📋 Configuration: {config.name}")
        print(f"📝 Description: {config.description}")
        print(f"🔧 Strategy: {config.processing_strategy.value}")
        print(f"🎯 Primary OCR: {config.primary_ocr.provider.value}")
        if config.fallback_ocr:
            print(f"🔄 Fallback OCR: {config.fallback_ocr.provider.value}")
        print("=" * 60)
        
        # Run benchmarks for each dataset type
        all_results = []
        
        # If dataset_path is provided, use it directly instead of predefined types
        if dataset_path and Path(dataset_path).exists():
            print(f"\n📊 Using provided dataset: {dataset_path}")
            dataset_type = "custom_dataset"
            
            try:
                # Load dataset from provided path
                dataset = await self._load_dataset(dataset_type, dataset_path, max_samples)
                
                if not dataset:
                    print(f"⚠️  No data available from {dataset_path}, falling back to synthetic data...")
                    # Fall back to synthetic data generation
                    dataset_types_to_use = config.dataset_types
                else:
                    # Use the loaded dataset
                    dataset_types_to_use = [dataset_type]
                    
            except Exception as e:
                print(f"❌ Error loading dataset from {dataset_path}: {e}")
                print("   Falling back to synthetic data generation...")
                dataset_types_to_use = config.dataset_types
        else:
            print(f"\n📊 Using synthetic data generation for dataset types: {config.dataset_types}")
            dataset_types_to_use = config.dataset_types
        
        # Process each dataset type
        for dataset_type in dataset_types_to_use:
            print(f"\n📊 Evaluating dataset type: {dataset_type}")
            
            try:
                # Load or generate dataset (only if not already loaded above)
                if dataset_path and Path(dataset_path).exists() and dataset_type == "custom_dataset":
                    # Dataset already loaded above
                    pass
                else:
                    dataset = await self._load_dataset(dataset_type, dataset_path, max_samples)
                
                if not dataset:
                    print(f"⚠️  No data available for {dataset_type}, skipping...")
                    continue
                
                # Run evaluation
                result = await self._evaluate_dataset(config, dataset_type, dataset)
                all_results.append(result)
                
                # Print intermediate results
                self._print_dataset_results(result)
                
            except Exception as e:
                print(f"❌ Error evaluating {dataset_type}: {e}")
                continue
        
        # Complete session
        session.end_time = datetime.now()
        session.results = all_results
        session.summary = self._generate_session_summary(session)
        
        # Save results
        if save_results:
            self._save_session_results(session)
        
        # Print final summary
        self._print_session_summary(session)
        
        return session
    
    async def _load_dataset(self, dataset_type: str, dataset_path: str = None, max_samples: int = None) -> Dict[str, Any]:
        """Load or generate dataset for evaluation."""
        
        # If custom_dataset type with path, load from external source
        if dataset_type == "custom_dataset" and dataset_path and Path(dataset_path).exists():
            return await self._load_external_dataset(dataset_path, dataset_type)
        
        elif dataset_type == "sample_data":
            # Use built-in sample data generator
            return await self._generate_sample_data(max_samples or 20)
        
        elif dataset_type == "geometry":
            # Generate geometry-specific test data
            return await self._generate_geometry_data(max_samples or 15)
        
        elif dataset_type == "math_expressions":
            # Generate mathematical expression data
            return await self._generate_expression_data(max_samples or 25)
        
        elif dataset_type == "test_papers":
            # Generate test paper simulation data
            return await self._generate_test_paper_data(max_samples or 10)
        
        elif dataset_path and Path(dataset_path).exists():
            # Load from provided dataset path
            return await self._load_external_dataset(dataset_path, dataset_type)
        
        else:
            print(f"⚠️  Dataset type '{dataset_type}' not implemented, generating sample data")
            return await self._generate_sample_data(max_samples or 10)
    
    async def _generate_sample_data(self, num_samples: int) -> Dict[str, Any]:
        """Generate sample data for benchmarking."""
        
        sample_problems = [
            {
                "id": f"sample_{i+1}",
                "type": "algebra" if i % 3 == 0 else "geometry" if i % 3 == 1 else "arithmetic",
                "question": f"Sample math problem {i+1}",
                "ground_truth_answer": f"answer_{i+1}",
                "ground_truth_text": f"Problem {i+1}: Solve for x",
                "difficulty": "easy" if i < num_samples//3 else "medium" if i < 2*num_samples//3 else "hard",
                "expected_shapes": ["triangle", "circle"] if i % 3 == 1 else [],
                "expected_text": [f"x = {i+1}", f"answer = {i+1}"],
                "confidence_threshold": 0.7
            }
            for i in range(num_samples)
        ]
        
        return {
            "dataset_type": "sample_data",
            "total_samples": num_samples,
            "problems": sample_problems,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Generated sample data for benchmarking"
            }
        }
    
    async def _generate_geometry_data(self, num_samples: int) -> Dict[str, Any]:
        """Generate geometry-specific test data."""
        
        geometry_problems = [
            {
                "id": f"geo_{i+1}",
                "type": "geometry",
                "question": f"Find the area of triangle with vertices A(0,0), B({i+1},0), C(0,{i+2})",
                "ground_truth_answer": f"{(i+1)*(i+2)/2}",
                "ground_truth_text": f"Area = {(i+1)*(i+2)/2} square units",
                "difficulty": "medium",
                "expected_shapes": ["triangle", "coordinate_system"],
                "expected_text": [f"A(0,0)", f"B({i+1},0)", f"C(0,{i+2})", "Area"],
                "coordinates_expected": [f"(0,0)", f"({i+1},0)", f"(0,{i+2})"],
                "confidence_threshold": 0.6
            }
            for i in range(num_samples)
        ]
        
        return {
            "dataset_type": "geometry",
            "total_samples": num_samples,
            "problems": geometry_problems,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Generated geometry problems for OCR evaluation"
            }
        }
    
    async def _generate_expression_data(self, num_samples: int) -> Dict[str, Any]:
        """Generate mathematical expression data."""
        
        expressions = [
            {
                "id": f"expr_{i+1}",
                "type": "expression",
                "question": f"Simplify: {i+1}x + {i+2}x",
                "ground_truth_answer": f"{2*i+3}x",
                "ground_truth_text": f"{2*i+3}x",
                "difficulty": "easy",
                "expected_shapes": [],
                "expected_text": [f"{i+1}x", f"{i+2}x", f"{2*i+3}x"],
                "confidence_threshold": 0.8
            }
            for i in range(num_samples)
        ]
        
        return {
            "dataset_type": "math_expressions",
            "total_samples": num_samples,
            "problems": expressions,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Generated mathematical expressions for OCR evaluation"
            }
        }
    
    async def _generate_test_paper_data(self, num_samples: int) -> Dict[str, Any]:
        """Generate test paper simulation data."""
        
        test_papers = [
            {
                "id": f"paper_{i+1}",
                "type": "test_paper",
                "questions": [
                    {
                        "question_id": f"q{j+1}",
                        "question_text": f"Question {j+1}: Solve {j+1}x = {(j+1)*2}",
                        "student_answer": f"x = {2}",
                        "correct_answer": f"x = {2}",
                        "points": 10
                    }
                    for j in range(3)  # 3 questions per paper
                ],
                "total_questions": 3,
                "expected_score": 30,
                "difficulty": "medium"
            }
            for i in range(num_samples)
        ]
        
        return {
            "dataset_type": "test_papers",
            "total_samples": num_samples,
            "problems": test_papers,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "description": "Generated test papers for complete analysis evaluation"
            }
        }
    
    async def _load_external_dataset(self, dataset_path: str, dataset_type: str) -> Dict[str, Any]:
        """Load dataset from external file or directory."""
        
        dataset_path_obj = Path(dataset_path)
        
        if dataset_path_obj.is_file() and dataset_path_obj.suffix == '.json':
            # Load from JSON file
            with open(dataset_path_obj, 'r') as f:
                data = json.load(f)
            
            # Ensure image paths are absolute
            problems = data.get("problems", [])
            for problem in problems:
                if "image_path" in problem and not Path(problem["image_path"]).is_absolute():
                    problem["image_path"] = str(dataset_path_obj.parent / problem["image_path"])
            
            return {
                "dataset_type": dataset_type,
                "total_samples": len(problems),
                "problems": problems,
                "metadata": {
                    "loaded_from": str(dataset_path_obj),
                    "loaded_at": datetime.now().isoformat()
                }
            }
            
        elif dataset_path_obj.is_dir():
            # Load from directory with images
            return await self._load_dataset_from_directory(dataset_path_obj, dataset_type)
        
        else:
            raise ValueError(f"Unsupported dataset path: {dataset_path}")
    
    async def _load_dataset_from_directory(self, dataset_dir: Path, dataset_type: str) -> Dict[str, Any]:
        """Load dataset from directory containing images."""
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_dir.glob(f"*{ext}"))
            image_files.extend(dataset_dir.glob(f"*{ext.upper()}"))
        
        # Sort for consistent ordering
        image_files.sort()
        
        # Create problems from image files
        problems = []
        for i, image_file in enumerate(image_files):
            problem = {
                "id": f"{dataset_type}_{i+1}",
                "type": dataset_type,
                "image_path": str(image_file),
                "question": f"Analyze mathematical content in {image_file.name}",
                "ground_truth_answer": "",  # Would need to be provided separately
                "ground_truth_text": "",
                "difficulty": "medium",
                "expected_shapes": [],
                "expected_text": [],
                "confidence_threshold": 0.7
            }
            problems.append(problem)
        
        # Try to load metadata if available
        metadata_file = dataset_dir / "metadata.json"
        summary_file = dataset_dir / "test_problems_summary.json"
        
        if summary_file.exists():
            # Load test problems summary format
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
                
            # Update problems with summary metadata
            for problem in problems:
                image_name = Path(problem["image_path"]).name
                # Find matching problem in summary
                for prob_data in summary_data.get("problems", []):
                    if prob_data.get("filename") == image_name:
                        problem.update({
                            "ground_truth_answer": prob_data.get("solution", ""),
                            "ground_truth_text": prob_data.get("problem", ""),
                            "category": prob_data.get("category", ""),
                            "title": prob_data.get("title", ""),
                            "difficulty": "medium",  # Default difficulty
                            "expected_shapes": [],  # Would need to be specified
                            "expected_text": [],    # Would need to be specified
                        })
                        break
                        
        elif metadata_file.exists():
            # Load generic metadata format
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Update problems with metadata if available
            for problem in problems:
                image_name = Path(problem["image_path"]).name
                if image_name in metadata.get("problems", {}):
                    problem_meta = metadata["problems"][image_name]
                    problem.update(problem_meta)
        
        return {
            "dataset_type": dataset_type,
            "total_samples": len(problems),
            "problems": problems,
            "metadata": {
                "loaded_from": str(dataset_dir),
                "loaded_at": datetime.now().isoformat(),
                "image_count": len(image_files)
            }
        }
    
    async def _evaluate_dataset(self, config: BenchmarkConfig, dataset_type: str, dataset: Dict[str, Any]) -> BenchmarkResult:
        """Evaluate a dataset using the specified configuration."""
        
        start_time = time.time()
        problems = dataset["problems"]
        
        # Simulate OCR processing based on configuration
        predictions = await self._process_problems(config, problems)
        
        # Prepare ground truth in the format expected by evaluator
        ground_truth = self._prepare_ground_truth(dataset_type, problems)
        
        # Use real OCR evaluation for better accuracy
        # Flatten predictions to get individual OCR results
        all_ocr_results = []
        for category_results in predictions.values():
            all_ocr_results.extend(category_results)
        
        # Run evaluation with real OCR results
        result = self.evaluator.evaluate_real_ocr_results(
            ocr_results=all_ocr_results,
            dataset_name=dataset_type,
            system_name=f"{config.name} ({config.processing_strategy.value})"
        )
        
        processing_time = time.time() - start_time
        result.processing_time = processing_time
        
        return result
    
    async def _process_problems(self, config: BenchmarkConfig, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process problems using real OCR APIs."""
        
        predictions = {"expressions": [], "test_papers": [], "solutions": []}
        
        for problem in problems:
            # Use real OCR processing instead of simulation
            real_result = await self._process_with_real_ocr(config, problem)
            
            # Categorize based on problem type
            if problem.get("type") in ["expression", "algebra", "arithmetic"]:
                predictions["expressions"].append(real_result)
            elif problem.get("type") == "test_paper":
                predictions["test_papers"].append(real_result)
            else:
                predictions["solutions"].append(real_result)
        
        return predictions
    
    async def _process_with_real_ocr(self, config: BenchmarkConfig, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Process a problem using real OCR APIs based on configuration."""
        
        # Get image path from problem data
        image_path = problem.get("image_path")
        if not image_path or not Path(image_path).exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}",
                "extracted_text": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "shapes_detected": [],
                "coordinates_extracted": [],
                "text_recognized": []
            }
        
        # Map configuration to OCR method
        config_name = self._get_config_name_from_config(config)
        
        try:
            # Process with real OCR processor
            result = await self.real_ocr_processor.process_problem_with_config(
                image_path=image_path,
                config_name=config_name,
                problem_data=problem
            )
            
            # Add ground truth comparison for evaluation
            result["ground_truth"] = {
                "answer": problem.get("ground_truth_answer", ""),
                "text": problem.get("ground_truth_text", ""),
                "shapes": problem.get("expected_shapes", []),
                "coordinates": problem.get("coordinates_expected", []),
                "difficulty": problem.get("difficulty", "medium")
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {image_path} with {config_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "confidence": 0.0,
                "processing_time": 0.0,
                "shapes_detected": [],
                "coordinates_extracted": [],
                "text_recognized": [],
                "ground_truth": {
                    "answer": problem.get("ground_truth_answer", ""),
                    "text": problem.get("ground_truth_text", ""),
                    "shapes": problem.get("expected_shapes", []),
                    "coordinates": problem.get("coordinates_expected", []),
                    "difficulty": problem.get("difficulty", "medium")
                }
            }
    
    def _get_config_name_from_config(self, config: BenchmarkConfig) -> str:
        """Extract configuration name from BenchmarkConfig object."""
        
        # Map OCR provider to configuration name
        provider_map = {
            OCRProvider.GPT5_VISION: "gpt-5",
            OCRProvider.MATHPIX: "mathpix_gpt5_hybrid",
            OCRProvider.GOT_OCR2: "geometry_specialist",
            OCRProvider.UNIMERNET: "math_expression_expert"
        }
        
        # Use primary provider or default to GPT-5
        primary_provider = config.primary_ocr.provider if config.primary_ocr else OCRProvider.GPT5_VISION
        return provider_map.get(primary_provider, "gpt-5")
    
    async def _simulate_ocr_processing(self, config: BenchmarkConfig, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate OCR processing with realistic results."""
        
        # Base accuracy depends on OCR provider
        base_accuracy = {
            OCRProvider.MATHPIX: 0.85,
            OCRProvider.GPT4_VISION: 0.80,
            OCRProvider.GOT_OCR2: 0.75,
            OCRProvider.UNIMERNET: 0.90,
            OCRProvider.MONKEY_OCR: 0.70
        }.get(config.primary_ocr.provider, 0.75)
        
        # Adjust based on problem difficulty
        difficulty_modifier = {
            "easy": 0.1,
            "medium": 0.0,
            "hard": -0.1
        }.get(problem.get("difficulty", "medium"), 0.0)
        
        # Adjust based on processing strategy
        strategy_modifier = {
            ProcessingStrategy.PARALLEL: 0.05,
            ProcessingStrategy.HYBRID_BEST: 0.08,
            ProcessingStrategy.GEOMETRY_FIRST: 0.03 if problem.get("type") == "geometry" else -0.02
        }.get(config.processing_strategy, 0.0)
        
        final_accuracy = min(0.95, base_accuracy + difficulty_modifier + strategy_modifier)
        
        # Simulate processing time
        processing_time = {
            ProcessingStrategy.PARALLEL: 0.8,
            ProcessingStrategy.VISION_ONLY: 1.2,
            ProcessingStrategy.OCR_ONLY: 0.5,
            ProcessingStrategy.GEOMETRY_FIRST: 1.0
        }.get(config.processing_strategy, 0.7)
        
        # Generate simulated result
        success = final_accuracy > 0.6
        
        if success and final_accuracy > 0.8:
            # High accuracy - return ground truth with minor variations
            recognized_text = problem.get("ground_truth_text", "")
            recognized_answer = problem.get("ground_truth_answer", "")
            detected_shapes = problem.get("expected_shapes", [])
        else:
            # Lower accuracy - introduce errors
            recognized_text = problem.get("ground_truth_text", "")[:int(len(problem.get("ground_truth_text", "")) * final_accuracy)]
            recognized_answer = problem.get("ground_truth_answer", "") if final_accuracy > 0.7 else "unknown"
            detected_shapes = problem.get("expected_shapes", [])[:max(1, int(len(problem.get("expected_shapes", [])) * final_accuracy))]
        
        return {
            "id": problem["id"],
            "success": success,
            "confidence": final_accuracy,
            "processing_time": processing_time,
            "recognized_text": recognized_text,
            "recognized_answer": recognized_answer,
            "detected_shapes": detected_shapes,
            "extracted_coordinates": problem.get("coordinates_expected", [])[:2] if final_accuracy > 0.7 else [],
            "provider_used": config.primary_ocr.provider.value
        }
    
    def _prepare_ground_truth(self, dataset_type: str, problems: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare ground truth data for evaluation."""
        
        ground_truth = {"expressions": [], "test_papers": [], "solutions": []}
        
        for problem in problems:
            gt_item = {
                "id": problem["id"],
                "text": problem.get("ground_truth_text", ""),
                "answer": problem.get("ground_truth_answer", ""),
                "shapes": problem.get("expected_shapes", []),
                "coordinates": problem.get("coordinates_expected", [])
            }
            
            if problem.get("type") in ["expression", "algebra", "arithmetic"]:
                ground_truth["expressions"].append(gt_item)
            elif problem.get("type") == "test_paper":
                ground_truth["test_papers"].append(gt_item)
            else:
                ground_truth["solutions"].append(gt_item)
        
        return ground_truth
    
    def _print_dataset_results(self, result: BenchmarkResult):
        """Print results for a single dataset."""
        
        print(f"\n📊 Results for {result.dataset_name}:")
        print(f"   Overall Accuracy: {result.overall_accuracy:.1%}")
        print(f"   Processing Time: {result.processing_time:.2f}s")
        print(f"   Total Samples: {result.total_samples}")
        
        for metric in result.metrics:
            print(f"   {metric.metric_name}: {metric.accuracy:.1%} ({metric.score:.1f}/{metric.max_score:.1f})")
    
    def _generate_session_summary(self, session: BenchmarkSession) -> Dict[str, Any]:
        """Generate summary for the benchmark session."""
        
        if not session.results:
            return {"error": "No results available"}
        
        total_samples = sum(r.total_samples for r in session.results)
        avg_accuracy = sum(r.overall_accuracy for r in session.results) / len(session.results)
        total_time = sum(r.processing_time for r in session.results)
        
        # Collect all metrics
        all_metrics = {}
        for result in session.results:
            for metric in result.metrics:
                if metric.metric_name not in all_metrics:
                    all_metrics[metric.metric_name] = []
                all_metrics[metric.metric_name].append(metric.accuracy)
        
        # Calculate average metrics
        avg_metrics = {
            name: sum(values) / len(values)
            for name, values in all_metrics.items()
        }
        
        duration = (session.end_time - session.start_time).total_seconds()
        
        return {
            "session_id": session.session_id,
            "configuration": session.config_name,
            "duration_seconds": duration,
            "total_samples_processed": total_samples,
            "overall_accuracy": avg_accuracy,
            "total_processing_time": total_time,
            "average_metrics": avg_metrics,
            "datasets_evaluated": [r.dataset_name for r in session.results],
            "primary_ocr_provider": session.config.primary_ocr.provider.value,
            "processing_strategy": session.config.processing_strategy.value,
            "timestamp": session.start_time.isoformat()
        }
    
    def _print_session_summary(self, session: BenchmarkSession):
        """Print final session summary."""
        
        summary = session.summary
        
        print("\n" + "=" * 80)
        print("🎯 BENCHMARK SESSION SUMMARY")
        print("=" * 80)
        
        if not summary or "error" in summary:
            print("❌ Session completed with errors")
            if summary and "error" in summary:
                print(f"   Error: {summary['error']}")
            print(f"📋 Session ID: {session.session_id}")
            print(f"⚙️  Configuration: {session.config_name}")
            print(f"📊 Results: {len(session.results)} datasets processed")
            return
        
        print(f"📋 Session ID: {summary['session_id']}")
        print(f"⚙️  Configuration: {summary['configuration']}")
        print(f"🔧 Primary OCR: {summary['primary_ocr_provider']}")
        print(f"📊 Strategy: {summary['processing_strategy']}")
        print(f"⏱️  Duration: {summary['duration_seconds']:.1f}s")
        print(f"📈 Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"📊 Total Samples: {summary['total_samples_processed']}")
        print(f"⏱️  Processing Time: {summary['total_processing_time']:.2f}s")
        
        print(f"\n📊 DETAILED METRICS:")
        for metric_name, accuracy in summary['average_metrics'].items():
            print(f"   {metric_name}: {accuracy:.1%}")
        
        print(f"\n📁 DATASETS EVALUATED:")
        for dataset in summary['datasets_evaluated']:
            print(f"   ✅ {dataset}")
        
        print("\n" + "=" * 80)
    
    def _save_session_results(self, session: BenchmarkSession):
        """Save session results to file."""
        
        # Convert session to dictionary for JSON serialization
        session_dict = {
            "session_id": session.session_id,
            "config_name": session.config_name,
            "config": asdict(session.config),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "results": [asdict(result) for result in session.results],
            "summary": session.summary
        }
        
        # Handle enums in config
        session_dict["config"]["primary_ocr"]["provider"] = session.config.primary_ocr.provider.value
        session_dict["config"]["processing_strategy"] = session.config.processing_strategy.value
        if session.config.fallback_ocr:
            session_dict["config"]["fallback_ocr"]["provider"] = session.config.fallback_ocr.provider.value
        
        # Save to file
        output_file = self.output_dir / f"benchmark_session_{session.session_id}.json"
        
        with open(output_file, 'w') as f:
            json.dump(session_dict, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
    
    def compare_configurations(self, config_names: List[str], dataset_type: str = "sample_data") -> Dict[str, Any]:
        """Compare multiple configurations side by side."""
        
        print(f"\n🔍 CONFIGURATION COMPARISON")
        print("=" * 60)
        
        comparison_results = {}
        
        for config_name in config_names:
            print(f"\n🧪 Testing configuration: {config_name}")
            
            try:
                # Run benchmark for this configuration
                session = asyncio.run(self.run_benchmark(
                    config_name=config_name,
                    max_samples=10,  # Small sample for comparison
                    save_results=False
                ))
                
                comparison_results[config_name] = session.summary
                
            except Exception as e:
                print(f"❌ Error testing {config_name}: {e}")
                comparison_results[config_name] = {"error": str(e)}
        
        # Print comparison table
        self._print_comparison_table(comparison_results)
        
        return comparison_results
    
    def _print_comparison_table(self, results: Dict[str, Any]):
        """Print comparison table for multiple configurations."""
        
        print(f"\n📊 CONFIGURATION COMPARISON RESULTS")
        print("=" * 100)
        
        # Headers
        print(f"{'Configuration':<25} {'Accuracy':<12} {'Time (s)':<10} {'Samples':<10} {'Primary OCR':<15}")
        print("-" * 100)
        
        # Results
        for config_name, result in results.items():
            if "error" in result:
                print(f"{config_name:<25} {'ERROR':<12} {'-':<10} {'-':<10} {'-':<15}")
            else:
                accuracy = f"{result['overall_accuracy']:.1%}"
                time_str = f"{result['total_processing_time']:.2f}"
                samples = str(result['total_samples_processed'])
                ocr = result['primary_ocr_provider']
                
                print(f"{config_name:<25} {accuracy:<12} {time_str:<10} {samples:<10} {ocr:<15}")
        
        print("=" * 100)

async def main():
    """Main function for command line usage."""
    
    parser = argparse.ArgumentParser(description="Improved Math Recognition Benchmark Runner")
    parser.add_argument("--config", "-c", help="Configuration name to use")
    parser.add_argument("--dataset", "-d", help="Dataset path (optional)")
    parser.add_argument("--samples", "-s", type=int, help="Maximum samples to process")
    parser.add_argument("--output", "-o", help="Output directory for results")
    parser.add_argument("--list-configs", action="store_true", help="List available configurations")
    parser.add_argument("--compare", nargs="+", help="Compare multiple configurations")
    parser.add_argument("--validate", help="Validate a configuration")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = ImprovedBenchmarkRunner(output_dir=args.output)
    
    # List configurations
    if args.list_configs:
        print("\n📋 Available Configurations:")
        print("=" * 50)
        configs = runner.config_manager.list_configs()
        for name, description in configs.items():
            print(f"• {name}: {description}")
        return
    
    # Validate configuration
    if args.validate:
        try:
            config = runner.config_manager.get_config(args.validate)
            validation = runner.config_manager.validate_config(config)
            
            print(f"\n✅ Validation for '{args.validate}':")
            print(f"Valid: {validation['valid']}")
            if validation['errors']:
                print("Errors:")
                for error in validation['errors']:
                    print(f"  ❌ {error}")
            if validation['warnings']:
                print("Warnings:")
                for warning in validation['warnings']:
                    print(f"  ⚠️  {warning}")
            if validation['missing_requirements']:
                print("Missing Requirements:")
                for req in validation['missing_requirements']:
                    print(f"  🔧 {req}")
        except Exception as e:
            print(f"❌ Error validating configuration: {e}")
        return
    
    # Compare configurations
    if args.compare:
        runner.compare_configurations(args.compare)
        return
    
    # Run single benchmark
    if args.config:
        try:
            session = await runner.run_benchmark(
                config_name=args.config,
                dataset_path=args.dataset,
                max_samples=args.samples
            )
            print(f"\n✅ Benchmark completed successfully!")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return 1
    elif not args.list_configs and not args.validate and not args.compare:
        print("❌ Error: --config is required when running benchmarks")
        print("Use --list-configs to see available configurations")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main())) 