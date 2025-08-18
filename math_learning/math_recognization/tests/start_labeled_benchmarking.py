#!/usr/bin/env python3
"""
🎯 Labeled Dataset Benchmarking Starter Script

This script begins the process of downloading and setting up labeled datasets
for rigorous accuracy benchmarking of our math test analysis system.

Usage:
    python start_labeled_benchmarking.py --dataset all
    python start_labeled_benchmarking.py --dataset unimer
    python start_labeled_benchmarking.py --dataset mario
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import requests
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class LabeledDatasetManager:
    """
    Manager for downloading and setting up labeled datasets for benchmarking
    """
    
    def __init__(self, base_dir: str = "labeled_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "unimer": {
                "name": "UniMER-1M",
                "url": "https://github.com/opendatalab/UniMERNet",
                "description": "1M mathematical expressions with LaTeX ground truth",
                "priority": "HIGH",
                "format": "Image → LaTeX",
                "use_case": "Mathematical expression recognition"
            },
            "mathwriting": {
                "name": "MathWriting",
                "url": "https://github.com/google-research/mathwriting",
                "description": "230k human-written + 400k synthetic math expressions",
                "priority": "HIGH", 
                "format": "Touch screen → LaTeX",
                "use_case": "Handwritten math recognition"
            },
            "mario": {
                "name": "MARIO-EVAL",
                "url": "https://github.com/MARIO-Math-Reasoning/MARIO_EVAL",
                "description": "Mathematical reasoning evaluation toolkit",
                "priority": "MEDIUM",
                "format": "Problem → Solution verification",
                "use_case": "Mathematical equivalence checking"
            },
            "umath": {
                "name": "U-MATH",
                "url": "https://toloka.ai/math-benchmark",
                "description": "1,100 university-level math problems",
                "priority": "HIGH",
                "format": "Text + Visual problems",
                "use_case": "Educational assessment"
            },
            "mathv": {
                "name": "MATH-Vision",
                "url": "https://github.com/mathvision-cuhk/MATH-V",
                "description": "3,040 math problems with visual contexts",
                "priority": "MEDIUM",
                "format": "Image-based problems",
                "use_case": "Multimodal math reasoning"
            },
            "numina": {
                "name": "NuminaMath-groundtruth",
                "url": "https://huggingface.co/datasets/PrimeIntellect/NuminaMath-groundtruth",
                "description": "859k mathematical problems with solutions",
                "priority": "MEDIUM",
                "format": "Problem → Solution → Ground truth",
                "use_case": "Math problem solving"
            }
        }
        
    def print_banner(self):
        """Print startup banner"""
        print("🎯" + "="*70)
        print("    LABELED DATASET BENCHMARKING SYSTEM")
        print("    Math Test Analysis Accuracy Evaluation")
        print("="*72)
        print()
        
    def list_available_datasets(self):
        """List all available datasets with their details"""
        print("📚 Available Labeled Datasets:")
        print("-" * 50)
        
        for key, dataset in self.datasets.items():
            priority_icon = "⭐" if dataset["priority"] == "HIGH" else "📋"
            print(f"{priority_icon} {dataset['name']} ({key})")
            print(f"   📝 {dataset['description']}")
            print(f"   🔗 {dataset['url']}")
            print(f"   📊 Use Case: {dataset['use_case']}")
            print(f"   📋 Priority: {dataset['priority']}")
            print()
            
    def download_mario_eval(self) -> bool:
        """Download and setup MARIO-EVAL toolkit"""
        print("🔧 Setting up MARIO-EVAL toolkit...")
        
        mario_dir = self.base_dir / "mario_eval"
        
        try:
            if mario_dir.exists():
                print("   ✅ MARIO-EVAL already exists")
                return True
                
            # Clone repository
            print("   📥 Cloning MARIO-EVAL repository...")
            subprocess.run([
                "git", "clone", 
                "https://github.com/MARIO-Math-Reasoning/MARIO_EVAL.git",
                str(mario_dir)
            ], check=True, capture_output=True)
            
            # Install dependencies
            print("   📦 Installing dependencies...")
            subprocess.run([
                "pip", "install", "-e", "."
            ], cwd=mario_dir, check=True, capture_output=True)
            
            print("   ✅ MARIO-EVAL setup complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error setting up MARIO-EVAL: {e}")
            return False
            
    def download_mathv_dataset(self) -> bool:
        """Download MATH-Vision dataset"""
        print("🔧 Setting up MATH-Vision dataset...")
        
        mathv_dir = self.base_dir / "math_vision"
        
        try:
            if mathv_dir.exists():
                print("   ✅ MATH-Vision already exists")
                return True
                
            # Clone repository
            print("   📥 Cloning MATH-Vision repository...")
            subprocess.run([
                "git", "clone",
                "https://github.com/mathvision-cuhk/MATH-V.git",
                str(mathv_dir)
            ], check=True, capture_output=True)
            
            print("   ✅ MATH-Vision setup complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error setting up MATH-Vision: {e}")
            return False
            
    def setup_huggingface_dataset(self, dataset_name: str, hf_path: str) -> bool:
        """Setup a Hugging Face dataset"""
        print(f"🔧 Setting up {dataset_name} from Hugging Face...")
        
        try:
            # Check if datasets library is installed
            subprocess.run(["pip", "install", "datasets"], check=True, capture_output=True)
            
            # Create download script
            script_content = f'''
import os
from datasets import load_dataset
from pathlib import Path

# Create directory
dataset_dir = Path("{self.base_dir}") / "{dataset_name.lower().replace('-', '_')}"
dataset_dir.mkdir(exist_ok=True)

print("📥 Downloading {dataset_name} dataset...")
try:
    dataset = load_dataset("{hf_path}")
    
    # Save dataset info
    with open(dataset_dir / "dataset_info.json", "w") as f:
        import json
        json.dump({{
            "name": "{dataset_name}",
            "source": "{hf_path}",
            "splits": list(dataset.keys()),
            "features": str(dataset.features) if hasattr(dataset, 'features') else "Unknown",
            "download_date": "{datetime.now().isoformat()}"
        }}, f, indent=2)
    
    # Save sample data for inspection
    if "train" in dataset:
        sample = dataset["train"].select(range(min(10, len(dataset["train"]))))
        sample.to_json(dataset_dir / "sample_data.jsonl")
    
    print(f"✅ {dataset_name} downloaded successfully!")
    print(f"📁 Saved to: {{dataset_dir}}")
    
except Exception as e:
    print(f"❌ Error downloading {dataset_name}: {{e}}")
'''
            
            # Execute download script
            exec(script_content)
            return True
            
        except Exception as e:
            print(f"   ❌ Error setting up {dataset_name}: {e}")
            return False
            
    def create_benchmark_framework(self):
        """Create basic benchmarking framework"""
        print("🏗️ Creating benchmarking framework...")
        
        framework_dir = self.base_dir / "benchmark_framework"
        framework_dir.mkdir(exist_ok=True)
        
        # Create evaluation metrics module
        metrics_code = '''
"""
Comprehensive evaluation metrics for math test analysis benchmarking
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class BenchmarkResults:
    """Container for benchmark results"""
    dataset_name: str
    model_name: str
    metrics: Dict[str, float]
    timestamp: str
    sample_count: int
    
class BenchmarkMetrics:
    """
    Comprehensive evaluation metrics for math test analysis
    """
    
    @staticmethod
    def character_error_rate(predicted: str, ground_truth: str) -> float:
        """Calculate Character Error Rate (CER)"""
        if not ground_truth:
            return 1.0 if predicted else 0.0
            
        # Simple edit distance calculation
        def edit_distance(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                        
            return dp[m][n]
        
        errors = edit_distance(predicted, ground_truth)
        return errors / len(ground_truth)
    
    @staticmethod
    def word_error_rate(predicted: str, ground_truth: str) -> float:
        """Calculate Word Error Rate (WER)"""
        pred_words = predicted.split()
        gt_words = ground_truth.split()
        
        if not gt_words:
            return 1.0 if pred_words else 0.0
            
        # Calculate word-level edit distance
        def word_edit_distance(w1, w2):
            m, n = len(w1), len(w2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j
                
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if w1[i-1] == w2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
                        
            return dp[m][n]
        
        errors = word_edit_distance(pred_words, gt_words)
        return errors / len(gt_words)
    
    @staticmethod
    def exact_match_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
        """Calculate exact match accuracy"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
            
        matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
        return matches / len(predictions)
    
    @staticmethod
    def classification_metrics(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Calculate precision, recall, F1 for classification tasks"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")
        
        # Convert to binary classification (correct/incorrect)
        pred_binary = [1 if p.strip() == g.strip() else 0 for p, g in zip(predictions, ground_truths)]
        true_binary = [1] * len(ground_truths)  # All ground truths are correct by definition
        
        tp = sum(pred_binary)
        fp = len(predictions) - tp
        fn = 0  # No false negatives in this setup
        tn = 0  # No true negatives in this setup
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": tp / len(predictions)
        }

def run_benchmark_evaluation(dataset_name: str, model_predictions: List[str], 
                           ground_truths: List[str]) -> BenchmarkResults:
    """
    Run comprehensive benchmark evaluation
    """
    metrics = {}
    
    # Calculate all metrics
    metrics["exact_match_accuracy"] = BenchmarkMetrics.exact_match_accuracy(
        model_predictions, ground_truths
    )
    
    # Calculate average CER and WER
    cers = [BenchmarkMetrics.character_error_rate(p, g) 
            for p, g in zip(model_predictions, ground_truths)]
    wers = [BenchmarkMetrics.word_error_rate(p, g) 
            for p, g in zip(model_predictions, ground_truths)]
    
    metrics["character_error_rate"] = np.mean(cers)
    metrics["word_error_rate"] = np.mean(wers)
    
    # Classification metrics
    classification_results = BenchmarkMetrics.classification_metrics(
        model_predictions, ground_truths
    )
    metrics.update(classification_results)
    
    return BenchmarkResults(
        dataset_name=dataset_name,
        model_name="math_test_analyzer_v1",
        metrics=metrics,
        timestamp=datetime.now().isoformat(),
        sample_count=len(model_predictions)
    )
'''
        
        with open(framework_dir / "metrics.py", "w") as f:
            f.write(metrics_code)
            
        # Create evaluation runner
        runner_code = '''
"""
Benchmark evaluation runner for math test analysis system
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from metrics import run_benchmark_evaluation, BenchmarkResults

class BenchmarkRunner:
    """
    Main benchmark evaluation runner
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load a labeled dataset"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
        # Try to load different formats
        if dataset_path.suffix == ".json":
            with open(dataset_path) as f:
                return json.load(f)
        elif dataset_path.suffix == ".jsonl":
            data = []
            with open(dataset_path) as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return {"data": data}
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    
    def run_evaluation(self, dataset_name: str, dataset_path: str, 
                      model_predictions: List[str]) -> BenchmarkResults:
        """
        Run evaluation on a specific dataset
        """
        print(f"🧪 Running evaluation on {dataset_name}...")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Extract ground truths (this will vary by dataset format)
        ground_truths = self.extract_ground_truths(dataset)
        
        # Run benchmark
        results = run_benchmark_evaluation(dataset_name, model_predictions, ground_truths)
        
        # Save results
        results_file = self.results_dir / f"{dataset_name}_results_{results.timestamp.replace(':', '-')}.json"
        with open(results_file, "w") as f:
            json.dump({
                "dataset_name": results.dataset_name,
                "model_name": results.model_name,
                "metrics": results.metrics,
                "timestamp": results.timestamp,
                "sample_count": results.sample_count
            }, f, indent=2)
        
        print(f"✅ Results saved to: {results_file}")
        return results
    
    def extract_ground_truths(self, dataset: Dict[str, Any]) -> List[str]:
        """
        Extract ground truth labels from dataset
        (This method needs to be customized for each dataset format)
        """
        # Default implementation - override for specific datasets
        if "data" in dataset:
            return [item.get("ground_truth", "") for item in dataset["data"]]
        else:
            return [dataset.get("ground_truth", "")]
    
    def print_results(self, results: BenchmarkResults):
        """Print benchmark results in a formatted way"""
        print("\\n" + "="*60)
        print(f"📊 BENCHMARK RESULTS: {results.dataset_name}")
        print("="*60)
        print(f"📅 Timestamp: {results.timestamp}")
        print(f"🔢 Sample Count: {results.sample_count}")
        print(f"🤖 Model: {results.model_name}")
        print("\\n📈 Metrics:")
        
        for metric_name, value in results.metrics.items():
            if isinstance(value, float):
                print(f"   • {metric_name}: {value:.4f}")
            else:
                print(f"   • {metric_name}: {value}")
        print("="*60)

if __name__ == "__main__":
    runner = BenchmarkRunner()
    
    # Example usage
    print("🎯 Benchmark Runner Ready!")
    print("Use this script to evaluate your model against labeled datasets.")
    print("\\nExample usage:")
    print("  from benchmark_runner import BenchmarkRunner")
    print("  runner = BenchmarkRunner()")
    print("  results = runner.run_evaluation('test_dataset', 'path/to/dataset.json', predictions)")
'''
        
        with open(framework_dir / "benchmark_runner.py", "w") as f:
            f.write(runner_code)
            
        print("   ✅ Benchmarking framework created!")
        
    def setup_dataset(self, dataset_key: str) -> bool:
        """Setup a specific dataset"""
        if dataset_key not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_key}")
            return False
            
        dataset = self.datasets[dataset_key]
        print(f"🔧 Setting up {dataset['name']}...")
        
        success = False
        
        if dataset_key == "mario":
            success = self.download_mario_eval()
        elif dataset_key == "mathv":
            success = self.download_mathv_dataset()
        elif dataset_key == "numina":
            success = self.setup_huggingface_dataset("NuminaMath-groundtruth", 
                                                   "PrimeIntellect/NuminaMath-groundtruth")
        else:
            print(f"   ⚠️  Manual setup required for {dataset['name']}")
            print(f"   🔗 Visit: {dataset['url']}")
            success = True
            
        return success
        
    def setup_all_datasets(self):
        """Setup all available datasets"""
        print("🚀 Setting up all labeled datasets...")
        
        success_count = 0
        total_count = len(self.datasets)
        
        for dataset_key in self.datasets:
            if self.setup_dataset(dataset_key):
                success_count += 1
                
        print(f"\\n📊 Setup Summary: {success_count}/{total_count} datasets ready")
        
        if success_count == total_count:
            print("🎉 All datasets setup successfully!")
        else:
            print("⚠️  Some datasets require manual setup")
            
    def create_status_report(self):
        """Create a status report of available datasets"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "base_directory": str(self.base_dir),
            "datasets": {}
        }
        
        for key, dataset in self.datasets.items():
            dataset_dir = self.base_dir / key
            report["datasets"][key] = {
                "name": dataset["name"],
                "priority": dataset["priority"],
                "status": "Downloaded" if dataset_dir.exists() else "Not Downloaded",
                "local_path": str(dataset_dir) if dataset_dir.exists() else None,
                "url": dataset["url"]
            }
            
        # Save report
        report_file = self.base_dir / "dataset_status_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
            
        print(f"📋 Status report saved to: {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Setup labeled datasets for benchmarking")
    parser.add_argument("--dataset", choices=["all", "unimer", "mathwriting", "mario", 
                                            "umath", "mathv", "numina"], 
                       default="all", help="Dataset to setup")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    
    args = parser.parse_args()
    
    manager = LabeledDatasetManager()
    manager.print_banner()
    
    if args.list:
        manager.list_available_datasets()
        return
        
    if args.status:
        manager.create_status_report()
        return
        
    # Create benchmarking framework
    manager.create_benchmark_framework()
    
    if args.dataset == "all":
        manager.setup_all_datasets()
    else:
        manager.setup_dataset(args.dataset)
        
    # Create final status report
    print("\\n📋 Creating final status report...")
    manager.create_status_report()
    
    print("\\n🎯 Next Steps:")
    print("1. Review the dataset status report")
    print("2. Manually download any missing datasets")
    print("3. Run pilot benchmarks with available datasets")
    print("4. Begin comprehensive evaluation process")

if __name__ == "__main__":
    main() 