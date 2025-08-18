#!/usr/bin/env python3
"""
📥 Dataset Downloader for Math Test Analysis Benchmarking

Downloads real benchmark datasets from various sources including:
- GitHub repositories
- Hugging Face datasets
- Direct URLs
- Research institution datasets

Usage:
    python dataset_downloader.py --dataset all --output ../datasets/
    python dataset_downloader.py --dataset unimer --force-download
    python dataset_downloader.py --list-available
"""

import os
import sys
import json
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
from urllib.parse import urlparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class DatasetDownloader:
    """Downloads and manages benchmark datasets."""
    
    def __init__(self, output_dir: Path, verbose: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Dataset registry with download information
        self.datasets = {
            "unimer": {
                "name": "UniMER-1M Mathematical Expression Recognition",
                "description": "1M mathematical expressions with LaTeX ground truth",
                "source": "github", 
                "url": "https://github.com/opendatalab/UniMERNet",
                "size": "~2GB",
                "format": "images + json",
                "access": "public",
                "download_method": "git_clone"
            },
            "math_vision": {
                "name": "MATH-Vision Multimodal Mathematical Reasoning",
                "description": "3,040 visual math problems from competitions",
                "source": "huggingface",
                "url": "https://huggingface.co/datasets/AI4Math/MathVision",
                "size": "~500MB", 
                "format": "images + json",
                "access": "public",
                "download_method": "huggingface"
            },
            "mario_eval": {
                "name": "MARIO-EVAL Mathematical Reasoning",
                "description": "Mathematical reasoning evaluation toolkit",
                "source": "github",
                "url": "https://github.com/MARIO-Math-Reasoning/MARIO-EVAL",
                "size": "~100MB",
                "format": "json",
                "access": "public", 
                "download_method": "git_clone"
            },
            "numina_math": {
                "name": "NuminaMath Problem Dataset",
                "description": "859k mathematical problems with verified solutions",
                "source": "huggingface",
                "url": "https://huggingface.co/datasets/AI-MO/NuminaMath-CoT",
                "size": "~1GB",
                "format": "json",
                "access": "public",
                "download_method": "huggingface"
            },
            "math_qa": {
                "name": "MathQA Dataset",
                "description": "Mathematical word problems with step-by-step solutions",
                "source": "direct",
                "url": "https://math-qa.github.io/math-QA/",
                "size": "~50MB",
                "format": "json",
                "access": "public",
                "download_method": "direct_download"
            }
        }
    
    def list_available_datasets(self) -> None:
        """List all available datasets."""
        print("📊 Available Benchmark Datasets:")
        print("="*60)
        
        for dataset_id, info in self.datasets.items():
            status = "✅ Available" if info["access"] == "public" else "🔒 Restricted"
            print(f"\n🔹 {dataset_id.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Source: {info['source']} - {info['url']}")
            print(f"   Size: {info['size']}")
            print(f"   Format: {info['format']}")
            print(f"   Status: {status}")
        
        print("="*60)
    
    def check_git_available(self) -> bool:
        """Check if git is available."""
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def check_huggingface_available(self) -> bool:
        """Check if Hugging Face datasets library is available."""
        try:
            import datasets
            return True
        except ImportError:
            return False
    
    def download_via_git_clone(self, dataset_id: str, info: Dict[str, Any]) -> bool:
        """Download dataset via git clone."""
        if not self.check_git_available():
            print(f"❌ Git not available. Cannot download {dataset_id}")
            return False
        
        dataset_dir = self.output_dir / dataset_id
        
        if dataset_dir.exists():
            print(f"⚠️  Dataset {dataset_id} already exists at {dataset_dir}")
            return True
        
        print(f"📥 Cloning {info['name']}...")
        print(f"   URL: {info['url']}")
        print(f"   Destination: {dataset_dir}")
        
        try:
            result = subprocess.run([
                "git", "clone", info["url"], str(dataset_dir)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ Successfully cloned {dataset_id}")
                self._create_dataset_info(dataset_id, info, dataset_dir)
                return True
            else:
                print(f"❌ Failed to clone {dataset_id}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout while cloning {dataset_id}")
            return False
        except Exception as e:
            print(f"❌ Error cloning {dataset_id}: {e}")
            return False
    
    def download_via_huggingface(self, dataset_id: str, info: Dict[str, Any]) -> bool:
        """Download dataset via Hugging Face datasets library."""
        if not self.check_huggingface_available():
            print(f"❌ Hugging Face datasets library not available.")
            print("   Install with: pip install datasets")
            return False
        
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {info['name']} via Hugging Face...")
        print(f"   URL: {info['url']}")
        print(f"   Destination: {dataset_dir}")
        
        try:
            from datasets import load_dataset
            
            # Extract dataset name from URL
            dataset_name = info["url"].split("/")[-1]
            if "/" in dataset_name:
                dataset_name = info["url"].split("/")[-2] + "/" + dataset_name
            
            # Download dataset
            dataset = load_dataset(dataset_name, cache_dir=str(dataset_dir))
            
            # Save as JSON for consistency
            for split_name, split_data in dataset.items():
                output_file = dataset_dir / f"{split_name}.json"
                split_data.to_json(str(output_file))
                print(f"   Saved {split_name} split to {output_file}")
            
            print(f"✅ Successfully downloaded {dataset_id}")
            self._create_dataset_info(dataset_id, info, dataset_dir)
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_id}: {e}")
            return False
    
    def download_via_direct_url(self, dataset_id: str, info: Dict[str, Any]) -> bool:
        """Download dataset via direct URL."""
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {info['name']} via direct URL...")
        print(f"   URL: {info['url']}")
        print(f"   Destination: {dataset_dir}")
        
        # For now, create a placeholder since direct URLs vary
        placeholder_file = dataset_dir / "download_instructions.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"Dataset: {info['name']}\n")
            f.write(f"Description: {info['description']}\n")
            f.write(f"Source URL: {info['url']}\n")
            f.write(f"\nTo download this dataset:\n")
            f.write(f"1. Visit the URL above\n")
            f.write(f"2. Follow the download instructions on the website\n")
            f.write(f"3. Extract files to this directory: {dataset_dir}\n")
        
        print(f"✅ Created download instructions for {dataset_id}")
        print(f"   See: {placeholder_file}")
        self._create_dataset_info(dataset_id, info, dataset_dir)
        return True
    
    def _create_dataset_info(self, dataset_id: str, info: Dict[str, Any], dataset_dir: Path) -> None:
        """Create dataset information file."""
        info_file = dataset_dir / "dataset_info.json"
        dataset_info = {
            "dataset_id": dataset_id,
            "download_date": datetime.now().isoformat(),
            "source_info": info,
            "local_path": str(dataset_dir),
            "status": "downloaded"
        }
        
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def download_dataset(self, dataset_id: str, force: bool = False) -> bool:
        """Download a specific dataset."""
        if dataset_id not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_id}")
            print(f"Available datasets: {', '.join(self.datasets.keys())}")
            return False
        
        info = self.datasets[dataset_id]
        
        if info["access"] != "public":
            print(f"🔒 Dataset {dataset_id} requires special access")
            print(f"   Please visit: {info['url']}")
            return False
        
        # Check if already downloaded
        dataset_dir = self.output_dir / dataset_id
        if dataset_dir.exists() and not force:
            print(f"⚠️  Dataset {dataset_id} already exists. Use --force to re-download.")
            return True
        
        # Download based on method
        method = info["download_method"]
        
        if method == "git_clone":
            return self.download_via_git_clone(dataset_id, info)
        elif method == "huggingface":
            return self.download_via_huggingface(dataset_id, info)
        elif method == "direct_download":
            return self.download_via_direct_url(dataset_id, info)
        else:
            print(f"❌ Unknown download method: {method}")
            return False
    
    def download_all_datasets(self, force: bool = False) -> Dict[str, bool]:
        """Download all available datasets."""
        print("🚀 Downloading All Available Datasets...")
        print("="*60)
        
        results = {}
        
        for dataset_id in self.datasets:
            print(f"\n📦 Processing {dataset_id}...")
            success = self.download_dataset(dataset_id, force)
            results[dataset_id] = success
            
            if success:
                print(f"✅ {dataset_id} completed successfully")
            else:
                print(f"❌ {dataset_id} failed")
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total Datasets: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {successful/total:.1%}")
        
        for dataset_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {dataset_id}")
        
        print("="*60)
        
        return results
    
    def create_benchmark_config(self) -> None:
        """Create a configuration file for downloaded datasets."""
        config = {
            "created_date": datetime.now().isoformat(),
            "datasets_dir": str(self.output_dir),
            "available_datasets": {}
        }
        
        for dataset_id in self.datasets:
            dataset_dir = self.output_dir / dataset_id
            info_file = dataset_dir / "dataset_info.json"
            
            if info_file.exists():
                with open(info_file, 'r') as f:
                    dataset_info = json.load(f)
                config["available_datasets"][dataset_id] = dataset_info
        
        config_file = self.output_dir / "benchmark_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📋 Benchmark configuration saved to: {config_file}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    
    # Get available dataset names without instantiating the class
    temp_datasets = {
        "unimer": {},
        "math_vision": {},
        "mario_eval": {},
        "numina_math": {},
        "math_qa": {}
    }
    
    parser.add_argument("--dataset", choices=["all"] + list(temp_datasets.keys()),
                       default="all", help="Dataset to download")
    parser.add_argument("--output", default="../datasets", help="Output directory")
    parser.add_argument("--list-available", action="store_true", help="List available datasets")
    parser.add_argument("--force", action="store_true", help="Force re-download existing datasets")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / args.output
    
    # Create downloader
    downloader = DatasetDownloader(output_dir, verbose=args.verbose)
    
    if args.list_available:
        downloader.list_available_datasets()
        return
    
    if args.dataset == "all":
        downloader.download_all_datasets(force=args.force)
    else:
        downloader.download_dataset(args.dataset, force=args.force)
    
    # Create benchmark configuration
    downloader.create_benchmark_config()

if __name__ == "__main__":
    main() 