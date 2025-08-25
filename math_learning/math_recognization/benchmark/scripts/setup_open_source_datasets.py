#!/usr/bin/env python3
"""
🌟 OPEN SOURCE MATH DATASETS SETUP
================================================================================
Comprehensive script to download and setup open source datasets for handwritten
mathematical expression recognition and K-12 homework problems.

Based on latest research and available datasets (2024).
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import requests
from urllib.parse import urlparse
import zipfile
import tarfile

class OpenSourceDatasetManager:
    """Manager for downloading and setting up open source math datasets."""
    
    def __init__(self, base_dir: str = "real_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 📊 Available open source datasets
        self.datasets = {
            "mathwriting": {
                "name": "MathWriting Dataset",
                "description": "230K human-written + 400K synthetic handwritten math expressions",
                "size": "630,000 samples",
                "source": "Hugging Face",
                "url": "https://huggingface.co/datasets/deepcopy/MathWriting-human",
                "type": "online_handwritten",
                "format": "Image + LaTeX pairs",
                "difficulty": "Elementary to University",
                "priority": "HIGH",
                "download_method": "huggingface",
                "local_path": "MathWriting"
            },
            
            "hme100k": {
                "name": "HME100K Dataset", 
                "description": "Large-scale real-world handwritten math expressions",
                "size": "99,109 samples (74K train + 24K test)",
                "source": "GitHub + Official Site",
                "url": "https://github.com/Phymond/HME100K",
                "official_url": "https://ai.100tal.com/dataset",
                "type": "offline_handwritten",
                "format": "Image + LaTeX pairs",
                "difficulty": "K-12",
                "priority": "HIGH",
                "download_method": "manual",
                "local_path": "HME100K"
            },
            
            "crohme2023": {
                "name": "CROHME 2023 Dataset",
                "description": "Competition dataset for handwritten math expression recognition",
                "size": "~10,000 samples",
                "source": "Zenodo",
                "url": "https://zenodo.org/records/8428035",
                "type": "online_handwritten",
                "format": "InkML + LaTeX pairs",
                "difficulty": "Elementary to High School",
                "priority": "HIGH",
                "download_method": "zenodo",
                "local_path": "CROHME2023"
            },
            
            "mqhs": {
                "name": "Math Questions with Handwritten Solutions",
                "description": "Handwritten math expressions with bounding box annotations",
                "size": "Variable",
                "source": "GitHub",
                "url": "https://github.com/Jzliu-dl/MQHSdataset",
                "type": "offline_handwritten",
                "format": "Image + Bounding boxes + LaTeX",
                "difficulty": "Elementary to High School",
                "priority": "MEDIUM",
                "download_method": "github",
                "local_path": "MQHS"
            },
            
            "bhmsds": {
                "name": "Basic Handwritten Math Symbols Dataset",
                "description": "27K images of 18 handwritten math symbols",
                "size": "27,000 samples",
                "source": "GitHub",
                "url": "https://github.com/wblachowski/bhmsds",
                "type": "symbol_recognition",
                "format": "Image + Symbol labels",
                "difficulty": "Basic symbols",
                "priority": "LOW",
                "download_method": "github",
                "local_path": "BHMSDS"
            },
            
            "maxtex": {
                "name": "MaxTex Dataset",
                "description": "Large-scale printed mathematical formulas",
                "size": "223,000 samples",
                "source": "Figshare",
                "url": "https://figshare.com/articles/dataset/MaxTex_P_/27321012/2",
                "type": "printed_formulas",
                "format": "Image + LaTeX pairs",
                "difficulty": "Academic",
                "priority": "MEDIUM",
                "download_method": "figshare",
                "local_path": "MaxTex"
            },
            
            "aida_calculus": {
                "name": "Aida Calculus Dataset",
                "description": "100K synthetic handwritten calculus expressions",
                "size": "100,000 samples",
                "source": "NeurIPS 2020",
                "url": "https://nips.cc/virtual/2020/20458",
                "type": "synthetic_handwritten",
                "format": "Image + LaTeX pairs",
                "difficulty": "Calculus",
                "priority": "MEDIUM",
                "download_method": "neurips",
                "local_path": "AidaCalculus"
            }
        }
    
    def list_datasets(self):
        """📊 Display all available open source datasets."""
        print("\n🌟 OPEN SOURCE MATH DATASETS")
        print("=" * 80)
        
        for dataset_id, info in self.datasets.items():
            status = "✅ Available" if (self.base_dir / info['local_path']).exists() else "📥 Not Downloaded"
            
            print(f"\n🗂️  {dataset_id.upper()}")
            print(f"   Name: {info['name']}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Type: {info['type'].replace('_', ' ').title()}")
            print(f"   Difficulty: {info['difficulty']}")
            print(f"   Priority: {info['priority']}")
            print(f"   Source: {info['source']}")
            print(f"   Status: {status}")
    
    def show_download_instructions(self, dataset_id: str):
        """📝 Show detailed download instructions for a specific dataset."""
        if dataset_id not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_id}")
            return
        
        info = self.datasets[dataset_id]
        print(f"\n📥 DOWNLOAD INSTRUCTIONS: {info['name'].upper()}")
        print("=" * 80)
        
        if info['download_method'] == 'huggingface':
            self._show_huggingface_instructions(dataset_id, info)
        elif info['download_method'] == 'manual':
            self._show_manual_instructions(dataset_id, info)
        elif info['download_method'] == 'zenodo':
            self._show_zenodo_instructions(dataset_id, info)
        elif info['download_method'] == 'github':
            self._show_github_instructions(dataset_id, info)
        else:
            self._show_general_instructions(dataset_id, info)
    
    def _show_huggingface_instructions(self, dataset_id: str, info: Dict):
        """Show Hugging Face download instructions."""
        print(f"🤗 **Hugging Face Dataset**: {info['name']}")
        print(f"📊 **Size**: {info['size']}")
        print(f"🔗 **URL**: {info['url']}")
        print("\n**Installation Steps:**")
        print("```bash")
        print("# Install Hugging Face datasets library")
        print("pip install datasets")
        print("")
        print("# Download using Python script")
        print(f"python scripts/download_huggingface_dataset.py --dataset {dataset_id}")
        print("```")
        print("\n**Python Code:**")
        print("```python")
        print("from datasets import load_dataset")
        print(f"dataset = load_dataset('deepcopy/MathWriting-human')")
        print("```")
    
    def _show_manual_instructions(self, dataset_id: str, info: Dict):
        """Show manual download instructions."""
        print(f"📋 **Manual Download Required**: {info['name']}")
        print(f"📊 **Size**: {info['size']}")
        print(f"🔗 **Official URL**: {info.get('official_url', info['url'])}")
        print(f"🔗 **GitHub**: {info['url']}")
        print("\n**Steps:**")
        print("1. 📝 **Register for Academic Access**")
        print(f"   • Visit: {info.get('official_url', info['url'])}")
        print("   • Fill research application form")
        print("   • Provide institutional affiliation")
        print("   • Wait for approval (1-3 days)")
        print("\n2. 💾 **Download Dataset Files**")
        print("   • Download image files (.zip)")
        print("   • Download labels/annotations")
        print("   • Download documentation")
        print(f"\n3. 📁 **Extract to**: real_datasets/{info['local_path']}/")
    
    def _show_zenodo_instructions(self, dataset_id: str, info: Dict):
        """Show Zenodo download instructions."""
        print(f"🏛️  **Zenodo Dataset**: {info['name']}")
        print(f"📊 **Size**: {info['size']}")
        print(f"🔗 **URL**: {info['url']}")
        print("\n**Download Steps:**")
        print("```bash")
        print(f"# Create directory")
        print(f"mkdir -p real_datasets/{info['local_path']}")
        print(f"cd real_datasets/{info['local_path']}")
        print("")
        print("# Download from Zenodo (replace with actual download links)")
        print("wget [ZENODO_DOWNLOAD_URL]")
        print("unzip *.zip")
        print("```")
    
    def _show_github_instructions(self, dataset_id: str, info: Dict):
        """Show GitHub download instructions."""
        print(f"🐙 **GitHub Dataset**: {info['name']}")
        print(f"📊 **Size**: {info['size']}")
        print(f"🔗 **URL**: {info['url']}")
        print("\n**Download Steps:**")
        print("```bash")
        print(f"# Clone repository")
        print(f"git clone {info['url']}.git real_datasets/{info['local_path']}")
        print(f"cd real_datasets/{info['local_path']}")
        print("")
        print("# Check for additional download scripts")
        print("ls *.py *.sh")
        print("# Follow repository-specific instructions")
        print("```")
    
    def _show_general_instructions(self, dataset_id: str, info: Dict):
        """Show general download instructions."""
        print(f"🌐 **Dataset**: {info['name']}")
        print(f"📊 **Size**: {info['size']}")
        print(f"🔗 **URL**: {info['url']}")
        print(f"📝 **Method**: {info['download_method']}")
        print("\n**Steps:**")
        print(f"1. Visit: {info['url']}")
        print("2. Follow website-specific download instructions")
        print(f"3. Extract to: real_datasets/{info['local_path']}/")
    
    def create_benchmark_config(self):
        """📋 Create benchmark configuration for all available datasets."""
        config_file = self.base_dir / "open_source_datasets_config.json"
        
        benchmark_datasets = {}
        for dataset_id, info in self.datasets.items():
            if (self.base_dir / info['local_path']).exists():
                benchmark_datasets[f"{dataset_id}_sample"] = {
                    "name": f"{info['name']} Sample",
                    "description": info['description'],
                    "path": f"./real_datasets/{info['local_path']}/samples/current/images",
                    "samples": "Variable (50-200 typical)",
                    "types": [info['type'], "expressions", "equations"],
                    "difficulty": info['difficulty'],
                    "best_for": f"{info['type'].replace('_', ' ').title()} evaluation with ground truth"
                }
        
        with open(config_file, 'w') as f:
            json.dump(benchmark_datasets, f, indent=2)
        
        print(f"📋 Benchmark configuration saved to: {config_file}")
        return benchmark_datasets
    
    def verify_dataset(self, dataset_id: str) -> bool:
        """✅ Verify if a dataset is properly installed."""
        if dataset_id not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_id}")
            return False
        
        info = self.datasets[dataset_id]
        dataset_path = self.base_dir / info['local_path']
        
        if not dataset_path.exists():
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        # Check for common files
        required_files = ['images', 'labels', 'annotations']
        found_files = []
        
        for item in dataset_path.iterdir():
            if item.is_dir() and item.name in required_files:
                found_files.append(item.name)
            elif item.is_file() and any(ext in item.name for ext in ['.txt', '.json', '.csv']):
                found_files.append(item.name)
        
        print(f"✅ Dataset found: {dataset_path}")
        print(f"📁 Contents: {', '.join(found_files[:5])}")
        if len(found_files) > 5:
            print(f"   ... and {len(found_files) - 5} more files/directories")
        
        return True

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="🌟 Open Source Math Datasets Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python scripts/setup_open_source_datasets.py --list

  # Show download instructions for specific dataset
  python scripts/setup_open_source_datasets.py --info mathwriting
  python scripts/setup_open_source_datasets.py --info hme100k
  python scripts/setup_open_source_datasets.py --info crohme2023

  # Verify dataset installation
  python scripts/setup_open_source_datasets.py --verify mathwriting

  # Create benchmark configuration
  python scripts/setup_open_source_datasets.py --create-config

Priority Datasets for K-12 Homework:
  1. HME100K (Real K-12 handwritten expressions)
  2. MathWriting (Largest collection, 630K samples)
  3. CROHME 2023 (Standard benchmark)
        """
    )
    
    parser.add_argument("--list", action="store_true",
                       help="List all available open source datasets")
    parser.add_argument("--info", type=str, metavar="DATASET",
                       help="Show download instructions for specific dataset")
    parser.add_argument("--verify", type=str, metavar="DATASET", 
                       help="Verify dataset installation")
    parser.add_argument("--create-config", action="store_true",
                       help="Create benchmark configuration for available datasets")
    parser.add_argument("--base-dir", type=str, default="real_datasets",
                       help="Base directory for datasets (default: real_datasets)")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    manager = OpenSourceDatasetManager(args.base_dir)
    
    if args.list:
        manager.list_datasets()
    
    if args.info:
        manager.show_download_instructions(args.info)
    
    if args.verify:
        manager.verify_dataset(args.verify)
    
    if args.create_config:
        manager.create_benchmark_config()

if __name__ == "__main__":
    main()
