#!/usr/bin/env python3
"""
🚀 HME100K Dataset Setup Script
==============================

Downloads and sets up the HME100K dataset for mathematical OCR benchmarking.
Provides random sampling capabilities for manageable testing.

Usage:
    python setup_hme100k.py --download                    # Download full dataset
    python setup_hme100k.py --sample 100                  # Create random sample of 100
    python setup_hme100k.py --sample 50 --seed 42         # Reproducible sample with seed
    python setup_hme100k.py --info                        # Show dataset info
"""

import os
import sys
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any
import subprocess
from datetime import datetime

class HME100KSetup:
    """Setup and manage HME100K dataset with random sampling."""
    
    def __init__(self):
        self.dataset_dir = Path("./real_datasets/HME100K")
        self.samples_dir = self.dataset_dir / "samples"
        self.original_dir = self.dataset_dir / "original"
        
    def download_dataset(self) -> bool:
        """Provide instructions for HME100K dataset download."""
        print("📥 HME100K Dataset Download Instructions")
        print("=" * 60)
        
        print("❗ The HME100K dataset requires manual download from the official website.")
        print()
        print("🌐 Official Download Site: https://ai.100tal.com/dataset")
        print()
        print("📋 Steps to download:")
        print("   1. Visit: https://ai.100tal.com/dataset")
        print("   2. Register/login to access the dataset")
        print("   3. Download the HME100K dataset files")
        print("   4. Extract to: ./real_datasets/HME100K/original/")
        print()
        print("📁 Expected structure after extraction:")
        print("   ./real_datasets/HME100K/original/")
        print("   ├── train/")
        print("   │   ├── train_images/")
        print("   │   └── train_labels.txt")
        print("   └── test/")
        print("       ├── test_images/")
        print("       └── test_labels.txt")
        print()
        print("🔄 Alternative: Use our demo dataset for testing")
        print("   python setup_hme100k.py --create-demo")
        
        return False
    
    def create_demo_dataset(self) -> bool:
        """Create a small demo dataset for testing purposes."""
        print("🎯 Creating HME100K Demo Dataset...")
        print("=" * 50)
        
        try:
            # Create directory structure
            demo_dir = self.original_dir
            demo_dir.mkdir(parents=True, exist_ok=True)
            
            train_dir = demo_dir / "train"
            test_dir = demo_dir / "test"
            train_dir.mkdir(exist_ok=True)
            test_dir.mkdir(exist_ok=True)
            
            # Create some sample LaTeX expressions (K-12 appropriate)
            sample_expressions = [
                ("demo_001.png", "x + 2 = 5"),
                ("demo_002.png", "2x - 3 = 7"),
                ("demo_003.png", "\\frac{1}{2} + \\frac{1}{3}"),
                ("demo_004.png", "x^2 + 3x + 2 = 0"),
                ("demo_005.png", "\\sqrt{16} = 4"),
                ("demo_006.png", "2 \\times 3 = 6"),
                ("demo_007.png", "\\frac{x}{2} = 4"),
                ("demo_008.png", "y = 2x + 1"),
                ("demo_009.png", "3x + 4y = 12"),
                ("demo_010.png", "\\sin(30°) = \\frac{1}{2}"),
            ]
            
            # Create train labels (first 8)
            train_labels = []
            for i, (img_name, latex) in enumerate(sample_expressions[:8]):
                train_labels.append(f"{img_name}\t{latex}\n")
            
            # Create test labels (last 2)
            test_labels = []
            for i, (img_name, latex) in enumerate(sample_expressions[8:]):
                test_labels.append(f"{img_name}\t{latex}\n")
            
            # Write label files
            with open(demo_dir / "train_labels.txt", 'w', encoding='utf-8') as f:
                f.writelines(train_labels)
            
            with open(demo_dir / "test_labels.txt", 'w', encoding='utf-8') as f:
                f.writelines(test_labels)
            
            # Create placeholder image files (we'll use our existing test images as templates)
            import shutil
            test_images_dir = Path("./test_images")
            
            if test_images_dir.exists():
                existing_images = list(test_images_dir.glob("*.png"))
                
                # Copy existing images as demo samples
                for i, (img_name, _) in enumerate(sample_expressions[:8]):
                    if i < len(existing_images):
                        src = existing_images[i % len(existing_images)]
                        dst = train_dir / img_name
                        shutil.copy2(src, dst)
                
                for i, (img_name, _) in enumerate(sample_expressions[8:]):
                    if i < len(existing_images):
                        src = existing_images[i % len(existing_images)]
                        dst = test_dir / img_name
                        shutil.copy2(src, dst)
            else:
                # Create empty placeholder files
                for img_name, _ in sample_expressions[:8]:
                    (train_dir / img_name).touch()
                for img_name, _ in sample_expressions[8:]:
                    (test_dir / img_name).touch()
            
            # Create dataset info
            self._create_dataset_info()
            
            print("✅ Demo dataset created successfully!")
            print(f"📁 Location: {demo_dir}")
            print(f"🖼️  Training samples: 8")
            print(f"🖼️  Testing samples: 2")
            print(f"📝 LaTeX labels included")
            print()
            print("🚀 Ready for testing:")
            print("   python setup_hme100k.py --sample 5")
            print("   python hme100k_manager.py --new-sample 8")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create demo dataset: {e}")
            return False
    
    def _verify_dataset(self) -> bool:
        """Verify the downloaded dataset has expected structure."""
        expected_files = [
            "train_labels.txt",
            "test_labels.txt"
        ]
        
        expected_dirs = [
            "train",
            "test"
        ]
        
        for file in expected_files:
            if not (self.original_dir / file).exists():
                print(f"❌ Missing file: {file}")
                return False
        
        for dir_name in expected_dirs:
            if not (self.original_dir / dir_name).exists():
                print(f"❌ Missing directory: {dir_name}")
                return False
        
        return True
    
    def _create_dataset_info(self):
        """Create dataset information file."""
        info = {
            "name": "HME100K",
            "description": "Handwritten Mathematical Expression 100K Dataset",
            "source": "https://github.com/Phymond/HME100K",
            "downloaded": datetime.now().isoformat(),
            "structure": {
                "train_images": "train/ directory",
                "test_images": "test/ directory", 
                "train_labels": "train_labels.txt",
                "test_labels": "test_labels.txt"
            },
            "format": {
                "images": "PNG files",
                "labels": "LaTeX expressions in text files"
            }
        }
        
        with open(self.dataset_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    def create_random_sample(self, sample_size: int, seed: int = None, split: str = "train") -> bool:
        """Create a random sample from the dataset."""
        print(f"🎲 Creating Random Sample of {sample_size} images...")
        print("=" * 50)
        
        if not self.original_dir.exists():
            print("❌ Original dataset not found. Run --download first.")
            return False
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            print(f"🌱 Using random seed: {seed}")
        
        # Load labels
        labels_file = self.original_dir / f"{split}_labels.txt"
        if not labels_file.exists():
            print(f"❌ Labels file not found: {labels_file}")
            return False
        
        # Read all labels
        with open(labels_file, 'r', encoding='utf-8') as f:
            all_labels = f.readlines()
        
        print(f"📊 Total {split} samples available: {len(all_labels)}")
        
        if sample_size > len(all_labels):
            print(f"⚠️  Requested sample size ({sample_size}) larger than available ({len(all_labels)})")
            sample_size = len(all_labels)
        
        # Random sampling
        selected_indices = random.sample(range(len(all_labels)), sample_size)
        selected_labels = [all_labels[i] for i in selected_indices]
        
        # Create sample directory
        sample_name = f"sample_{sample_size}_{split}"
        if seed is not None:
            sample_name += f"_seed{seed}"
        
        sample_dir = self.samples_dir / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected images and create labels file
        images_dir = sample_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        copied_labels = []
        successful_copies = 0
        
        for idx, label_line in enumerate(selected_labels):
            # Parse label line (format: image_name.png	latex_expression)
            parts = label_line.strip().split('\t')
            if len(parts) < 2:
                continue
                
            image_name = parts[0]
            latex_expr = parts[1]
            
            # Source and destination paths
            src_image = self.original_dir / split / image_name
            dst_image = images_dir / image_name
            
            if src_image.exists():
                shutil.copy2(src_image, dst_image)
                copied_labels.append(f"{image_name}\t{latex_expr}\n")
                successful_copies += 1
            else:
                print(f"⚠️  Image not found: {image_name}")
        
        # Save labels file
        labels_path = sample_dir / "labels.txt"
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.writelines(copied_labels)
        
        # Create sample info
        sample_info = {
            "sample_name": sample_name,
            "original_split": split,
            "requested_size": sample_size,
            "actual_size": successful_copies,
            "seed": seed,
            "created": datetime.now().isoformat(),
            "images_dir": str(images_dir.relative_to(self.dataset_dir)),
            "labels_file": str(labels_path.relative_to(self.dataset_dir))
        }
        
        with open(sample_dir / "sample_info.json", 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        print(f"✅ Sample created successfully!")
        print(f"   📁 Location: {sample_dir}")
        print(f"   📊 Images copied: {successful_copies}/{sample_size}")
        print(f"   🏷️  Labels file: {labels_path}")
        
        return True
    
    def list_samples(self):
        """List all created samples."""
        if not self.samples_dir.exists():
            print("📂 No samples directory found.")
            return
        
        samples = list(self.samples_dir.glob("sample_*"))
        if not samples:
            print("📂 No samples found.")
            return
        
        print("📊 Available Samples:")
        print("=" * 50)
        
        for sample_dir in sorted(samples):
            info_file = sample_dir / "sample_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                print(f"📁 {sample_dir.name}")
                print(f"   Size: {info['actual_size']} images")
                print(f"   Split: {info['original_split']}")
                if info.get('seed'):
                    print(f"   Seed: {info['seed']}")
                print(f"   Created: {info['created'][:19]}")
                print()
    
    def show_dataset_info(self):
        """Show dataset information."""
        print("📊 HME100K Dataset Information")
        print("=" * 50)
        
        if not self.original_dir.exists():
            print("❌ Dataset not downloaded. Run --download first.")
            return
        
        # Count files
        train_dir = self.original_dir / "train"
        test_dir = self.original_dir / "test"
        
        train_count = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
        test_count = len(list(test_dir.glob("*.png"))) if test_dir.exists() else 0
        
        print(f"📁 Dataset Location: {self.original_dir}")
        print(f"🖼️  Training Images: {train_count:,}")
        print(f"🖼️  Testing Images: {test_count:,}")
        print(f"📊 Total Images: {train_count + test_count:,}")
        
        # Show sample labels
        train_labels = self.original_dir / "train_labels.txt"
        if train_labels.exists():
            with open(train_labels, 'r', encoding='utf-8') as f:
                sample_lines = [f.readline().strip() for _ in range(3)]
            
            print(f"\n📝 Sample Labels (first 3):")
            for i, line in enumerate(sample_lines, 1):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        print(f"   {i}. {parts[0]} → {parts[1][:50]}...")
        
        # List available samples
        print()
        self.list_samples()

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="🚀 HME100K Dataset Setup - Download and create random samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_hme100k.py --download                    # Download full dataset
  python setup_hme100k.py --sample 100                  # Random sample of 100 images
  python setup_hme100k.py --sample 50 --seed 42         # Reproducible sample
  python setup_hme100k.py --sample 75 --split test      # Sample from test set
  python setup_hme100k.py --info                        # Show dataset information
  python setup_hme100k.py --list-samples                # List created samples
        """
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Show download instructions for HME100K dataset')
    parser.add_argument('--create-demo', action='store_true',
                       help='Create demo dataset for testing (10 K-12 math samples)')
    parser.add_argument('--sample', type=int,
                       help='Create random sample of specified size')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible sampling')
    parser.add_argument('--split', choices=['train', 'test'], default='train',
                       help='Dataset split to sample from (default: train)')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset information')
    parser.add_argument('--list-samples', action='store_true',
                       help='List all created samples')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if dataset exists')
    
    args = parser.parse_args()
    
    setup = HME100KSetup()
    
    if args.download:
        setup.download_dataset()
    
    elif args.create_demo:
        if args.force and setup.original_dir.exists():
            print("🗑️  Removing existing dataset...")
            shutil.rmtree(setup.original_dir)
        
        success = setup.create_demo_dataset()
        if not success:
            sys.exit(1)
    
    elif args.sample:
        success = setup.create_random_sample(args.sample, args.seed, args.split)
        if not success:
            sys.exit(1)
    
    elif args.info:
        setup.show_dataset_info()
    
    elif args.list_samples:
        setup.list_samples()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
