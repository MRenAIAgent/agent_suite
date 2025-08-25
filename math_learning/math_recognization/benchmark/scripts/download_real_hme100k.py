#!/usr/bin/env python3
"""
Real HME100K Dataset Download and Setup Script

This script helps you download and set up the real HME100K dataset with actual handwritten
mathematical expressions from tens of thousands of writers.
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple

class RealHME100KDownloader:
    def __init__(self):
        self.base_path = Path("./real_datasets/HME100K")
        self.original_path = self.base_path / "original"
        self.samples_path = self.base_path / "samples"
        
    def show_download_instructions(self):
        """Show detailed download instructions for the real HME100K dataset."""
        print("\n" + "="*80)
        print("📥 REAL HME100K DATASET DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print("❗ The real HME100K dataset contains 99,109 handwritten math expressions")
        print("   from tens of thousands of writers who wrote on paper and uploaded them.")
        print()
        print("🌐 Official Download Site: https://ai.100tal.com/dataset")
        print()
        print("📋 Steps to download:")
        print("   1. Visit: https://ai.100tal.com/dataset")
        print("   2. Register/login to access the dataset")
        print("   3. Download the HME100K dataset files")
        print("   4. Extract to:", str(self.original_path))
        print()
        print("📁 Expected structure after extraction:")
        print(f"   {self.original_path}/")
        print("   ├── train/")
        print("   │   ├── train_images/     # 74,502 real handwritten images")
        print("   │   └── train_labels.txt  # LaTeX ground truth labels")
        print("   └── test/")
        print("       ├── test_images/      # 24,607 real handwritten images")
        print("       └── test_labels.txt   # LaTeX ground truth labels")
        print()
        print("📊 Dataset Statistics:")
        print("   • Total Images: 99,109 handwritten math expressions")
        print("   • Training Set: 74,502 images")
        print("   • Test Set: 24,607 images")
        print("   • Writers: Tens of thousands of different people")
        print("   • Content: Real K-12 mathematical expressions")
        print("   • Format: Handwritten on paper, scanned/photographed")
        print()
        print("🔄 After download, run:")
        print(f"   python {sys.argv[0]} --verify")
        print(f"   python {sys.argv[0]} --create-sample 200")
        print()
        
    def verify_dataset(self) -> bool:
        """Verify that the real HME100K dataset is properly downloaded."""
        print("\n🔍 Verifying HME100K Dataset...")
        print("="*50)
        
        # Check directory structure
        train_images = self.original_path / "train" / "train_images"
        train_labels = self.original_path / "train" / "train_labels.txt"
        test_images = self.original_path / "test" / "test_images"
        test_labels = self.original_path / "test" / "test_labels.txt"
        
        missing_items = []
        
        if not train_images.exists():
            missing_items.append(f"❌ Missing: {train_images}")
        if not train_labels.exists():
            missing_items.append(f"❌ Missing: {train_labels}")
        if not test_images.exists():
            missing_items.append(f"❌ Missing: {test_images}")
        if not test_labels.exists():
            missing_items.append(f"❌ Missing: {test_labels}")
            
        if missing_items:
            print("❌ Dataset verification failed!")
            for item in missing_items:
                print(f"   {item}")
            print()
            print("💡 Please download the real HME100K dataset first:")
            print("   python download_real_hme100k.py --download")
            return False
            
        # Count images
        try:
            train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
            test_count = len(list(test_images.glob("*.jpg"))) + len(list(test_images.glob("*.png")))
            
            print(f"✅ Training images: {train_count:,}")
            print(f"✅ Test images: {test_count:,}")
            print(f"✅ Total images: {train_count + test_count:,}")
            
            # Check labels
            with open(train_labels, 'r', encoding='utf-8') as f:
                train_label_count = len(f.readlines())
            with open(test_labels, 'r', encoding='utf-8') as f:
                test_label_count = len(f.readlines())
                
            print(f"✅ Training labels: {train_label_count:,}")
            print(f"✅ Test labels: {test_label_count:,}")
            
            if train_count > 50000 and test_count > 20000:
                print("\n🎉 Real HME100K dataset verified successfully!")
                print("   This appears to be the full dataset with real handwritten math.")
                return True
            else:
                print("\n⚠️  Warning: Dataset seems smaller than expected.")
                print("   Expected: ~74K training + ~24K test images")
                print("   You might have a demo/subset version.")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying dataset: {e}")
            return False
    
    def load_labels(self, split: str = "train") -> List[Tuple[str, str]]:
        """Load image-label pairs from the dataset."""
        labels_file = self.original_path / split / f"{split}_labels.txt"
        
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
            
        pairs = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    image_name, latex_label = line.split('\t', 1)
                    pairs.append((image_name, latex_label))
                    
        return pairs
    
    def create_handwritten_sample(self, sample_size: int = 200, seed: int = None, split: str = "train"):
        """Create a sample of real handwritten math expressions."""
        print(f"\n🎲 Creating sample of {sample_size} real handwritten math expressions...")
        print("="*70)
        
        if seed is not None:
            random.seed(seed)
            print(f"🌱 Using random seed: {seed}")
        
        # Verify dataset first
        if not self.verify_dataset():
            print("❌ Cannot create sample - dataset not properly downloaded.")
            return False
            
        try:
            # Load all image-label pairs
            pairs = self.load_labels(split)
            print(f"📊 Total {split} samples available: {len(pairs):,}")
            
            if len(pairs) < sample_size:
                print(f"⚠️  Warning: Requested {sample_size} samples, but only {len(pairs)} available.")
                sample_size = len(pairs)
                
            # Random sample
            selected_pairs = random.sample(pairs, sample_size)
            
            # Create sample directory
            sample_name = f"real_handwritten_{sample_size}_{split}_seed{seed if seed else 'random'}"
            sample_dir = self.samples_path / sample_name
            sample_images_dir = sample_dir / "images"
            
            # Clean up existing sample
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
                
            sample_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images and create labels
            source_images_dir = self.original_path / split / f"{split}_images"
            copied_count = 0
            labels_content = []
            
            print(f"📁 Creating sample directory: {sample_dir}")
            print(f"🖼️  Copying {sample_size} handwritten math images...")
            
            for image_name, latex_label in selected_pairs:
                source_path = source_images_dir / image_name
                dest_path = sample_images_dir / image_name
                
                if source_path.exists():
                    shutil.copy2(source_path, dest_path)
                    labels_content.append(f"{image_name}\t{latex_label}")
                    copied_count += 1
                else:
                    print(f"⚠️  Warning: Image not found: {source_path}")
            
            # Save labels file
            labels_file = sample_dir / "labels.txt"
            with open(labels_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(labels_content))
                
            # Create metadata
            metadata = {
                "name": sample_name,
                "size": copied_count,
                "split": split,
                "seed": seed,
                "created": str(Path().absolute()),
                "source": "Real HME100K Dataset",
                "content_type": "handwritten_math_expressions",
                "ground_truth": "latex_labels"
            }
            
            metadata_file = sample_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
                
            # Update current sample link
            current_link = self.samples_path / "current"
            if current_link.exists():
                current_link.unlink()
            current_link.symlink_to(sample_name)
            
            print(f"✅ Sample created successfully!")
            print(f"   📁 Location: {sample_dir}")
            print(f"   📊 Images copied: {copied_count}/{sample_size}")
            print(f"   🏷️  Labels file: {labels_file}")
            print(f"   📋 Metadata: {metadata_file}")
            print()
            print("🎯 Sample Statistics:")
            
            # Show sample of labels
            sample_labels = labels_content[:5]
            for i, label_line in enumerate(sample_labels, 1):
                image_name, latex = label_line.split('\t', 1)
                print(f"   {i}. {image_name}: {latex}")
            if len(labels_content) > 5:
                print(f"   ... and {len(labels_content) - 5} more")
                
            print()
            print("🚀 Ready for benchmarking!")
            print("   python run_benchmark.py -d hme100k_sample -s gpt-5")
            print("   python run_benchmark.py -d hme100k_sample -s all --comparison-table")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample: {e}")
            return False
    
    def list_samples(self):
        """List all available samples."""
        print("\n📋 Available HME100K Samples:")
        print("="*50)
        
        if not self.samples_path.exists():
            print("❌ No samples directory found.")
            return
            
        samples = [d for d in self.samples_path.iterdir() if d.is_dir()]
        
        if not samples:
            print("❌ No samples found.")
            print("💡 Create a sample first:")
            print("   python download_real_hme100k.py --create-sample 200")
            return
            
        current_sample = None
        current_link = self.samples_path / "current"
        if current_link.exists():
            current_sample = current_link.resolve().name
            
        for sample_dir in sorted(samples):
            metadata_file = sample_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    marker = "🎯" if sample_dir.name == current_sample else "📁"
                    print(f"{marker} {sample_dir.name}")
                    print(f"   Size: {metadata.get('size', 'unknown')} images")
                    print(f"   Split: {metadata.get('split', 'unknown')}")
                    print(f"   Seed: {metadata.get('seed', 'unknown')}")
                    print(f"   Type: {metadata.get('content_type', 'unknown')}")
                    
                except Exception as e:
                    print(f"📁 {sample_dir.name} (metadata error: {e})")
            else:
                marker = "🎯" if sample_dir.name == current_sample else "📁"
                print(f"{marker} {sample_dir.name} (no metadata)")

def main():
    parser = argparse.ArgumentParser(
        description="Download and setup real HME100K dataset with handwritten math expressions"
    )
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Show download instructions for real HME100K dataset"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify that the real HME100K dataset is properly downloaded"
    )
    parser.add_argument(
        "--create-sample", 
        type=int, 
        metavar="SIZE",
        help="Create a sample of SIZE real handwritten math expressions (e.g., 200)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--split", 
        choices=["train", "test"], 
        default="train",
        help="Dataset split to sample from (default: train)"
    )
    parser.add_argument(
        "--list", 
        action="store_true", 
        help="List all available samples"
    )
    
    args = parser.parse_args()
    
    downloader = RealHME100KDownloader()
    
    if args.download:
        downloader.show_download_instructions()
    elif args.verify:
        downloader.verify_dataset()
    elif args.create_sample:
        downloader.create_handwritten_sample(args.create_sample, args.seed, args.split)
    elif args.list:
        downloader.list_samples()
    else:
        # Default: show download instructions
        downloader.show_download_instructions()

if __name__ == "__main__":
    main()
