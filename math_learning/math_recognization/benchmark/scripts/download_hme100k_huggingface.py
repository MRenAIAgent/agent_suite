#!/usr/bin/env python3
"""
HME100K Alternative Download Script - Hugging Face Dataset

Downloads the HME100k-400 dataset from Hugging Face which contains 400 real handwritten
mathematical expressions from the original HME100K dataset.
"""

import os
import sys
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from datasets import load_dataset
    from PIL import Image
    import requests
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

class HME100KHuggingFaceDownloader:
    def __init__(self):
        self.base_path = Path("./real_datasets/HME100K")
        self.hf_path = self.base_path / "huggingface"
        self.samples_path = self.base_path / "samples"
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        if not HUGGINGFACE_AVAILABLE:
            print("❌ Missing dependencies. Please install:")
            print("   pip install datasets pillow requests")
            print("   # or")
            print("   pip install huggingface_hub[datasets]")
            return False
        return True
        
    def show_alternative_info(self):
        """Show information about the alternative HME100K dataset."""
        print("\n" + "="*80)
        print("📥 ALTERNATIVE HME100K DATASET - HUGGING FACE")
        print("="*80)
        print("✅ Available: HME100k-400 dataset with 400 real handwritten math expressions")
        print("🌐 Source: https://huggingface.co/datasets/VLM-Perception/HME100k-400")
        print()
        print("📊 Dataset Information:")
        print("   • Total Images: 400 real handwritten math expressions")
        print("   • Content: Subset of original HME100K dataset")
        print("   • Format: Real handwritten mathematical expressions")
        print("   • Ground Truth: LaTeX labels included")
        print("   • License: Research/Educational use")
        print()
        print("🚀 Quick Setup:")
        print("   # Install dependencies")
        print("   pip install datasets pillow requests")
        print()
        print("   # Download dataset")
        print(f"   python {sys.argv[0]} --download")
        print()
        print("   # Create 200 sample")
        print(f"   python {sys.argv[0]} --create-sample 200")
        print()
        print("💡 Advantages:")
        print("   ✅ No registration required")
        print("   ✅ Smaller download size (~400 images vs 99K)")
        print("   ✅ Real handwritten math expressions")
        print("   ✅ LaTeX ground truth labels")
        print("   ✅ Ready for immediate use")
        print()
        
    def download_hf_dataset(self):
        """Download the HME100k-400 dataset from Hugging Face."""
        if not self.check_dependencies():
            return False
            
        print("\n📥 Downloading HME100k-400 from Hugging Face...")
        print("="*60)
        
        try:
            # Create directories
            self.hf_path.mkdir(parents=True, exist_ok=True)
            images_dir = self.hf_path / "images"
            images_dir.mkdir(exist_ok=True)
            
            print("🔄 Loading dataset from Hugging Face...")
            dataset = load_dataset("VLM-Perception/HME100k-400")
            
            # The dataset might have train/test splits or be a single split
            if 'train' in dataset:
                data = dataset['train']
            else:
                # If no splits, use the first available split
                split_name = list(dataset.keys())[0]
                data = dataset[split_name]
                print(f"📊 Using split: {split_name}")
            
            print(f"📊 Total samples in dataset: {len(data)}")
            
            # Download and save images with labels
            labels_content = []
            downloaded_count = 0
            
            print("🖼️  Downloading images and labels...")
            
            for i, sample in enumerate(data):
                try:
                    # Get image and label
                    image = sample['image']  # PIL Image
                    latex_label = sample.get('latex', sample.get('label', sample.get('text', f'sample_{i}')))
                    
                    # Save image
                    image_filename = f"hf_sample_{i:04d}.png"
                    image_path = images_dir / image_filename
                    
                    # Convert to RGB if needed and save
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path, 'PNG')
                    
                    # Add to labels
                    labels_content.append(f"{image_filename}\t{latex_label}")
                    downloaded_count += 1
                    
                    if (i + 1) % 50 == 0:
                        print(f"   Downloaded {i + 1}/{len(data)} images...")
                        
                except Exception as e:
                    print(f"⚠️  Error processing sample {i}: {e}")
                    continue
            
            # Save labels file
            labels_file = self.hf_path / "labels.txt"
            with open(labels_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(labels_content))
            
            # Create metadata
            metadata = {
                "source": "Hugging Face - VLM-Perception/HME100k-400",
                "total_samples": downloaded_count,
                "content_type": "handwritten_math_expressions",
                "ground_truth": "latex_labels",
                "download_date": str(Path().absolute()),
                "url": "https://huggingface.co/datasets/VLM-Perception/HME100k-400"
            }
            
            metadata_file = self.hf_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Download completed!")
            print(f"   📁 Location: {self.hf_path}")
            print(f"   📊 Images downloaded: {downloaded_count}")
            print(f"   🏷️  Labels file: {labels_file}")
            print(f"   📋 Metadata: {metadata_file}")
            print()
            print("🎯 Sample Labels:")
            for i, label_line in enumerate(labels_content[:5]):
                image_name, latex = label_line.split('\t', 1)
                print(f"   {i+1}. {image_name}: {latex}")
            if len(labels_content) > 5:
                print(f"   ... and {len(labels_content) - 5} more")
            
            print()
            print("🚀 Next steps:")
            print(f"   python {sys.argv[0]} --create-sample 200")
            print("   python run_benchmark.py -d hme100k_sample -s gpt-5")
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("\n💡 Troubleshooting:")
            print("   1. Check internet connection")
            print("   2. Install dependencies: pip install datasets pillow requests")
            print("   3. Try again in a few minutes")
            return False
    
    def verify_hf_dataset(self) -> bool:
        """Verify the Hugging Face dataset is downloaded."""
        print("\n🔍 Verifying Hugging Face HME100K Dataset...")
        print("="*50)
        
        images_dir = self.hf_path / "images"
        labels_file = self.hf_path / "labels.txt"
        metadata_file = self.hf_path / "metadata.json"
        
        if not images_dir.exists():
            print("❌ Images directory not found")
            return False
            
        if not labels_file.exists():
            print("❌ Labels file not found")
            return False
        
        try:
            # Count images
            image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            image_count = len(image_files)
            
            # Count labels
            with open(labels_file, 'r', encoding='utf-8') as f:
                label_count = len(f.readlines())
            
            print(f"✅ Images found: {image_count}")
            print(f"✅ Labels found: {label_count}")
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ Source: {metadata.get('source', 'Unknown')}")
                print(f"✅ Content: {metadata.get('content_type', 'Unknown')}")
            
            if image_count >= 300:  # Should have ~400 images
                print("\n🎉 Hugging Face HME100K dataset verified successfully!")
                print("   This dataset contains real handwritten math expressions.")
                return True
            else:
                print(f"\n⚠️  Warning: Expected ~400 images, found {image_count}")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying dataset: {e}")
            return False
    
    def create_sample_from_hf(self, sample_size: int = 200, seed: int = None):
        """Create a sample from the Hugging Face dataset."""
        print(f"\n🎲 Creating sample of {sample_size} handwritten math expressions...")
        print("="*70)
        
        if not self.verify_hf_dataset():
            print("❌ Cannot create sample - Hugging Face dataset not available.")
            print("💡 Download first:")
            print(f"   python {sys.argv[0]} --download")
            return False
        
        if seed is not None:
            random.seed(seed)
            print(f"🌱 Using random seed: {seed}")
        
        try:
            # Load labels
            labels_file = self.hf_path / "labels.txt"
            with open(labels_file, 'r', encoding='utf-8') as f:
                all_labels = [line.strip() for line in f if line.strip()]
            
            print(f"📊 Total samples available: {len(all_labels)}")
            
            if len(all_labels) < sample_size:
                print(f"⚠️  Warning: Requested {sample_size} samples, but only {len(all_labels)} available.")
                sample_size = len(all_labels)
            
            # Random sample
            selected_labels = random.sample(all_labels, sample_size)
            
            # Create sample directory
            sample_name = f"hf_handwritten_{sample_size}_seed{seed if seed else 'random'}"
            sample_dir = self.samples_path / sample_name
            sample_images_dir = sample_dir / "images"
            
            # Clean up existing sample
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            
            sample_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy selected images
            source_images_dir = self.hf_path / "images"
            copied_count = 0
            final_labels = []
            
            print(f"📁 Creating sample directory: {sample_dir}")
            print(f"🖼️  Copying {sample_size} handwritten math images...")
            
            for label_line in selected_labels:
                if '\t' in label_line:
                    image_name, latex_label = label_line.split('\t', 1)
                    source_path = source_images_dir / image_name
                    dest_path = sample_images_dir / image_name
                    
                    if source_path.exists():
                        shutil.copy2(source_path, dest_path)
                        final_labels.append(label_line)
                        copied_count += 1
                    else:
                        print(f"⚠️  Warning: Image not found: {source_path}")
            
            # Save labels file
            labels_file = sample_dir / "labels.txt"
            with open(labels_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(final_labels))
            
            # Create metadata
            metadata = {
                "name": sample_name,
                "size": copied_count,
                "source": "Hugging Face HME100k-400",
                "seed": seed,
                "created": str(Path().absolute()),
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
            print()
            print("🎯 Sample Statistics:")
            
            # Show sample of labels
            for i, label_line in enumerate(final_labels[:5], 1):
                if '\t' in label_line:
                    image_name, latex = label_line.split('\t', 1)
                    print(f"   {i}. {image_name}: {latex}")
            if len(final_labels) > 5:
                print(f"   ... and {len(final_labels) - 5} more")
            
            print()
            print("🚀 Ready for benchmarking!")
            print("   python run_benchmark.py -d hme100k_sample -s gpt-5")
            print("   python run_benchmark.py -d hme100k_sample -s all --comparison-table")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Download HME100K dataset from Hugging Face (400 handwritten math expressions)"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show information about the alternative HME100K dataset"
    )
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download HME100k-400 dataset from Hugging Face"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify the Hugging Face dataset is downloaded"
    )
    parser.add_argument(
        "--create-sample", 
        type=int, 
        metavar="SIZE",
        help="Create a sample of SIZE handwritten math expressions (e.g., 200)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    downloader = HME100KHuggingFaceDownloader()
    
    if args.info:
        downloader.show_alternative_info()
    elif args.download:
        downloader.download_hf_dataset()
    elif args.verify:
        downloader.verify_hf_dataset()
    elif args.create_sample:
        downloader.create_sample_from_hf(args.create_sample, args.seed)
    else:
        # Default: show info
        downloader.show_alternative_info()

if __name__ == "__main__":
    main()
