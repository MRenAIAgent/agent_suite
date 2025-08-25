#!/usr/bin/env python3
"""
🤗 MATHWRITING DATASET DOWNLOADER
================================================================================
Download and setup the MathWriting dataset from Hugging Face.
- 230,000 human-written handwritten math expressions
- 400,000 synthetic handwritten math expressions  
- Each sample paired with LaTeX ground truth
- Largest available online handwritten math dataset
================================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import random

try:
    from datasets import load_dataset
    from PIL import Image
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

class MathWritingDownloader:
    """Download and manage MathWriting dataset from Hugging Face."""
    
    def __init__(self, base_dir: str = "real_datasets"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "MathWriting"
        self.samples_dir = self.dataset_dir / "samples"
        
        # Create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
    
    def check_requirements(self) -> bool:
        """✅ Check if required packages are installed."""
        if not DATASETS_AVAILABLE:
            print("❌ Missing required packages. Install with:")
            print("pip install datasets pillow")
            return False
        return True
    
    def download_dataset(self, subset: str = "human", sample_size: Optional[int] = None):
        """📥 Download MathWriting dataset from Hugging Face."""
        if not self.check_requirements():
            return False
        
        print(f"\n🤗 DOWNLOADING MATHWRITING DATASET")
        print("=" * 60)
        print(f"📊 Subset: {subset}")
        print(f"📦 Sample Size: {sample_size or 'Full dataset'}")
        print(f"💾 Destination: {self.dataset_dir}")
        
        try:
            # Download dataset
            print("\n📥 Downloading from Hugging Face...")
            if subset == "human":
                dataset = load_dataset("deepcopy/MathWriting-human")
            else:
                dataset = load_dataset("deepcopy/MathWriting-synthetic")
            
            print(f"✅ Downloaded successfully!")
            print(f"📊 Dataset info: {dataset}")
            
            # Save dataset info
            self._save_dataset_info(dataset, subset)
            
            # Create samples if requested
            if sample_size:
                self._create_sample(dataset, sample_size, subset)
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def _save_dataset_info(self, dataset, subset: str):
        """💾 Save dataset metadata."""
        info = {
            "name": f"MathWriting-{subset}",
            "source": "Hugging Face",
            "url": f"https://huggingface.co/datasets/deepcopy/MathWriting-{subset}",
            "subset": subset,
            "splits": list(dataset.keys()),
            "total_samples": sum(len(dataset[split]) for split in dataset.keys()),
            "description": "Large-scale handwritten mathematical expression dataset",
            "format": "Image + LaTeX pairs",
            "download_date": str(Path().resolve())
        }
        
        info_file = self.dataset_dir / f"dataset_info_{subset}.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"📋 Dataset info saved to: {info_file}")
    
    def _create_sample(self, dataset, sample_size: int, subset: str):
        """📸 Create a sample subset for benchmarking."""
        print(f"\n📸 Creating sample of {sample_size} expressions...")
        
        # Create sample directory
        sample_name = f"{subset}_{sample_size}_samples"
        sample_dir = self.samples_dir / sample_name / "current"
        images_dir = sample_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Get train split (usually the largest)
        train_split = dataset['train'] if 'train' in dataset else list(dataset.values())[0]
        
        # Sample random indices
        total_samples = len(train_split)
        if sample_size > total_samples:
            print(f"⚠️  Requested {sample_size} samples, but only {total_samples} available")
            sample_size = total_samples
        
        indices = random.sample(range(total_samples), sample_size)
        
        # Create labels file
        labels_file = sample_dir / "labels.txt"
        labels_data = []
        
        print(f"💾 Extracting {sample_size} samples...")
        for i, idx in enumerate(indices):
            try:
                sample = train_split[idx]
                
                # Get image and LaTeX
                image = sample['image']
                latex = sample['latex']
                
                # Save image
                image_name = f"mathwriting_{subset}_{i+1:04d}.png"
                image_path = images_dir / image_name
                
                if hasattr(image, 'save'):
                    image.save(image_path)
                else:
                    # Convert to PIL Image if needed
                    Image.fromarray(image).save(image_path)
                
                # Add to labels
                labels_data.append(f"{image_name}\t{latex}")
                
                if (i + 1) % 50 == 0:
                    print(f"   Processed {i + 1}/{sample_size} samples...")
                
            except Exception as e:
                print(f"⚠️  Error processing sample {idx}: {e}")
                continue
        
        # Save labels
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels_data))
        
        # Save metadata
        metadata = {
            "name": f"MathWriting {subset.title()} Sample",
            "size": len(labels_data),
            "source": f"MathWriting-{subset}",
            "sample_method": "random",
            "content_type": f"{subset}_handwritten_math",
            "ground_truth": "latex_expressions",
            "description": f"Random sample of {len(labels_data)} expressions from MathWriting-{subset}",
            "focus": "handwritten_expressions"
        }
        
        metadata_file = sample_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Sample created successfully!")
        print(f"📁 Images: {images_dir}")
        print(f"📝 Labels: {labels_file}")
        print(f"📋 Metadata: {metadata_file}")
        print(f"📊 Total samples: {len(labels_data)}")
        
        # Create symlink to current
        current_link = self.samples_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(sample_name)
        
        print(f"🔗 Current sample link updated: {current_link} -> {sample_name}")
    
    def list_samples(self):
        """📋 List all available samples."""
        print(f"\n📊 MATHWRITING SAMPLES")
        print("=" * 50)
        
        if not self.samples_dir.exists():
            print("❌ No samples directory found")
            return
        
        samples = [d for d in self.samples_dir.iterdir() if d.is_dir() and d.name != "current"]
        
        if not samples:
            print("❌ No samples found")
            return
        
        for sample_dir in sorted(samples):
            metadata_file = sample_dir / "current" / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                print(f"\n🗂️  {sample_dir.name}")
                print(f"   Name: {metadata['name']}")
                print(f"   Size: {metadata['size']} samples")
                print(f"   Type: {metadata['content_type']}")
                print(f"   Description: {metadata['description']}")
            else:
                print(f"\n🗂️  {sample_dir.name}")
                print(f"   Status: ⚠️  Missing metadata")
        
        # Show current sample
        current_link = self.samples_dir / "current"
        if current_link.exists():
            target = current_link.resolve().name
            print(f"\n🔗 Current sample: {target}")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(
        description="🤗 MathWriting Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download human-written subset (230K samples)
  python scripts/download_mathwriting_dataset.py --download human

  # Download and create 1000-sample benchmark
  python scripts/download_mathwriting_dataset.py --download human --sample-size 1000

  # Download synthetic subset (400K samples) 
  python scripts/download_mathwriting_dataset.py --download synthetic --sample-size 500

  # List available samples
  python scripts/download_mathwriting_dataset.py --list-samples

  # Check requirements
  python scripts/download_mathwriting_dataset.py --check-requirements

Dataset Info:
  • Total: 630,000 handwritten math expressions
  • Human-written: 230,000 samples
  • Synthetic: 400,000 samples
  • Format: Image + LaTeX ground truth
  • Source: Hugging Face (deepcopy/MathWriting-*)
        """
    )
    
    parser.add_argument("--download", type=str, choices=["human", "synthetic"],
                       help="Download dataset subset (human or synthetic)")
    parser.add_argument("--sample-size", type=int, 
                       help="Create sample with specified number of expressions")
    parser.add_argument("--list-samples", action="store_true",
                       help="List all available samples")
    parser.add_argument("--check-requirements", action="store_true",
                       help="Check if required packages are installed")
    parser.add_argument("--base-dir", type=str, default="real_datasets",
                       help="Base directory for datasets (default: real_datasets)")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    downloader = MathWritingDownloader(args.base_dir)
    
    if args.check_requirements:
        if downloader.check_requirements():
            print("✅ All requirements satisfied")
        else:
            sys.exit(1)
    
    if args.download:
        success = downloader.download_dataset(args.download, args.sample_size)
        if not success:
            sys.exit(1)
    
    if args.list_samples:
        downloader.list_samples()

if __name__ == "__main__":
    main()
