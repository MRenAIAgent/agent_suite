#!/usr/bin/env python3
"""
🏛️ CROHME 2023 DATASET DOWNLOADER
================================================================================
Download and setup the CROHME 2023 dataset for handwritten mathematical 
expression recognition benchmarking.

CROHME (Competition on Recognition of Online Handwritten Mathematical Expressions)
- Standard competition benchmark dataset
- Online handwritten expressions (stroke-based)
- InkML format with LaTeX ground truth
- ~3,905 new handwritten equations (2023)
================================================================================
"""

import os
import sys
import json
import argparse
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import random

class CROHME2023Downloader:
    """Download and manage CROHME 2023 dataset."""
    
    def __init__(self, base_dir: str = "real_datasets"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "CROHME2023"
        self.samples_dir = self.dataset_dir / "samples"
        
        # Create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(exist_ok=True)
        
        # Known CROHME dataset sources
        self.download_sources = {
            "official": {
                "name": "Official CROHME Website",
                "url": "http://www.isical.ac.in/~crohme/",
                "description": "Official competition website",
                "method": "manual"
            },
            "zenodo_2019": {
                "name": "CROHME 2019 (Zenodo)",
                "url": "https://zenodo.org/record/3648396",
                "description": "CROHME 2019 dataset on Zenodo",
                "method": "zenodo"
            },
            "github_tamer": {
                "name": "TAMER Project (GitHub)",
                "url": "https://github.com/qingzhenduyu/TAMER",
                "description": "TAMER project with CROHME processing tools",
                "method": "github"
            },
            "alternative": {
                "name": "Alternative Research Sources",
                "url": "Multiple research repositories",
                "description": "Various academic sources",
                "method": "research"
            }
        }
    
    def show_download_instructions(self):
        """📝 Show comprehensive download instructions for CROHME 2023."""
        print(f"\n🏛️ CROHME 2023 DATASET DOWNLOAD GUIDE")
        print("=" * 80)
        print("📊 **Dataset Info**:")
        print("   • Size: ~10,000 handwritten math expressions")
        print("   • Type: Online handwritten (stroke-based)")
        print("   • Format: InkML + LaTeX ground truth")
        print("   • Difficulty: Elementary to High School")
        print("   • Use: Standard competition benchmark")
        
        print(f"\n📥 **Download Options**:")
        
        for source_id, info in self.download_sources.items():
            print(f"\n🔗 **{info['name']}**")
            print(f"   URL: {info['url']}")
            print(f"   Description: {info['description']}")
            print(f"   Method: {info['method']}")
        
        print(f"\n⚠️  **Important Notes**:")
        print("   • CROHME 2023 may require registration for download")
        print("   • Some sources may have older versions (2019, 2016)")
        print("   • InkML format requires conversion to images")
        print("   • Ground truth is provided in LaTeX format")
        
        print(f"\n🛠️  **Manual Download Steps**:")
        print("1. **Visit Official Website**:")
        print("   http://www.isical.ac.in/~crohme/")
        print("   Look for CROHME 2023 competition data")
        print("")
        print("2. **Alternative: Use CROHME 2019 (Available)**:")
        print("   wget https://zenodo.org/record/3648396/files/CROHME2019_dataset.zip")
        print("   unzip CROHME2019_dataset.zip")
        print("")
        print("3. **Extract to correct location**:")
        print(f"   mv CROHME* {self.dataset_dir}/")
        print("")
        print("4. **Verify installation**:")
        print("   python scripts/download_crohme2023_dataset.py --verify")
    
    def create_demo_dataset(self):
        """🎨 Create a demo CROHME-style dataset using available data."""
        print(f"\n🎨 CREATING CROHME DEMO DATASET")
        print("=" * 60)
        print("Since CROHME 2023 requires manual download, creating demo dataset...")
        
        # Create demo directory
        demo_dir = self.samples_dir / "crohme_demo" / "current"
        images_dir = demo_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create demo InkML-style expressions (simulated)
        demo_expressions = [
            {
                "filename": "crohme_demo_001.png",
                "latex": "x + 2 = 5",
                "description": "Simple linear equation"
            },
            {
                "filename": "crohme_demo_002.png", 
                "latex": "\\frac{x}{2} + 3 = 7",
                "description": "Fraction equation"
            },
            {
                "filename": "crohme_demo_003.png",
                "latex": "x^2 - 4 = 0",
                "description": "Quadratic equation"
            },
            {
                "filename": "crohme_demo_004.png",
                "latex": "\\sqrt{x + 1} = 3",
                "description": "Square root equation"
            },
            {
                "filename": "crohme_demo_005.png",
                "latex": "\\sin(x) = \\frac{1}{2}",
                "description": "Trigonometric equation"
            },
            {
                "filename": "crohme_demo_006.png",
                "latex": "\\int_{0}^{1} x dx = \\frac{1}{2}",
                "description": "Integral expression"
            }
        ]
        
        # Try to copy from existing test images if available
        test_images_dir = Path("test_images")
        if test_images_dir.exists():
            available_images = list(test_images_dir.glob("*.png"))
            if available_images:
                print(f"📸 Using {len(available_images)} existing test images...")
                
                for i, expr in enumerate(demo_expressions):
                    if i < len(available_images):
                        src_image = available_images[i]
                        dst_image = images_dir / expr["filename"]
                        
                        # Copy image
                        import shutil
                        shutil.copy2(src_image, dst_image)
                        print(f"   Copied: {src_image.name} -> {expr['filename']}")
        
        # Create labels file
        labels_file = demo_dir / "labels.txt"
        labels_data = []
        for expr in demo_expressions:
            labels_data.append(f"{expr['filename']}\t{expr['latex']}")
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(labels_data))
        
        # Create metadata
        metadata = {
            "name": "CROHME Demo Dataset",
            "size": len(demo_expressions),
            "source": "Demo simulation of CROHME format",
            "content_type": "online_handwritten_math",
            "ground_truth": "latex_expressions",
            "description": "Demo dataset simulating CROHME competition format",
            "focus": "competition_benchmark",
            "format": "Image + LaTeX (simulating InkML conversion)",
            "note": "Replace with real CROHME 2023 data when available"
        }
        
        metadata_file = demo_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlink to current
        current_link = self.samples_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to("crohme_demo")
        
        print(f"✅ Demo dataset created successfully!")
        print(f"📁 Images: {images_dir}")
        print(f"📝 Labels: {labels_file}")
        print(f"📋 Metadata: {metadata_file}")
        print(f"📊 Total samples: {len(demo_expressions)}")
        print(f"🔗 Current sample link: {current_link} -> crohme_demo")
        
        print(f"\n⚠️  **Next Steps**:")
        print("1. Download real CROHME 2023 dataset manually")
        print("2. Replace demo data with real CROHME data")
        print("3. Use InkML to image conversion tools")
        print("4. Run benchmark with real data")
        
        return True
    
    def verify_dataset(self) -> bool:
        """✅ Verify CROHME dataset installation."""
        print(f"\n✅ VERIFYING CROHME 2023 DATASET")
        print("=" * 50)
        
        if not self.dataset_dir.exists():
            print(f"❌ Dataset directory not found: {self.dataset_dir}")
            return False
        
        # Check for common CROHME files
        expected_files = [
            "*.inkml",  # InkML files
            "*.xml",    # XML files
            "*.lst",    # List files
            "*.txt"     # Text files
        ]
        
        found_files = []
        for pattern in expected_files:
            files = list(self.dataset_dir.rglob(pattern))
            if files:
                found_files.extend([f.name for f in files[:3]])  # Show first 3
        
        if found_files:
            print(f"✅ Dataset found: {self.dataset_dir}")
            print(f"📁 Sample files: {', '.join(found_files)}")
            if len(found_files) >= 3:
                print("   ... and more files")
        else:
            print(f"⚠️  Dataset directory exists but no CROHME files found")
            print(f"📁 Directory contents:")
            for item in self.dataset_dir.iterdir():
                print(f"   - {item.name}")
        
        # Check samples
        if (self.samples_dir / "current").exists():
            current_sample = self.samples_dir / "current"
            metadata_file = current_sample / "current" / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                print(f"\n🎯 Current sample: {metadata['name']}")
                print(f"📊 Size: {metadata['size']} samples")
                print(f"📝 Type: {metadata['content_type']}")
            else:
                print(f"\n🎯 Current sample: Available but no metadata")
        
        return True
    
    def list_samples(self):
        """📋 List all available CROHME samples."""
        print(f"\n📊 CROHME 2023 SAMPLES")
        print("=" * 40)
        
        if not self.samples_dir.exists():
            print("❌ No samples directory found")
            return
        
        samples = [d for d in self.samples_dir.iterdir() if d.is_dir() and d.name != "current"]
        
        if not samples:
            print("❌ No samples found")
            print("💡 Create demo dataset: --create-demo")
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
        description="🏛️ CROHME 2023 Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download instructions
  python scripts/download_crohme2023_dataset.py --info

  # Create demo dataset (for testing)
  python scripts/download_crohme2023_dataset.py --create-demo

  # Verify dataset installation
  python scripts/download_crohme2023_dataset.py --verify

  # List available samples
  python scripts/download_crohme2023_dataset.py --list-samples

Dataset Info:
  • CROHME: Competition on Recognition of Online Handwritten Mathematical Expressions
  • Format: InkML (stroke data) + LaTeX ground truth
  • Size: ~10,000 handwritten expressions
  • Standard benchmark for handwritten math recognition
  • Used in academic competitions since 2011

Note: CROHME 2023 requires manual download from official sources.
      This script provides demo data and integration tools.
        """
    )
    
    parser.add_argument("--info", action="store_true",
                       help="Show download instructions and information")
    parser.add_argument("--create-demo", action="store_true",
                       help="Create demo CROHME dataset for testing")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset installation")
    parser.add_argument("--list-samples", action="store_true",
                       help="List all available samples")
    parser.add_argument("--base-dir", type=str, default="real_datasets",
                       help="Base directory for datasets (default: real_datasets)")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    downloader = CROHME2023Downloader(args.base_dir)
    
    if args.info:
        downloader.show_download_instructions()
    
    if args.create_demo:
        downloader.create_demo_dataset()
    
    if args.verify:
        downloader.verify_dataset()
    
    if args.list_samples:
        downloader.list_samples()

if __name__ == "__main__":
    main()

