#!/usr/bin/env python3
"""
MLHME-38K Dataset Setup Script

Downloads and sets up the MLHME-38K dataset which contains multi-line handwritten
mathematical expressions - the closest available dataset to complete homework problems.
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from typing import List, Dict, Any

class MLHME38KSetup:
    def __init__(self):
        self.base_path = Path("./real_datasets/MLHME38K")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def show_dataset_info(self):
        """Show information about the MLHME-38K dataset."""
        print("\n" + "="*80)
        print("📚 MLHME-38K DATASET - Multi-Line Handwritten Math Expressions")
        print("="*80)
        print("🎯 **Closest Available Dataset to Complete Homework Problems**")
        print()
        print("📊 Dataset Statistics:")
        print("   • Total Images: 38,000 handwritten mathematical expressions")
        print("   • Multi-line: 9,931 images (complex, homework-like problems)")
        print("   • Single-line: 28,069 images (simple expressions)")
        print("   • Real handwritten: From actual users in real scenarios")
        print("   • Ground Truth: LaTeX labels for each expression")
        print()
        print("🎓 Content Type:")
        print("   • Multi-step mathematical problems")
        print("   • Complex equations spanning multiple lines")
        print("   • Real-world handwriting variations")
        print("   • Various mathematical topics (algebra, calculus, etc.)")
        print()
        print("💡 Why This is Best for Homework-like Problems:")
        print("   ✅ Multi-line expressions (like homework solutions)")
        print("   ✅ Real handwritten content from users")
        print("   ✅ Complex mathematical content")
        print("   ✅ Variations in color, blur, backgrounds")
        print("   ✅ LaTeX ground truth for accurate evaluation")
        print()
        print("🌐 Official Source: https://ai.100tal.com/icdar")
        print("📄 Paper: Multi-Line Handwritten Mathematical Expression Recognition")
        print()
        print("⚠️  Note: This dataset requires manual download and registration")
        print("   due to academic licensing requirements.")
        print()
        
    def show_download_instructions(self):
        """Show detailed download instructions."""
        print("\n" + "="*80)
        print("📥 MLHME-38K DOWNLOAD INSTRUCTIONS")
        print("="*80)
        print()
        print("🔗 **Step 1: Visit Official Website**")
        print("   URL: https://ai.100tal.com/icdar")
        print("   This is the official 100tal AI research page")
        print()
        print("📝 **Step 2: Register for Academic Access**")
        print("   • Fill out the research application form")
        print("   • Provide institutional affiliation")
        print("   • Describe your research purpose")
        print("   • Wait for approval (usually 1-3 days)")
        print()
        print("💾 **Step 3: Download Dataset**")
        print("   After approval, download:")
        print("   • MLHME-38K_images.zip (image files)")
        print("   • MLHME-38K_labels.txt (LaTeX ground truth)")
        print("   • Dataset documentation")
        print()
        print("📁 **Step 4: Extract to Correct Location**")
        print(f"   Extract to: {self.base_path}/")
        print("   Expected structure:")
        print(f"   {self.base_path}/")
        print("   ├── images/")
        print("   │   ├── multiline/     # 9,931 multi-line expressions")
        print("   │   └── singleline/    # 28,069 single-line expressions")
        print("   ├── labels.txt         # LaTeX ground truth")
        print("   └── metadata.json      # Dataset information")
        print()
        print("🚀 **Step 5: Verify Installation**")
        print(f"   python {sys.argv[0]} --verify")
        print()
        
    def verify_dataset(self) -> bool:
        """Verify that MLHME-38K dataset is properly installed."""
        print("\n🔍 Verifying MLHME-38K Dataset...")
        print("="*50)
        
        # Check directory structure
        images_dir = self.base_path / "images"
        multiline_dir = images_dir / "multiline"
        singleline_dir = images_dir / "singleline"
        labels_file = self.base_path / "labels.txt"
        
        missing_items = []
        
        if not images_dir.exists():
            missing_items.append(f"❌ Missing: {images_dir}")
        if not multiline_dir.exists():
            missing_items.append(f"❌ Missing: {multiline_dir}")
        if not singleline_dir.exists():
            missing_items.append(f"❌ Missing: {singleline_dir}")
        if not labels_file.exists():
            missing_items.append(f"❌ Missing: {labels_file}")
            
        if missing_items:
            print("❌ Dataset verification failed!")
            for item in missing_items:
                print(f"   {item}")
            print()
            print("💡 Please download the MLHME-38K dataset first:")
            print(f"   python {sys.argv[0]} --download")
            return False
            
        # Count images
        try:
            multiline_count = len(list(multiline_dir.glob("*.jpg"))) + len(list(multiline_dir.glob("*.png")))
            singleline_count = len(list(singleline_dir.glob("*.jpg"))) + len(list(singleline_dir.glob("*.png")))
            
            print(f"✅ Multi-line images: {multiline_count:,}")
            print(f"✅ Single-line images: {singleline_count:,}")
            print(f"✅ Total images: {multiline_count + singleline_count:,}")
            
            # Check labels
            with open(labels_file, 'r', encoding='utf-8') as f:
                label_count = len(f.readlines())
                
            print(f"✅ Labels: {label_count:,}")
            
            if multiline_count > 5000 and singleline_count > 20000:
                print("\n🎉 MLHME-38K dataset verified successfully!")
                print("   This dataset contains multi-line handwritten math expressions")
                print("   that are closest to real homework problems.")
                return True
            else:
                print(f"\n⚠️  Warning: Dataset seems smaller than expected.")
                print("   Expected: ~9,931 multi-line + ~28,069 single-line images")
                print("   You might have a partial download.")
                return False
                
        except Exception as e:
            print(f"❌ Error verifying dataset: {e}")
            return False
    
    def create_homework_sample(self, sample_size: int = 100, focus_multiline: bool = True):
        """Create a sample focused on homework-like problems."""
        print(f"\n🎲 Creating homework-like sample ({sample_size} images)...")
        print("="*60)
        
        if not self.verify_dataset():
            print("❌ Cannot create sample - dataset not available.")
            return False
            
        try:
            # Focus on multi-line expressions for homework-like content
            if focus_multiline:
                source_dir = self.base_path / "images" / "multiline"
                sample_name = f"homework_multiline_{sample_size}"
                print("🎯 Focusing on multi-line expressions (most homework-like)")
            else:
                source_dir = self.base_path / "images"
                sample_name = f"homework_mixed_{sample_size}"
                print("🎯 Using mixed single-line and multi-line expressions")
            
            # Create sample directory
            sample_dir = self.base_path / "samples" / sample_name
            sample_images_dir = sample_dir / "images"
            
            if sample_dir.exists():
                import shutil
                shutil.rmtree(sample_dir)
                
            sample_images_dir.mkdir(parents=True, exist_ok=True)
            
            # Get available images
            if focus_multiline:
                available_images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
            else:
                multiline_images = list((source_dir / "multiline").glob("*.jpg")) + list((source_dir / "multiline").glob("*.png"))
                singleline_images = list((source_dir / "singleline").glob("*.jpg")) + list((source_dir / "singleline").glob("*.png"))
                # Prefer multiline images for homework-like content
                available_images = multiline_images + singleline_images[:sample_size//4]
            
            if len(available_images) < sample_size:
                print(f"⚠️  Warning: Only {len(available_images)} images available, using all.")
                sample_size = len(available_images)
            
            # Random sample
            import random
            selected_images = random.sample(available_images, sample_size)
            
            # Copy images
            copied_count = 0
            for img_path in selected_images:
                dest_path = sample_images_dir / img_path.name
                import shutil
                shutil.copy2(img_path, dest_path)
                copied_count += 1
            
            # Create metadata
            metadata = {
                "name": sample_name,
                "size": copied_count,
                "focus": "multiline" if focus_multiline else "mixed",
                "source": "MLHME-38K Dataset",
                "content_type": "homework_like_problems",
                "description": "Multi-line handwritten mathematical expressions resembling homework problems"
            }
            
            metadata_file = sample_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Sample created successfully!")
            print(f"   📁 Location: {sample_dir}")
            print(f"   📊 Images: {copied_count}")
            print(f"   🎯 Focus: {'Multi-line (homework-like)' if focus_multiline else 'Mixed content'}")
            print()
            print("🚀 Ready for benchmarking!")
            print("   # Add to benchmark configuration and run:")
            print("   python run_benchmark.py -d mlhme38k_sample -s gpt-5")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating sample: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Setup MLHME-38K dataset - closest available to real homework problems"
    )
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show information about MLHME-38K dataset"
    )
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Show download instructions for MLHME-38K dataset"
    )
    parser.add_argument(
        "--verify", 
        action="store_true", 
        help="Verify that MLHME-38K dataset is properly installed"
    )
    parser.add_argument(
        "--create-sample", 
        type=int, 
        metavar="SIZE",
        help="Create a homework-like sample of SIZE images (focuses on multi-line expressions)"
    )
    parser.add_argument(
        "--mixed", 
        action="store_true", 
        help="Include both single-line and multi-line expressions in sample"
    )
    
    args = parser.parse_args()
    
    setup = MLHME38KSetup()
    
    if args.info:
        setup.show_dataset_info()
    elif args.download:
        setup.show_download_instructions()
    elif args.verify:
        setup.verify_dataset()
    elif args.create_sample:
        setup.create_homework_sample(args.create_sample, focus_multiline=not args.mixed)
    else:
        # Default: show dataset info
        setup.show_dataset_info()

if __name__ == "__main__":
    main()
