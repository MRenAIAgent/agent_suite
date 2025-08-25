#!/usr/bin/env python3
"""
🎲 HME100K Sample Manager for Benchmarking
==========================================

Manages HME100K samples for benchmark testing with easy random sampling.
Creates symlinks to 'current' sample for seamless benchmark integration.

Usage:
    python hme100k_manager.py --new-sample 100            # Create new random sample of 100
    python hme100k_manager.py --new-sample 50 --seed 42   # Reproducible sample
    python hme100k_manager.py --use-sample sample_100_train_seed42  # Use existing sample
    python hme100k_manager.py --list                      # List available samples
    python hme100k_manager.py --current                   # Show current sample info
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

class HME100KManager:
    """Manage HME100K samples for benchmarking."""
    
    def __init__(self):
        self.dataset_dir = Path("./real_datasets/HME100K")
        self.samples_dir = self.dataset_dir / "samples"
        self.current_link = self.samples_dir / "current"
        
    def create_new_sample(self, size: int, seed: int = None, split: str = "train") -> bool:
        """Create a new random sample and set it as current."""
        print(f"🎲 Creating new HME100K sample ({size} images)...")
        
        # Run the setup script to create sample
        cmd = ["python", "setup_hme100k.py", "--sample", str(size), "--split", split]
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to create sample: {result.stderr}")
            return False
        
        print(result.stdout)
        
        # Find the newly created sample
        sample_name = f"sample_{size}_{split}"
        if seed is not None:
            sample_name += f"_seed{seed}"
        
        sample_dir = self.samples_dir / sample_name
        if not sample_dir.exists():
            print(f"❌ Sample directory not found: {sample_dir}")
            return False
        
        # Set as current
        return self.set_current_sample(sample_name)
    
    def set_current_sample(self, sample_name: str) -> bool:
        """Set a specific sample as the current one for benchmarking."""
        sample_dir = self.samples_dir / sample_name
        
        if not sample_dir.exists():
            print(f"❌ Sample not found: {sample_name}")
            return False
        
        # Remove existing current link
        if self.current_link.exists() or self.current_link.is_symlink():
            self.current_link.unlink()
        
        # Create new symlink
        try:
            self.current_link.symlink_to(sample_dir.name)
            print(f"✅ Set current sample to: {sample_name}")
            
            # Show sample info
            self.show_current_info()
            return True
            
        except Exception as e:
            print(f"❌ Failed to set current sample: {e}")
            return False
    
    def list_samples(self):
        """List all available samples."""
        if not self.samples_dir.exists():
            print("📂 No samples directory found. Run setup first.")
            return
        
        samples = list(self.samples_dir.glob("sample_*"))
        if not samples:
            print("📂 No samples found. Create one with --new-sample.")
            return
        
        print("📊 Available HME100K Samples:")
        print("=" * 60)
        
        current_sample = None
        if self.current_link.exists():
            current_sample = self.current_link.resolve().name
        
        for sample_dir in sorted(samples):
            info_file = sample_dir / "sample_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                marker = "👉 CURRENT" if sample_dir.name == current_sample else "  "
                print(f"{marker} 📁 {sample_dir.name}")
                print(f"     Size: {info['actual_size']} images")
                print(f"     Split: {info['original_split']}")
                if info.get('seed'):
                    print(f"     Seed: {info['seed']} (reproducible)")
                print(f"     Created: {info['created'][:19]}")
                print()
        
        if current_sample:
            print(f"🎯 Current sample for benchmarking: {current_sample}")
        else:
            print("⚠️  No current sample set. Use --use-sample to set one.")
    
    def show_current_info(self):
        """Show information about the current sample."""
        if not self.current_link.exists():
            print("⚠️  No current sample set.")
            print("   Use --new-sample or --use-sample to set one.")
            return
        
        current_dir = self.current_link.resolve()
        info_file = current_dir / "sample_info.json"
        
        if not info_file.exists():
            print(f"❌ Sample info not found for: {current_dir.name}")
            return
        
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print("🎯 Current HME100K Sample:")
        print("=" * 40)
        print(f"📁 Name: {current_dir.name}")
        print(f"📊 Size: {info['actual_size']} images")
        print(f"📚 Split: {info['original_split']}")
        if info.get('seed'):
            print(f"🌱 Seed: {info['seed']} (reproducible)")
        print(f"📅 Created: {info['created'][:19]}")
        print(f"📂 Path: {current_dir}")
        
        # Show benchmark command
        print(f"\n🚀 Ready for benchmarking!")
        print(f"   python run_benchmark.py -d hme100k_sample -s gpt-5")
        print(f"   python run_benchmark.py -d hme100k_sample -s all --comparison-table")
    
    def quick_setup(self, size: int = 100):
        """Quick setup: download dataset if needed, create sample, set as current."""
        print(f"🚀 Quick HME100K Setup ({size} samples)...")
        print("=" * 50)
        
        # Check if dataset exists
        if not (self.dataset_dir / "original").exists():
            print("📥 Dataset not found. Downloading...")
            result = subprocess.run([
                "python", "setup_hme100k.py", "--download"
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                return False
            
            print("✅ Dataset downloaded!")
        
        # Create sample
        return self.create_new_sample(size)

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="🎲 HME100K Sample Manager - Manage samples for benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hme100k_manager.py --new-sample 100            # New random sample of 100
  python hme100k_manager.py --new-sample 50 --seed 42   # Reproducible sample  
  python hme100k_manager.py --use-sample sample_100_train  # Use existing sample
  python hme100k_manager.py --list                      # List all samples
  python hme100k_manager.py --current                   # Show current sample
  python hme100k_manager.py --quick-setup 75            # Download + create + set sample
        """
    )
    
    parser.add_argument('--new-sample', type=int,
                       help='Create new random sample of specified size')
    parser.add_argument('--seed', type=int,
                       help='Random seed for reproducible sampling')
    parser.add_argument('--split', choices=['train', 'test'], default='train',
                       help='Dataset split to sample from (default: train)')
    parser.add_argument('--use-sample', type=str,
                       help='Set existing sample as current for benchmarking')
    parser.add_argument('--list', action='store_true',
                       help='List all available samples')
    parser.add_argument('--current', action='store_true',
                       help='Show current sample information')
    parser.add_argument('--quick-setup', type=int,
                       help='Quick setup: download dataset + create sample + set as current')
    
    args = parser.parse_args()
    
    manager = HME100KManager()
    
    if args.new_sample:
        success = manager.create_new_sample(args.new_sample, args.seed, args.split)
        if not success:
            sys.exit(1)
    
    elif args.use_sample:
        success = manager.set_current_sample(args.use_sample)
        if not success:
            sys.exit(1)
    
    elif args.list:
        manager.list_samples()
    
    elif args.current:
        manager.show_current_info()
    
    elif args.quick_setup:
        success = manager.quick_setup(args.quick_setup)
        if not success:
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
