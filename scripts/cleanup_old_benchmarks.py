#!/usr/bin/env python3
"""
Cleanup Old Benchmarks Script

This script helps clean up old benchmark files that have been moved to the new
benchmark directory structure. It creates a backup of these files before removing them.
"""

import os
import shutil
import argparse
from datetime import datetime


def backup_and_remove(path, backup_dir):
    """
    Backup a file or directory and then remove it.
    
    Args:
        path: Path to backup and remove
        backup_dir: Directory to store backups
    """
    if not os.path.exists(path):
        print(f"Warning: {path} does not exist, skipping")
        return
    
    # Create backup path
    rel_path = os.path.relpath(path, os.getcwd())
    backup_path = os.path.join(backup_dir, rel_path)
    
    # Ensure backup directory exists
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    
    # Backup file or directory
    if os.path.isdir(path):
        if os.path.exists(backup_path):
            shutil.rmtree(backup_path)
        shutil.copytree(path, backup_path)
        print(f"Backed up directory {path} to {backup_path}")
        shutil.rmtree(path)
        print(f"Removed directory {path}")
    else:
        shutil.copy2(path, backup_path)
        print(f"Backed up file {path} to {backup_path}")
        os.remove(path)
        print(f"Removed file {path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up old benchmark files")
    parser.add_argument("--backup-dir", default=f"backup/cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Directory to store backups")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Don't actually remove anything, just print what would be done")
    args = parser.parse_args()
    
    # Create backup directory
    os.makedirs(args.backup_dir, exist_ok=True)
    
    # List of paths to clean up
    paths_to_clean = [
        "agents/eval/taubench",
        "agents/eval/agent_benchmark.py",
        "agents/eval/README_BENCHMARK.md",
    ]
    
    for path in paths_to_clean:
        if args.dry_run:
            print(f"Would backup and remove {path}")
        else:
            backup_and_remove(path, args.backup_dir)
    
    print(f"Cleanup complete. Backups stored in {args.backup_dir}")


if __name__ == "__main__":
    main() 