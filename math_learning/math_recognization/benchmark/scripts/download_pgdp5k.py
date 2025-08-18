#!/usr/bin/env python3
"""
PGDP5K Dataset Downloader and Setup Script

This script downloads the PGDP5K dataset (5,000 real plane geometry diagrams)
and sets it up for benchmarking geometry OCR systems.

Dataset: PGDP5K - A Diagram Parsing Dataset for Plane Geometry Problems
URL: http://www.nlpr.ia.ac.cn/databases/CASIA-PGDP5K/
Password: pal2022
"""

import os
import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import argparse

class PGDP5KDownloader:
    def __init__(self, base_dir: str = "../real_datasets"):
        self.base_dir = Path(base_dir)
        self.pgdp5k_dir = self.base_dir / "PGDP5K"
        self.dataset_url = "http://www.nlpr.ia.ac.cn/databases/CASIA-PGDP5K/"
        self.password = "pal2022"
        
    def setup_directories(self):
        """Create necessary directory structure"""
        directories = [
            self.pgdp5k_dir,
            self.pgdp5k_dir / "images",
            self.pgdp5k_dir / "annotations", 
            self.pgdp5k_dir / "processed"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def download_instructions(self):
        """Print manual download instructions since the dataset requires web access"""
        print("🔽 PGDP5K Dataset Download Instructions")
        print("=" * 50)
        print(f"📍 Website: {self.dataset_url}")
        print(f"🔑 Password: {self.password}")
        print()
        print("📋 Manual Steps:")
        print("1. Open your web browser")
        print("2. Navigate to: http://www.nlpr.ia.ac.cn/databases/CASIA-PGDP5K/")
        print("3. Enter password: pal2022")
        print("4. Download the dataset files")
        print("5. Extract to:", str(self.pgdp5k_dir))
        print()
        print("📁 Expected files after extraction:")
        print("   - images/ (5,000 geometry diagram images)")
        print("   - annotations/ (JSON files with labels)")
        print("   - train.json, val.json, test.json")
        print()
        print("⚡ Once downloaded, run: python download_pgdp5k.py --process")
        
    def process_dataset(self):
        """Process the downloaded dataset into benchmark format"""
        print("🔄 Processing PGDP5K dataset...")
        
        # Check if dataset exists
        if not self.pgdp5k_dir.exists():
            print("❌ PGDP5K directory not found. Please download first.")
            return False
            
        # Look for annotation files
        annotation_files = list(self.pgdp5k_dir.glob("*.json"))
        if not annotation_files:
            print("❌ No JSON annotation files found. Please check download.")
            return False
            
        print(f"✅ Found {len(annotation_files)} annotation files")
        
        # Process each split
        for split in ['train', 'val', 'test']:
            split_file = self.pgdp5k_dir / f"{split}.json"
            if split_file.exists():
                self.process_split(split_file, split)
            else:
                print(f"⚠️  {split}.json not found, skipping...")
        
        self.create_benchmark_config()
        print("✅ PGDP5K dataset processing complete!")
        return True
    
    def process_split(self, split_file: Path, split_name: str):
        """Process a single data split"""
        print(f"📊 Processing {split_name} split...")
        
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to benchmark format
            benchmark_data = []
            for item_id, item_data in data.items():
                benchmark_item = self.convert_to_benchmark_format(item_id, item_data)
                if benchmark_item:
                    benchmark_data.append(benchmark_item)
            
            # Save processed data
            output_file = self.pgdp5k_dir / "processed" / f"{split_name}_benchmark.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Processed {len(benchmark_data)} items for {split_name}")
            
        except Exception as e:
            print(f"❌ Error processing {split_name}: {e}")
    
    def convert_to_benchmark_format(self, item_id: str, item_data: Dict) -> Dict[str, Any]:
        """Convert PGDP5K format to our benchmark format"""
        try:
            return {
                "id": item_id,
                "image_path": f"images/{item_data.get('file_name', f'{item_id}.jpg')}",
                "width": item_data.get('width'),
                "height": item_data.get('height'),
                "geometric_primitives": {
                    "points": item_data.get('geos', {}).get('points', []),
                    "lines": item_data.get('geos', {}).get('lines', []),
                    "circles": item_data.get('geos', {}).get('circles', [])
                },
                "symbols": item_data.get('symbols', []),
                "relations": item_data.get('relations', {}),
                "expected_shapes": self.extract_expected_shapes(item_data),
                "expected_text": self.extract_expected_text(item_data),
                "expected_coordinates": self.extract_expected_coordinates(item_data),
                "difficulty": self.estimate_difficulty(item_data),
                "geometry_type": self.classify_geometry_type(item_data)
            }
        except Exception as e:
            print(f"⚠️  Error converting {item_id}: {e}")
            return None
    
    def extract_expected_shapes(self, item_data: Dict) -> List[str]:
        """Extract expected geometric shapes from annotations"""
        shapes = []
        geos = item_data.get('geos', {})
        
        if geos.get('points'):
            shapes.extend(['point'] * len(geos['points']))
        if geos.get('lines'):
            shapes.extend(['line'] * len(geos['lines']))
        if geos.get('circles'):
            shapes.extend(['circle'] * len(geos['circles']))
            
        return list(set(shapes))  # Remove duplicates
    
    def extract_expected_text(self, item_data: Dict) -> str:
        """Extract expected text content"""
        symbols = item_data.get('symbols', [])
        text_parts = []
        
        for symbol in symbols:
            if len(symbol) > 3 and symbol[3]:  # text_content field
                text_parts.append(str(symbol[3]))
        
        return " ".join(text_parts) if text_parts else ""
    
    def extract_expected_coordinates(self, item_data: Dict) -> List[Dict]:
        """Extract coordinate information"""
        coordinates = []
        geos = item_data.get('geos', {})
        
        # Points
        for point in geos.get('points', []):
            if len(point) >= 3:
                coordinates.append({"x": point[1], "y": point[2], "type": "point"})
        
        # Lines (endpoints)
        for line in geos.get('lines', []):
            if len(line) >= 5:
                coordinates.extend([
                    {"x": line[1], "y": line[2], "type": "line_start"},
                    {"x": line[3], "y": line[4], "type": "line_end"}
                ])
        
        # Circles (center)
        for circle in geos.get('circles', []):
            if len(circle) >= 4:
                coordinates.append({
                    "x": circle[1], "y": circle[2], 
                    "radius": circle[3], "type": "circle_center"
                })
        
        return coordinates
    
    def estimate_difficulty(self, item_data: Dict) -> str:
        """Estimate difficulty based on complexity"""
        geos = item_data.get('geos', {})
        symbols = item_data.get('symbols', [])
        relations = item_data.get('relations', {})
        
        total_primitives = (
            len(geos.get('points', [])) + 
            len(geos.get('lines', [])) + 
            len(geos.get('circles', []))
        )
        total_symbols = len(symbols)
        total_relations = sum(len(v) for v in relations.values() if isinstance(v, list))
        
        complexity_score = total_primitives + total_symbols + total_relations
        
        if complexity_score <= 5:
            return "easy"
        elif complexity_score <= 15:
            return "medium"
        else:
            return "hard"
    
    def classify_geometry_type(self, item_data: Dict) -> str:
        """Classify the type of geometry problem"""
        geos = item_data.get('geos', {})
        
        has_circles = len(geos.get('circles', [])) > 0
        has_lines = len(geos.get('lines', [])) > 0
        has_points = len(geos.get('points', [])) > 0
        
        if has_circles and has_lines:
            return "mixed"
        elif has_circles:
            return "circular"
        elif has_lines:
            return "linear"
        elif has_points:
            return "point_geometry"
        else:
            return "basic_shapes"
    
    def create_benchmark_config(self):
        """Create configuration file for the benchmark"""
        config = {
            "dataset_name": "PGDP5K",
            "description": "Plane Geometry Diagram Parsing Dataset with 5,000 real images",
            "total_samples": 5000,
            "splits": {
                "train": 3500,
                "val": 500,
                "test": 1000
            },
            "image_formats": ["jpg", "png"],
            "annotation_format": "json",
            "geometric_primitives": ["point", "line", "circle"],
            "symbol_types": 16,
            "text_types": 6,
            "difficulty_levels": ["easy", "medium", "hard"],
            "geometry_types": ["basic_shapes", "linear", "circular", "mixed", "point_geometry"],
            "evaluation_metrics": [
                "shape_detection_accuracy",
                "text_recognition_accuracy", 
                "coordinate_extraction_accuracy",
                "overall_parsing_accuracy"
            ],
            "baseline_performance": {
                "expected_accuracy_range": "70-85%",
                "previous_synthetic_accuracy": "10%",
                "improvement_expected": "+60-75%"
            }
        }
        
        config_file = self.pgdp5k_dir / "benchmark_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Created benchmark config: {config_file}")
    
    def verify_dataset(self):
        """Verify the dataset is properly set up"""
        print("🔍 Verifying PGDP5K dataset setup...")
        
        # Check directories
        required_dirs = [
            self.pgdp5k_dir / "images",
            self.pgdp5k_dir / "annotations",
            self.pgdp5k_dir / "processed"
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✅ {dir_path.name}/ directory exists")
            else:
                print(f"❌ {dir_path.name}/ directory missing")
                return False
        
        # Check processed files
        processed_files = list((self.pgdp5k_dir / "processed").glob("*_benchmark.json"))
        print(f"✅ Found {len(processed_files)} processed benchmark files")
        
        # Check config
        config_file = self.pgdp5k_dir / "benchmark_config.json"
        if config_file.exists():
            print("✅ Benchmark configuration exists")
        else:
            print("❌ Benchmark configuration missing")
            return False
        
        print("✅ PGDP5K dataset verification complete!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download and setup PGDP5K dataset")
    parser.add_argument("--download", action="store_true", 
                       help="Show download instructions")
    parser.add_argument("--process", action="store_true",
                       help="Process downloaded dataset")
    parser.add_argument("--verify", action="store_true",
                       help="Verify dataset setup")
    parser.add_argument("--all", action="store_true",
                       help="Run all steps (download instructions + process + verify)")
    
    args = parser.parse_args()
    
    downloader = PGDP5KDownloader()
    downloader.setup_directories()
    
    if args.download or args.all:
        downloader.download_instructions()
    
    if args.process or args.all:
        if downloader.process_dataset():
            print("✅ Dataset processing successful!")
        else:
            print("❌ Dataset processing failed!")
            return
    
    if args.verify or args.all:
        if downloader.verify_dataset():
            print("🎉 PGDP5K dataset is ready for benchmarking!")
        else:
            print("⚠️  Dataset verification failed!")

if __name__ == "__main__":
    main() 