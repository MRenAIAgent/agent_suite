#!/usr/bin/env python3
"""
Results Verification Script

Shows images, annotations, and benchmark results side by side for manual verification.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import glob

def load_test_annotations():
    """Load test annotations"""
    annotations_file = "../real_datasets/PGDP5K/processed/test_benchmark.json"
    
    try:
        with open(annotations_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return []

def load_latest_results():
    """Load the most recent benchmark results"""
    result_files = glob.glob("real_benchmark_results_*.json")
    
    if not result_files:
        print("❌ No benchmark results found. Run benchmark first.")
        return None
    
    # Get the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📊 Loading results from: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None

def display_image_info(image_path, annotation, result=None):
    """Display information about an image"""
    print(f"\n🖼️  IMAGE: {image_path}")
    print("=" * 60)
    
    # Check if image exists
    full_path = f"../real_datasets/PGDP5K/{image_path}"
    if not os.path.exists(full_path):
        print(f"❌ Image not found: {full_path}")
        return
    
    # Image info
    try:
        with Image.open(full_path) as img:
            print(f"📏 Dimensions: {img.size[0]}x{img.size[1]}")
            print(f"📁 Size: {os.path.getsize(full_path)} bytes")
    except Exception as e:
        print(f"❌ Error reading image: {e}")
    
    # Expected annotations
    print(f"\n📋 EXPECTED ANNOTATIONS:")
    print(f"   Shapes: {annotation.get('expected_shapes', [])}")
    print(f"   Text: '{annotation.get('expected_text', '')}'")
    print(f"   Coordinates: {len(annotation.get('expected_coordinates', []))} points")
    print(f"   Difficulty: {annotation.get('difficulty', 'unknown')}")
    print(f"   Geometry Type: {annotation.get('geometry_type', 'unknown')}")
    
    # Geometric primitives details
    primitives = annotation.get('geometric_primitives', {})
    print(f"\n🔍 GEOMETRIC PRIMITIVES:")
    print(f"   Points: {len(primitives.get('points', []))}")
    print(f"   Lines: {len(primitives.get('lines', []))}")
    print(f"   Circles: {len(primitives.get('circles', []))}")
    
    # Show detailed primitives
    if primitives.get('points'):
        print(f"   Point coordinates: {primitives['points']}")
    if primitives.get('lines'):
        print(f"   Line coordinates: {primitives['lines']}")
    if primitives.get('circles'):
        print(f"   Circle data: {primitives['circles']}")
    
    # Symbols
    symbols = annotation.get('symbols', [])
    if symbols:
        print(f"\n🏷️  SYMBOLS ({len(symbols)}):")
        for i, symbol in enumerate(symbols):
            if len(symbol) >= 4:
                print(f"   {i+1}. Type: {symbol[1]}, Text: '{symbol[3]}'")
    
    # Benchmark results
    if result:
        print(f"\n🎯 BENCHMARK RESULTS:")
        print(f"   Success: {'✅' if result.get('success') else '❌'}")
        print(f"   Extracted Text: '{result.get('extracted_text', '')}'")
        print(f"   Detected Shapes: {result.get('detected_shapes', [])}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
        print(f"   Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        
        # Comparison
        expected_shapes = set(annotation.get('expected_shapes', []))
        detected_shapes = set(result.get('detected_shapes', []))
        expected_text = annotation.get('expected_text', '').strip()
        extracted_text = result.get('extracted_text', '').strip()
        
        print(f"\n📊 COMPARISON:")
        print(f"   Shape Match: {'✅' if expected_shapes == detected_shapes else '❌'}")
        print(f"     Expected: {sorted(expected_shapes)}")
        print(f"     Detected: {sorted(detected_shapes)}")
        
        print(f"   Text Match: {'✅' if expected_text.lower() == extracted_text.lower() else '❌'}")
        print(f"     Expected: '{expected_text}'")
        print(f"     Extracted: '{extracted_text}'")

def create_annotated_image(image_path, annotation, output_path=None):
    """Create an annotated version of the image showing expected elements"""
    full_path = f"../real_datasets/PGDP5K/{image_path}"
    
    try:
        # Load image
        img = Image.open(full_path).copy()
        draw = ImageDraw.Draw(img)
        
        # Try to load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Draw geometric primitives
        primitives = annotation.get('geometric_primitives', {})
        
        # Draw points
        for point in primitives.get('points', []):
            if len(point) >= 3:
                x, y = point[1], point[2]
                draw.ellipse([x-5, y-5, x+5, y+5], fill='red', outline='darkred', width=2)
        
        # Draw lines
        for line in primitives.get('lines', []):
            if len(line) >= 5:
                x1, y1, x2, y2 = line[1], line[2], line[3], line[4]
                draw.line([(x1, y1), (x2, y2)], fill='blue', width=3)
        
        # Draw circles
        for circle in primitives.get('circles', []):
            if len(circle) >= 4:
                cx, cy, radius = circle[1], circle[2], circle[3]
                bbox = [cx-radius, cy-radius, cx+radius, cy+radius]
                draw.ellipse(bbox, outline='green', width=3)
                # Mark center
                draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill='green')
        
        # Draw symbols
        for symbol in annotation.get('symbols', []):
            if len(symbol) >= 5:
                text = str(symbol[3])
                bbox = symbol[4]
                if len(bbox) >= 2:
                    x, y = bbox[0], bbox[1]
                    draw.text((x, y), text, fill='purple', font=font)
        
        # Add title
        title = f"ID: {annotation.get('id', 'unknown')} | Type: {annotation.get('geometry_type', 'unknown')}"
        draw.text((10, 10), title, fill='black', font=font)
        
        # Save or show
        if output_path:
            img.save(output_path)
            print(f"💾 Annotated image saved: {output_path}")
        else:
            # Try to display (works in some environments)
            try:
                img.show()
            except:
                print("💡 Cannot display image directly. Save with --output flag.")
        
        return img
        
    except Exception as e:
        print(f"❌ Error creating annotated image: {e}")
        return None

def main():
    """Main verification function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify benchmark results")
    parser.add_argument("--show-images", action="store_true",
                       help="Create annotated images showing expected elements")
    parser.add_argument("--output-dir", default="verification_images",
                       help="Directory to save annotated images")
    parser.add_argument("--sample-id", help="Show specific sample ID only")
    
    args = parser.parse_args()
    
    print("🔍 BENCHMARK RESULTS VERIFICATION")
    print("=" * 50)
    
    # Load data
    annotations = load_test_annotations()
    results_data = load_latest_results()
    
    if not annotations:
        print("❌ No annotations found")
        return
    
    # Extract results by ID
    results_by_id = {}
    if results_data:
        strategy_results = list(results_data.values())[0]  # Get first strategy
        for result in strategy_results.get('results', []):
            results_by_id[result['test_case_id']] = result
    
    # Create output directory if needed
    if args.show_images:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each annotation
    for annotation in annotations:
        sample_id = annotation.get('id')
        
        # Filter by sample ID if specified
        if args.sample_id and sample_id != args.sample_id:
            continue
        
        # Get corresponding result
        result = results_by_id.get(sample_id)
        
        # Display info
        display_image_info(annotation['image_path'], annotation, result)
        
        # Create annotated image if requested
        if args.show_images:
            output_path = os.path.join(args.output_dir, f"{sample_id}_annotated.jpg")
            create_annotated_image(annotation['image_path'], annotation, output_path)
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("=" * 30)
    print(f"Total test samples: {len(annotations)}")
    print(f"Benchmark results available: {len(results_by_id)}")
    
    if results_data:
        strategy_results = list(results_data.values())[0]
        metrics = strategy_results.get('metrics', {})
        print(f"Overall accuracy: {metrics.get('overall_accuracy', 0):.1f}%")
        print(f"Success rate: {metrics.get('success_rate', 0):.1f}%")
    
    if args.show_images:
        print(f"Annotated images saved to: {args.output_dir}/")

if __name__ == "__main__":
    main() 