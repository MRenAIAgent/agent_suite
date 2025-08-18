#!/usr/bin/env python3
"""
Sample PGDP5K Dataset Generator

Creates realistic sample data in PGDP5K format for immediate testing
while waiting for the real dataset download.
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import math

class SamplePGDP5KGenerator:
    def __init__(self, base_dir: str = "../real_datasets/PGDP5K"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.annotations_dir = self.base_dir / "annotations"
        
    def setup_directories(self):
        """Create directory structure"""
        for dir_path in [self.base_dir, self.images_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_path}")
    
    def generate_sample_dataset(self, num_samples: int = 50):
        """Generate sample dataset in PGDP5K format"""
        print(f"🎨 Generating {num_samples} sample PGDP5K-format images...")
        
        train_data = {}
        val_data = {}
        test_data = {}
        
        for i in range(num_samples):
            sample_id = f"sample_{i:03d}"
            
            # Generate image and annotation
            image_data, annotation_data = self.create_sample_geometry(sample_id, i)
            
            # Save image
            image_path = self.images_dir / f"{sample_id}.jpg"
            image_data.save(image_path, "JPEG")
            
            # Distribute into splits
            if i < num_samples * 0.7:  # 70% train
                train_data[sample_id] = annotation_data
            elif i < num_samples * 0.85:  # 15% val
                val_data[sample_id] = annotation_data
            else:  # 15% test
                test_data[sample_id] = annotation_data
        
        # Save annotation files
        self.save_split_data(train_data, "train")
        self.save_split_data(val_data, "val") 
        self.save_split_data(test_data, "test")
        
        print(f"✅ Generated {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        return True
    
    def create_sample_geometry(self, sample_id: str, index: int):
        """Create a realistic geometry diagram with annotations"""
        # Create image
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Initialize annotation data
        annotation = {
            "file_name": f"{sample_id}.jpg",
            "width": width,
            "height": height,
            "geos": {"points": [], "lines": [], "circles": []},
            "symbols": [],
            "relations": {"geo2geo": [], "sym2geo": []}
        }
        
        # Generate geometry based on pattern
        geometry_type = index % 5
        
        if geometry_type == 0:  # Triangle
            self.create_triangle(draw, annotation, width, height, font)
        elif geometry_type == 1:  # Circle with center
            self.create_circle_diagram(draw, annotation, width, height, font)
        elif geometry_type == 2:  # Parallel lines
            self.create_parallel_lines(draw, annotation, width, height, font)
        elif geometry_type == 3:  # Rectangle
            self.create_rectangle(draw, annotation, width, height, font)
        else:  # Mixed geometry
            self.create_mixed_geometry(draw, annotation, width, height, font)
        
        return image, annotation
    
    def create_triangle(self, draw, annotation, width, height, font):
        """Create triangle with labeled vertices"""
        # Triangle vertices
        vertices = [
            (width//3, height*2//3),      # A
            (width*2//3, height*2//3),    # B  
            (width//2, height//3)         # C
        ]
        
        # Draw triangle
        draw.polygon(vertices, outline='black', width=2)
        
        # Add points to annotation
        for i, (x, y) in enumerate(vertices):
            point_id = i + 1
            annotation["geos"]["points"].append([point_id, x, y])
            
            # Draw vertex labels
            label = chr(65 + i)  # A, B, C
            draw.text((x-10, y-20), label, fill='black', font=font)
            
            # Add symbol annotation
            annotation["symbols"].append([
                len(annotation["symbols"]) + 1,
                "vertex_label", 
                "text",
                label,
                [x-10, y-20, x+10, y]
            ])
        
        # Add triangle sides as lines
        for i in range(3):
            start = vertices[i]
            end = vertices[(i+1) % 3]
            line_id = i + 1
            annotation["geos"]["lines"].append([
                line_id, start[0], start[1], end[0], end[1]
            ])
    
    def create_circle_diagram(self, draw, annotation, width, height, font):
        """Create circle with center point"""
        center_x, center_y = width//2, height//2
        radius = min(width, height) // 4
        
        # Draw circle
        bbox = [center_x-radius, center_y-radius, center_x+radius, center_y+radius]
        draw.ellipse(bbox, outline='black', width=2)
        
        # Draw center point
        draw.ellipse([center_x-3, center_y-3, center_x+3, center_y+3], fill='black')
        draw.text((center_x+10, center_y-10), "O", fill='black', font=font)
        
        # Add to annotations
        annotation["geos"]["circles"].append([1, center_x, center_y, radius])
        annotation["geos"]["points"].append([1, center_x, center_y])
        annotation["symbols"].append([1, "center_point", "text", "O", [center_x+10, center_y-10, center_x+20, center_y]])
    
    def create_parallel_lines(self, draw, annotation, width, height, font):
        """Create two parallel lines"""
        y1 = height // 3
        y2 = height * 2 // 3
        x_start = width // 4
        x_end = width * 3 // 4
        
        # Draw parallel lines
        draw.line([(x_start, y1), (x_end, y1)], fill='black', width=2)
        draw.line([(x_start, y2), (x_end, y2)], fill='black', width=2)
        
        # Add parallel symbols
        mid_x = width // 2
        draw.text((mid_x-20, y1-25), "l₁", fill='black', font=font)
        draw.text((mid_x-20, y2+10), "l₂", fill='black', font=font)
        draw.text((mid_x+30, (y1+y2)//2), "l₁ ∥ l₂", fill='black', font=font)
        
        # Add to annotations
        annotation["geos"]["lines"].append([1, x_start, y1, x_end, y1])
        annotation["geos"]["lines"].append([2, x_start, y2, x_end, y2])
        annotation["symbols"].extend([
            [1, "line_label", "text", "l₁", [mid_x-20, y1-25, mid_x, y1-10]],
            [2, "line_label", "text", "l₂", [mid_x-20, y2+10, mid_x, y2+25]],
            [3, "parallel_symbol", "text", "l₁ ∥ l₂", [mid_x+30, (y1+y2)//2-10, mid_x+80, (y1+y2)//2+10]]
        ])
    
    def create_rectangle(self, draw, annotation, width, height, font):
        """Create rectangle with labeled vertices"""
        x1, y1 = width//4, height//3
        x2, y2 = width*3//4, height*2//3
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='black', width=2)
        
        # Vertices
        vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        labels = ['A', 'B', 'C', 'D']
        
        # Add vertices and labels
        for i, ((x, y), label) in enumerate(zip(vertices, labels)):
            annotation["geos"]["points"].append([i+1, x, y])
            draw.text((x-10, y-20), label, fill='black', font=font)
            annotation["symbols"].append([
                i+1, "vertex_label", "text", label, [x-10, y-20, x+10, y]
            ])
        
        # Add rectangle sides
        for i in range(4):
            start = vertices[i]
            end = vertices[(i+1) % 4]
            annotation["geos"]["lines"].append([
                i+1, start[0], start[1], end[0], end[1]
            ])
    
    def create_mixed_geometry(self, draw, annotation, width, height, font):
        """Create mixed geometry with multiple elements"""
        # Circle
        cx, cy = width//3, height//2
        radius = 50
        draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], outline='blue', width=2)
        annotation["geos"]["circles"].append([1, cx, cy, radius])
        
        # Triangle
        vertices = [
            (width*2//3, height//3),
            (width*5//6, height*2//3), 
            (width//2, height*2//3)
        ]
        draw.polygon(vertices, outline='red', width=2)
        
        for i, (x, y) in enumerate(vertices):
            annotation["geos"]["points"].append([i+1, x, y])
        
        for i in range(3):
            start = vertices[i]
            end = vertices[(i+1) % 3]
            annotation["geos"]["lines"].append([
                i+1, start[0], start[1], end[0], end[1]
            ])
        
        # Labels
        draw.text((cx-10, cy-radius-20), "Circle", fill='blue', font=font)
        draw.text((width*2//3-20, height//3-20), "Triangle", fill='red', font=font)
        
        annotation["symbols"].extend([
            [1, "shape_label", "text", "Circle", [cx-10, cy-radius-20, cx+30, cy-radius]],
            [2, "shape_label", "text", "Triangle", [width*2//3-20, height//3-20, width*2//3+40, height//3]]
        ])
    
    def save_split_data(self, data, split_name):
        """Save split data to JSON file"""
        output_file = self.base_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {len(data)} samples to {split_name}.json")

def main():
    """Generate sample PGDP5K dataset"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample PGDP5K dataset")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of sample images to generate")
    parser.add_argument("--output-dir", default="../real_datasets/PGDP5K",
                       help="Output directory for sample dataset")
    
    args = parser.parse_args()
    
    print("🎨 Creating Sample PGDP5K Dataset")
    print("=" * 40)
    print(f"📊 Samples: {args.samples}")
    print(f"📁 Output: {args.output_dir}")
    print()
    
    generator = SamplePGDP5KGenerator(args.output_dir)
    generator.setup_directories()
    
    if generator.generate_sample_dataset(args.samples):
        print()
        print("✅ Sample PGDP5K dataset created successfully!")
        print("🚀 You can now run:")
        print("   python download_pgdp5k.py --process")
        print("   python real_geometry_benchmark_runner.py --simulate --samples 20")
    else:
        print("❌ Failed to create sample dataset")

if __name__ == "__main__":
    main() 