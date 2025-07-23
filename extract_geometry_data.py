#!/usr/bin/env python3
"""
Extract geometry exercises and save them as data files
"""

import json
import os
from typing import List, Dict, Any

# Sample geometry exercises data based on educational standards
geometry_exercises_data = {
    "basic_geometry": {
        "name": "Basic Geometry Exercises (K-8)",
        "description": "Fundamental geometry concepts for elementary and middle school",
        "exercises": [
            {
                "id": "GEO-K5-001",
                "title": "Identify Basic Shapes",
                "problem": "Name these shapes: □ △ ○ ◇",
                "solution": "Square, Triangle, Circle, Diamond",
                "difficulty": 1,
                "estimated_time": 2,
                "grade_level": "K-5",
                "format_type": "multiple-choice",
                "concepts": ["basic_shapes", "shape_recognition"],
                "hints": ["Look at the number of sides", "Count the corners", "Some shapes are curved"]
            },
            {
                "id": "GEO-K5-002",
                "title": "Count Sides and Vertices",
                "problem": "How many sides and vertices does a pentagon have?",
                "solution": "5 sides and 5 vertices",
                "difficulty": 1,
                "estimated_time": 2,
                "grade_level": "K-5",
                "format_type": "free-response",
                "concepts": ["polygons", "vertices", "sides"],
                "hints": ["Count each straight edge", "Vertices are corners where sides meet"]
            },
            {
                "id": "GEO-K5-003",
                "title": "Identify Angle Types",
                "problem": "Classify this angle as acute, right, or obtuse: An angle measuring 45°",
                "solution": "Acute",
                "difficulty": 2,
                "estimated_time": 3,
                "grade_level": "K-5",
                "format_type": "multiple-choice",
                "concepts": ["angles", "angle_types"],
                "hints": ["Acute angles are less than 90°", "Right angles are exactly 90°", "Obtuse angles are greater than 90°"]
            },
            {
                "id": "GEO-68-001",
                "title": "Triangle Classification by Angles",
                "problem": "A triangle has angles of 60°, 60°, and 60°. What type of triangle is this?",
                "solution": "Equilateral triangle",
                "difficulty": 2,
                "estimated_time": 3,
                "grade_level": "6-8",
                "format_type": "multiple-choice",
                "concepts": ["triangles", "triangle_classification", "angles"],
                "hints": ["All angles are equal", "Equilateral triangles have all equal angles", "Each angle is 60°"]
            },
            {
                "id": "GEO-68-002",
                "title": "Triangle Classification by Sides",
                "problem": "A triangle has sides of length 3 cm, 4 cm, and 5 cm. What type of triangle is this?",
                "solution": "Scalene triangle",
                "difficulty": 2,
                "estimated_time": 3,
                "grade_level": "6-8",
                "format_type": "multiple-choice",
                "concepts": ["triangles", "triangle_classification", "sides"],
                "hints": ["All sides are different lengths", "Scalene = all sides different", "Check if any sides are equal"]
            },
            {
                "id": "GEO-68-003",
                "title": "Quadrilateral Properties",
                "problem": "What makes a parallelogram special compared to other quadrilaterals?",
                "solution": "Opposite sides are parallel and equal in length",
                "difficulty": 3,
                "estimated_time": 4,
                "grade_level": "6-8",
                "format_type": "free-response",
                "concepts": ["quadrilaterals", "parallelograms", "parallel_lines"],
                "hints": ["Look at opposite sides", "Parallel means never intersecting", "Equal length sides"]
            },
            {
                "id": "GEO-68-004",
                "title": "Rectangle Area",
                "problem": "Find the area of a rectangle with length 8 cm and width 5 cm.",
                "solution": "40 square cm",
                "difficulty": 2,
                "estimated_time": 3,
                "grade_level": "6-8",
                "format_type": "calculation",
                "concepts": ["area", "rectangles", "multiplication"],
                "hints": ["Area = length × width", "8 × 5 = ?", "Don't forget the units"]
            },
            {
                "id": "GEO-68-005",
                "title": "Triangle Area",
                "problem": "Find the area of a triangle with base 6 cm and height 4 cm.",
                "solution": "12 square cm",
                "difficulty": 3,
                "estimated_time": 4,
                "grade_level": "6-8",
                "format_type": "calculation",
                "concepts": ["area", "triangles", "base_height"],
                "hints": ["Area = ½ × base × height", "½ × 6 × 4 = ?", "Remember to multiply by ½"]
            },
            {
                "id": "GEO-68-006",
                "title": "Circle Area",
                "problem": "Find the area of a circle with radius 3 cm. (Use π ≈ 3.14)",
                "solution": "28.26 square cm",
                "difficulty": 3,
                "estimated_time": 5,
                "grade_level": "6-8",
                "format_type": "calculation",
                "concepts": ["area", "circles", "pi", "radius"],
                "hints": ["Area = π × r²", "r = 3, so r² = 9", "π × 9 ≈ 3.14 × 9"]
            },
            {
                "id": "GEO-68-007",
                "title": "Coordinate Plotting",
                "problem": "Plot the point (3, -2) on a coordinate plane. In which quadrant is it located?",
                "solution": "Quadrant IV (fourth quadrant)",
                "difficulty": 2,
                "estimated_time": 3,
                "grade_level": "6-8",
                "format_type": "coordinate-plotting",
                "concepts": ["coordinates", "quadrants", "plotting"],
                "hints": ["First number is x-coordinate", "Second number is y-coordinate", "Positive x, negative y = Quadrant IV"]
            },
            {
                "id": "GEO-68-008",
                "title": "Pythagorean Theorem Basic",
                "problem": "In a right triangle, if the legs are 3 and 4, what is the hypotenuse?",
                "solution": "5",
                "difficulty": 3,
                "estimated_time": 5,
                "grade_level": "6-8",
                "format_type": "calculation",
                "concepts": ["pythagorean_theorem", "right_triangles", "hypotenuse"],
                "hints": ["a² + b² = c²", "3² + 4² = ?", "√(9 + 16) = √25 = 5"]
            }
        ]
    },
    "advanced_geometry": {
        "name": "Advanced Geometry Exercises (9-12)",
        "description": "Advanced geometry concepts for high school",
        "exercises": [
            {
                "id": "GEO-912-001",
                "title": "Volume of a Cylinder",
                "problem": "Find the volume of a cylinder with radius 4 cm and height 10 cm. (Use π ≈ 3.14)",
                "solution": "502.4 cubic cm",
                "difficulty": 4,
                "estimated_time": 6,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["volume", "cylinders", "pi", "3d_geometry"],
                "hints": ["Volume = π × r² × h", "r = 4, h = 10", "π × 16 × 10"]
            },
            {
                "id": "GEO-912-002",
                "title": "Surface Area of a Sphere",
                "problem": "Find the surface area of a sphere with radius 5 cm. (Use π ≈ 3.14)",
                "solution": "314 square cm",
                "difficulty": 4,
                "estimated_time": 6,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["surface_area", "spheres", "pi", "3d_geometry"],
                "hints": ["Surface Area = 4π × r²", "r = 5, so r² = 25", "4 × 3.14 × 25"]
            },
            {
                "id": "GEO-912-003",
                "title": "Trigonometric Ratios",
                "problem": "In a right triangle, if the opposite side is 3 and the hypotenuse is 5, find sin θ.",
                "solution": "0.6 or 3/5",
                "difficulty": 4,
                "estimated_time": 5,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["trigonometry", "sine", "right_triangles"],
                "hints": ["sin θ = opposite/hypotenuse", "opposite = 3, hypotenuse = 5", "3/5 = 0.6"]
            },
            {
                "id": "GEO-912-004",
                "title": "Circle Theorems",
                "problem": "If an inscribed angle intercepts an arc of 80°, what is the measure of the inscribed angle?",
                "solution": "40°",
                "difficulty": 4,
                "estimated_time": 5,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["circles", "inscribed_angles", "arc_measures"],
                "hints": ["Inscribed angle = ½ × intercepted arc", "Arc = 80°", "80° ÷ 2 = 40°"]
            },
            {
                "id": "GEO-912-005",
                "title": "Polygon Interior Angles",
                "problem": "What is the sum of interior angles in a regular octagon?",
                "solution": "1080°",
                "difficulty": 4,
                "estimated_time": 5,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["polygons", "interior_angles", "octagons"],
                "hints": ["Formula: (n-2) × 180°", "Octagon has 8 sides", "(8-2) × 180° = 6 × 180°"]
            },
            {
                "id": "GEO-912-006",
                "title": "Law of Sines",
                "problem": "In triangle ABC, angle A = 30°, angle B = 45°, and side a = 10. Find side b.",
                "solution": "14.14",
                "difficulty": 5,
                "estimated_time": 8,
                "grade_level": "9-12",
                "format_type": "calculation",
                "concepts": ["trigonometry", "law_of_sines", "triangles"],
                "hints": ["a/sin(A) = b/sin(B)", "10/sin(30°) = b/sin(45°)", "sin(30°) = 0.5, sin(45°) ≈ 0.707"]
            }
        ]
    }
}

def save_geometry_exercises():
    """Save geometry exercises to JSON files"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save basic geometry exercises
    basic_file = 'data/geometry_exercises_basic.json'
    with open(basic_file, 'w', encoding='utf-8') as f:
        json.dump(geometry_exercises_data["basic_geometry"], f, indent=2, ensure_ascii=False)
    print(f"✅ Saved basic geometry exercises to {basic_file}")
    
    # Save advanced geometry exercises  
    advanced_file = 'data/geometry_exercises_advanced.json'
    with open(advanced_file, 'w', encoding='utf-8') as f:
        json.dump(geometry_exercises_data["advanced_geometry"], f, indent=2, ensure_ascii=False)
    print(f"✅ Saved advanced geometry exercises to {advanced_file}")
    
    # Save combined exercises
    combined_data = {
        "name": "Complete Geometry Exercise Bank",
        "description": "Comprehensive geometry exercises for K-12",
        "total_exercises": len(geometry_exercises_data["basic_geometry"]["exercises"]) + 
                          len(geometry_exercises_data["advanced_geometry"]["exercises"]),
        "categories": {
            "basic": geometry_exercises_data["basic_geometry"],
            "advanced": geometry_exercises_data["advanced_geometry"]
        }
    }
    
    combined_file = 'data/geometry_exercises_complete.json'
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved complete geometry exercises to {combined_file}")
    
    # Print summary
    basic_count = len(geometry_exercises_data["basic_geometry"]["exercises"])
    advanced_count = len(geometry_exercises_data["advanced_geometry"]["exercises"])
    total_count = basic_count + advanced_count
    
    print(f"\n📊 Summary:")
    print(f"   Basic exercises (K-8): {basic_count}")
    print(f"   Advanced exercises (9-12): {advanced_count}")
    print(f"   Total exercises: {total_count}")
    
    # Show concepts covered
    all_concepts = set()
    for category in geometry_exercises_data.values():
        for exercise in category["exercises"]:
            all_concepts.update(exercise["concepts"])
    
    print(f"   Concepts covered: {len(all_concepts)}")
    print(f"   Concept list: {sorted(all_concepts)}")
    
    return {
        'basic_file': basic_file,
        'advanced_file': advanced_file, 
        'combined_file': combined_file,
        'total_exercises': total_count
    }

if __name__ == "__main__":
    result = save_geometry_exercises()
    print(f"\n🎉 Geometry exercise data files created successfully!")
    print(f"📁 Files saved in: {os.path.abspath('data')}") 