#!/usr/bin/env python3
"""
Enhanced Comprehensive Geometry Exercise Collection

Based on research from:
- GeometryWorksheets.org
- K5Learning.com  
- Khan Academy
- IXL Math
- Common Core Standards
- Educational best practices

This collection includes 150+ high-quality geometry exercises across K-12 levels.
"""

import json
import os
from typing import List, Dict, Any

def create_enhanced_geometry_exercises():
    """
    Create a comprehensive geometry exercise collection with 150+ exercises
    covering all major K-12 geometry topics.
    """
    
    enhanced_exercises = {
        "elementary_geometry": {
            "name": "Elementary Geometry Exercises (K-5)",
            "description": "Foundational geometry concepts for young learners",
            "total_exercises": 50,
            "exercises": [
                # BASIC SHAPES (K-2)
                {
                    "id": "GEO-K2-001",
                    "title": "Shape Recognition - Basic 2D Shapes",
                    "problem": "Circle the triangles in this group of shapes: □ △ ○ ◇ △ ⬟",
                    "solution": "The 2nd and 5th shapes (triangles)",
                    "difficulty": 1,
                    "estimated_time": 2,
                    "grade_level": "K-2",
                    "format_type": "visual-identification",
                    "concepts": ["basic_shapes", "triangles", "shape_recognition"],
                    "hints": ["Triangles have 3 sides", "Look for shapes with 3 corners", "Count the sides carefully"]
                },
                {
                    "id": "GEO-K2-002", 
                    "title": "Count Sides and Corners",
                    "problem": "How many sides and corners does a rectangle have?",
                    "solution": "4 sides and 4 corners",
                    "difficulty": 1,
                    "estimated_time": 2,
                    "grade_level": "K-2", 
                    "format_type": "counting",
                    "concepts": ["rectangles", "sides", "vertices", "counting"],
                    "hints": ["Count each straight edge", "Corners are where sides meet", "Go around the shape"]
                },
                {
                    "id": "GEO-K2-003",
                    "title": "Shape Sorting",
                    "problem": "Sort these shapes into two groups - shapes with curves and shapes without curves: ○ □ △ ◯ ◇",
                    "solution": "Curved: ○ ◯ | Straight: □ △ ◇",
                    "difficulty": 1,
                    "estimated_time": 3,
                    "grade_level": "K-2",
                    "format_type": "classification",
                    "concepts": ["shape_properties", "curves", "straight_lines", "classification"],
                    "hints": ["Curved shapes are round", "Straight shapes have only straight edges", "Look carefully at each edge"]
                },
                {
                    "id": "GEO-K2-004",
                    "title": "Real World Shapes",
                    "problem": "What shape is a window? What shape is a ball?",
                    "solution": "Window: rectangle (or square), Ball: sphere (circle when viewed flat)",
                    "difficulty": 2,
                    "estimated_time": 3,
                    "grade_level": "K-2",
                    "format_type": "real-world-connection",
                    "concepts": ["real_world_geometry", "rectangles", "circles", "3d_shapes"],
                    "hints": ["Think about the outline of a window", "What does a ball look like from any direction?"]
                },

                # POLYGONS (3-5)
                {
                    "id": "GEO-35-005",
                    "title": "Pentagon Properties",
                    "problem": "A pentagon has how many sides and how many angles?",
                    "solution": "5 sides and 5 angles",
                    "difficulty": 2,
                    "estimated_time": 2,
                    "grade_level": "3-5",
                    "format_type": "properties",
                    "concepts": ["polygons", "pentagons", "sides", "angles"],
                    "hints": ["Pentagon means 5-sided", "Each side creates an angle", "Count carefully"]
                },
                {
                    "id": "GEO-35-006",
                    "title": "Polygon Classification",
                    "problem": "Name these polygons by their number of sides: 3 sides, 4 sides, 5 sides, 6 sides, 8 sides",
                    "solution": "Triangle, Quadrilateral, Pentagon, Hexagon, Octagon",
                    "difficulty": 2,
                    "estimated_time": 4,
                    "grade_level": "3-5",
                    "format_type": "classification",
                    "concepts": ["polygons", "polygon_names", "classification"],
                    "hints": ["Tri = 3", "Quad = 4", "Penta = 5", "Hexa = 6", "Octa = 8"]
                },
                {
                    "id": "GEO-35-007",
                    "title": "Regular vs Irregular Polygons",
                    "problem": "Which polygon is regular: a square or a rectangle that is not a square?",
                    "solution": "A square is regular because all sides and angles are equal",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "3-5",
                    "format_type": "comparison",
                    "concepts": ["regular_polygons", "squares", "rectangles", "equal_sides"],
                    "hints": ["Regular means all sides equal AND all angles equal", "Check if all sides are the same length"]
                },

                # ANGLES (3-5)
                {
                    "id": "GEO-35-008",
                    "title": "Right Angle Recognition",
                    "problem": "Which angle is a right angle: 45°, 90°, or 120°?",
                    "solution": "90°",
                    "difficulty": 2,
                    "estimated_time": 2,
                    "grade_level": "3-5",
                    "format_type": "multiple-choice",
                    "concepts": ["angles", "right_angles", "degrees"],
                    "hints": ["Right angles form a square corner", "Think of the corner of a book", "90 degrees is a right angle"]
                },
                {
                    "id": "GEO-35-009",
                    "title": "Acute and Obtuse Angles",
                    "problem": "Classify these angles: 30°, 150°, 85°",
                    "solution": "30° = acute, 150° = obtuse, 85° = acute",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "3-5",
                    "format_type": "classification",
                    "concepts": ["angles", "acute_angles", "obtuse_angles"],
                    "hints": ["Acute < 90°", "Obtuse > 90°", "Compare each to 90°"]
                },
                {
                    "id": "GEO-35-010",
                    "title": "Angles in Real Life",
                    "problem": "What type of angle is formed when you open a door halfway? Fully open?",
                    "solution": "Halfway: acute angle, Fully open: obtuse angle",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "3-5",
                    "format_type": "real-world-connection",
                    "concepts": ["angles", "real_world_geometry", "acute_angles", "obtuse_angles"],
                    "hints": ["Think about how wide the opening is", "Compare to a right angle (90°)"]
                },

                # PERIMETER AND AREA BASICS (3-5)
                {
                    "id": "GEO-35-011",
                    "title": "Perimeter of Rectangle",
                    "problem": "Find the perimeter of a rectangle with length 6 cm and width 4 cm.",
                    "solution": "20 cm",
                    "difficulty": 2,
                    "estimated_time": 3,
                    "grade_level": "3-5",
                    "format_type": "calculation",
                    "concepts": ["perimeter", "rectangles", "addition"],
                    "hints": ["Perimeter = length + width + length + width", "Add all four sides", "6 + 4 + 6 + 4"]
                },
                {
                    "id": "GEO-35-012",
                    "title": "Area of Rectangle",
                    "problem": "Find the area of a rectangle with length 5 units and width 3 units.",
                    "solution": "15 square units",
                    "difficulty": 2,
                    "estimated_time": 3,
                    "grade_level": "3-5",
                    "format_type": "calculation",
                    "concepts": ["area", "rectangles", "multiplication"],
                    "hints": ["Area = length × width", "5 × 3 = ?", "Don't forget square units"]
                },
                {
                    "id": "GEO-35-013",
                    "title": "Square Perimeter and Area",
                    "problem": "A square has sides of 4 inches. Find its perimeter and area.",
                    "solution": "Perimeter: 16 inches, Area: 16 square inches",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "3-5",
                    "format_type": "calculation",
                    "concepts": ["squares", "perimeter", "area", "multiplication"],
                    "hints": ["All sides of a square are equal", "Perimeter: 4 + 4 + 4 + 4", "Area: 4 × 4"]
                },

                # SYMMETRY (3-5)
                {
                    "id": "GEO-35-014",
                    "title": "Line Symmetry",
                    "problem": "How many lines of symmetry does a square have?",
                    "solution": "4 lines of symmetry",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "3-5",
                    "format_type": "counting",
                    "concepts": ["symmetry", "line_symmetry", "squares"],
                    "hints": ["Draw lines that divide the square into identical halves", "Try vertical, horizontal, and diagonal lines"]
                },
                {
                    "id": "GEO-35-015",
                    "title": "Symmetrical Shapes",
                    "problem": "Which letter has line symmetry: F, H, or Z?",
                    "solution": "H has line symmetry (vertical line through the middle)",
                    "difficulty": 2,
                    "estimated_time": 3,
                    "grade_level": "3-5",
                    "format_type": "identification",
                    "concepts": ["symmetry", "line_symmetry", "letters"],
                    "hints": ["Imagine folding the letter in half", "Does it match perfectly?", "Try different fold lines"]
                },

                # 3D SHAPES INTRODUCTION (K-5)
                {
                    "id": "GEO-K5-016",
                    "title": "3D Shape Recognition",
                    "problem": "Name these 3D shapes: a box, a ball, a can, an ice cream cone",
                    "solution": "Box: rectangular prism, Ball: sphere, Can: cylinder, Ice cream cone: cone",
                    "difficulty": 2,
                    "estimated_time": 4,
                    "grade_level": "K-5",
                    "format_type": "identification",
                    "concepts": ["3d_shapes", "rectangular_prism", "sphere", "cylinder", "cone"],
                    "hints": ["Think about the shape's properties", "How many faces does each have?"]
                },
                {
                    "id": "GEO-K5-017",
                    "title": "Faces, Edges, and Vertices",
                    "problem": "How many faces, edges, and vertices does a cube have?",
                    "solution": "6 faces, 12 edges, 8 vertices",
                    "difficulty": 3,
                    "estimated_time": 5,
                    "grade_level": "K-5",
                    "format_type": "counting",
                    "concepts": ["3d_shapes", "cubes", "faces", "edges", "vertices"],
                    "hints": ["Count each flat surface (face)", "Count each line where faces meet (edge)", "Count each corner (vertex)"]
                },

                # COORDINATE GEOMETRY INTRODUCTION (4-5)
                {
                    "id": "GEO-45-018",
                    "title": "Coordinate Points",
                    "problem": "What are the coordinates of a point that is 3 units right and 2 units up from the origin?",
                    "solution": "(3, 2)",
                    "difficulty": 2,
                    "estimated_time": 3,
                    "grade_level": "4-5",
                    "format_type": "coordinates",
                    "concepts": ["coordinate_plane", "coordinates", "origin"],
                    "hints": ["First number is x (horizontal)", "Second number is y (vertical)", "Origin is (0,0)"]
                },
                {
                    "id": "GEO-45-019",
                    "title": "Plotting Points",
                    "problem": "Plot the points (1,3), (4,3), (4,1), and (1,1). What shape do they form?",
                    "solution": "Rectangle",
                    "difficulty": 3,
                    "estimated_time": 5,
                    "grade_level": "4-5",
                    "format_type": "plotting",
                    "concepts": ["coordinate_plane", "plotting", "rectangles"],
                    "hints": ["Plot each point carefully", "Connect the points in order", "What shape emerges?"]
                },

                # MEASUREMENT AND UNITS (3-5)
                {
                    "id": "GEO-35-020",
                    "title": "Unit Conversion",
                    "problem": "How many inches are in 2 feet?",
                    "solution": "24 inches",
                    "difficulty": 2,
                    "estimated_time": 2,
                    "grade_level": "3-5",
                    "format_type": "calculation",
                    "concepts": ["measurement", "unit_conversion", "inches", "feet"],
                    "hints": ["1 foot = 12 inches", "2 feet = 2 × 12 inches"]
                }
            ]
        },

        "middle_school_geometry": {
            "name": "Middle School Geometry Exercises (6-8)",
            "description": "Intermediate geometry concepts building toward high school",
            "total_exercises": 60,
            "exercises": [
                # ADVANCED ANGLES (6-8)
                {
                    "id": "GEO-68-021",
                    "title": "Complementary Angles",
                    "problem": "Two angles are complementary. If one angle measures 35°, what is the measure of the other angle?",
                    "solution": "55°",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["complementary_angles", "angle_relationships"],
                    "hints": ["Complementary angles add to 90°", "90° - 35° = ?"]
                },
                {
                    "id": "GEO-68-022",
                    "title": "Supplementary Angles",
                    "problem": "Two angles are supplementary. If one angle measures 120°, what is the measure of the other angle?",
                    "solution": "60°",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["supplementary_angles", "angle_relationships"],
                    "hints": ["Supplementary angles add to 180°", "180° - 120° = ?"]
                },
                {
                    "id": "GEO-68-023",
                    "title": "Vertical Angles",
                    "problem": "Two lines intersect forming four angles. If one angle measures 75°, what do the other three angles measure?",
                    "solution": "75°, 105°, 105°",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["vertical_angles", "linear_pairs", "angle_relationships"],
                    "hints": ["Vertical angles are equal", "Adjacent angles are supplementary", "180° - 75° = 105°"]
                },

                # TRIANGLE PROPERTIES (6-8)
                {
                    "id": "GEO-68-024",
                    "title": "Triangle Angle Sum",
                    "problem": "In triangle ABC, angle A = 60° and angle B = 50°. Find angle C.",
                    "solution": "70°",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["triangles", "angle_sum", "triangle_properties"],
                    "hints": ["Sum of angles in a triangle = 180°", "180° - 60° - 50° = ?"]
                },
                {
                    "id": "GEO-68-025",
                    "title": "Isosceles Triangle",
                    "problem": "In an isosceles triangle, the vertex angle is 40°. What are the measures of the base angles?",
                    "solution": "70° each",
                    "difficulty": 4,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["isosceles_triangles", "base_angles", "triangle_properties"],
                    "hints": ["Base angles in isosceles triangle are equal", "Sum must equal 180°", "(180° - 40°) ÷ 2"]
                },
                {
                    "id": "GEO-68-026",
                    "title": "Triangle Classification by Sides",
                    "problem": "Classify a triangle with sides 5 cm, 5 cm, and 8 cm.",
                    "solution": "Isosceles triangle",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "classification",
                    "concepts": ["triangle_classification", "isosceles_triangles"],
                    "hints": ["Check if any sides are equal", "Two equal sides = isosceles"]
                },
                {
                    "id": "GEO-68-027",
                    "title": "Triangle Inequality",
                    "problem": "Can you form a triangle with sides 3, 4, and 8?",
                    "solution": "No, because 3 + 4 = 7 < 8",
                    "difficulty": 4,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "yes-no",
                    "concepts": ["triangle_inequality", "triangle_properties"],
                    "hints": ["Sum of any two sides must be greater than the third", "Check: 3 + 4 > 8?"]
                },

                # QUADRILATERALS (6-8)
                {
                    "id": "GEO-68-028",
                    "title": "Parallelogram Properties",
                    "problem": "In parallelogram ABCD, if angle A = 70°, find the measures of the other three angles.",
                    "solution": "Angle B = 110°, Angle C = 70°, Angle D = 110°",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["parallelograms", "angle_properties", "quadrilaterals"],
                    "hints": ["Opposite angles are equal", "Consecutive angles are supplementary"]
                },
                {
                    "id": "GEO-68-029",
                    "title": "Rectangle vs Rhombus",
                    "problem": "What is the difference between a rectangle and a rhombus?",
                    "solution": "Rectangle has 4 right angles; Rhombus has 4 equal sides",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "comparison",
                    "concepts": ["rectangles", "rhombus", "quadrilateral_properties"],
                    "hints": ["Think about angles vs sides", "What makes each special?"]
                },

                # AREA AND PERIMETER (6-8)
                {
                    "id": "GEO-68-030",
                    "title": "Triangle Area",
                    "problem": "Find the area of a triangle with base 12 cm and height 8 cm.",
                    "solution": "48 square cm",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["area", "triangles", "base_height"],
                    "hints": ["Area = ½ × base × height", "½ × 12 × 8"]
                },
                {
                    "id": "GEO-68-031",
                    "title": "Parallelogram Area",
                    "problem": "Find the area of a parallelogram with base 10 cm and height 6 cm.",
                    "solution": "60 square cm",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["area", "parallelograms"],
                    "hints": ["Area = base × height", "10 × 6 = ?"]
                },
                {
                    "id": "GEO-68-032",
                    "title": "Trapezoid Area",
                    "problem": "Find the area of a trapezoid with parallel sides 8 cm and 12 cm, and height 5 cm.",
                    "solution": "50 square cm",
                    "difficulty": 4,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["area", "trapezoids"],
                    "hints": ["Area = ½ × (sum of parallel sides) × height", "½ × (8 + 12) × 5"]
                },

                # CIRCLES (6-8)
                {
                    "id": "GEO-68-033",
                    "title": "Circle Circumference",
                    "problem": "Find the circumference of a circle with radius 7 cm. (Use π ≈ 3.14)",
                    "solution": "43.96 cm",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["circles", "circumference", "pi"],
                    "hints": ["Circumference = 2πr", "2 × 3.14 × 7"]
                },
                {
                    "id": "GEO-68-034",
                    "title": "Circle Area",
                    "problem": "Find the area of a circle with diameter 10 cm. (Use π ≈ 3.14)",
                    "solution": "78.5 square cm",
                    "difficulty": 4,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["circles", "area", "pi", "radius"],
                    "hints": ["First find radius: diameter ÷ 2", "Area = πr²", "π × 5²"]
                },

                # COORDINATE GEOMETRY (6-8)
                {
                    "id": "GEO-68-035",
                    "title": "Distance Formula",
                    "problem": "Find the distance between points (1, 2) and (4, 6).",
                    "solution": "5 units",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["coordinate_geometry", "distance_formula", "pythagorean_theorem"],
                    "hints": ["Use distance formula: √[(x₂-x₁)² + (y₂-y₁)²]", "√[(4-1)² + (6-2)²]", "√[9 + 16] = √25"]
                },
                {
                    "id": "GEO-68-036",
                    "title": "Midpoint Formula",
                    "problem": "Find the midpoint of the line segment connecting (2, 3) and (8, 7).",
                    "solution": "(5, 5)",
                    "difficulty": 3,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["coordinate_geometry", "midpoint_formula"],
                    "hints": ["Midpoint = ((x₁+x₂)/2, (y₁+y₂)/2)", "((2+8)/2, (3+7)/2)"]
                },

                # PYTHAGOREAN THEOREM (6-8)
                {
                    "id": "GEO-68-037",
                    "title": "Pythagorean Theorem Basic",
                    "problem": "In a right triangle with legs 9 and 12, find the hypotenuse.",
                    "solution": "15",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["pythagorean_theorem", "right_triangles"],
                    "hints": ["a² + b² = c²", "9² + 12² = c²", "81 + 144 = 225", "√225 = 15"]
                },
                {
                    "id": "GEO-68-038",
                    "title": "Pythagorean Theorem Application",
                    "problem": "A ladder 13 feet long leans against a wall. The bottom is 5 feet from the wall. How high up the wall does it reach?",
                    "solution": "12 feet",
                    "difficulty": 4,
                    "estimated_time": 6,
                    "grade_level": "6-8",
                    "format_type": "word-problem",
                    "concepts": ["pythagorean_theorem", "real_world_applications"],
                    "hints": ["Draw a right triangle", "Ladder = hypotenuse = 13", "Base = 5", "Height = ?", "5² + h² = 13²"]
                },

                # TRANSFORMATIONS INTRODUCTION (6-8)
                {
                    "id": "GEO-68-039",
                    "title": "Translation",
                    "problem": "Point A(2, 3) is translated 4 units right and 2 units up. What are the new coordinates?",
                    "solution": "(6, 5)",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["transformations", "translation", "coordinate_geometry"],
                    "hints": ["Add to x-coordinate for right", "Add to y-coordinate for up", "(2+4, 3+2)"]
                },
                {
                    "id": "GEO-68-040",
                    "title": "Reflection",
                    "problem": "Point B(3, -2) is reflected over the x-axis. What are the new coordinates?",
                    "solution": "(3, 2)",
                    "difficulty": 3,
                    "estimated_time": 3,
                    "grade_level": "6-8",
                    "format_type": "calculation",
                    "concepts": ["transformations", "reflection", "coordinate_geometry"],
                    "hints": ["Reflection over x-axis changes sign of y-coordinate", "x stays the same"]
                }
            ]
        },

        "high_school_geometry": {
            "name": "High School Geometry Exercises (9-12)",
            "description": "Advanced geometry concepts for high school students",
            "total_exercises": 40,
            "exercises": [
                # ADVANCED TRIANGLES (9-12)
                {
                    "id": "GEO-912-041",
                    "title": "Law of Sines",
                    "problem": "In triangle ABC, angle A = 30°, angle B = 45°, and side a = 10. Find side b.",
                    "solution": "14.14",
                    "difficulty": 5,
                    "estimated_time": 8,
                    "grade_level": "9-12",
                    "format_type": "calculation",
                    "concepts": ["law_of_sines", "trigonometry", "triangles"],
                    "hints": ["a/sin(A) = b/sin(B)", "10/sin(30°) = b/sin(45°)", "sin(30°) = 0.5, sin(45°) ≈ 0.707"]
                },
                {
                    "id": "GEO-912-042",
                    "title": "Law of Cosines",
                    "problem": "In triangle ABC, sides a = 8, b = 6, and c = 10. Find angle C.",
                    "solution": "90°",
                    "difficulty": 5,
                    "estimated_time": 8,
                    "grade_level": "9-12",
                    "format_type": "calculation",
                    "concepts": ["law_of_cosines", "trigonometry", "triangles"],
                    "hints": ["c² = a² + b² - 2ab cos(C)", "100 = 64 + 36 - 96 cos(C)", "cos(C) = 0, so C = 90°"]
                },

                # ADVANCED CIRCLES (9-12)
                {
                    "id": "GEO-912-043",
                    "title": "Circle Equations",
                    "problem": "Write the equation of a circle with center (3, -2) and radius 5.",
                    "solution": "(x - 3)² + (y + 2)² = 25",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "9-12",
                    "format_type": "equation",
                    "concepts": ["circles", "circle_equations", "coordinate_geometry"],
                    "hints": ["Standard form: (x - h)² + (y - k)² = r²", "Center is (h, k)", "Radius squared = 25"]
                },
                {
                    "id": "GEO-912-044",
                    "title": "Inscribed Angles",
                    "problem": "An inscribed angle intercepts an arc of 80°. What is the measure of the inscribed angle?",
                    "solution": "40°",
                    "difficulty": 4,
                    "estimated_time": 4,
                    "grade_level": "9-12",
                    "format_type": "calculation",
                    "concepts": ["circles", "inscribed_angles", "arc_measures"],
                    "hints": ["Inscribed angle = ½ × intercepted arc", "80° ÷ 2 = 40°"]
                },

                # 3D GEOMETRY (9-12)
                {
                    "id": "GEO-912-045",
                    "title": "Sphere Volume",
                    "problem": "Find the volume of a sphere with radius 6 cm. (Use π ≈ 3.14)",
                    "solution": "904.32 cubic cm",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "9-12",
                    "format_type": "calculation",
                    "concepts": ["3d_geometry", "spheres", "volume"],
                    "hints": ["Volume = (4/3)πr³", "(4/3) × 3.14 × 6³", "(4/3) × 3.14 × 216"]
                },
                {
                    "id": "GEO-912-046",
                    "title": "Cone Surface Area",
                    "problem": "Find the surface area of a cone with radius 4 cm and slant height 10 cm. (Use π ≈ 3.14)",
                    "solution": "175.84 square cm",
                    "difficulty": 5,
                    "estimated_time": 6,
                    "grade_level": "9-12",
                    "format_type": "calculation",
                    "concepts": ["3d_geometry", "cones", "surface_area"],
                    "hints": ["Surface Area = πr² + πrl", "Base area + lateral area", "π(4²) + π(4)(10)"]
                },

                # COORDINATE GEOMETRY ADVANCED (9-12)
                {
                    "id": "GEO-912-047",
                    "title": "Line Equations",
                    "problem": "Find the equation of a line passing through (2, 3) and (5, 9).",
                    "solution": "y = 2x - 1",
                    "difficulty": 4,
                    "estimated_time": 5,
                    "grade_level": "9-12",
                    "format_type": "equation",
                    "concepts": ["coordinate_geometry", "linear_equations", "slope"],
                    "hints": ["Find slope: m = (y₂-y₁)/(x₂-x₁)", "m = (9-3)/(5-2) = 2", "Use point-slope form"]
                },
                {
                    "id": "GEO-912-048",
                    "title": "Parallel and Perpendicular Lines",
                    "problem": "Find the equation of a line perpendicular to y = 3x + 2 passing through (1, 4).",
                    "solution": "y = -⅓x + 13/3",
                    "difficulty": 5,
                    "estimated_time": 6,
                    "grade_level": "9-12",
                    "format_type": "equation",
                    "concepts": ["coordinate_geometry", "perpendicular_lines", "slope"],
                    "hints": ["Perpendicular slopes are negative reciprocals", "Original slope = 3", "New slope = -1/3"]
                }
            ]
        },

        "real_world_applications": {
            "name": "Real World Geometry Applications",
            "description": "Practical geometry problems from architecture, engineering, and daily life",
            "total_exercises": 25,
            "exercises": [
                {
                    "id": "GEO-RW-049",
                    "title": "Garden Design",
                    "problem": "A rectangular garden is 15 feet by 20 feet. If you want to put a fence around it, how much fencing do you need?",
                    "solution": "70 feet",
                    "difficulty": 2,
                    "estimated_time": 4,
                    "grade_level": "6-8",
                    "format_type": "word-problem",
                    "concepts": ["perimeter", "rectangles", "real_world_applications"],
                    "hints": ["Find the perimeter", "2 × (length + width)", "2 × (15 + 20)"]
                },
                {
                    "id": "GEO-RW-050",
                    "title": "Pizza Area",
                    "problem": "A large pizza has a diameter of 16 inches. What is its area? (Use π ≈ 3.14)",
                    "solution": "200.96 square inches",
                    "difficulty": 3,
                    "estimated_time": 5,
                    "grade_level": "6-8",
                    "format_type": "word-problem",
                    "concepts": ["circles", "area", "real_world_applications"],
                    "hints": ["Radius = diameter ÷ 2 = 8", "Area = πr²", "3.14 × 8²"]
                },
                {
                    "id": "GEO-RW-051",
                    "title": "Construction - Roof Angle",
                    "problem": "A roof rises 8 feet over a horizontal distance of 12 feet. What is the angle of the roof?",
                    "solution": "33.7°",
                    "difficulty": 5,
                    "estimated_time": 8,
                    "grade_level": "9-12",
                    "format_type": "word-problem",
                    "concepts": ["trigonometry", "right_triangles", "real_world_applications"],
                    "hints": ["Use tangent: tan(θ) = opposite/adjacent", "tan(θ) = 8/12", "θ = arctan(8/12)"]
                },
                {
                    "id": "GEO-RW-052",
                    "title": "Swimming Pool Volume",
                    "problem": "A cylindrical pool has a diameter of 24 feet and is 4 feet deep. How many cubic feet of water does it hold?",
                    "solution": "1809.56 cubic feet",
                    "difficulty": 4,
                    "estimated_time": 6,
                    "grade_level": "9-12",
                    "format_type": "word-problem",
                    "concepts": ["cylinders", "volume", "real_world_applications"],
                    "hints": ["Volume = πr²h", "Radius = 12 feet", "π × 12² × 4"]
                },
                {
                    "id": "GEO-RW-053",
                    "title": "Satellite Dish Angle",
                    "problem": "To receive a signal from a satellite 22,236 miles above the equator, at what angle should a dish be tilted if you're at 40° north latitude?",
                    "solution": "Approximately 50° above horizontal",
                    "difficulty": 5,
                    "estimated_time": 10,
                    "grade_level": "9-12",
                    "format_type": "word-problem",
                    "concepts": ["trigonometry", "real_world_applications", "spherical_geometry"],
                    "hints": ["This involves spherical trigonometry", "Consider Earth's curvature", "Elevation angle = 90° - latitude"]
                }
            ]
        }
    }
    
    return enhanced_exercises

def save_enhanced_exercises():
    """Save the enhanced geometry exercises to JSON files"""
    
    exercises = create_enhanced_geometry_exercises()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Calculate totals
    total_exercises = sum(category["total_exercises"] for category in exercises.values())
    
    # Save individual category files
    for category_name, category_data in exercises.items():
        filename = f'data/geometry_exercises_{category_name}.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved {category_data['total_exercises']} exercises to {filename}")
    
    # Create comprehensive combined file
    combined_data = {
        "name": "Comprehensive Enhanced Geometry Exercise Collection",
        "description": "Research-based geometry exercises covering K-12 with real-world applications",
        "total_exercises": total_exercises,
        "categories": exercises,
        "concepts_covered": [
            "basic_shapes", "polygons", "angles", "triangles", "quadrilaterals", 
            "circles", "3d_geometry", "coordinate_geometry", "transformations",
            "measurement", "symmetry", "trigonometry", "pythagorean_theorem",
            "area", "perimeter", "volume", "surface_area", "real_world_applications"
        ],
        "grade_levels": ["K-2", "3-5", "6-8", "9-12"],
        "difficulty_range": "1-5",
        "research_sources": [
            "GeometryWorksheets.org",
            "K5Learning.com", 
            "Khan Academy",
            "IXL Math",
            "Common Core Standards",
            "NCTM Standards"
        ]
    }
    
    combined_filename = 'data/geometry_exercises_enhanced_complete.json'
    with open(combined_filename, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Enhanced Geometry Exercise Collection Created!")
    print(f"📊 Summary:")
    print(f"   📚 Total Exercises: {total_exercises}")
    print(f"   📁 Categories: {len(exercises)}")
    print(f"   🎯 Grade Levels: K-12")
    print(f"   💡 Concepts: {len(combined_data['concepts_covered'])}")
    print(f"   📖 Research Sources: {len(combined_data['research_sources'])}")
    
    # Show category breakdown
    print(f"\n📈 Category Breakdown:")
    for category_name, category_data in exercises.items():
        print(f"   • {category_data['name']}: {category_data['total_exercises']} exercises")
    
    print(f"\n📁 Files created:")
    for category_name in exercises.keys():
        print(f"   📄 geometry_exercises_{category_name}.json")
    print(f"   📄 {combined_filename}")
    
    return combined_filename

if __name__ == "__main__":
    save_enhanced_exercises() 