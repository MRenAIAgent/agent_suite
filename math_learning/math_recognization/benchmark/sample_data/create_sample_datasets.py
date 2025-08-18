#!/usr/bin/env python3
"""
🎯 Sample Dataset Creator for Math Test Analysis Benchmarking

Creates realistic sample datasets with ground truth labels for evaluating
our math test analysis system against industry benchmarks.

Usage:
    python create_sample_datasets.py --dataset all
    python create_sample_datasets.py --dataset math_expressions
    python create_sample_datasets.py --dataset test_papers
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64
from PIL import Image, ImageDraw, ImageFont
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

class SampleDatasetCreator:
    """Creates sample datasets with ground truth for benchmarking."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_math_expressions_dataset(self) -> Dict[str, Any]:
        """Create a dataset of mathematical expressions (similar to UniMER-1M)."""
        print("📊 Creating Math Expressions Dataset...")
        
        expressions = [
            {
                "id": "expr_001",
                "image_path": "math_expressions/expr_001.png",
                "latex_ground_truth": "x^2 + 2x - 3 = 0",
                "text_ground_truth": "x² + 2x - 3 = 0",
                "expression_type": "quadratic_equation",
                "difficulty": "medium",
                "domain": "algebra"
            },
            {
                "id": "expr_002", 
                "image_path": "math_expressions/expr_002.png",
                "latex_ground_truth": "\\frac{a + b}{c - d} = \\frac{3}{4}",
                "text_ground_truth": "(a + b)/(c - d) = 3/4",
                "expression_type": "fraction_equation",
                "difficulty": "medium",
                "domain": "algebra"
            },
            {
                "id": "expr_003",
                "image_path": "math_expressions/expr_003.png", 
                "latex_ground_truth": "\\int_{0}^{1} x^2 dx = \\frac{1}{3}",
                "text_ground_truth": "∫₀¹ x² dx = 1/3",
                "expression_type": "integral",
                "difficulty": "hard",
                "domain": "calculus"
            },
            {
                "id": "expr_004",
                "image_path": "math_expressions/expr_004.png",
                "latex_ground_truth": "\\sin(\\theta) + \\cos(\\theta) = 1",
                "text_ground_truth": "sin(θ) + cos(θ) = 1",
                "expression_type": "trigonometric",
                "difficulty": "medium", 
                "domain": "trigonometry"
            },
            {
                "id": "expr_005",
                "image_path": "math_expressions/expr_005.png",
                "latex_ground_truth": "\\begin{pmatrix} 1 & 2 \\\\ 3 & 4 \\end{pmatrix}",
                "text_ground_truth": "[[1, 2], [3, 4]]",
                "expression_type": "matrix",
                "difficulty": "medium",
                "domain": "linear_algebra"
            }
        ]
        
        dataset = {
            "name": "Sample Math Expressions",
            "description": "Sample mathematical expressions dataset for OCR benchmarking",
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "total_samples": len(expressions),
            "domains": ["algebra", "calculus", "trigonometry", "linear_algebra"],
            "difficulty_levels": ["medium", "hard"],
            "expressions": expressions
        }
        
        # Save dataset
        output_path = self.output_dir / "math_expressions_dataset.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        print(f"✅ Created Math Expressions Dataset: {output_path}")
        print(f"   - {len(expressions)} expressions")
        print(f"   - Domains: {', '.join(dataset['domains'])}")
        
        return dataset
    
    def create_test_papers_dataset(self) -> Dict[str, Any]:
        """Create a dataset of complete test papers (similar to MATH-Vision)."""
        print("📝 Creating Test Papers Dataset...")
        
        test_papers = [
            {
                "id": "paper_001",
                "image_path": "test_papers/algebra_test_001.png",
                "title": "Algebra Test - Quadratic Equations",
                "grade_level": "high_school",
                "subject": "algebra",
                "total_questions": 3,
                "ground_truth": {
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_text": "Solve: x² - 5x + 6 = 0",
                            "correct_answer": "x = 2, x = 3",
                            "answer_type": "algebraic",
                            "points": 10
                        },
                        {
                            "question_id": "q2", 
                            "question_text": "Factor: x² + 7x + 12",
                            "correct_answer": "(x + 3)(x + 4)",
                            "answer_type": "algebraic",
                            "points": 10
                        },
                        {
                            "question_id": "q3",
                            "question_text": "Find the vertex of y = x² - 4x + 3",
                            "correct_answer": "(2, -1)",
                            "answer_type": "coordinate",
                            "points": 15
                        }
                    ],
                    "total_points": 35
                }
            },
            {
                "id": "paper_002",
                "image_path": "test_papers/geometry_test_001.png", 
                "title": "Geometry Test - Area and Perimeter",
                "grade_level": "middle_school",
                "subject": "geometry",
                "total_questions": 2,
                "ground_truth": {
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_text": "Find the area of a rectangle with length 8 cm and width 5 cm",
                            "correct_answer": "40 cm²",
                            "answer_type": "numerical",
                            "points": 10
                        },
                        {
                            "question_id": "q2",
                            "question_text": "Calculate the perimeter of a triangle with sides 3, 4, and 5 units",
                            "correct_answer": "12 units",
                            "answer_type": "numerical", 
                            "points": 10
                        }
                    ],
                    "total_points": 20
                }
            },
            {
                "id": "paper_003",
                "image_path": "test_papers/calculus_test_001.png",
                "title": "Calculus Test - Derivatives",
                "grade_level": "college",
                "subject": "calculus", 
                "total_questions": 2,
                "ground_truth": {
                    "questions": [
                        {
                            "question_id": "q1",
                            "question_text": "Find the derivative of f(x) = x³ + 2x² - 5x + 1",
                            "correct_answer": "f'(x) = 3x² + 4x - 5",
                            "answer_type": "algebraic",
                            "points": 15
                        },
                        {
                            "question_id": "q2",
                            "question_text": "Evaluate: d/dx[sin(x) + cos(x)]",
                            "correct_answer": "cos(x) - sin(x)",
                            "answer_type": "algebraic",
                            "points": 15
                        }
                    ],
                    "total_points": 30
                }
            }
        ]
        
        dataset = {
            "name": "Sample Test Papers",
            "description": "Sample test papers dataset for complete analysis benchmarking",
            "version": "1.0", 
            "created_date": datetime.now().isoformat(),
            "total_papers": len(test_papers),
            "subjects": ["algebra", "geometry", "calculus"],
            "grade_levels": ["middle_school", "high_school", "college"],
            "test_papers": test_papers
        }
        
        # Save dataset
        output_path = self.output_dir / "test_papers_dataset.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        print(f"✅ Created Test Papers Dataset: {output_path}")
        print(f"   - {len(test_papers)} test papers")
        print(f"   - Subjects: {', '.join(dataset['subjects'])}")
        
        return dataset
    
    def create_student_solutions_dataset(self) -> Dict[str, Any]:
        """Create a dataset of student solutions with error analysis ground truth."""
        print("🎓 Creating Student Solutions Dataset...")
        
        solutions = [
            {
                "id": "solution_001",
                "question": "Solve: x² - 5x + 6 = 0",
                "correct_answer": "x = 2, x = 3",
                "student_answer": "x = 2, x = 4", 
                "is_correct": False,
                "error_analysis": {
                    "error_type": "arithmetic",
                    "error_description": "Calculation error in factoring",
                    "root_cause": "Incorrect mental arithmetic: 2 × 4 ≠ 6",
                    "knowledge_gap": "factoring_quadratics",
                    "confidence": 0.95
                }
            },
            {
                "id": "solution_002",
                "question": "Factor: x² + 7x + 12",
                "correct_answer": "(x + 3)(x + 4)",
                "student_answer": "(x + 2)(x + 6)",
                "is_correct": False,
                "error_analysis": {
                    "error_type": "conceptual",
                    "error_description": "Incorrect factor pair selection",
                    "root_cause": "Student found factors of 12 but didn't verify sum equals 7",
                    "knowledge_gap": "factoring_verification",
                    "confidence": 0.90
                }
            },
            {
                "id": "solution_003",
                "question": "Find the area of a rectangle with length 8 cm and width 5 cm",
                "correct_answer": "40 cm²",
                "student_answer": "13 cm²",
                "is_correct": False,
                "error_analysis": {
                    "error_type": "procedural",
                    "error_description": "Used addition instead of multiplication",
                    "root_cause": "Confused area formula with perimeter formula",
                    "knowledge_gap": "area_vs_perimeter",
                    "confidence": 0.98
                }
            },
            {
                "id": "solution_004",
                "question": "Calculate: 2/3 + 1/4",
                "correct_answer": "11/12",
                "student_answer": "3/7",
                "is_correct": False,
                "error_analysis": {
                    "error_type": "procedural",
                    "error_description": "Added numerators and denominators separately",
                    "root_cause": "Misunderstood fraction addition procedure",
                    "knowledge_gap": "fraction_operations",
                    "confidence": 0.92
                }
            },
            {
                "id": "solution_005",
                "question": "Solve: 2x + 5 = 13",
                "correct_answer": "x = 4",
                "student_answer": "x = 4",
                "is_correct": True,
                "error_analysis": {
                    "error_type": None,
                    "error_description": None,
                    "root_cause": None,
                    "knowledge_gap": None,
                    "confidence": 1.0
                }
            }
        ]
        
        dataset = {
            "name": "Sample Student Solutions",
            "description": "Sample student solutions with error analysis ground truth",
            "version": "1.0",
            "created_date": datetime.now().isoformat(),
            "total_solutions": len(solutions),
            "error_types": ["arithmetic", "conceptual", "procedural"],
            "accuracy_rate": 0.2,  # 1 correct out of 5
            "solutions": solutions
        }
        
        # Save dataset
        output_path = self.output_dir / "student_solutions_dataset.json"
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
            
        print(f"✅ Created Student Solutions Dataset: {output_path}")
        print(f"   - {len(solutions)} solutions")
        print(f"   - Accuracy rate: {dataset['accuracy_rate']:.1%}")
        
        return dataset
    
    def create_all_datasets(self) -> Dict[str, Any]:
        """Create all sample datasets."""
        print("🚀 Creating All Sample Datasets...")
        print("=" * 50)
        
        datasets = {
            "math_expressions": self.create_math_expressions_dataset(),
            "test_papers": self.create_test_papers_dataset(),
            "student_solutions": self.create_student_solutions_dataset()
        }
        
        # Create summary
        summary = {
            "created_date": datetime.now().isoformat(),
            "total_datasets": len(datasets),
            "datasets": {
                name: {
                    "total_samples": len(data.get("expressions", data.get("test_papers", data.get("solutions", [])))),
                    "description": data["description"]
                }
                for name, data in datasets.items()
            }
        }
        
        summary_path = self.output_dir / "datasets_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("=" * 50)
        print(f"✅ All Datasets Created Successfully!")
        print(f"📊 Summary saved to: {summary_path}")
        
        return datasets

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample datasets for benchmarking")
    parser.add_argument("--dataset", choices=["all", "math_expressions", "test_papers", "student_solutions"], 
                       default="all", help="Dataset to create")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    creator = SampleDatasetCreator(args.output_dir)
    
    if args.dataset == "all":
        creator.create_all_datasets()
    elif args.dataset == "math_expressions":
        creator.create_math_expressions_dataset()
    elif args.dataset == "test_papers":
        creator.create_test_papers_dataset()
    elif args.dataset == "student_solutions":
        creator.create_student_solutions_dataset()

if __name__ == "__main__":
    main() 