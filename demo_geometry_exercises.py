#!/usr/bin/env python3
"""
Demo: Geometry Exercises from Educational Websites

This script demonstrates the comprehensive geometry exercise bank
downloaded and curated from high-quality educational websites.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exercises.geometry_exercises import create_geometry_exercises, create_advanced_geometry_exercises


def main():
    """
    Demonstrate the geometry exercise bank.
    """
    print("🔺 GEOMETRY EXERCISES DEMONSTRATION")
    print("=" * 60)
    print("📚 Source: High-quality educational websites")
    print("   - GeometryWorksheets.org")
    print("   - K5Learning.com") 
    print("   - Various educational institutions")
    print()
    
    # Load geometry exercises
    print("📖 Loading Geometry Exercise Banks...")
    basic_bank = create_geometry_exercises()
    advanced_bank = create_advanced_geometry_exercises()
    
    print(f"✅ Basic Geometry Bank: {len(basic_bank.get_all_exercises())} exercises")
    print(f"✅ Advanced Geometry Bank: {len(advanced_bank.get_all_exercises())} exercises")
    print()
    
    # ============================================================================
    # BASIC GEOMETRY EXERCISES SHOWCASE
    # ============================================================================
    
    print("🎯 BASIC GEOMETRY EXERCISES SHOWCASE")
    print("-" * 50)
    
    basic_exercises = basic_bank.get_all_exercises()
    
    # Group exercises by difficulty
    by_difficulty = {}
    for ex in basic_exercises:
        diff = ex.difficulty
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(ex)
    
    for difficulty in sorted(by_difficulty.keys()):
        exercises = by_difficulty[difficulty]
        print(f"\n📊 Difficulty Level {difficulty} ({len(exercises)} exercises):")
        
        for i, ex in enumerate(exercises[:3], 1):  # Show first 3 exercises
            print(f"\n   {i}. {ex.title}")
            print(f"      Problem: {ex.problem}")
            print(f"      Solution: {ex.solution}")
            print(f"      Time: {ex.estimated_time} minutes")
            print(f"      Format: {ex.format_type}")
            
            # Show concept relationships
            concepts = []
            for concept_id, relationship in ex.concept_relationships.items():
                concepts.append(f"{concept_id} ({relationship.weight:.1f})")
            print(f"      Concepts: {', '.join(concepts)}")
            
            if ex.hints:
                print(f"      Hints: {'; '.join(ex.hints[:2])}")  # Show first 2 hints
    
    # ============================================================================
    # ADVANCED GEOMETRY EXERCISES SHOWCASE
    # ============================================================================
    
    print("\n\n🎓 ADVANCED GEOMETRY EXERCISES SHOWCASE")
    print("-" * 50)
    
    advanced_exercises = advanced_bank.get_all_exercises()
    
    for i, ex in enumerate(advanced_exercises, 1):
        print(f"\n{i}. {ex.title}")
        print(f"   Problem: {ex.problem}")
        print(f"   Solution: {ex.solution}")
        print(f"   Difficulty: {ex.difficulty}/5")
        print(f"   Time: {ex.estimated_time} minutes")
        
        # Show concept relationships
        concepts = []
        for concept_id, relationship in ex.concept_relationships.items():
            concepts.append(f"{concept_id} ({relationship.weight:.1f})")
        print(f"   Concepts: {', '.join(concepts)}")
        
        if ex.hints:
            print(f"   Hints: {'; '.join(ex.hints)}")
    
    # ============================================================================
    # EXERCISE STATISTICS
    # ============================================================================
    
    print("\n\n📊 GEOMETRY EXERCISE STATISTICS")
    print("-" * 50)
    
    all_exercises = basic_exercises + advanced_exercises
    
    # Concept coverage
    all_concepts = set()
    for ex in all_exercises:
        all_concepts.update(ex.concept_relationships.keys())
    
    print(f"📈 Total Exercises: {len(all_exercises)}")
    print(f"🎯 Concepts Covered: {len(all_concepts)}")
    print(f"⏱️  Average Time: {sum(ex.estimated_time for ex in all_exercises) / len(all_exercises):.1f} minutes")
    
    # Difficulty distribution
    difficulty_counts = {}
    for ex in all_exercises:
        diff = ex.difficulty
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    
    print(f"\n📊 Difficulty Distribution:")
    for diff in sorted(difficulty_counts.keys()):
        count = difficulty_counts[diff]
        percentage = (count / len(all_exercises)) * 100
        print(f"   Level {diff}: {count} exercises ({percentage:.1f}%)")
    
    # Format types
    format_counts = {}
    for ex in all_exercises:
        fmt = ex.format_type
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    print(f"\n📝 Exercise Formats:")
    for fmt, count in format_counts.items():
        percentage = (count / len(all_exercises)) * 100
        print(f"   {fmt}: {count} exercises ({percentage:.1f}%)")
    
    # Top concepts
    concept_counts = {}
    for ex in all_exercises:
        for concept_id in ex.concept_relationships.keys():
            concept_counts[concept_id] = concept_counts.get(concept_id, 0) + 1
    
    print(f"\n🏆 Top 10 Concepts Covered:")
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (concept, count) in enumerate(sorted_concepts[:10], 1):
        print(f"   {i:2d}. {concept}: {count} exercises")
    
    # ============================================================================
    # SAMPLE EXERCISE SELECTION
    # ============================================================================
    
    print("\n\n🎲 SAMPLE EXERCISE SELECTION")
    print("-" * 50)
    
    # Select exercises by concept
    target_concepts = ["triangles", "area_perimeter", "pythagorean_theorem", "angles"]
    
    for concept in target_concepts:
        concept_exercises = [ex for ex in all_exercises 
                           if concept in ex.concept_relationships]
        
        if concept_exercises:
            print(f"\n📐 {concept.replace('_', ' ').title()} Exercises ({len(concept_exercises)} available):")
            
            # Show one example
            ex = concept_exercises[0]
            print(f"   Example: {ex.title}")
            print(f"   Problem: {ex.problem}")
            print(f"   Solution: {ex.solution}")
            print(f"   Difficulty: {ex.difficulty}/5, Time: {ex.estimated_time}min")
    
    print(f"\n✅ Geometry exercise demonstration complete!")
    print(f"📚 Total exercises available: {len(all_exercises)}")
    print(f"🌐 Sources: Educational websites with high-quality content")
    print(f"🎯 Ready for integration with the math learning system!")


if __name__ == "__main__":
    main() 