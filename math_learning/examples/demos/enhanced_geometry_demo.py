#!/usr/bin/env python3
"""
Enhanced Geometry Exercise Collection Demo

This script demonstrates the comprehensive, research-based geometry exercise 
collection with 175+ exercises covering K-12 levels.
"""

import json
import random
from pathlib import Path

def load_exercise_data():
    """Load the enhanced geometry exercise collection"""
    data_path = Path("data/geometry_exercises_enhanced_complete.json")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {data_path}")
        return None

def display_collection_overview(data):
    """Display overview of the enhanced exercise collection"""
    print("🎯 ENHANCED GEOMETRY EXERCISE COLLECTION")
    print("=" * 60)
    print(f"📚 {data['name']}")
    print(f"📝 {data['description']}")
    print(f"🔢 Total Exercises: {data['total_exercises']}")
    print(f"📊 Categories: {len(data['categories'])}")
    print(f"🎓 Grade Levels: {', '.join(data['grade_levels'])}")
    print(f"🔧 Difficulty Range: {data['difficulty_range']}")
    
    print(f"\n📖 Research Sources:")
    for source in data['research_sources']:
        print(f"   • {source}")
    
    print(f"\n💡 Concepts Covered ({len(data['concepts_covered'])}):")
    concepts_per_line = 4
    for i in range(0, len(data['concepts_covered']), concepts_per_line):
        concepts_line = data['concepts_covered'][i:i+concepts_per_line]
        print(f"   • {' • '.join(concepts_line)}")

def display_category_breakdown(data):
    """Display detailed breakdown by category"""
    print(f"\n📈 CATEGORY BREAKDOWN")
    print("=" * 60)
    
    for category_name, category_data in data['categories'].items():
        print(f"\n📁 {category_data['name']}")
        print(f"   📝 {category_data['description']}")
        print(f"   🔢 Exercises: {category_data['total_exercises']}")
        
        # Sample exercise types
        exercise_types = set()
        grade_levels = set()
        concepts = set()
        
        for exercise in category_data['exercises'][:10]:  # Sample first 10
            exercise_types.add(exercise['format_type'])
            grade_levels.add(exercise['grade_level'])
            concepts.update(exercise['concepts'])
        
        print(f"   🎯 Grade Levels: {', '.join(sorted(grade_levels))}")
        print(f"   📋 Exercise Types: {', '.join(sorted(exercise_types))}")
        print(f"   💡 Key Concepts: {', '.join(sorted(list(concepts)[:5]))}...")

def show_sample_exercises(data):
    """Show sample exercises from each category"""
    print(f"\n🎲 SAMPLE EXERCISES")
    print("=" * 60)
    
    for category_name, category_data in data['categories'].items():
        print(f"\n📚 {category_data['name']}")
        print("-" * 50)
        
        # Show 2 random exercises from each category
        sample_exercises = random.sample(category_data['exercises'], 
                                       min(2, len(category_data['exercises'])))
        
        for i, exercise in enumerate(sample_exercises, 1):
            print(f"\n{i}. {exercise['title']} (ID: {exercise['id']})")
            print(f"   📊 Difficulty: {exercise['difficulty']}/5 | ⏱️ Time: {exercise['estimated_time']} min")
            print(f"   🎓 Grade: {exercise['grade_level']} | 📝 Type: {exercise['format_type']}")
            print(f"   ❓ Problem: {exercise['problem']}")
            print(f"   ✅ Solution: {exercise['solution']}")
            print(f"   💡 Concepts: {', '.join(exercise['concepts'])}")
            if exercise['hints']:
                print(f"   💭 Hints: {'; '.join(exercise['hints'][:2])}")

def show_difficulty_distribution(data):
    """Show distribution of exercises by difficulty"""
    print(f"\n📊 DIFFICULTY DISTRIBUTION")
    print("=" * 60)
    
    difficulty_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    total_exercises = 0
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            difficulty_counts[exercise['difficulty']] += 1
            total_exercises += 1
    
    for difficulty, count in difficulty_counts.items():
        percentage = (count / total_exercises) * 100
        bar = "█" * int(percentage / 2)
        print(f"   Difficulty {difficulty}: {count:3d} exercises ({percentage:5.1f}%) {bar}")

def show_grade_level_distribution(data):
    """Show distribution of exercises by grade level"""
    print(f"\n🎓 GRADE LEVEL DISTRIBUTION")
    print("=" * 60)
    
    grade_counts = {}
    total_exercises = 0
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            grade = exercise['grade_level']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            total_exercises += 1
    
    for grade, count in sorted(grade_counts.items()):
        percentage = (count / total_exercises) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {grade:8s}: {count:3d} exercises ({percentage:5.1f}%) {bar}")

def show_concept_coverage(data):
    """Show which concepts are covered most frequently"""
    print(f"\n💡 CONCEPT COVERAGE ANALYSIS")
    print("=" * 60)
    
    concept_counts = {}
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            for concept in exercise['concepts']:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
    
    # Sort by frequency
    sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 15 Most Covered Concepts:")
    for i, (concept, count) in enumerate(sorted_concepts[:15], 1):
        print(f"   {i:2d}. {concept.replace('_', ' ').title():25s}: {count:3d} exercises")

def show_exercise_format_types(data):
    """Show distribution of exercise format types"""
    print(f"\n📝 EXERCISE FORMAT TYPES")
    print("=" * 60)
    
    format_counts = {}
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            format_type = exercise['format_type']
            format_counts[format_type] = format_counts.get(format_type, 0) + 1
    
    for format_type, count in sorted(format_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {format_type.replace('_', ' ').replace('-', ' ').title():25s}: {count:3d} exercises")

def show_comparison_with_original(data):
    """Compare with the original smaller collection"""
    print(f"\n📈 ENHANCEMENT COMPARISON")
    print("=" * 60)
    
    print("Original Collection vs Enhanced Collection:")
    print(f"   📊 Exercises:        17 → {data['total_exercises']} (+{data['total_exercises'] - 17})")
    print(f"   📁 Categories:       2 → {len(data['categories'])} (+{len(data['categories']) - 2})")
    print(f"   💡 Concepts:         37 → {len(data['concepts_covered'])} (reorganized & expanded)")
    print(f"   🎓 Grade Coverage:   K-12 → K-12 (maintained)")
    print(f"   📖 Research Sources: 0 → {len(data['research_sources'])} (added)")
    
    print(f"\nNew Features Added:")
    print(f"   ✨ Real-world applications category")
    print(f"   ✨ Comprehensive K-2 foundational exercises")
    print(f"   ✨ Advanced high school topics (Law of Sines/Cosines)")
    print(f"   ✨ Multiple exercise format types")
    print(f"   ✨ Research-based content from 6 educational sources")
    print(f"   ✨ Detailed hints and step-by-step guidance")

def interactive_exercise_explorer(data):
    """Interactive exercise exploration"""
    print(f"\n🔍 INTERACTIVE EXERCISE EXPLORER")
    print("=" * 60)
    
    while True:
        print(f"\nChoose an option:")
        print(f"1. Browse by Category")
        print(f"2. Browse by Grade Level")
        print(f"3. Browse by Difficulty")
        print(f"4. Random Exercise")
        print(f"5. Search by Concept")
        print(f"0. Exit")
        
        choice = input(f"\nEnter your choice (0-5): ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            browse_by_category(data)
        elif choice == '2':
            browse_by_grade_level(data)
        elif choice == '3':
            browse_by_difficulty(data)
        elif choice == '4':
            show_random_exercise(data)
        elif choice == '5':
            search_by_concept(data)
        else:
            print("❌ Invalid choice. Please try again.")

def browse_by_category(data):
    """Browse exercises by category"""
    print(f"\n📁 Categories:")
    categories = list(data['categories'].keys())
    for i, category in enumerate(categories, 1):
        print(f"{i}. {data['categories'][category]['name']}")
    
    try:
        choice = int(input(f"\nSelect category (1-{len(categories)}): "))
        if 1 <= choice <= len(categories):
            category_name = categories[choice - 1]
            category_data = data['categories'][category_name]
            
            print(f"\n📚 {category_data['name']}")
            exercises = category_data['exercises'][:5]  # Show first 5
            for i, exercise in enumerate(exercises, 1):
                print(f"\n{i}. {exercise['title']}")
                print(f"   {exercise['problem']}")
                print(f"   💡 Difficulty: {exercise['difficulty']}/5")
    except (ValueError, IndexError):
        print("❌ Invalid selection.")

def browse_by_grade_level(data):
    """Browse exercises by grade level"""
    grade_exercises = {}
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            grade = exercise['grade_level']
            if grade not in grade_exercises:
                grade_exercises[grade] = []
            grade_exercises[grade].append(exercise)
    
    print(f"\n🎓 Grade Levels:")
    grades = sorted(grade_exercises.keys())
    for i, grade in enumerate(grades, 1):
        print(f"{i}. {grade} ({len(grade_exercises[grade])} exercises)")
    
    try:
        choice = int(input(f"\nSelect grade level (1-{len(grades)}): "))
        if 1 <= choice <= len(grades):
            grade = grades[choice - 1]
            exercises = grade_exercises[grade][:3]  # Show first 3
            
            print(f"\n🎓 Grade Level {grade} Exercises:")
            for i, exercise in enumerate(exercises, 1):
                print(f"\n{i}. {exercise['title']}")
                print(f"   {exercise['problem']}")
    except (ValueError, IndexError):
        print("❌ Invalid selection.")

def browse_by_difficulty(data):
    """Browse exercises by difficulty level"""
    difficulty_exercises = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            difficulty_exercises[exercise['difficulty']].append(exercise)
    
    print(f"\n📊 Difficulty Levels:")
    for difficulty in range(1, 6):
        count = len(difficulty_exercises[difficulty])
        print(f"{difficulty}. Difficulty {difficulty} ({count} exercises)")
    
    try:
        choice = int(input(f"\nSelect difficulty (1-5): "))
        if 1 <= choice <= 5:
            exercises = difficulty_exercises[choice][:3]  # Show first 3
            
            print(f"\n📊 Difficulty {choice} Exercises:")
            for i, exercise in enumerate(exercises, 1):
                print(f"\n{i}. {exercise['title']}")
                print(f"   {exercise['problem']}")
                print(f"   ⏱️ Time: {exercise['estimated_time']} minutes")
    except (ValueError, IndexError):
        print("❌ Invalid selection.")

def show_random_exercise(data):
    """Show a random exercise"""
    all_exercises = []
    for category_data in data['categories'].values():
        all_exercises.extend(category_data['exercises'])
    
    exercise = random.choice(all_exercises)
    
    print(f"\n🎲 Random Exercise:")
    print(f"📝 {exercise['title']} (ID: {exercise['id']})")
    print(f"🎓 Grade: {exercise['grade_level']} | 📊 Difficulty: {exercise['difficulty']}/5")
    print(f"❓ Problem: {exercise['problem']}")
    
    show_solution = input(f"\nShow solution? (y/n): ").lower().strip() == 'y'
    if show_solution:
        print(f"✅ Solution: {exercise['solution']}")
        if exercise['hints']:
            print(f"💭 Hints: {'; '.join(exercise['hints'])}")

def search_by_concept(data):
    """Search exercises by concept"""
    all_concepts = set()
    concept_exercises = {}
    
    for category_data in data['categories'].values():
        for exercise in category_data['exercises']:
            for concept in exercise['concepts']:
                all_concepts.add(concept)
                if concept not in concept_exercises:
                    concept_exercises[concept] = []
                concept_exercises[concept].append(exercise)
    
    search_term = input(f"\nEnter concept to search for: ").lower().strip()
    
    matching_concepts = [c for c in all_concepts if search_term in c.lower()]
    
    if matching_concepts:
        print(f"\n🔍 Found {len(matching_concepts)} matching concepts:")
        for concept in sorted(matching_concepts)[:5]:
            count = len(concept_exercises[concept])
            print(f"   • {concept.replace('_', ' ').title()} ({count} exercises)")
            
            # Show first exercise for this concept
            if concept_exercises[concept]:
                exercise = concept_exercises[concept][0]
                print(f"     Example: {exercise['title']}")
    else:
        print(f"❌ No concepts found matching '{search_term}'")

def main():
    """Main demo function"""
    print("🎯 ENHANCED GEOMETRY EXERCISE COLLECTION DEMO")
    print("=" * 70)
    
    # Load data
    data = load_exercise_data()
    if not data:
        return
    
    # Display comprehensive overview
    display_collection_overview(data)
    display_category_breakdown(data)
    show_difficulty_distribution(data)
    show_grade_level_distribution(data)
    show_concept_coverage(data)
    show_exercise_format_types(data)
    show_comparison_with_original(data)
    show_sample_exercises(data)
    
    # Interactive exploration
    explore = input(f"\n🔍 Would you like to explore exercises interactively? (y/n): ").lower().strip()
    if explore == 'y':
        interactive_exercise_explorer(data)
    
    print(f"\n🎉 Demo Complete! The enhanced collection is ready for use.")
    print(f"📁 Access files in: math_learning/data/")
    print(f"📊 Total: {data['total_exercises']} exercises across K-12")

if __name__ == "__main__":
    main() 