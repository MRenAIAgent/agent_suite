"""
Concept Identification System Demo

This demo shows how to use the standalone concept identification system
to analyze mathematical exercises and identify courses, topics, and concepts.
"""

import asyncio
import json
from pathlib import Path

# Import the concept identification system
from math_learning.ai_agent.tools.concept_identification_system import ConceptIdentificationSystem

# Optional: Import LLM if available
try:
    from llm.openai.openai_llm import OpenAILLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  LLM not available - using pattern-based analysis only")


async def demo_basic_usage():
    """Demonstrate basic usage of the concept identification system."""
    print("🔍 Concept Identification System - Basic Usage Demo")
    print("=" * 60)
    
    # Initialize the system (with or without LLM)
    if LLM_AVAILABLE:
        try:
            # Try to initialize with OpenAI API key from environment
            llm = OpenAILLM(api_key="your-api-key-here")  # Replace with actual API key
            system = ConceptIdentificationSystem(llm=llm)
            use_llm = True
            print("✅ System initialized with LLM support")
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            system = ConceptIdentificationSystem(llm=None)
            use_llm = False
            print("✅ Falling back to pattern-based analysis only")
    else:
        system = ConceptIdentificationSystem(llm=None)
        use_llm = False
        print("✅ System initialized with pattern-based analysis only")
    
    # Test exercises from different subjects
    test_exercises = [
        {
            "title": "Linear Equation",
            "content": "Solve for x: 2x + 5 = 13",
            "content_type": "text"
        },
        {
            "title": "Geometry Problem",
            "content": "Find the area of a triangle with base 8 cm and height 6 cm. Use the formula A = (1/2) × base × height.",
            "content_type": "text"
        },
        {
            "title": "Word Problem",
            "content": "Sarah has 24 apples. She gives away 1/3 of them to her friends and eats 2 more. How many apples does she have left?",
            "content_type": "text"
        },
        {
            "title": "Calculus Problem",
            "content": "Find the derivative of f(x) = 3x² + 2x - 5",
            "content_type": "text"
        },
        {
            "title": "Statistics Problem", 
            "content": "Calculate the mean and standard deviation of the following data set: 12, 15, 18, 22, 25, 28, 30",
            "content_type": "text"
        }
    ]
    
    print(f"\n📚 Analyzing {len(test_exercises)} exercises...")
    print("-" * 60)
    
    results = []
    for i, exercise in enumerate(test_exercises, 1):
        print(f"\n📝 Exercise {i}: {exercise['title']}")
        print(f"Content: {exercise['content'][:80]}...")
        
        # Analyze the exercise
        result = await system.identify_concepts(
            exercise["content"], 
            exercise["content_type"],
            use_llm=use_llm
        )
        
        results.append(result)
        
        # Display results
        print(f"🎓 Course: {result.course}")
        print(f"📖 Topic: {result.topic}")
        print(f"🧠 Concepts: {[c['concept_name'] for c in result.concepts[:3]]}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"📋 Exercise Type: {result.exercise_type}")
        print(f"📈 Difficulty: {result.difficulty_level}")
        print(f"🔬 Analysis Method: {result.analysis_method}")
    
    return results


async def demo_markdown_analysis():
    """Demonstrate analysis of markdown-formatted content."""
    print("\n\n📝 Markdown Content Analysis Demo")
    print("=" * 60)
    
    system = ConceptIdentificationSystem(llm=None)  # Use pattern-based for demo
    
    markdown_content = """
    # Quadratic Equations Practice
    
    ## Problem 1
    Solve the following quadratic equation using the **quadratic formula**:
    
    x² - 5x + 6 = 0
    
    ## Problem 2  
    Factor the following quadratic expression:
    
    2x² + 7x + 3
    
    ## Problem 3
    A ball is thrown upward with an initial velocity of 48 ft/sec from a height of 6 feet.
    The height h (in feet) after t seconds is given by:
    
    h(t) = -16t² + 48t + 6
    
    When will the ball hit the ground?
    """
    
    print("📄 Analyzing markdown content...")
    result = await system.identify_concepts(markdown_content, "markdown", use_llm=False)
    
    print(f"\n🎓 Course: {result.course}")
    print(f"📖 Topic: {result.topic}")
    print(f"🧠 Concepts found: {len(result.concepts)}")
    
    for i, concept in enumerate(result.concepts[:5], 1):
        print(f"  {i}. {concept['concept_name']} (confidence: {concept['confidence']:.2f})")
    
    print(f"📊 Overall Confidence: {result.confidence:.2f}")
    print(f"📋 Exercise Type: {result.exercise_type}")
    print(f"📈 Difficulty: {result.difficulty_level}")


async def demo_batch_processing():
    """Demonstrate batch processing of multiple exercises."""
    print("\n\n📦 Batch Processing Demo")
    print("=" * 60)
    
    system = ConceptIdentificationSystem(llm=None)
    
    # Multiple exercises to process at once
    exercises = [
        {"content": "Graph the line y = 2x + 3", "content_type": "text"},
        {"content": "Find the volume of a cylinder with radius 4 cm and height 10 cm", "content_type": "text"},
        {"content": "Simplify: (3x² + 2x - 1) + (x² - 4x + 5)", "content_type": "text"},
        {"content": "What is the probability of rolling a sum of 7 with two dice?", "content_type": "text"},
        {"content": "Solve the system: x + y = 5, 2x - y = 1", "content_type": "text"}
    ]
    
    print(f"🔄 Processing {len(exercises)} exercises in batch...")
    
    results = await system.batch_identify_concepts(exercises, use_llm=False)
    
    # Analyze results
    courses = [r.course for r in results]
    course_counts = {}
    for course in courses:
        course_counts[course] = course_counts.get(course, 0) + 1
    
    print(f"\n📊 Results Summary:")
    print(f"Total exercises processed: {len(results)}")
    print(f"Courses identified:")
    for course, count in course_counts.items():
        print(f"  • {course}: {count} exercises")
    
    print(f"\nAverage confidence: {sum(r.confidence for r in results) / len(results):.2f}")


async def demo_export_report():
    """Demonstrate exporting analysis results to a report."""
    print("\n\n📄 Export Report Demo")
    print("=" * 60)
    
    system = ConceptIdentificationSystem(llm=None)
    
    # Analyze some exercises
    exercises = [
        {"content": "Solve for x: 3x - 7 = 14", "content_type": "text"},
        {"content": "Find the area of a circle with radius 5 meters", "content_type": "text"},
        {"content": "What is the integral of 2x + 3?", "content_type": "text"}
    ]
    
    results = await system.batch_identify_concepts(exercises, use_llm=False)
    
    # Export report
    output_path = "concept_analysis_report.json"
    report = system.export_analysis_report(results, output_path)
    
    print(f"📊 Analysis report exported to: {output_path}")
    print(f"Report summary:")
    print(f"  • Total exercises: {report['analysis_summary']['total_exercises']}")
    print(f"  • Courses identified: {len(report['analysis_summary']['courses_identified'])}")
    print(f"  • Average confidence: {report['analysis_summary']['average_confidence']:.2f}")
    print(f"  • Unique concepts found: {len(report['concept_frequency'])}")
    
    # Show top concepts
    concept_freq = report['concept_frequency']
    if concept_freq:
        top_concepts = sorted(concept_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  • Top concepts:")
        for concept, freq in top_concepts:
            print(f"    - {concept}: {freq} times")


def demo_system_info():
    """Show information about the system capabilities."""
    print("\n\n🔧 System Information")
    print("=" * 60)
    
    system = ConceptIdentificationSystem(llm=None)
    
    # Show supported courses
    courses = system.get_supported_courses()
    print(f"📚 Supported Courses ({len(courses)}):")
    for course in courses:
        print(f"  • {course}")
    
    # Show supported concepts (first 10)
    concepts = system.get_supported_concepts()
    print(f"\n🧠 Supported Concepts ({len(concepts)} total, showing first 10):")
    for concept in concepts[:10]:
        concept_name = concept.replace("_", " ").title()
        print(f"  • {concept_name}")
    
    if len(concepts) > 10:
        print(f"  ... and {len(concepts) - 10} more")


async def main():
    """Run all demos."""
    print("🚀 Concept Identification System - Complete Demo")
    print("=" * 80)
    
    try:
        # Run all demo functions
        await demo_basic_usage()
        await demo_markdown_analysis()
        await demo_batch_processing()
        await demo_export_report()
        demo_system_info()
        
        print("\n\n✅ All demos completed successfully!")
        print("\n💡 Usage Tips:")
        print("  • Use LLM analysis for better accuracy when available")
        print("  • Pattern-based analysis is fast and works offline")
        print("  • Batch processing is efficient for multiple exercises")
        print("  • Export reports for detailed analysis and record-keeping")
        print("  • Supports both text and image inputs (with OCR)")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 