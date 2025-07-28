"""
Simple Usage Example for Math Recognition System

This file shows the easiest ways to use the integrated math test analysis system.
"""

import asyncio
from math_learning.math_recognization import (
    create_complete_analyzer,
    create_hybrid_analyzer,
    CompleteTestAnalyzer,
    HybridErrorAnalyzer
)

# ============================================================================
# METHOD 1: Complete Test Analysis (Recommended for full workflow)
# ============================================================================

async def analyze_complete_test():
    """Analyze a complete test image with all components integrated."""
    
    # Create analyzer (simplest way)
    analyzer = create_complete_analyzer()
    
    # Or create with LLM client for AI features
    # analyzer = create_complete_analyzer(llm_client=your_llm_client)
    
    # Analyze test image
    with open("student_test.jpg", "rb") as f:
        test_image = f.read()
    
    answer_key = {
        "Q1": "x = 4",
        "Q2": "5/6", 
        "Q3": "(x-3)(x+3)"
    }
    
    # Get complete analysis
    result = await analyzer.analyze_test(
        image_data=test_image,
        test_id="TEST_001",
        student_id="ALICE_123", 
        answer_key=answer_key
    )
    
    # Access results
    print(f"Overall Score: {result.overall_score_percentage:.1f}%")
    print(f"Concept Gaps: {result.concept_gaps_identified}")
    print(f"Recommendations: {result.immediate_remediation}")
    
    # Generate reports
    student_report = analyzer.export_analysis_report(result, format="student")
    teacher_report = analyzer.export_analysis_report(result, format="teacher")
    json_report = analyzer.export_analysis_report(result, format="json")
    
    return result

# ============================================================================
# METHOD 2: Individual Question Analysis (For single questions)
# ============================================================================

async def analyze_single_question():
    """Analyze a single question with hybrid AI + rule-based approach."""
    
    # Create hybrid analyzer
    analyzer = create_hybrid_analyzer()
    
    # Analyze one question
    result = await analyzer.analyze_comprehensive(
        question="Solve for x: 2x + 3 = 11",
        student_answer="x = 5",
        correct_answer="x = 4",
        work_shown="2x + 3 = 11\n2x = 8\nx = 4"
    )
    
    # Access detailed analysis
    print(f"Is Correct: {result.correctness_result.is_correct}")
    print(f"Primary Issue: {result.primary_diagnosis}")
    print(f"Recommendations: {result.recommended_actions}")
    print(f"Learning Priority: {result.learning_priority}")
    
    return result

# ============================================================================
# METHOD 3: Step-by-Step Component Usage (Advanced)
# ============================================================================

def analyze_step_by_step():
    """Use individual components step by step for custom workflows."""
    
    from math_learning.math_recognization import (
        AnswerCorrectnessEngine,
        ErrorAnalysisEngine,
        RootCauseAnalyzer,
        KnowledgeGapIdentifier
    )
    
    # Step 1: Check answer correctness
    correctness_engine = AnswerCorrectnessEngine()
    correctness_result = correctness_engine.check_correctness(
        student_answer="x = 5",
        expected_answer="x = 4",
        question="Solve for x: 2x + 3 = 11"
    )
    
    if not correctness_result.is_correct:
        # Step 2: Analyze the error
        error_engine = ErrorAnalysisEngine()
        error_result = error_engine.analyze_error(
            question="Solve for x: 2x + 3 = 11",
            student_answer="x = 5", 
            correct_answer="x = 4",
            work_shown="2x + 3 = 11\n2x = 8\nx = 4"
        )
        
        # Step 3: Find root cause
        root_cause_analyzer = RootCauseAnalyzer()
        root_cause_result = root_cause_analyzer.analyze_root_cause(error_result)
        
        # Step 4: Identify knowledge gaps
        gap_identifier = KnowledgeGapIdentifier()
        gap_result = gap_identifier.identify_gaps(error_result, root_cause_result)
        
        print(f"Error Type: {error_result.error_type}")
        print(f"Root Cause: {root_cause_result.primary_cause}")
        print(f"Knowledge Gaps: {[gap.concept_name for gap in gap_result.identified_gaps]}")
    
    return correctness_result

# ============================================================================
# METHOD 4: Batch Processing (Multiple tests)
# ============================================================================

async def analyze_multiple_tests():
    """Process multiple tests efficiently."""
    
    analyzer = create_complete_analyzer()
    
    test_files = ["test1.jpg", "test2.jpg", "test3.jpg"]
    answer_keys = [
        {"Q1": "x = 4", "Q2": "5/6"},
        {"Q1": "y = 2", "Q2": "3/4"}, 
        {"Q1": "z = 7", "Q2": "1/2"}
    ]
    
    results = []
    for i, test_file in enumerate(test_files):
        try:
            with open(test_file, "rb") as f:
                test_image = f.read()
            
            result = await analyzer.analyze_test(
                image_data=test_image,
                test_id=f"TEST_{i+1:03d}",
                student_id=f"STUDENT_{i+1:03d}",
                answer_key=answer_keys[i]
            )
            results.append(result)
            
        except FileNotFoundError:
            print(f"File {test_file} not found, skipping...")
            continue
    
    # Aggregate results
    total_score = sum(r.overall_score_percentage for r in results)
    avg_score = total_score / len(results) if results else 0
    
    print(f"Processed {len(results)} tests")
    print(f"Average Score: {avg_score:.1f}%")
    
    return results

# ============================================================================
# Usage Examples
# ============================================================================

async def main():
    """Run example analyses."""
    
    print("🧮 Math Recognition System - Usage Examples")
    print("=" * 60)
    
    # Example 1: Complete test analysis (most common use case)
    print("\n📊 Example 1: Complete Test Analysis")
    try:
        # result = await analyze_complete_test()
        print("   → Use analyze_complete_test() with actual image file")
    except Exception as e:
        print(f"   → Demo: {e}")
    
    # Example 2: Single question analysis
    print("\n🔍 Example 2: Single Question Analysis")
    try:
        result = await analyze_single_question()
        print("   ✅ Single question analysis completed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Example 3: Step-by-step component usage
    print("\n⚙️  Example 3: Step-by-Step Analysis")
    try:
        result = analyze_step_by_step()
        print("   ✅ Step-by-step analysis completed")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Example 4: Batch processing
    print("\n📦 Example 4: Batch Processing")
    try:
        # results = await analyze_multiple_tests()
        print("   → Use analyze_multiple_tests() with actual image files")
    except Exception as e:
        print(f"   → Demo: {e}")
    
    print("\n✨ All examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 