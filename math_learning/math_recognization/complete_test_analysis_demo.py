"""
Complete Test Analysis Demo

This demo shows how to use the complete math test analysis system
to process student test images and generate detailed analysis reports.
"""

import asyncio
import base64
from pathlib import Path
import sys
import json
from PIL import Image, ImageDraw, ImageFont
import io

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from math_learning.math_recognization.complete_test_analyzer import CompleteTestAnalyzer

def create_sample_test_image() -> bytes:
    """Create a sample test image for demonstration."""
    # Create a realistic test image
    img = Image.new('RGB', (800, 1000), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add test header
    draw.text((50, 30), "Math Test - Student: Alice", fill='black', font=title_font)
    draw.text((50, 60), "Date: March 15, 2024", fill='black', font=font)
    
    # Add questions and student answers
    questions_data = [
        {
            'question': "1. Solve for x: 2x + 3 = 11",
            'student_answer': "x = 5",
            'work': "2x + 3 = 11\n   2x = 11 - 3\n   2x = 8\n   x = 4",
            'y_pos': 120
        },
        {
            'question': "2. What is 1/2 + 1/3?",
            'student_answer': "Answer: 2/5",
            'work': "1/2 + 1/3 = 2/5",
            'y_pos': 300
        },
        {
            'question': "3. Factor: x² - 9",
            'student_answer': "(x-3)(x-3)",
            'work': "x² - 9 = (x-3)(x-3)",
            'y_pos': 450
        },
        {
            'question': "4. Simplify: 3(x + 2) - 2x",
            'student_answer': "x + 6",
            'work': "3(x + 2) - 2x\n= 3x + 6 - 2x\n= x + 6",
            'y_pos': 600
        }
    ]
    
    for q_data in questions_data:
        # Draw question
        draw.text((50, q_data['y_pos']), q_data['question'], fill='black', font=font)
        
        # Draw student work
        work_lines = q_data['work'].split('\n')
        for i, line in enumerate(work_lines):
            draw.text((70, q_data['y_pos'] + 30 + i*20), line, fill='blue', font=font)
        
        # Draw student answer
        answer_y = q_data['y_pos'] + 30 + len(work_lines)*20 + 10
        draw.text((70, answer_y), q_data['student_answer'], fill='red', font=font)
        
        # Add a line separator
        draw.line([(50, q_data['y_pos'] + 120), (750, q_data['y_pos'] + 120)], fill='gray', width=1)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

async def demo_complete_analysis():
    """Demonstrate the complete test analysis workflow."""
    print("🧮 Math Test Analysis Demo")
    print("=" * 50)
    
    # Initialize the analyzer
    print("📊 Initializing Complete Test Analyzer...")
    analyzer = CompleteTestAnalyzer()
    
    # Create sample test image
    print("🖼️  Creating sample test image...")
    test_image = create_sample_test_image()
    
    # Define answer key
    answer_key = {
        "Q1": "x = 4",  # Student got x = 5 (wrong)
        "Q2": "5/6",    # Student got 2/5 (wrong - common fraction error)
        "Q3": "(x-3)(x+3)",  # Student got (x-3)(x-3) (wrong - difference of squares error)
        "Q4": "x + 6"   # Student got x + 6 (correct)
    }
    
    print("🔍 Processing test image and analyzing answers...")
    print("   This includes:")
    print("   - Image processing and OCR")
    print("   - Question-answer extraction")
    print("   - Answer correctness checking")
    print("   - Error analysis and root cause identification")
    print("   - Knowledge gap analysis")
    print("   - AI-powered insights")
    
    # Perform complete analysis
    analysis_result = await analyzer.analyze_test(
        image_data=test_image,
        test_id="TEST_001",
        student_id="ALICE_123",
        answer_key=answer_key,
        image_format="png"
    )
    
    print(f"✅ Analysis completed with status: {analysis_result.processing_status.value}")
    print()
    
    # Display results
    print("📋 ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"📊 Overall Score: {analysis_result.overall_score_percentage:.1f}%")
    print(f"✅ Correct Answers: {analysis_result.correct_answers}/{analysis_result.total_questions}")
    print(f"⚠️  Partially Correct: {analysis_result.partially_correct}")
    print(f"❌ Incorrect Answers: {analysis_result.incorrect_answers}")
    print(f"🎯 Analysis Confidence: {analysis_result.analysis_confidence:.2f}")
    print()
    
    # Show concept gaps
    if analysis_result.concept_gaps_identified:
        print("🧠 IDENTIFIED KNOWLEDGE GAPS:")
        for i, gap in enumerate(analysis_result.concept_gaps_identified[:5], 1):
            print(f"   {i}. {gap}")
        print()
    
    # Show prerequisite gaps
    if analysis_result.prerequisite_gaps:
        print("📚 PREREQUISITE GAPS:")
        for i, prereq in enumerate(analysis_result.prerequisite_gaps[:3], 1):
            print(f"   {i}. {prereq}")
        print()
    
    # Show immediate recommendations
    if analysis_result.immediate_remediation:
        print("🎯 IMMEDIATE RECOMMENDATIONS:")
        for i, rec in enumerate(analysis_result.immediate_remediation, 1):
            print(f"   {i}. {rec}")
        print()
    
    # Show detailed question analysis
    print("🔍 DETAILED QUESTION ANALYSIS:")
    print("-" * 40)
    
    for i, pair in enumerate(analysis_result.image_analysis.question_answer_pairs):
        print(f"\n📝 Question {i+1}: {pair.question_text}")
        print(f"👤 Student Answer: {pair.student_answer}")
        print(f"✅ Expected Answer: {pair.expected_answer}")
        
        if i < len(analysis_result.answer_analyses):
            analysis = analysis_result.answer_analyses[i]
            
            if analysis.correctness_result.is_correct:
                print("✅ Status: CORRECT")
            else:
                print("❌ Status: INCORRECT")
                print(f"🔍 Primary Issue: {analysis.primary_diagnosis}")
                print(f"📈 Learning Priority: {analysis.learning_priority}")
                
                if analysis.recommended_actions:
                    print(f"💡 Recommendation: {analysis.recommended_actions[0]}")
        print("-" * 40)
    
    print(f"\n⏱️  Estimated Remediation Time: {analysis_result.estimated_remediation_time:.1f} hours")
    
    return analysis_result

async def demo_report_generation(analysis_result):
    """Demonstrate different report formats."""
    print("\n📄 REPORT GENERATION DEMO")
    print("=" * 50)
    
    analyzer = CompleteTestAnalyzer()
    
    # Generate student-friendly report
    print("👤 STUDENT REPORT:")
    print("-" * 20)
    student_report = analyzer.export_analysis_report(analysis_result, format="student")
    
    print(f"Hello {student_report['student_name']}!")
    print(f"Test Date: {student_report['test_date']}")
    print(f"Your Score: {student_report['overall_score']}")
    print(f"Questions Correct: {student_report['questions_correct']}")
    print(f"\n💬 {student_report['encouragement_message']}")
    
    if student_report.get('areas_to_improve'):
        print(f"\n📚 Areas to focus on:")
        for area in student_report['areas_to_improve']:
            print(f"   • {area}")
    
    if student_report.get('next_steps'):
        print(f"\n🎯 Next steps:")
        for step in student_report['next_steps']:
            print(f"   • {step}")
    
    print(f"\n⏱️  Suggested study time: {student_report['estimated_study_time']}")
    
    # Generate teacher report
    print("\n\n👩‍🏫 TEACHER REPORT:")
    print("-" * 20)
    teacher_report = analyzer.export_analysis_report(analysis_result, format="teacher")
    
    print(f"Student ID: {teacher_report['student_id']}")
    summary = teacher_report['test_analysis_summary']
    print(f"Overall Score: {summary['overall_score']:.1f}%")
    print(f"Questions: {summary['correct']} correct, {summary['partial']} partial, {summary['incorrect']} incorrect")
    print(f"Analysis Confidence: {summary['analysis_confidence']:.2f}")
    
    gaps = teacher_report['learning_gaps_analysis']
    if gaps['concept_gaps']:
        print(f"\n🧠 Concept Gaps: {', '.join(gaps['concept_gaps'][:3])}")
    
    if gaps['prerequisite_gaps']:
        print(f"📚 Prerequisite Gaps: {', '.join(gaps['prerequisite_gaps'][:3])}")
    
    recommendations = teacher_report['instructional_recommendations']
    if recommendations['immediate_interventions']:
        print(f"\n🎯 Immediate Interventions:")
        for intervention in recommendations['immediate_interventions'][:2]:
            print(f"   • {intervention}")
    
    print(f"\n⏱️  Estimated Remediation: {gaps['estimated_remediation_time']:.1f} hours")

async def demo_json_export(analysis_result):
    """Demonstrate JSON export for integration."""
    print("\n\n💾 JSON EXPORT DEMO")
    print("=" * 50)
    
    analyzer = CompleteTestAnalyzer()
    json_report = analyzer.export_analysis_report(analysis_result, format="json")
    
    print("📄 JSON Report Structure:")
    print(f"   • Test Metadata: {list(json_report['test_metadata'].keys())}")
    print(f"   • Image Processing: {list(json_report['image_processing'].keys())}")
    print(f"   • Overall Performance: {list(json_report['overall_performance'].keys())}")
    print(f"   • Learning Analysis: {list(json_report['learning_analysis'].keys())}")
    print(f"   • Recommendations: {list(json_report['recommendations'].keys())}")
    print(f"   • Detailed Questions: {len(json_report['detailed_question_analyses'])} questions")
    
    print(f"\n📊 Key Metrics from JSON:")
    print(f"   • Score: {json_report['overall_performance']['score_percentage']:.1f}%")
    print(f"   • Processing Confidence: {json_report['confidence_metrics']['overall_confidence']:.2f}")
    print(f"   • Remediation Time: {json_report['learning_analysis']['estimated_remediation_hours']:.1f}h")
    
    # Save to file for demonstration
    output_file = Path(__file__).parent / "sample_analysis_output.json"
    with open(output_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"\n💾 Full JSON report saved to: {output_file}")

async def main():
    """Run the complete demo."""
    print("🚀 Starting Complete Math Test Analysis Demo")
    print("This demo shows the full workflow from image to insights\n")
    
    try:
        # Run main analysis
        analysis_result = await demo_complete_analysis()
        
        # Generate reports
        await demo_report_generation(analysis_result)
        
        # Export JSON
        await demo_json_export(analysis_result)
        
        print("\n✨ Demo completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("   ✅ Image processing and OCR")
        print("   ✅ Question-answer extraction")
        print("   ✅ Multi-method answer validation")
        print("   ✅ Comprehensive error analysis")
        print("   ✅ Root cause identification")
        print("   ✅ Knowledge gap mapping")
        print("   ✅ AI-powered insights")
        print("   ✅ Multiple report formats")
        print("   ✅ JSON export for integration")
        
        print("\n💡 Next Steps:")
        print("   • Integrate with real OCR service (Mathpix, UniMERNet)")
        print("   • Connect to LLM for AI analysis")
        print("   • Add database integration")
        print("   • Build web interface")
        print("   • Add student progress tracking")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 