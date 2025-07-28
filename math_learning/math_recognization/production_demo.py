"""
Production Demo: Mathpix OCR + GPT-4 Vision Math Test Analysis

This demo shows how to use the production-ready image processing pipeline
that combines Mathpix OCR and GPT-4 Vision for comprehensive math test analysis.
"""

import asyncio
import base64
import os
from pathlib import Path
from typing import Dict, Any

# Import the production components
from math_learning.math_recognization import (
    create_complete_analyzer,
    HybridImageProcessor,
    ProcessingStrategy,
    process_math_test_image,
    extract_math_from_image,
    analyze_math_test_image
)

async def demo_individual_components():
    """Demo using individual OCR and Vision components."""
    
    print("🔍 Demo 1: Individual Components")
    print("=" * 50)
    
    # Sample image data (in practice, load from file)
    # For demo purposes, we'll use a placeholder
    sample_image_path = "sample_math_test.jpg"
    
    if not os.path.exists(sample_image_path):
        print("⚠️  Sample image not found. Creating placeholder...")
        # In practice, you would have a real math test image
        sample_image_data = b"placeholder_image_data"
    else:
        with open(sample_image_path, 'rb') as f:
            sample_image_data = f.read()
    
    # Demo 1A: Mathpix OCR only
    print("\n📝 Testing Mathpix OCR...")
    try:
        ocr_result = await extract_math_from_image(
            image_data=sample_image_data,
            # Credentials from environment variables
            app_id=os.getenv('MATHPIX_APP_ID'),
            app_key=os.getenv('MATHPIX_APP_KEY')
        )
        print(f"✅ OCR Success: {ocr_result.confidence:.2f} confidence")
        print(f"   Text: {ocr_result.text[:100]}...")
        print(f"   LaTeX: {ocr_result.latex[:100]}...")
        print(f"   Processing Time: {ocr_result.processing_time_ms:.1f}ms")
    except Exception as e:
        print(f"❌ OCR Error: {e}")
    
    # Demo 1B: GPT-4 Vision only
    print("\n👁️  Testing GPT-4 Vision...")
    try:
        vision_result = await analyze_math_test_image(
            image_data=sample_image_data,
            api_key=os.getenv('OPENAI_API_KEY')
        )
        print(f"✅ Vision Success: {vision_result.processing_confidence:.2f} confidence")
        print(f"   Questions Found: {vision_result.total_questions_detected}")
        for i, pair in enumerate(vision_result.question_answer_pairs):
            print(f"   Q{i+1}: {pair.question_text[:50]}...")
            print(f"        Answer: {pair.student_answer}")
        print(f"   Processing Time: {vision_result.processing_time_ms:.1f}ms")
    except Exception as e:
        print(f"❌ Vision Error: {e}")

async def demo_hybrid_processor():
    """Demo using the hybrid processor with different strategies."""
    
    print("\n🔄 Demo 2: Hybrid Processing Strategies")
    print("=" * 50)
    
    # Sample image data
    sample_image_data = b"placeholder_image_data"  # Replace with real image
    
    # Test different processing strategies
    strategies = [
        ProcessingStrategy.PARALLEL,
        ProcessingStrategy.OCR_FIRST,
        ProcessingStrategy.VISION_FIRST,
        ProcessingStrategy.OCR_ONLY,
        ProcessingStrategy.VISION_ONLY
    ]
    
    processor = HybridImageProcessor(
        mathpix_app_id=os.getenv('MATHPIX_APP_ID'),
        mathpix_app_key=os.getenv('MATHPIX_APP_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    for strategy in strategies:
        print(f"\n🧪 Testing {strategy.value}...")
        try:
            result = await processor.process_test_image(
                image_data=sample_image_data,
                strategy=strategy
            )
            
            if result.success:
                print(f"✅ Success: {len(result.question_answer_pairs)} questions found")
                print(f"   Confidence: {result.confidence_score:.2f}")
                print(f"   Processing Time: {result.total_processing_time_ms:.1f}ms")
                
                # Show first question if available
                if result.question_answer_pairs:
                    qa = result.question_answer_pairs[0]
                    print(f"   Sample Q: {qa.get('question_text', '')[:50]}...")
                    print(f"   Sample A: {qa.get('student_answer', '')}")
            else:
                print(f"❌ Failed: {result.error_messages}")
                
        except Exception as e:
            print(f"❌ Error with {strategy.value}: {e}")
    
    # Show processing statistics
    print(f"\n📊 Processing Statistics:")
    stats = processor.get_processing_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

async def demo_complete_analyzer():
    """Demo using the complete test analyzer with full workflow."""
    
    print("\n🎯 Demo 3: Complete Test Analysis Workflow")
    print("=" * 50)
    
    # Create analyzer with all credentials
    analyzer = create_complete_analyzer(
        mathpix_app_id=os.getenv('MATHPIX_APP_ID'),
        mathpix_app_key=os.getenv('MATHPIX_APP_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Sample test data
    sample_image_data = b"placeholder_image_data"  # Replace with real image
    test_id = "DEMO_TEST_001"
    student_id = "STUDENT_123"
    
    # Optional answer key for validation
    answer_key = {
        "Q1": "x = 4",
        "Q2": "5/6", 
        "Q3": "(x-3)*(x+3)"
    }
    
    print("🔄 Processing test image...")
    try:
        # Run complete analysis
        analysis = await analyzer.analyze_test(
            image_data=sample_image_data,
            test_id=test_id,
            student_id=student_id,
            answer_key=answer_key,
            image_format="jpg"
        )
        
        print(f"✅ Analysis Complete!")
        print(f"   Status: {analysis.processing_status.value}")
        print(f"   Overall Score: {analysis.overall_score_percentage:.1f}%")
        print(f"   Questions: {analysis.total_questions}")
        print(f"   Correct: {analysis.correct_answers}")
        print(f"   Partially Correct: {analysis.partially_correct}")
        print(f"   Incorrect: {analysis.incorrect_answers}")
        print(f"   Confidence: {analysis.analysis_confidence:.2f}")
        
        # Show concept gaps identified
        if analysis.concept_gaps_identified:
            print(f"\n📚 Concept Gaps Identified:")
            for gap in analysis.concept_gaps_identified[:3]:
                print(f"   • {gap}")
        
        # Show remediation recommendations
        if analysis.immediate_remediation:
            print(f"\n💡 Immediate Remediation:")
            for rec in analysis.immediate_remediation[:2]:
                print(f"   • {rec}")
        
        # Show practice recommendations
        if analysis.practice_recommendations:
            print(f"\n📝 Practice Recommendations:")
            for rec in analysis.practice_recommendations[:2]:
                print(f"   • {rec}")
        
        # Generate reports
        print(f"\n📄 Generating Reports...")
        student_report = analyzer.report_generator.generate_student_report(analysis)
        teacher_report = analyzer.report_generator.generate_teacher_report(analysis)
        
        print(f"   Student Report: {len(str(student_report))} characters")
        print(f"   Teacher Report: {len(str(teacher_report))} characters")
        
        # Show sample from student report
        print(f"\n👨‍🎓 Student Report Sample:")
        print(f"   Score: {student_report['overall_score']}")
        print(f"   Message: {student_report['encouragement_message'][:100]}...")
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")

async def demo_batch_processing():
    """Demo batch processing of multiple test images."""
    
    print("\n📦 Demo 4: Batch Processing")
    print("=" * 50)
    
    # Create processor
    processor = HybridImageProcessor(
        mathpix_app_id=os.getenv('MATHPIX_APP_ID'),
        mathpix_app_key=os.getenv('MATHPIX_APP_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )
    
    # Sample batch of images (placeholders)
    sample_images = [
        b"placeholder_image_1",
        b"placeholder_image_2", 
        b"placeholder_image_3"
    ]
    
    print(f"🔄 Processing {len(sample_images)} images in batch...")
    try:
        results = await processor.batch_process_images(
            images=sample_images,
            strategy=ProcessingStrategy.PARALLEL,
            max_concurrent=2
        )
        
        successful = sum(1 for r in results if hasattr(r, 'success') and r.success)
        print(f"✅ Batch Complete: {successful}/{len(results)} successful")
        
        for i, result in enumerate(results):
            if hasattr(result, 'success'):
                status = "✅" if result.success else "❌"
                questions = len(result.question_answer_pairs) if result.success else 0
                print(f"   Image {i+1}: {status} {questions} questions")
            else:
                print(f"   Image {i+1}: ❌ Exception: {str(result)[:50]}...")
                
    except Exception as e:
        print(f"❌ Batch Error: {e}")

async def demo_error_handling():
    """Demo error handling and fallback scenarios."""
    
    print("\n⚠️  Demo 5: Error Handling & Fallbacks")
    print("=" * 50)
    
    # Test with invalid credentials
    print("🧪 Testing with invalid credentials...")
    processor = HybridImageProcessor(
        mathpix_app_id="invalid_id",
        mathpix_app_key="invalid_key",
        openai_api_key="invalid_key"
    )
    
    result = await processor.process_test_image(
        image_data=b"placeholder_image",
        strategy=ProcessingStrategy.PARALLEL
    )
    
    print(f"Result: {'✅ Success' if result.success else '❌ Failed'}")
    if not result.success:
        print(f"Error Messages: {result.error_messages}")
    
    # Test with partial credentials (OCR only)
    print("\n🧪 Testing OCR-only fallback...")
    processor_ocr_only = HybridImageProcessor(
        mathpix_app_id=os.getenv('MATHPIX_APP_ID'),
        mathpix_app_key=os.getenv('MATHPIX_APP_KEY'),
        openai_api_key=None  # No Vision API
    )
    
    result = await processor_ocr_only.process_test_image(
        image_data=b"placeholder_image",
        strategy=ProcessingStrategy.OCR_ONLY
    )
    
    print(f"OCR-only Result: {'✅ Success' if result.success else '❌ Failed'}")

def print_setup_instructions():
    """Print setup instructions for the demo."""
    
    print("🚀 Math Test Analysis Production Demo")
    print("=" * 50)
    print()
    print("📋 Setup Instructions:")
    print("1. Set environment variables:")
    print("   export MATHPIX_APP_ID='your_mathpix_app_id'")
    print("   export MATHPIX_APP_KEY='your_mathpix_app_key'") 
    print("   export OPENAI_API_KEY='your_openai_api_key'")
    print()
    print("2. Install required dependencies:")
    print("   pip install aiohttp openai")
    print()
    print("3. Place sample math test images in the current directory")
    print("   (or update the demo to use your image paths)")
    print()
    
    # Check if credentials are available
    missing_creds = []
    if not os.getenv('MATHPIX_APP_ID'):
        missing_creds.append('MATHPIX_APP_ID')
    if not os.getenv('MATHPIX_APP_KEY'):
        missing_creds.append('MATHPIX_APP_KEY')
    if not os.getenv('OPENAI_API_KEY'):
        missing_creds.append('OPENAI_API_KEY')
    
    if missing_creds:
        print(f"⚠️  Missing credentials: {', '.join(missing_creds)}")
        print("   Some demos will be skipped or show mock results.")
    else:
        print("✅ All credentials found!")
    
    print()

async def main():
    """Run all demos."""
    
    print_setup_instructions()
    
    try:
        await demo_individual_components()
        await demo_hybrid_processor()
        await demo_complete_analyzer()
        await demo_batch_processing()
        await demo_error_handling()
        
        print("\n🎉 All demos completed!")
        print("\n💡 Next Steps:")
        print("   • Replace placeholder images with real math test images")
        print("   • Integrate with your application's image upload flow")
        print("   • Customize the analysis pipeline for your specific needs")
        print("   • Add database integration to store results")
        print("   • Build a web interface for teachers and students")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 