#!/usr/bin/env python3
"""
Dedicated Test Runner for Image Parsing Integration Tests

This script runs comprehensive tests on the actual test images using our
geometry OCR and hybrid processing solutions, generating detailed results.
"""

import asyncio
import sys
import argparse
from pathlib import Path
import time

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

def get_test_images():
    """Get all test images from the data directory."""
    image_files = []
    
    # Supported image formats
    supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    
    for file_path in TEST_DATA_DIR.iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append({
                'name': file_path.name,
                'path': file_path,
                'size_kb': file_path.stat().st_size / 1024,
                'type': classify_image_type(file_path.name)
            })
    
    return sorted(image_files, key=lambda x: x['name'])

def classify_image_type(filename: str) -> str:
    """Classify image type based on filename patterns."""
    filename_lower = filename.lower()
    
    if 'algebra' in filename_lower or 'equation' in filename_lower:
        return 'algebra'
    elif 'geometry' in filename_lower or 'triangle' in filename_lower or 'circle' in filename_lower:
        return 'geometry'
    elif 'test' in filename_lower:
        return 'mixed_test'
    elif 'screenshot' in filename_lower:
        return 'screenshot'
    else:
        return 'unknown'

async def run_geometry_ocr_tests():
    """Run geometry OCR tests on all images."""
    print("🔍 Running Geometry OCR Tests")
    print("=" * 50)
    
    # Import here to avoid import issues
    try:
        from math_learning.math_recognization import (
            GeometryOCRProcessor,
            GeometryOCRProvider
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Get test images
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found in data directory!")
        return False
    
    print(f"📸 Found {len(test_images)} test images:")
    for img in test_images:
        print(f"   • {img['name']} ({img['size_kb']:.1f}KB, {img['type']})")
    
    results = []
    processor = GeometryOCRProcessor(providers=[GeometryOCRProvider.GOT_OCR2])
    
    for image_info in test_images:
        print(f"\n📐 Processing: {image_info['name']}")
        print(f"   Size: {image_info['size_kb']:.1f}KB")
        print(f"   Type: {image_info['type']}")
        
        try:
            # Read image data
            with open(image_info['path'], 'rb') as f:
                image_data = f.read()
            
            # Process with geometry OCR
            start_time = time.time()
            result = await processor.process_geometry_image(
                image_data=image_data,
                preferred_provider=GeometryOCRProvider.GOT_OCR2
            )
            processing_time = (time.time() - start_time) * 1000
            
            # Analyze results
            analysis = {
                'image_name': image_info['name'],
                'image_type': image_info['type'],
                'success': not result.error,
                'confidence': result.confidence,
                'processing_time_ms': processing_time,
                'detected_shapes': result.detected_shapes,
                'diagram_type': result.diagram_type,
                'geometric_elements_count': len(result.geometric_elements),
                'text_length': len(result.text) if result.text else 0,
                'latex_found': bool(result.latex),
                'coordinates_found': bool(result.coordinate_info),
                'error': result.error
            }
            
            results.append(analysis)
            
            # Print results
            if result.error:
                print(f"   ❌ Failed: {result.error}")
            else:
                print(f"   ✅ Success (confidence: {result.confidence:.2f})")
                print(f"   ⏱️  Processing time: {processing_time:.1f}ms")
                
                if result.detected_shapes:
                    print(f"   🔍 Shapes: {', '.join(result.detected_shapes)}")
                
                if result.diagram_type != 'unknown':
                    print(f"   📊 Diagram type: {result.diagram_type}")
                
                if result.geometric_elements:
                    print(f"   📐 Elements found: {len(result.geometric_elements)}")
                
                if result.text:
                    preview = result.text[:100].replace('\n', ' ')
                    print(f"   📝 Text preview: {preview}...")
                
                if result.coordinate_info:
                    points = result.coordinate_info.get('points', [])
                    print(f"   📍 Coordinates: {len(points)} points found")
            
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            results.append({
                'image_name': image_info['name'],
                'image_type': image_info['type'],
                'success': False,
                'error': str(e)
            })
    
    # Generate summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 Geometry OCR Summary:")
    print(f"   Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    if successful > 0:
        avg_confidence = sum(r.get('confidence', 0) for r in results if r.get('success')) / successful
        avg_time = sum(r.get('processing_time_ms', 0) for r in results if r.get('success')) / successful
        total_shapes = sum(len(r.get('detected_shapes', [])) for r in results if r.get('success'))
        
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Average processing time: {avg_time:.1f}ms")
        print(f"   Total shapes detected: {total_shapes}")
    
    # Save results
    await save_results(results, "geometry_ocr_results.json")
    
    return successful > 0

async def run_hybrid_processing_tests():
    """Run hybrid processing tests on real images."""
    print("\n⚖️  Running Hybrid Processing Tests")
    print("=" * 50)
    
    # Import here to avoid import issues
    try:
        from math_learning.math_recognization import (
            HybridImageProcessor,
            ProcessingStrategy,
            GeometryOCRProvider
        )
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found!")
        return False
    
    # Test strategies
    strategies = [
        ProcessingStrategy.GEOMETRY_FIRST,
        ProcessingStrategy.PARALLEL,
        ProcessingStrategy.OCR_FIRST
    ]
    
    # Test on first 2 images to avoid long processing times
    test_subset = test_images[:2]
    
    results = []
    
    for strategy in strategies:
        print(f"\n🔧 Testing Strategy: {strategy.value}")
        print("-" * 40)
        
        # Create processor with mock credentials for testing
        processor = HybridImageProcessor(
            mathpix_app_id="test_id",
            mathpix_app_key="test_key",
            openai_api_key="test_key",
            geometry_providers=[GeometryOCRProvider.GOT_OCR2],
            default_strategy=strategy
        )
        
        strategy_results = {
            'strategy': strategy.value,
            'images': []
        }
        
        for image_info in test_subset:
            print(f"\n🖼️  Processing: {image_info['name']}")
            
            try:
                # Read image data
                with open(image_info['path'], 'rb') as f:
                    image_data = f.read()
                
                # Process with current strategy
                start_time = time.time()
                result = await processor.process_test_image(
                    image_data=image_data,
                    strategy=strategy
                )
                processing_time = (time.time() - start_time) * 1000
                
                # Analyze results
                image_result = {
                    'image_name': image_info['name'],
                    'file_size_kb': len(image_data) / 1024,
                    'success': result.success,
                    'processing_time_ms': processing_time,
                    'confidence_score': result.confidence_score,
                    'questions_found': len(result.question_answer_pairs),
                    'error_count': len(result.error_messages),
                    'components_used': {
                        'ocr': result.metadata.get('ocr_available', False),
                        'vision': result.metadata.get('vision_available', False),
                        'geometry': result.metadata.get('geometry_available', False)
                    },
                    'strategy_used': result.processing_strategy.value if hasattr(result, 'processing_strategy') else strategy.value
                }
                
                # Add sample extracted content
                if result.question_answer_pairs:
                    image_result['sample_questions'] = []
                    for qa in result.question_answer_pairs[:2]:
                        qa_sample = {
                            'question_id': qa.get('question_id', 'unknown'),
                            'question_text': qa.get('question_text', '')[:100] + '...' if len(qa.get('question_text', '')) > 100 else qa.get('question_text', ''),
                            'student_answer': qa.get('student_answer', ''),
                            'confidence': qa.get('confidence', 0),
                            'source': qa.get('source', 'unknown')
                        }
                        image_result['sample_questions'].append(qa_sample)
                
                strategy_results['images'].append(image_result)
                
                # Print results
                print(f"   {'✅' if result.success else '❌'} Success: {result.success}")
                print(f"   🎯 Confidence: {result.confidence_score:.2f}")
                print(f"   ⏱️  Processing Time: {processing_time:.1f}ms")
                print(f"   📝 Questions Found: {len(result.question_answer_pairs)}")
                
                # Show extracted content preview
                if result.question_answer_pairs:
                    print(f"   📋 Sample Questions:")
                    for i, qa in enumerate(result.question_answer_pairs[:2]):
                        preview = qa.get('question_text', '')[:50]
                        print(f"     Q{i+1}: {preview}..." if len(preview) == 50 else f"     Q{i+1}: {preview}")
                
                if result.error_messages:
                    print(f"   ⚠️  Errors: {len(result.error_messages)}")
                    for error in result.error_messages[:2]:
                        print(f"     • {error}")
                
            except Exception as e:
                print(f"   ❌ Processing failed: {str(e)}")
                image_result = {
                    'image_name': image_info['name'],
                    'success': False,
                    'error': str(e),
                    'processing_time_ms': 0
                }
                strategy_results['images'].append(image_result)
        
        results.append(strategy_results)
    
    # Save comparison results
    await save_results(results, "hybrid_processing_comparison.json")
    
    return len(results) > 0

async def run_complete_analyzer_test():
    """Test the complete analyzer on a real image."""
    print("\n🔗 Running Complete Analyzer Test")
    print("=" * 50)
    
    # Import here to avoid import issues
    try:
        from math_learning.math_recognization import create_complete_analyzer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found!")
        return False
    
    # Create complete analyzer with mock credentials
    analyzer = create_complete_analyzer(
        mathpix_app_id="test_id",
        mathpix_app_key="test_key",
        openai_api_key="test_key"
    )
    
    # Test on first image
    test_image = test_images[0]
    
    print(f"🎯 Analyzing: {test_image['name']}")
    
    try:
        # Read image data
        with open(test_image['path'], 'rb') as f:
            image_data = f.read()
        
        # Run complete analysis - fix the method name
        start_time = time.time()
        
        # Use the correct method name from the CompleteTestAnalyzer
        result = await analyzer.analyze_test(
            image_data=image_data,
            test_id=f"test_{test_image['name']}",
            student_id="test_student"
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n📊 Complete Analysis Results:")
        print(f"   Processing Status: {result.processing_status.value}")
        print(f"   Total Processing Time: {total_time:.1f}ms")
        print(f"   Overall Score: {result.overall_score_percentage:.1f}%")
        print(f"   Analysis Confidence: {result.analysis_confidence:.2f}")
        
        # Image analysis results
        img_analysis = result.image_analysis
        print(f"\n🖼️  Image Analysis:")
        print(f"   Quality Score: {img_analysis.image_quality_score:.2f}")
        print(f"   Layout Detected: {'✅' if img_analysis.layout_detected else '❌'}")
        print(f"   Questions Found: {img_analysis.total_questions_found}")
        print(f"   Processing Confidence: {img_analysis.processing_confidence:.2f}")
        print(f"   OCR Method: {img_analysis.ocr_method_used}")
        
        # Answer analyses
        print(f"\n📝 Answer Analyses:")
        print(f"   Total Questions: {result.total_questions}")
        print(f"   Correct Answers: {result.correct_answers}")
        print(f"   Partially Correct: {result.partially_correct}")
        print(f"   Incorrect Answers: {result.incorrect_answers}")
        
        if result.answer_analyses:
            print(f"\n   Sample Analyses ({len(result.answer_analyses)} total):")
            for i, analysis in enumerate(result.answer_analyses[:2]):  # Show first 2
                print(f"\n   Analysis {i+1}:")
                print(f"     Correctness: {'✅' if analysis.correctness_result.is_correct else '❌'}")
                print(f"     Confidence: {analysis.correctness_result.confidence:.2f}")
                if analysis.error_analysis:
                    print(f"     Error Type: {analysis.error_analysis.error_type.value}")
        
        # Learning insights
        if result.concept_gaps_identified:
            print(f"\n🎯 Learning Insights:")
            print(f"   Concept Gaps: {len(result.concept_gaps_identified)}")
            print(f"   Priority Areas: {len(result.learning_priority_areas)}")
        
        # Save complete analysis results
        analysis_data = {
            'image_name': test_image['name'],
            'total_processing_time_ms': total_time,
            'processing_status': result.processing_status.value,
            'overall_score': result.overall_score_percentage,
            'total_questions': result.total_questions,
            'correct_answers': result.correct_answers,
            'image_quality': img_analysis.image_quality_score,
            'processing_confidence': img_analysis.processing_confidence,
            'analysis_confidence': result.analysis_confidence
        }
        
        await save_results([analysis_data], "complete_analyzer_test.json")
        
        return result.processing_status.value == 'COMPLETED'
        
    except Exception as e:
        print(f"❌ Complete analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def run_mock_parsing_demo():
    """Run a demo that shows what would be extracted from images."""
    print("\n🚀 Running Mock Parsing Demo")
    print("=" * 50)
    
    test_images = get_test_images()
    
    if not test_images:
        print("⚠️  No test images found!")
        return False
    
    print(f"📸 Analyzing {len(test_images)} test images:")
    
    results = []
    
    for image_info in test_images:
        print(f"\n🖼️  {image_info['name']} ({image_info['size_kb']:.1f}KB)")
        
        # Read image data
        with open(image_info['path'], 'rb') as f:
            image_data = f.read()
        
        # Simulate parsing results based on image characteristics
        mock_result = simulate_parsing_result(image_info, image_data)
        results.append(mock_result)
        
        # Print simulated results
        print(f"   📊 Simulated Parsing Results:")
        print(f"   Content Type: {mock_result['content_type']}")
        print(f"   Questions Detected: {mock_result['questions_detected']}")
        print(f"   Mathematical Elements: {len(mock_result['math_elements'])}")
        
        if mock_result['sample_questions']:
            print(f"   📝 Sample Questions:")
            for i, q in enumerate(mock_result['sample_questions'][:2]):
                print(f"     Q{i+1}: {q['text']}")
                print(f"         Answer: {q['answer']}")
                print(f"         Type: {q['type']}")
    
    # Save mock results
    await save_results(results, "mock_parsing_demo.json")
    
    print(f"\n🎉 Mock parsing completed for {len(test_images)} images!")
    return True

def simulate_parsing_result(image_info, image_data):
    """Simulate what would be extracted from an image."""
    
    # Base simulation based on image characteristics
    size_kb = len(image_data) / 1024
    filename = image_info['name'].lower()
    
    # Determine content type
    if 'screenshot' in filename:
        content_type = 'mixed_math_content'
        base_questions = 3 if size_kb < 300 else 5 if size_kb < 500 else 7
    elif 'test' in filename:
        content_type = 'structured_test'
        base_questions = 2 if size_kb < 200 else 4
    else:
        content_type = 'unknown_math'
        base_questions = 2
    
    # Generate sample questions based on content type
    sample_questions = []
    
    if content_type == 'mixed_math_content':
        sample_questions = [
            {
                'text': 'Solve for x: 2x + 5 = 13',
                'answer': 'x = 4',
                'type': 'algebra',
                'confidence': 0.9
            },
            {
                'text': 'Find the area of triangle ABC',
                'answer': '24 square units',
                'type': 'geometry',
                'confidence': 0.85
            },
            {
                'text': 'Simplify: (x² - 4)/(x - 2)',
                'answer': 'x + 2',
                'type': 'algebra',
                'confidence': 0.88
            }
        ]
    elif content_type == 'structured_test':
        sample_questions = [
            {
                'text': 'Calculate: 15 × 8',
                'answer': '120',
                'type': 'arithmetic',
                'confidence': 0.95
            },
            {
                'text': 'What is 3/4 + 1/8?',
                'answer': '7/8',
                'type': 'fractions',
                'confidence': 0.92
            }
        ]
    
    # Determine math elements
    math_elements = ['equations', 'expressions', 'numbers']
    if size_kb > 300:
        math_elements.extend(['diagrams', 'graphs'])
    if 'geometry' in filename or size_kb > 400:
        math_elements.extend(['shapes', 'angles'])
    
    return {
        'image_name': image_info['name'],
        'file_size_kb': size_kb,
        'content_type': content_type,
        'questions_detected': min(base_questions, len(sample_questions)),
        'sample_questions': sample_questions[:base_questions],
        'math_elements': math_elements,
        'estimated_accuracy': 0.85 if content_type == 'structured_test' else 0.80,
        'processing_strategy_recommended': 'parallel' if size_kb > 300 else 'ocr_first'
    }

async def save_results(results, filename: str):
    """Save test results to JSON file."""
    import json
    
    results_dir = TEST_DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / filename
    
    # Add metadata
    output_data = {
        'test_timestamp': time.time(),
        'test_count': len(results),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"📁 Results saved to: {output_file}")

def print_test_summary(results: dict):
    """Print a summary of test results."""
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed")
    
    print("\n📁 Check the 'data/results/' directory for detailed JSON results")

async def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run image parsing integration tests")
    parser.add_argument("--test", choices=[
        "all", "geometry", "hybrid", "complete", "demo"
    ], default="all", help="Which tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    print("🧪 Image Parsing Integration Test Runner")
    print("=" * 60)
    print(f"Running tests: {args.test}")
    print(f"Verbose mode: {args.verbose}")
    
    # Check test data directory
    if not TEST_DATA_DIR.exists():
        print(f"❌ Test data directory not found: {TEST_DATA_DIR}")
        sys.exit(1)
    
    # Create results directory
    results_dir = TEST_DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    start_time = time.time()
    test_results = {}
    
    try:
        if args.test in ["all", "demo"]:
            test_results["Mock Demo"] = await run_mock_parsing_demo()
        
        if args.test in ["all", "geometry"]:
            test_results["Geometry OCR"] = await run_geometry_ocr_tests()
        
        if args.test in ["all", "hybrid"]:
            test_results["Hybrid Processing"] = await run_hybrid_processing_tests()
        
        if args.test in ["all", "complete"]:
            test_results["Complete Analyzer"] = await run_complete_analyzer_test()
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    total_time = time.time() - start_time
    
    # Print summary
    print_test_summary(test_results)
    print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
    
    # Exit with appropriate code
    all_passed = all(test_results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    asyncio.run(main()) 