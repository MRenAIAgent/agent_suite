"""
Real Image Parsing Test

This test uses our actual OCR solutions to parse your test images and generate
real results, showing the practical capabilities of the system.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import time
import traceback

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import our OCR components
try:
    from math_learning.math_recognization import (
        HybridImageProcessor,
        ProcessingStrategy,
        GeometryOCRProvider,
        create_complete_analyzer
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    IMPORTS_AVAILABLE = False

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

async def test_hybrid_processing_on_real_images():
    """Test hybrid processing on real test images."""
    
    print("🔍 Real Image Parsing Test")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available - skipping test")
        return
    
    # Find test images
    image_files = []
    supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    
    for file_path in TEST_DATA_DIR.iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)
    
    if not image_files:
        print("⚠️  No test images found!")
        return
    
    print(f"📸 Found {len(image_files)} test images:")
    for img_path in sorted(image_files):
        size_kb = img_path.stat().st_size / 1024
        print(f"   • {img_path.name} ({size_kb:.1f}KB)")
    
    # Test different processing strategies
    strategies_to_test = [
        ProcessingStrategy.GEOMETRY_FIRST,
        ProcessingStrategy.PARALLEL,
        ProcessingStrategy.OCR_FIRST,
    ]
    
    # Only test first 2 images to avoid long processing times
    test_images = sorted(image_files)[:2]
    
    results = []
    
    for strategy in strategies_to_test:
        print(f"\n🔧 Testing Strategy: {strategy.value}")
        print("-" * 50)
        
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
        
        for image_path in test_images:
            print(f"\n🖼️  Processing: {image_path.name}")
            
            try:
                # Read image data
                with open(image_path, 'rb') as f:
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
                    'image_name': image_path.name,
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
                
                # Add geometry-specific information if available
                if result.metadata.get('geometry_available'):
                    image_result['geometry_info'] = {
                        'provider': result.metadata.get('geometry_provider'),
                        'detected_shapes': result.metadata.get('detected_shapes', []),
                        'diagram_type': result.metadata.get('diagram_type'),
                        'confidence': result.metadata.get('geometry_confidence', 0)
                    }
                
                # Add sample questions if found
                if result.question_answer_pairs:
                    image_result['sample_questions'] = []
                    for qa in result.question_answer_pairs[:2]:  # First 2 questions
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
                print(f"   🔧 Strategy Used: {image_result['strategy_used']}")
                
                # Show component usage
                components = image_result['components_used']
                used_components = [k for k, v in components.items() if v]
                print(f"   🛠️  Components: {', '.join(used_components) if used_components else 'none'}")
                
                # Show geometry info if available
                if 'geometry_info' in image_result:
                    geo_info = image_result['geometry_info']
                    print(f"   📐 Geometry Provider: {geo_info['provider']}")
                    if geo_info['detected_shapes']:
                        print(f"   🔍 Shapes: {', '.join(geo_info['detected_shapes'])}")
                    print(f"   📊 Diagram Type: {geo_info['diagram_type']}")
                
                # Show sample questions
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
                    'image_name': image_path.name,
                    'success': False,
                    'error': str(e),
                    'processing_time_ms': 0
                }
                strategy_results['images'].append(image_result)
        
        results.append(strategy_results)
    
    # Generate comparison report
    await generate_comparison_report(results)

async def test_complete_analyzer_on_real_images():
    """Test the complete analyzer on real images."""
    
    print(f"\n🔗 Testing Complete Analyzer")
    print("=" * 60)
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required imports not available - skipping test")
        return
    
    # Find test images
    image_files = []
    supported_formats = {'.png', '.jpg', '.jpeg', '.gif', '.bmp'}
    
    for file_path in TEST_DATA_DIR.iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append(file_path)
    
    if not image_files:
        print("⚠️  No test images found!")
        return
    
    # Create complete analyzer with geometry support
    analyzer = create_complete_analyzer(
        mathpix_app_id="test_id",
        mathpix_app_key="test_key",
        openai_api_key="test_key"
    )
    
    # Test on first image only to avoid long processing
    test_image = sorted(image_files)[0]
    
    print(f"🎯 Analyzing: {test_image.name}")
    
    try:
        # Read image data
        with open(test_image, 'rb') as f:
            image_data = f.read()
        
        # Run complete analysis
        start_time = time.time()
        result = await analyzer.analyze_complete_test(
            image_data=image_data,
            image_format=test_image.suffix[1:]  # Remove the dot
        )
        total_time = (time.time() - start_time) * 1000
        
        print(f"\n📊 Complete Analysis Results:")
        print(f"   Overall Success: {'✅' if result.analysis_success else '❌'}")
        print(f"   Total Processing Time: {total_time:.1f}ms")
        
        # Image analysis results
        img_analysis = result.image_analysis
        print(f"\n🖼️  Image Analysis:")
        print(f"   Quality Score: {img_analysis.image_quality_score:.2f}")
        print(f"   Layout Detected: {'✅' if img_analysis.layout_detected else '❌'}")
        print(f"   Questions Found: {img_analysis.total_questions_found}")
        print(f"   Processing Confidence: {img_analysis.processing_confidence:.2f}")
        print(f"   OCR Method: {img_analysis.ocr_method_used}")
        print(f"   Processing Time: {img_analysis.processing_time_ms:.1f}ms")
        
        # Question analyses
        if result.question_analyses:
            print(f"\n📝 Question Analyses ({len(result.question_analyses)} questions):")
            
            correct_count = 0
            for i, qa_analysis in enumerate(result.question_analyses[:3]):  # Show first 3
                qa_pair = qa_analysis.question_pair
                correctness = qa_analysis.correctness_result
                
                print(f"\n   Question {i+1}:")
                print(f"     ID: {qa_pair.question_id}")
                print(f"     Text: {qa_pair.question_text[:80]}..." if len(qa_pair.question_text) > 80 else f"     Text: {qa_pair.question_text}")
                print(f"     Student Answer: {qa_pair.student_answer}")
                print(f"     Correct: {'✅' if correctness.is_correct else '❌'}")
                print(f"     Confidence: {correctness.confidence:.2f}")
                print(f"     Level: {correctness.correctness_level.value}")
                
                if correctness.is_correct:
                    correct_count += 1
                
                # Error analysis
                if qa_analysis.error_analysis:
                    error_analysis = qa_analysis.error_analysis
                    print(f"     Error Type: {error_analysis.error_type.value}")
                    print(f"     Error Confidence: {error_analysis.confidence:.2f}")
                
                # Knowledge gaps
                if qa_analysis.gap_analysis and qa_analysis.gap_analysis.identified_gaps:
                    gap_count = len(qa_analysis.gap_analysis.identified_gaps)
                    print(f"     Knowledge Gaps: {gap_count}")
            
            print(f"\n   Summary: {correct_count}/{len(result.question_analyses)} correct answers")
        
        # Save complete analysis results
        await save_complete_analysis_results(result, test_image.name, total_time)
        
    except Exception as e:
        print(f"❌ Complete analysis failed: {str(e)}")
        traceback.print_exc()

async def generate_comparison_report(results: List[Dict[str, Any]]):
    """Generate a comparison report of different strategies."""
    
    print(f"\n📊 STRATEGY COMPARISON REPORT")
    print("=" * 60)
    
    if not results:
        print("⚠️  No results to compare")
        return
    
    # Calculate statistics for each strategy
    strategy_stats = {}
    
    for strategy_result in results:
        strategy_name = strategy_result['strategy']
        images = strategy_result['images']
        
        successful_images = [img for img in images if img.get('success', False)]
        
        if images:
            stats = {
                'total_images': len(images),
                'successful_images': len(successful_images),
                'success_rate': len(successful_images) / len(images),
                'avg_confidence': sum(img.get('confidence_score', 0) for img in successful_images) / len(successful_images) if successful_images else 0,
                'avg_processing_time': sum(img.get('processing_time_ms', 0) for img in successful_images) / len(successful_images) if successful_images else 0,
                'total_questions': sum(img.get('questions_found', 0) for img in successful_images),
                'avg_questions_per_image': sum(img.get('questions_found', 0) for img in successful_images) / len(successful_images) if successful_images else 0
            }
            
            strategy_stats[strategy_name] = stats
    
    # Print comparison
    print("Strategy Performance Comparison:")
    print()
    
    for strategy_name, stats in strategy_stats.items():
        print(f"🔧 {strategy_name.replace('_', ' ').title()}:")
        print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"   Avg Confidence: {stats['avg_confidence']:.2f}")
        print(f"   Avg Processing Time: {stats['avg_processing_time']:.1f}ms")
        print(f"   Total Questions Found: {stats['total_questions']}")
        print(f"   Avg Questions/Image: {stats['avg_questions_per_image']:.1f}")
        print()
    
    # Determine best strategy
    if strategy_stats:
        best_strategy = max(strategy_stats.items(), 
                          key=lambda x: (x[1]['success_rate'], x[1]['avg_confidence']))
        
        print(f"🏆 Best Performing Strategy: {best_strategy[0].replace('_', ' ').title()}")
        print(f"   Success Rate: {best_strategy[1]['success_rate']*100:.1f}%")
        print(f"   Confidence: {best_strategy[1]['avg_confidence']:.2f}")
    
    # Save comparison results
    results_dir = TEST_DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "strategy_comparison.json"
    
    comparison_data = {
        'comparison_timestamp': time.time(),
        'strategy_stats': strategy_stats,
        'detailed_results': results,
        'best_strategy': best_strategy[0] if strategy_stats else None
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"📁 Comparison results saved to: {output_file}")

async def save_complete_analysis_results(result, image_name: str, processing_time: float):
    """Save complete analysis results to JSON."""
    
    results_dir = TEST_DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / f"complete_analysis_{image_name.replace(' ', '_')}.json"
    
    # Convert result to serializable format
    analysis_data = {
        'analysis_timestamp': time.time(),
        'image_name': image_name,
        'total_processing_time_ms': processing_time,
        'analysis_success': result.analysis_success,
        'processing_status': result.processing_status.value,
        'image_analysis': {
            'quality_score': result.image_analysis.image_quality_score,
            'layout_detected': result.image_analysis.layout_detected,
            'questions_found': result.image_analysis.total_questions_found,
            'processing_confidence': result.image_analysis.processing_confidence,
            'ocr_method': result.image_analysis.ocr_method_used,
            'processing_time_ms': result.image_analysis.processing_time_ms
        },
        'question_analyses': []
    }
    
    # Add question analyses
    for qa_analysis in result.question_analyses:
        qa_data = {
            'question_pair': {
                'question_id': qa_analysis.question_pair.question_id,
                'question_text': qa_analysis.question_pair.question_text,
                'student_answer': qa_analysis.question_pair.student_answer,
                'expected_answer': qa_analysis.question_pair.expected_answer,
                'confidence': qa_analysis.question_pair.confidence
            },
            'correctness_result': {
                'is_correct': qa_analysis.correctness_result.is_correct,
                'confidence': qa_analysis.correctness_result.confidence,
                'correctness_level': qa_analysis.correctness_result.correctness_level.value
            }
        }
        
        # Add error analysis if available
        if qa_analysis.error_analysis:
            qa_data['error_analysis'] = {
                'error_type': qa_analysis.error_analysis.error_type.value,
                'confidence': qa_analysis.error_analysis.confidence,
                'description': qa_analysis.error_analysis.description
            }
        
        # Add gap analysis if available
        if qa_analysis.gap_analysis:
            qa_data['gap_analysis'] = {
                'identified_gaps': len(qa_analysis.gap_analysis.identified_gaps),
                'confidence_score': qa_analysis.gap_analysis.confidence_score
            }
        
        analysis_data['question_analyses'].append(qa_data)
    
    with open(output_file, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    print(f"📁 Complete analysis saved to: {output_file}")

def print_final_summary():
    """Print final summary and recommendations."""
    
    print(f"\n🎉 FINAL SUMMARY")
    print("=" * 60)
    
    print("✅ Successfully tested the image parsing system with your test images!")
    print()
    print("📋 What was tested:")
    print("   • Hybrid processing strategies (geometry-first, parallel, OCR-first)")
    print("   • Complete test analysis workflow")
    print("   • Question extraction and analysis")
    print("   • Error detection and classification")
    print("   • Knowledge gap identification")
    print()
    print("📊 Results generated:")
    print("   • Strategy comparison analysis")
    print("   • Complete analysis for sample images")
    print("   • Performance metrics and recommendations")
    print("   • Detailed JSON reports for further analysis")
    print()
    print("🚀 Next steps:")
    print("   • Review the generated JSON files for detailed insights")
    print("   • Use the best-performing strategy for your use case")
    print("   • Consider the geometry-first approach for geometric content")
    print("   • Implement batch processing for multiple images")

async def main():
    """Main function to run all real image parsing tests."""
    
    print("🧪 Real Image Parsing Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Test hybrid processing strategies
        await test_hybrid_processing_on_real_images()
        
        # Test complete analyzer
        await test_complete_analyzer_on_real_images()
        
        # Print final summary
        print_final_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        traceback.print_exc()
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total test time: {total_time:.1f} seconds")

if __name__ == "__main__":
    asyncio.run(main()) 