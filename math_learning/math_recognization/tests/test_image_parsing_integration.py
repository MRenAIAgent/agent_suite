"""
Integration Tests for Image Parsing with Real Test Data

This module tests the complete image parsing pipeline using actual test images,
comparing different OCR strategies (Mathpix vs Geometry OCR) and generating
detailed analysis results.
"""

import pytest
import asyncio
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import time

# Import our OCR components
from math_learning.math_recognization import (
    HybridImageProcessor,
    ProcessingStrategy,
    GeometryOCRProvider,
    GeometryOCRProcessor,
    process_geometry_image_auto,
    get_geometry_ocr_recommendations,
    create_complete_analyzer
)

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"

class TestImageParsingIntegration:
    """Integration tests for image parsing with real test data."""
    
    @pytest.fixture
    def test_images(self):
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
                    'type': self._classify_image_type(file_path.name)
                })
        
        return sorted(image_files, key=lambda x: x['name'])
    
    def _classify_image_type(self, filename: str) -> str:
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
    
    @pytest.mark.asyncio
    async def test_geometry_ocr_on_all_images(self, test_images):
        """Test geometry OCR processing on all test images."""
        
        print(f"\n🔍 Testing Geometry OCR on {len(test_images)} images")
        print("=" * 60)
        
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
        await self._save_results(results, "geometry_ocr_results.json")
        
        # Assertions
        assert len(results) == len(test_images), "Should process all images"
        assert successful > 0, "At least one image should be processed successfully"
    
    @pytest.mark.asyncio
    async def test_hybrid_processing_comparison(self, test_images):
        """Compare different processing strategies on test images."""
        
        print(f"\n⚖️  Testing Hybrid Processing Strategies")
        print("=" * 60)
        
        # Test strategies to compare
        strategies = [
            ProcessingStrategy.GEOMETRY_FIRST,
            ProcessingStrategy.PARALLEL,
            ProcessingStrategy.GEOMETRY_ONLY,
            ProcessingStrategy.OCR_FIRST
        ]
        
        comparison_results = []
        
        # Create processor (with mock credentials for testing)
        processor = HybridImageProcessor(
            mathpix_app_id="test_id",
            mathpix_app_key="test_key",
            openai_api_key="test_key",
            geometry_providers=[GeometryOCRProvider.GOT_OCR2],
            default_strategy=ProcessingStrategy.GEOMETRY_FIRST
        )
        
        # Test on first few images to avoid long test times
        test_subset = test_images[:3] if len(test_images) > 3 else test_images
        
        for image_info in test_subset:
            print(f"\n🖼️  Testing: {image_info['name']}")
            
            # Read image data
            with open(image_info['path'], 'rb') as f:
                image_data = f.read()
            
            image_results = {
                'image_name': image_info['name'],
                'image_type': image_info['type'],
                'strategies': {}
            }
            
            for strategy in strategies:
                print(f"   🔄 Strategy: {strategy.value}")
                
                try:
                    start_time = time.time()
                    result = await processor.process_test_image(
                        image_data=image_data,
                        strategy=strategy
                    )
                    processing_time = (time.time() - start_time) * 1000
                    
                    strategy_result = {
                        'success': result.success,
                        'confidence': result.confidence_score,
                        'processing_time_ms': processing_time,
                        'questions_found': len(result.question_answer_pairs),
                        'strategy_used': result.processing_strategy.value if hasattr(result, 'processing_strategy') else strategy.value,
                        'components_used': {
                            'ocr': result.metadata.get('ocr_available', False),
                            'vision': result.metadata.get('vision_available', False),
                            'geometry': result.metadata.get('geometry_available', False)
                        },
                        'error_count': len(result.error_messages)
                    }
                    
                    # Add geometry-specific info if available
                    if result.metadata.get('geometry_available'):
                        strategy_result['geometry_info'] = {
                            'provider': result.metadata.get('geometry_provider'),
                            'shapes': result.metadata.get('detected_shapes', []),
                            'diagram_type': result.metadata.get('diagram_type')
                        }
                    
                    image_results['strategies'][strategy.value] = strategy_result
                    
                    print(f"      {'✅' if result.success else '❌'} Success: {result.success}")
                    print(f"      🎯 Confidence: {result.confidence_score:.2f}")
                    print(f"      ⏱️  Time: {processing_time:.1f}ms")
                    print(f"      📝 Questions: {len(result.question_answer_pairs)}")
                    
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
                    image_results['strategies'][strategy.value] = {
                        'success': False,
                        'error': str(e)
                    }
            
            comparison_results.append(image_results)
        
        # Analyze comparison results
        print(f"\n📊 Strategy Comparison Summary:")
        strategy_stats = {}
        
        for strategy in strategies:
            strategy_name = strategy.value
            successes = []
            confidences = []
            times = []
            
            for img_result in comparison_results:
                strategy_data = img_result['strategies'].get(strategy_name, {})
                if strategy_data.get('success'):
                    successes.append(1)
                    confidences.append(strategy_data.get('confidence', 0))
                    times.append(strategy_data.get('processing_time_ms', 0))
                else:
                    successes.append(0)
            
            if successes:
                strategy_stats[strategy_name] = {
                    'success_rate': sum(successes) / len(successes),
                    'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
                    'avg_time': sum(times) / len(times) if times else 0,
                    'total_tests': len(successes)
                }
        
        for strategy_name, stats in strategy_stats.items():
            print(f"\n🔧 {strategy_name.replace('_', ' ').title()}:")
            print(f"   Success Rate: {stats['success_rate']*100:.1f}%")
            print(f"   Avg Confidence: {stats['avg_confidence']:.2f}")
            print(f"   Avg Time: {stats['avg_time']:.1f}ms")
        
        # Save comparison results
        await self._save_results(comparison_results, "strategy_comparison_results.json")
        
        # Assertions
        assert len(comparison_results) > 0, "Should have comparison results"
        assert any(stats['success_rate'] > 0 for stats in strategy_stats.values()), "At least one strategy should have some success"
    
    @pytest.mark.asyncio 
    async def test_complete_analyzer_integration(self, test_images):
        """Test the complete analyzer with real images."""
        
        print(f"\n🔗 Testing Complete Analyzer Integration")
        print("=" * 60)
        
        # Create complete analyzer with geometry support
        analyzer = create_complete_analyzer(
            mathpix_app_id="test_id",
            mathpix_app_key="test_key", 
            openai_api_key="test_key"
        )
        
        integration_results = []
        
        # Test on subset of images
        test_subset = test_images[:2] if len(test_images) > 2 else test_images
        
        for image_info in test_subset:
            print(f"\n🎯 Analyzing: {image_info['name']}")
            
            try:
                # Read image data
                with open(image_info['path'], 'rb') as f:
                    image_data = f.read()
                
                # Run complete analysis
                start_time = time.time()
                result = await analyzer.analyze_complete_test(
                    image_data=image_data,
                    image_format=image_info['path'].suffix[1:]  # Remove the dot
                )
                total_time = (time.time() - start_time) * 1000
                
                # Analyze the complete result
                analysis = {
                    'image_name': image_info['name'],
                    'image_type': image_info['type'],
                    'success': result.analysis_success,
                    'total_processing_time_ms': total_time,
                    'image_analysis': {
                        'quality_score': result.image_analysis.image_quality_score,
                        'layout_detected': result.image_analysis.layout_detected,
                        'questions_found': result.image_analysis.total_questions_found,
                        'processing_confidence': result.image_analysis.processing_confidence,
                        'ocr_method': result.image_analysis.ocr_method_used
                    },
                    'question_analyses': []
                }
                
                # Analyze each question
                for qa_analysis in result.question_analyses:
                    qa_info = {
                        'question_id': qa_analysis.question_pair.question_id,
                        'question_text': qa_analysis.question_pair.question_text[:100],
                        'student_answer': qa_analysis.question_pair.student_answer,
                        'correctness': {
                            'is_correct': qa_analysis.correctness_result.is_correct,
                            'confidence': qa_analysis.correctness_result.confidence,
                            'level': qa_analysis.correctness_result.correctness_level.value
                        },
                        'error_analysis': {
                            'error_type': qa_analysis.error_analysis.error_type.value if qa_analysis.error_analysis else None,
                            'confidence': qa_analysis.error_analysis.confidence if qa_analysis.error_analysis else 0
                        },
                        'knowledge_gaps': len(qa_analysis.gap_analysis.identified_gaps) if qa_analysis.gap_analysis else 0
                    }
                    analysis['question_analyses'].append(qa_info)
                
                integration_results.append(analysis)
                
                # Print results
                print(f"   {'✅' if result.analysis_success else '❌'} Analysis Success: {result.analysis_success}")
                print(f"   ⏱️  Total Time: {total_time:.1f}ms")
                print(f"   🖼️  Image Quality: {result.image_analysis.image_quality_score:.2f}")
                print(f"   📝 Questions Found: {result.image_analysis.total_questions_found}")
                print(f"   🔧 OCR Method: {result.image_analysis.ocr_method_used}")
                
                if result.question_analyses:
                    correct_answers = sum(1 for qa in result.question_analyses if qa.correctness_result.is_correct)
                    print(f"   ✅ Correct Answers: {correct_answers}/{len(result.question_analyses)}")
                    
                    if any(qa.error_analysis for qa in result.question_analyses):
                        error_types = [qa.error_analysis.error_type.value for qa in result.question_analyses if qa.error_analysis]
                        print(f"   🚨 Error Types Found: {', '.join(set(error_types))}")
                
            except Exception as e:
                print(f"   ❌ Integration Error: {str(e)}")
                integration_results.append({
                    'image_name': image_info['name'],
                    'image_type': image_info['type'],
                    'success': False,
                    'error': str(e)
                })
        
        # Generate integration summary
        successful_analyses = sum(1 for r in integration_results if r.get('success', False))
        print(f"\n📊 Complete Analysis Summary:")
        print(f"   Success Rate: {successful_analyses}/{len(integration_results)} ({successful_analyses/len(integration_results)*100:.1f}%)")
        
        if successful_analyses > 0:
            avg_questions = sum(r['image_analysis']['questions_found'] for r in integration_results if r.get('success')) / successful_analyses
            avg_quality = sum(r['image_analysis']['quality_score'] for r in integration_results if r.get('success')) / successful_analyses
            
            print(f"   Average Questions per Image: {avg_questions:.1f}")
            print(f"   Average Image Quality: {avg_quality:.2f}")
        
        # Save integration results
        await self._save_results(integration_results, "complete_analyzer_results.json")
        
        # Assertions
        assert len(integration_results) > 0, "Should have integration results"
        assert successful_analyses > 0, "At least one complete analysis should succeed"
    
    @pytest.mark.asyncio
    async def test_image_type_classification(self, test_images):
        """Test automatic classification of image content types."""
        
        print(f"\n🏷️  Testing Image Type Classification")
        print("=" * 60)
        
        classification_results = []
        processor = GeometryOCRProcessor()
        
        for image_info in test_images:
            print(f"\n🔍 Classifying: {image_info['name']}")
            
            try:
                # Read image data
                with open(image_info['path'], 'rb') as f:
                    image_data = f.read()
                
                # Process with geometry OCR to get content analysis
                result = await processor.process_geometry_image(image_data)
                
                # Classify based on detected content
                detected_type = self._classify_from_ocr_result(result)
                expected_type = image_info['type']
                
                classification = {
                    'image_name': image_info['name'],
                    'expected_type': expected_type,
                    'detected_type': detected_type,
                    'match': detected_type == expected_type or expected_type == 'unknown',
                    'confidence': result.confidence if not result.error else 0,
                    'detected_shapes': result.detected_shapes if not result.error else [],
                    'diagram_type': result.diagram_type if not result.error else 'unknown',
                    'has_text': bool(result.text) if not result.error else False,
                    'has_coordinates': bool(result.coordinate_info) if not result.error else False
                }
                
                classification_results.append(classification)
                
                print(f"   Expected: {expected_type}")
                print(f"   Detected: {detected_type}")
                print(f"   Match: {'✅' if classification['match'] else '❌'}")
                
                if result.detected_shapes:
                    print(f"   Shapes: {', '.join(result.detected_shapes)}")
                
            except Exception as e:
                print(f"   ❌ Classification Error: {str(e)}")
                classification_results.append({
                    'image_name': image_info['name'],
                    'expected_type': image_info['type'],
                    'detected_type': 'error',
                    'match': False,
                    'error': str(e)
                })
        
        # Calculate classification accuracy
        matches = sum(1 for r in classification_results if r.get('match', False))
        accuracy = matches / len(classification_results) if classification_results else 0
        
        print(f"\n📊 Classification Summary:")
        print(f"   Accuracy: {matches}/{len(classification_results)} ({accuracy*100:.1f}%)")
        
        # Show type distribution
        type_counts = {}
        for result in classification_results:
            detected = result.get('detected_type', 'unknown')
            type_counts[detected] = type_counts.get(detected, 0) + 1
        
        print(f"   Type Distribution:")
        for type_name, count in sorted(type_counts.items()):
            print(f"     {type_name}: {count}")
        
        # Save classification results
        await self._save_results(classification_results, "image_classification_results.json")
        
        # Assertions
        assert len(classification_results) == len(test_images), "Should classify all images"
        assert accuracy > 0.3, f"Classification accuracy should be > 30%, got {accuracy*100:.1f}%"
    
    def _classify_from_ocr_result(self, ocr_result) -> str:
        """Classify image type based on OCR results."""
        if ocr_result.error:
            return 'error'
        
        # Check for geometry indicators
        geometry_shapes = {'triangle', 'circle', 'square', 'rectangle', 'polygon', 'angle', 'line'}
        if any(shape in ocr_result.detected_shapes for shape in geometry_shapes):
            return 'geometry'
        
        # Check diagram type
        if ocr_result.diagram_type and 'geometry' in ocr_result.diagram_type:
            return 'geometry'
        
        # Check for coordinates
        if ocr_result.coordinate_info and ocr_result.coordinate_info.get('points'):
            return 'geometry'
        
        # Check text content for math indicators
        if ocr_result.text:
            text_lower = ocr_result.text.lower()
            if any(word in text_lower for word in ['equation', 'solve', 'algebra', 'calculate']):
                return 'algebra'
            elif any(word in text_lower for word in ['triangle', 'circle', 'angle', 'geometry']):
                return 'geometry'
        
        # Check for LaTeX
        if ocr_result.latex:
            return 'algebra'
        
        # Default classification
        if ocr_result.text and len(ocr_result.text) > 50:
            return 'mixed_test'
        
        return 'unknown'
    
    async def _save_results(self, results: List[Dict[str, Any]], filename: str):
        """Save test results to JSON file."""
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

@pytest.mark.asyncio
async def test_quick_image_parsing_demo():
    """Quick demo test that can be run independently."""
    
    print("\n🚀 Quick Image Parsing Demo")
    print("=" * 40)
    
    test_data_dir = Path(__file__).parent / "data"
    
    # Find first available image
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(test_data_dir.glob(f"*{ext}"))
    
    if not image_files:
        print("⚠️  No test images found - skipping demo")
        return
    
    test_image = image_files[0]
    print(f"📸 Testing with: {test_image.name}")
    
    # Read image
    with open(test_image, 'rb') as f:
        image_data = f.read()
    
    # Test geometry OCR
    try:
        result = await process_geometry_image_auto(
            image_data=image_data,
            preferred_provider="got_ocr2"
        )
        
        print(f"✅ Geometry OCR Result:")
        print(f"   Success: {not result.error}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Shapes: {result.detected_shapes}")
        print(f"   Diagram Type: {result.diagram_type}")
        
        if result.text:
            print(f"   Text Preview: {result.text[:100]}...")
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    print("🎉 Demo completed!")

if __name__ == "__main__":
    # Run quick demo
    asyncio.run(test_quick_image_parsing_demo()) 