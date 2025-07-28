"""
Geometry OCR Demo: Advanced Alternatives to Mathpix for Geometry Recognition

This demo showcases state-of-the-art OCR systems that excel at geometry recognition,
including shapes, diagrams, and geometric constructions - areas where Mathpix has limitations.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any

# Import the geometry OCR components
from math_learning.math_recognization import (
    GeometryOCRProcessor,
    GeometryOCRProvider,
    process_geometry_image_auto,
    get_geometry_ocr_recommendations,
    HybridImageProcessor,
    ProcessingStrategy
)

async def demo_geometry_ocr_providers():
    """Demo different geometry OCR providers and their capabilities."""
    
    print("🔍 Demo 1: Geometry OCR Providers Comparison")
    print("=" * 60)
    
    # Get provider information
    processor = GeometryOCRProcessor()
    provider_info = processor.get_provider_info()
    
    print("📋 Available Geometry OCR Providers:")
    for provider_name, info in provider_info.items():
        status = "✅ Available" if info.get("available", False) else "❌ Not Available"
        print(f"\n🔧 {info.get('name', provider_name)}")
        print(f"   Status: {status}")
        print(f"   Description: {info.get('description', 'N/A')}")
        if info.get('strengths'):
            print(f"   Strengths: {', '.join(info['strengths'])}")
        if info.get('best_for'):
            print(f"   Best For: {info['best_for']}")
    
    # Show recommendations
    print(f"\n💡 Geometry OCR Recommendations:")
    recommendations = get_geometry_ocr_recommendations()
    for geometry_type, recommendation in recommendations.items():
        print(f"   • {geometry_type.replace('_', ' ').title()}: {recommendation}")

async def demo_got_ocr_geometry():
    """Demo GOT-OCR2.0 for geometry recognition."""
    
    print("\n🎯 Demo 2: GOT-OCR2.0 for Geometry Recognition")
    print("=" * 60)
    
    # Sample geometry image (placeholder)
    sample_image_data = b"placeholder_geometry_image"
    
    print("📐 Testing GOT-OCR2.0 with geometric diagrams...")
    try:
        result = await process_geometry_image_auto(
            image_data=sample_image_data,
            preferred_provider="got_ocr2"
        )
        
        if result.success and not result.error:
            print(f"✅ GOT-OCR2.0 Success!")
            print(f"   Provider: {result.provider.value}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Processing Time: {result.processing_time_ms:.1f}ms")
            print(f"   Detected Shapes: {', '.join(result.detected_shapes)}")
            print(f"   Diagram Type: {result.diagram_type}")
            
            if result.text:
                print(f"   Extracted Text: {result.text[:100]}...")
            
            if result.latex:
                print(f"   LaTeX: {result.latex[:100]}...")
            
            if result.geometric_elements:
                print(f"   Geometric Elements Found: {len(result.geometric_elements)}")
                for i, element in enumerate(result.geometric_elements[:3]):
                    print(f"     {i+1}. {element.get('type', 'unknown')}: {element.get('text', '')}")
            
            if result.coordinate_info:
                print(f"   Coordinate Information: {result.coordinate_info}")
                
        else:
            print(f"❌ GOT-OCR2.0 Failed: {result.error}")
            
    except Exception as e:
        print(f"❌ GOT-OCR2.0 Error: {e}")

async def demo_geometry_vs_standard_ocr():
    """Compare geometry OCR vs standard OCR for geometric content."""
    
    print("\n⚖️  Demo 3: Geometry OCR vs Standard OCR Comparison")
    print("=" * 60)
    
    # Sample geometric image
    sample_image_data = b"placeholder_geometry_diagram"
    
    # Test with geometry-first strategy
    print("🔬 Testing Geometry-First Strategy...")
    try:
        processor = HybridImageProcessor(
            geometry_providers=[GeometryOCRProvider.GOT_OCR2],
            default_strategy=ProcessingStrategy.GEOMETRY_FIRST
        )
        
        geometry_result = await processor.process_test_image(
            image_data=sample_image_data,
            strategy=ProcessingStrategy.GEOMETRY_FIRST
        )
        
        print(f"📊 Geometry-First Results:")
        print(f"   Success: {'✅' if geometry_result.success else '❌'}")
        print(f"   Questions Found: {len(geometry_result.question_answer_pairs)}")
        print(f"   Confidence: {geometry_result.confidence_score:.2f}")
        print(f"   Processing Time: {geometry_result.total_processing_time_ms:.1f}ms")
        
        if geometry_result.metadata.get('geometry_available'):
            print(f"   Geometry Provider: {geometry_result.metadata.get('geometry_provider', 'N/A')}")
            print(f"   Detected Shapes: {geometry_result.metadata.get('detected_shapes', [])}")
            print(f"   Diagram Type: {geometry_result.metadata.get('diagram_type', 'N/A')}")
        
        # Show sample questions
        if geometry_result.question_answer_pairs:
            print(f"   Sample Questions:")
            for i, qa in enumerate(geometry_result.question_answer_pairs[:2]):
                print(f"     Q{i+1}: {qa.get('question_text', '')[:50]}...")
                print(f"          Type: {qa.get('problem_type', 'unknown')}")
                print(f"          Source: {qa.get('source', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Geometry-First Error: {e}")
    
    # Compare with standard OCR approach
    print(f"\n🔬 Testing Standard OCR Approach...")
    try:
        standard_result = await processor.process_test_image(
            image_data=sample_image_data,
            strategy=ProcessingStrategy.PARALLEL
        )
        
        print(f"📊 Standard OCR Results:")
        print(f"   Success: {'✅' if standard_result.success else '❌'}")
        print(f"   Questions Found: {len(standard_result.question_answer_pairs)}")
        print(f"   Confidence: {standard_result.confidence_score:.2f}")
        print(f"   Processing Time: {standard_result.total_processing_time_ms:.1f}ms")
        
        # Performance comparison
        if geometry_result.success and standard_result.success:
            geo_time = geometry_result.total_processing_time_ms
            std_time = standard_result.total_processing_time_ms
            speed_diff = ((std_time - geo_time) / std_time) * 100 if std_time > 0 else 0
            
            print(f"\n📈 Performance Comparison:")
            print(f"   Geometry-First: {geo_time:.1f}ms")
            print(f"   Standard OCR: {std_time:.1f}ms")
            print(f"   Speed Difference: {speed_diff:+.1f}%")
            
            conf_diff = geometry_result.confidence_score - standard_result.confidence_score
            print(f"   Confidence Difference: {conf_diff:+.2f}")
        
    except Exception as e:
        print(f"❌ Standard OCR Error: {e}")

async def demo_specific_geometry_types():
    """Demo recognition of specific geometry types."""
    
    print("\n📐 Demo 4: Specific Geometry Type Recognition")
    print("=" * 60)
    
    # Different types of geometry problems
    geometry_samples = {
        "triangle_construction": {
            "description": "Triangle construction with angle measurements",
            "image": b"placeholder_triangle_image",
            "expected_shapes": ["triangle", "angle", "line"]
        },
        "circle_geometry": {
            "description": "Circle with inscribed shapes",
            "image": b"placeholder_circle_image",
            "expected_shapes": ["circle", "triangle", "angle"]
        },
        "coordinate_plane": {
            "description": "Coordinate geometry with points and lines",
            "image": b"placeholder_coordinate_image",
            "expected_shapes": ["line", "point"]
        },
        "polygon_properties": {
            "description": "Various polygons with measurements",
            "image": b"placeholder_polygon_image",
            "expected_shapes": ["polygon", "rectangle", "square"]
        }
    }
    
    processor = GeometryOCRProcessor(providers=[GeometryOCRProvider.GOT_OCR2])
    
    for geometry_type, sample in geometry_samples.items():
        print(f"\n🔍 Testing: {sample['description']}")
        
        try:
            result = await processor.process_geometry_image(
                image_data=sample['image'],
                preferred_provider=GeometryOCRProvider.GOT_OCR2
            )
            
            if result.success and not result.error:
                print(f"   ✅ Recognition Success")
                print(f"   Detected Shapes: {result.detected_shapes}")
                print(f"   Diagram Type: {result.diagram_type}")
                print(f"   Confidence: {result.confidence:.2f}")
                
                # Check if expected shapes were detected
                expected = set(sample['expected_shapes'])
                detected = set(result.detected_shapes)
                matched = expected.intersection(detected)
                
                if matched:
                    print(f"   ✅ Matched Expected Shapes: {', '.join(matched)}")
                if expected - detected:
                    print(f"   ⚠️  Missed Shapes: {', '.join(expected - detected)}")
                if detected - expected:
                    print(f"   ➕ Additional Shapes: {', '.join(detected - expected)}")
                
                if result.coordinate_info:
                    print(f"   📍 Coordinates Found: {len(result.coordinate_info.get('points', []))} points")
                
            else:
                print(f"   ❌ Recognition Failed: {result.error}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

async def demo_batch_geometry_processing():
    """Demo batch processing of multiple geometry images."""
    
    print("\n📦 Demo 5: Batch Geometry Processing")
    print("=" * 60)
    
    # Sample batch of geometry images
    geometry_batch = [
        b"placeholder_geometry_1",
        b"placeholder_geometry_2",
        b"placeholder_geometry_3",
        b"placeholder_geometry_4"
    ]
    
    print(f"🔄 Processing {len(geometry_batch)} geometry images in batch...")
    
    try:
        processor = GeometryOCRProcessor(providers=[GeometryOCRProvider.GOT_OCR2])
        results = await processor.batch_process_geometry(
            images=geometry_batch,
            max_concurrent=2
        )
        
        successful = 0
        total_shapes = 0
        total_time = 0
        
        print(f"📊 Batch Processing Results:")
        for i, result in enumerate(results):
            if hasattr(result, 'success') and not isinstance(result, Exception):
                status = "✅" if result.success else "❌"
                shapes = len(result.detected_shapes) if result.success else 0
                time_ms = result.processing_time_ms if result.success else 0
                
                print(f"   Image {i+1}: {status} {shapes} shapes, {time_ms:.1f}ms")
                
                if result.success:
                    successful += 1
                    total_shapes += shapes
                    total_time += time_ms
                    
                    if result.detected_shapes:
                        print(f"            Shapes: {', '.join(result.detected_shapes)}")
            else:
                print(f"   Image {i+1}: ❌ Exception: {str(result)[:50]}...")
        
        print(f"\n📈 Batch Summary:")
        print(f"   Success Rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"   Total Shapes Detected: {total_shapes}")
        print(f"   Average Processing Time: {total_time/successful:.1f}ms" if successful > 0 else "   Average Processing Time: N/A")
        
    except Exception as e:
        print(f"❌ Batch Processing Error: {e}")

async def demo_geometry_ocr_integration():
    """Demo integration with the complete test analysis system."""
    
    print("\n🔗 Demo 6: Integration with Complete Test Analysis")
    print("=" * 60)
    
    print("🎯 Testing integrated geometry recognition in complete workflow...")
    
    try:
        # Create hybrid processor with geometry support
        processor = HybridImageProcessor(
            geometry_providers=[GeometryOCRProvider.GOT_OCR2],
            default_strategy=ProcessingStrategy.GEOMETRY_FIRST
        )
        
        # Sample geometry test image
        sample_image = b"placeholder_geometry_test"
        
        result = await processor.process_test_image(
            image_data=sample_image,
            strategy=ProcessingStrategy.GEOMETRY_FIRST
        )
        
        print(f"🔍 Integration Results:")
        print(f"   Overall Success: {'✅' if result.success else '❌'}")
        print(f"   Processing Strategy: {result.processing_strategy.value}")
        print(f"   Total Processing Time: {result.total_processing_time_ms:.1f}ms")
        
        # Show what components were used
        metadata = result.metadata
        components_used = []
        if metadata.get('ocr_available'):
            components_used.append("Mathpix OCR")
        if metadata.get('vision_available'):
            components_used.append("GPT-4 Vision")
        if metadata.get('geometry_available'):
            components_used.append(f"Geometry OCR ({metadata.get('geometry_provider', 'unknown')})")
        
        print(f"   Components Used: {', '.join(components_used)}")
        
        if metadata.get('geometry_available'):
            print(f"   Geometry Details:")
            print(f"     Provider: {metadata.get('geometry_provider', 'N/A')}")
            print(f"     Confidence: {metadata.get('geometry_confidence', 0):.2f}")
            print(f"     Shapes: {metadata.get('detected_shapes', [])}")
            print(f"     Diagram Type: {metadata.get('diagram_type', 'N/A')}")
        
        # Show question-answer pairs
        if result.question_answer_pairs:
            print(f"   Questions Extracted: {len(result.question_answer_pairs)}")
            for i, qa in enumerate(result.question_answer_pairs[:2]):
                print(f"     Q{i+1} ({qa.get('source', 'unknown')}): {qa.get('question_text', '')[:50]}...")
                if qa.get('problem_type'):
                    print(f"          Type: {qa['problem_type']}")
        
    except Exception as e:
        print(f"❌ Integration Error: {e}")

def print_setup_instructions():
    """Print setup instructions for geometry OCR."""
    
    print("🚀 Advanced Geometry OCR Demo")
    print("=" * 60)
    print()
    print("📋 Setup Instructions:")
    print("1. Install required dependencies:")
    print("   pip install transformers torch")
    print("   # For GOT-OCR2.0:")
    print("   # pip install 'transformers>=4.30.0' torch torchvision")
    print()
    print("2. The system will automatically download models on first use:")
    print("   • GOT-OCR2.0: ~580M parameters")
    print("   • UniMERNet: Various sizes available")
    print()
    print("3. For best performance with geometry:")
    print("   • Use GOT-OCR2.0 for complex geometric diagrams")
    print("   • Use UniMERNet for mathematical geometry proofs")
    print("   • Combine with GPT-4 Vision for layout understanding")
    print()
    print("🎯 Why These Alternatives Beat Mathpix for Geometry:")
    print("   ✅ GOT-OCR2.0: Explicitly handles geometric shapes")
    print("   ✅ Better at complex diagram structures")
    print("   ✅ Improved coordinate and measurement extraction")
    print("   ✅ Superior handling of hand-drawn geometry")
    print("   ✅ No API costs or rate limits")
    print("   ✅ Can run locally for privacy")
    print()

async def main():
    """Run all geometry OCR demos."""
    
    print_setup_instructions()
    
    try:
        await demo_geometry_ocr_providers()
        await demo_got_ocr_geometry()
        await demo_geometry_vs_standard_ocr()
        await demo_specific_geometry_types()
        await demo_batch_geometry_processing()
        await demo_geometry_ocr_integration()
        
        print("\n🎉 All geometry OCR demos completed!")
        print("\n💡 Key Takeaways:")
        print("   • GOT-OCR2.0 excels at geometric shape recognition")
        print("   • Geometry-first strategy improves accuracy for geometric content")
        print("   • These alternatives handle complex diagrams better than Mathpix")
        print("   • Integration with existing pipeline is seamless")
        print("   • Batch processing enables efficient handling of multiple images")
        print("\n🚀 Next Steps:")
        print("   • Replace Mathpix with geometry OCR for geometric content")
        print("   • Use hybrid approach: geometry OCR + GPT-4 Vision")
        print("   • Fine-tune provider selection based on content type")
        print("   • Consider local deployment for privacy and cost savings")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 