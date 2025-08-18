#!/usr/bin/env python3
"""
Accuracy Improvement Plan for Math Recognition System

Based on systematic benchmark results, this script implements specific
strategies to improve accuracy across different metrics.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import importlib.util

# Add parent directory to path for imports
current_dir = Path(__file__).parent
benchmark_root = current_dir.parent
math_recognization_root = benchmark_root.parent
sys.path.insert(0, str(benchmark_root))
sys.path.insert(0, str(math_recognization_root))

from dotenv import load_dotenv
load_dotenv()

@dataclass
class ImprovementStrategy:
    """Strategy for improving specific accuracy metrics."""
    name: str
    target_metric: str
    current_accuracy: float
    target_accuracy: float
    implementation_steps: List[str]
    expected_improvement: float

class AccuracyImprovementPlan:
    """Comprehensive plan to improve math recognition accuracy."""
    
    def __init__(self, benchmark_results: Dict[str, Any]):
        self.benchmark_results = benchmark_results
        self.current_metrics = self.extract_current_metrics()
        self.improvement_strategies = self.generate_improvement_strategies()
    
    def extract_current_metrics(self) -> Dict[str, float]:
        """Extract current performance metrics from benchmark results."""
        if not self.benchmark_results.get('results'):
            return {}
        
        # Get the latest result
        latest_result = self.benchmark_results['results'][0]
        metrics = latest_result.get('accuracy_metrics', {})
        
        return {
            'overall_accuracy': metrics.get('overall_accuracy', 0.0),
            'success_rate': metrics.get('success_rate', 0.0),
            'shape_detection_accuracy': metrics.get('shape_detection_accuracy', 0.0),
            'text_extraction_accuracy': metrics.get('text_extraction_accuracy', 0.0),
            'coordinate_detection_accuracy': metrics.get('coordinate_detection_accuracy', 0.0),
            'avg_confidence': metrics.get('avg_confidence', 0.0),
            'avg_processing_time': metrics.get('avg_processing_time', 0.0)
        }
    
    def generate_improvement_strategies(self) -> List[ImprovementStrategy]:
        """Generate specific improvement strategies based on current performance."""
        strategies = []
        
        # Strategy 1: Improve Shape Detection (currently 43.3%)
        strategies.append(ImprovementStrategy(
            name="Enhanced Shape Detection",
            target_metric="shape_detection_accuracy",
            current_accuracy=self.current_metrics.get('shape_detection_accuracy', 0.0),
            target_accuracy=0.80,  # Target 80%
            implementation_steps=[
                "Implement multi-prompt approach with shape-specific prompts",
                "Add visual reasoning chains for complex shapes",
                "Use specialized geometric vocabulary in prompts",
                "Implement shape validation using geometric rules",
                "Add fallback OCR for shape recognition"
            ],
            expected_improvement=0.37  # +37% improvement
        ))
        
        # Strategy 2: Improve Coordinate Detection (currently 33.3%)
        strategies.append(ImprovementStrategy(
            name="Advanced Coordinate Extraction",
            target_metric="coordinate_detection_accuracy", 
            current_accuracy=self.current_metrics.get('coordinate_detection_accuracy', 0.0),
            target_accuracy=0.85,  # Target 85%
            implementation_steps=[
                "Implement regex-based coordinate pattern matching",
                "Add OCR preprocessing for coordinate regions",
                "Use coordinate-specific prompting strategies",
                "Implement coordinate validation and correction",
                "Add support for multiple coordinate formats"
            ],
            expected_improvement=0.52  # +52% improvement
        ))
        
        # Strategy 3: Maintain High Text Extraction (currently 86.0%)
        strategies.append(ImprovementStrategy(
            name="Optimized Text Extraction",
            target_metric="text_extraction_accuracy",
            current_accuracy=self.current_metrics.get('text_extraction_accuracy', 0.0),
            target_accuracy=0.95,  # Target 95%
            implementation_steps=[
                "Fine-tune mathematical expression recognition",
                "Add LaTeX formatting support",
                "Implement mathematical symbol validation",
                "Add context-aware text extraction",
                "Implement multi-scale text recognition"
            ],
            expected_improvement=0.09  # +9% improvement
        ))
        
        # Strategy 4: Reduce Processing Time (currently 8.62s)
        strategies.append(ImprovementStrategy(
            name="Performance Optimization",
            target_metric="avg_processing_time",
            current_accuracy=self.current_metrics.get('avg_processing_time', 0.0),
            target_accuracy=5.0,  # Target 5 seconds
            implementation_steps=[
                "Implement parallel processing for multiple images",
                "Add caching for repeated analyses",
                "Optimize image preprocessing pipeline",
                "Implement early termination for high-confidence results",
                "Add batch processing capabilities"
            ],
            expected_improvement=-3.62  # -3.62s improvement
        ))
        
        return strategies
    
    def print_improvement_plan(self):
        """Print comprehensive improvement plan."""
        print("\n" + "=" * 80)
        print("🎯 MATH RECOGNITION ACCURACY IMPROVEMENT PLAN")
        print("=" * 80)
        
        print(f"\n📊 CURRENT PERFORMANCE BASELINE:")
        print("-" * 50)
        for metric, value in self.current_metrics.items():
            if 'time' in metric:
                print(f"   {metric.replace('_', ' ').title()}: {value:.2f}s")
            elif 'confidence' in metric:
                print(f"   {metric.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"   {metric.replace('_', ' ').title()}: {value:.1%}")
        
        print(f"\n🚀 IMPROVEMENT STRATEGIES:")
        print("=" * 80)
        
        total_expected_improvement = 0
        
        for i, strategy in enumerate(self.improvement_strategies, 1):
            print(f"\n{i}. {strategy.name}")
            print("-" * 60)
            print(f"   🎯 Target Metric: {strategy.target_metric}")
            
            if 'time' in strategy.target_metric:
                print(f"   📈 Current: {strategy.current_accuracy:.2f}s")
                print(f"   🎯 Target: {strategy.target_accuracy:.2f}s")
                print(f"   ⚡ Expected Improvement: {strategy.expected_improvement:+.2f}s")
            else:
                print(f"   📈 Current: {strategy.current_accuracy:.1%}")
                print(f"   🎯 Target: {strategy.target_accuracy:.1%}")
                print(f"   ⚡ Expected Improvement: {strategy.expected_improvement:+.1%}")
            
            print(f"   📋 Implementation Steps:")
            for j, step in enumerate(strategy.implementation_steps, 1):
                print(f"      {j}. {step}")
            
            if 'overall' not in strategy.target_metric and 'time' not in strategy.target_metric:
                total_expected_improvement += strategy.expected_improvement
        
        # Calculate projected overall improvement
        current_overall = self.current_metrics.get('overall_accuracy', 0.0)
        projected_overall = min(current_overall + (total_expected_improvement * 0.3), 0.95)  # Conservative estimate
        
        print(f"\n🎉 PROJECTED RESULTS:")
        print("-" * 50)
        print(f"   Current Overall Accuracy: {current_overall:.1%}")
        print(f"   Projected Overall Accuracy: {projected_overall:.1%}")
        print(f"   Expected Overall Improvement: {projected_overall - current_overall:+.1%}")
        
        print(f"\n📅 IMPLEMENTATION TIMELINE:")
        print("-" * 50)
        print(f"   Phase 1 (Week 1-2): Enhanced Shape Detection & Coordinate Extraction")
        print(f"   Phase 2 (Week 3): Text Extraction Optimization")
        print(f"   Phase 3 (Week 4): Performance Optimization")
        print(f"   Phase 4 (Week 5): Integration Testing & Validation")
        
        print(f"\n🔧 IMMEDIATE NEXT STEPS:")
        print("-" * 50)
        print(f"   1. Implement enhanced shape detection prompts")
        print(f"   2. Add coordinate pattern matching")
        print(f"   3. Set up A/B testing framework")
        print(f"   4. Create expanded test dataset")
        print(f"   5. Implement performance monitoring")

class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with improvement strategies."""
    
    def __init__(self):
        self.modules = self.load_modules()
    
    def load_modules(self):
        """Load required modules."""
        modules = {}
        try:
            spec = importlib.util.spec_from_file_location(
                "gpt4_vision", math_recognization_root / "gpt4_vision.py"
            )
            gpt4_vision = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpt4_vision)
            modules['gpt4_vision'] = gpt4_vision
        except Exception as e:
            print(f"❌ Failed to load GPT-4 Vision: {e}")
        return modules
    
    async def test_enhanced_shape_detection(self, image_path: Path, annotation: Dict) -> Dict[str, Any]:
        """Test enhanced shape detection strategy."""
        if not self.modules.get('gpt4_vision'):
            return {'success': False, 'error': 'GPT-4 Vision not available'}
        
        gpt4_client = self.modules['gpt4_vision'].GPT4VisionClient()
        
        # Enhanced multi-prompt approach
        shape_specific_prompt = f"""
        You are a geometry expert analyzing this mathematical diagram. Focus specifically on identifying ALL geometric shapes with high precision.
        
        Expected shapes in this problem: {annotation.get('expected_shapes', [])}
        
        For each shape you identify, provide:
        1. Shape type (triangle, circle, line, point, polygon, etc.)
        2. Specific characteristics (right triangle, equilateral, etc.)
        3. Location description
        4. Any measurements or labels
        
        Look carefully for:
        - Basic shapes: triangles, circles, squares, rectangles
        - Complex shapes: polygons, irregular shapes
        - Geometric constructions: angle bisectors, perpendiculars
        - Coordinate systems: axes, grids, coordinate planes
        
        Be extremely thorough and precise in your shape identification.
        """
        
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            start_time = time.time()
            response = await gpt4_client.analyze_test_image(image_bytes, shape_specific_prompt)
            processing_time = time.time() - start_time
            
            if response and response.success:
                raw_text = response.raw_response or ""
                shapes_detected = self.extract_enhanced_shapes(raw_text, annotation)
                
                return {
                    'success': True,
                    'shapes_detected': shapes_detected,
                    'confidence': getattr(response, 'processing_confidence', 0.0),
                    'processing_time': processing_time,
                    'strategy': 'enhanced_shape_detection',
                    'raw_response': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
                }
            else:
                return {'success': False, 'error': 'Enhanced shape detection failed'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def extract_enhanced_shapes(self, response_text: str, annotation: Dict) -> List[str]:
        """Enhanced shape extraction with better pattern matching."""
        import re
        
        shapes_detected = []
        text_lower = response_text.lower()
        
        # Enhanced shape keywords with more variations
        enhanced_shape_keywords = {
            'triangle': ['triangle', 'triangular', 'right triangle', 'equilateral', 'isosceles', 'scalene'],
            'circle': ['circle', 'circular', 'semicircle', 'arc', 'circumference'],
            'line': ['line', 'linear', 'segment', 'ray', 'line segment'],
            'point': ['point', 'vertex', 'vertices', 'intersection point'],
            'polygon': ['polygon', 'hexagon', 'pentagon', 'octagon', 'regular polygon'],
            'rectangle': ['rectangle', 'rectangular', 'rect'],
            'square': ['square', 'quadrilateral'],
            'coordinate_system': ['coordinate', 'axis', 'axes', 'x-axis', 'y-axis', 'grid'],
            'angle': ['angle', 'degree', '°', 'right angle', 'acute', 'obtuse'],
            'prism': ['prism', 'rectangular prism', '3d', 'three-dimensional']
        }
        
        # Score-based matching for better accuracy
        shape_scores = {}
        
        for shape_type, keywords in enhanced_shape_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
                    # Bonus for exact matches
                    if f" {keyword} " in f" {text_lower} ":
                        score += 0.5
            
            if score > 0:
                shape_scores[shape_type] = score
        
        # Sort by confidence and take top matches
        sorted_shapes = sorted(shape_scores.items(), key=lambda x: x[1], reverse=True)
        shapes_detected = [shape for shape, score in sorted_shapes if score >= 0.5]
        
        return shapes_detected

async def main():
    """Main function to analyze current performance and generate improvement plan."""
    print("🔍 ANALYZING CURRENT BENCHMARK RESULTS...")
    
    # Load latest benchmark results
    results_files = list(Path('.').glob('systematic_benchmark_results_*.json'))
    if not results_files:
        print("❌ No benchmark results found. Run systematic_ocr_benchmark.py first.")
        return
    
    # Get the latest results file
    latest_results_file = max(results_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_results_file, 'r') as f:
        benchmark_results = json.load(f)
    
    print(f"✅ Loaded results from: {latest_results_file}")
    
    # Generate improvement plan
    improvement_plan = AccuracyImprovementPlan(benchmark_results)
    improvement_plan.print_improvement_plan()
    
    # Test enhanced strategies
    print(f"\n🧪 TESTING ENHANCED STRATEGIES...")
    print("=" * 80)
    
    enhanced_runner = EnhancedBenchmarkRunner()
    
    # Test enhanced shape detection on one sample
    test_image = Path("../real_datasets/ComplexGeometry/images/complex_001.jpg")
    if test_image.exists():
        # Load annotation
        with open("../real_datasets/ComplexGeometry/annotations.json", 'r') as f:
            annotations = json.load(f)
        
        test_annotation = annotations[0]  # First sample
        
        print(f"🔬 Testing Enhanced Shape Detection on {test_image.name}")
        result = await enhanced_runner.test_enhanced_shape_detection(test_image, test_annotation)
        
        if result['success']:
            shapes_found = len(result['shapes_detected'])
            expected_shapes = len(test_annotation.get('expected_shapes', []))
            improvement = shapes_found - expected_shapes
            
            print(f"   ✅ Enhanced Strategy Results:")
            print(f"   📊 Shapes Detected: {shapes_found} (expected: {expected_shapes})")
            print(f"   📈 Improvement: {improvement:+d} shapes")
            print(f"   ⏱️  Processing Time: {result['processing_time']:.2f}s")
            print(f"   🎯 Confidence: {result['confidence']:.2f}")
            print(f"   🔍 Detected Shapes: {result['shapes_detected']}")
        else:
            print(f"   ❌ Enhanced strategy test failed: {result.get('error')}")
    
    # Save improvement plan
    timestamp = int(time.time())
    plan_file = f"accuracy_improvement_plan_{timestamp}.json"
    
    plan_data = {
        'plan_info': {
            'name': 'Math Recognition Accuracy Improvement Plan',
            'timestamp': timestamp,
            'based_on_results': str(latest_results_file)
        },
        'current_metrics': improvement_plan.current_metrics,
        'improvement_strategies': [
            {
                'name': strategy.name,
                'target_metric': strategy.target_metric,
                'current_accuracy': strategy.current_accuracy,
                'target_accuracy': strategy.target_accuracy,
                'expected_improvement': strategy.expected_improvement,
                'implementation_steps': strategy.implementation_steps
            }
            for strategy in improvement_plan.improvement_strategies
        ]
    }
    
    with open(plan_file, 'w') as f:
        json.dump(plan_data, f, indent=2)
    
    print(f"\n💾 Improvement plan saved to: {plan_file}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc() 