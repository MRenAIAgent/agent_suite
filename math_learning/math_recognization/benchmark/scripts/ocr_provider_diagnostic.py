#!/usr/bin/env python3
"""
OCR Provider Diagnostic Script

This script tests each OCR provider individually to diagnose issues
and identify which providers are working correctly.
"""

import asyncio
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import importlib.util

# Add parent directory to path for imports
current_dir = Path(__file__).parent
math_recognization_root = current_dir.parent.parent
sys.path.insert(0, str(math_recognization_root))

from dotenv import load_dotenv
load_dotenv()

class OCRProviderDiagnostic:
    """Diagnostic tool for testing OCR providers individually."""
    
    def __init__(self):
        self.results = {}
        self.test_image_path = None
        self.setup_test_image()
    
    def setup_test_image(self):
        """Find a test image to use for diagnostics."""
        possible_paths = [
            "../real_datasets/ComplexGeometry/images/complex_001.jpg",
            "../real_datasets/PGDP5K/images/sample_000.jpg",
            "test_images/simple_math.jpg"
        ]
        
        for path in possible_paths:
            full_path = Path(path)
            if full_path.exists():
                self.test_image_path = full_path
                print(f"✅ Using test image: {full_path}")
                return
        
        print("⚠️  No test image found. Creating a simple test...")
        self.create_simple_test_image()
    
    def create_simple_test_image(self):
        """Create a simple test image if none exists."""
        # For now, just indicate we need a test image
        print("❌ No test images available. Please ensure ComplexGeometry dataset is present.")
        return None
    
    async def test_gpt4_vision(self) -> Dict[str, Any]:
        """Test GPT-4 Vision provider."""
        print("\n🔍 Testing GPT-4 Vision...")
        
        try:
            # Load GPT-4 Vision module
            spec = importlib.util.spec_from_file_location(
                "gpt4_vision", math_recognization_root / "gpt4_vision.py"
            )
            gpt4_vision = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gpt4_vision)
            
            # Check API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    'provider': 'GPT-4 Vision',
                    'status': 'failed',
                    'error': 'OPENAI_API_KEY not found in environment',
                    'suggestion': 'Set OPENAI_API_KEY environment variable'
                }
            
            print(f"   ✅ API key found: {api_key[:10]}...")
            
            # Initialize client
            client = gpt4_vision.GPT4VisionClient()
            print("   ✅ Client initialized")
            
            if not self.test_image_path or not self.test_image_path.exists():
                return {
                    'provider': 'GPT-4 Vision',
                    'status': 'failed',
                    'error': 'No test image available',
                    'suggestion': 'Ensure test images are present in the dataset'
                }
            
            # Load test image
            with open(self.test_image_path, 'rb') as f:
                image_bytes = f.read()
            
            print(f"   ✅ Loaded test image: {len(image_bytes)} bytes")
            
            # Test analysis
            start_time = time.time()
            prompt = "Analyze this math problem image. Identify any geometric shapes, mathematical expressions, coordinates, or text you can see."
            
            response = await client.analyze_test_image(image_bytes, prompt)
            processing_time = time.time() - start_time
            
            if response and response.success:
                return {
                    'provider': 'GPT-4 Vision',
                    'status': 'success',
                    'processing_time': processing_time,
                    'confidence': getattr(response, 'processing_confidence', 0.0),
                    'response_length': len(response.raw_response or ""),
                    'sample_response': (response.raw_response or "")[:200] + "..." if len(response.raw_response or "") > 200 else response.raw_response
                }
            else:
                return {
                    'provider': 'GPT-4 Vision',
                    'status': 'failed',
                    'error': 'API call failed or returned unsuccessful response',
                    'processing_time': processing_time,
                    'raw_response': str(response) if response else 'None'
                }
            
        except Exception as e:
            return {
                'provider': 'GPT-4 Vision',
                'status': 'failed',
                'error': str(e),
                'suggestion': 'Check module imports and API configuration'
            }
    
    async def test_mathpix_ocr(self) -> Dict[str, Any]:
        """Test Mathpix OCR provider."""
        print("\n🔍 Testing Mathpix OCR...")
        
        try:
            # Check for Mathpix credentials
            app_id = os.getenv('MATHPIX_APP_ID')
            app_key = os.getenv('MATHPIX_APP_KEY')
            
            if not app_id or not app_key:
                return {
                    'provider': 'Mathpix OCR',
                    'status': 'failed',
                    'error': 'MATHPIX_APP_ID or MATHPIX_APP_KEY not found',
                    'suggestion': 'Set Mathpix credentials in environment variables'
                }
            
            print(f"   ✅ Credentials found: {app_id[:10]}...")
            
            # Try to import and test Mathpix (if available)
            # This is a placeholder since we may not have the actual Mathpix client
            return {
                'provider': 'Mathpix OCR',
                'status': 'not_implemented',
                'error': 'Mathpix client not yet integrated',
                'suggestion': 'Implement MathpixOCRClient class'
            }
            
        except Exception as e:
            return {
                'provider': 'Mathpix OCR',
                'status': 'failed',
                'error': str(e)
            }
    
    async def test_got_ocr2(self) -> Dict[str, Any]:
        """Test GOT-OCR2.0 provider."""
        print("\n🔍 Testing GOT-OCR2.0...")
        
        try:
            # Load Advanced Geometry OCR module
            spec = importlib.util.spec_from_file_location(
                "advanced_geometry_ocr", math_recognization_root / "advanced_geometry_ocr.py"
            )
            advanced_geometry_ocr = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(advanced_geometry_ocr)
            
            print("   ✅ Module loaded")
            
            # Initialize GOT-OCR client
            client = advanced_geometry_ocr.GOTOCRClient()
            print("   ✅ Client initialized")
            
            if not self.test_image_path or not self.test_image_path.exists():
                return {
                    'provider': 'GOT-OCR2.0',
                    'status': 'failed',
                    'error': 'No test image available'
                }
            
            # Test extraction
            start_time = time.time()
            result = await client.extract_geometry(str(self.test_image_path))
            processing_time = time.time() - start_time
            
            if result and isinstance(result, dict):
                return {
                    'provider': 'GOT-OCR2.0',
                    'status': 'success',
                    'processing_time': processing_time,
                    'shapes_found': len(result.get('shapes', [])),
                    'text_found': len(result.get('text', [])),
                    'content_length': len(result.get('content', '')),
                    'sample_content': result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
                }
            else:
                return {
                    'provider': 'GOT-OCR2.0',
                    'status': 'failed',
                    'error': 'Invalid response format',
                    'raw_result': str(result)
                }
            
        except Exception as e:
            return {
                'provider': 'GOT-OCR2.0',
                'status': 'failed',
                'error': str(e),
                'suggestion': 'Check GOT-OCR2.0 installation and dependencies'
            }
    
    async def test_unimernet(self) -> Dict[str, Any]:
        """Test UniMERNet provider."""
        print("\n🔍 Testing UniMERNet...")
        
        try:
            # Check if UniMER dataset/model is available
            unimer_path = Path("../datasets/unimer")
            if not unimer_path.exists():
                return {
                    'provider': 'UniMERNet',
                    'status': 'not_available',
                    'error': 'UniMER dataset not found',
                    'suggestion': 'Download UniMER dataset first'
                }
            
            # This would require actual UniMERNet model integration
            return {
                'provider': 'UniMERNet',
                'status': 'not_implemented',
                'error': 'UniMERNet client not yet integrated',
                'suggestion': 'Implement UniMERNet inference client'
            }
            
        except Exception as e:
            return {
                'provider': 'UniMERNet',
                'status': 'failed',
                'error': str(e)
            }
    
    async def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run diagnostic tests on all OCR providers."""
        print("🚀 STARTING OCR PROVIDER DIAGNOSTIC")
        print("=" * 60)
        
        if not self.test_image_path:
            print("❌ No test image available. Cannot run diagnostics.")
            return {'error': 'No test image available'}
        
        print(f"📸 Test image: {self.test_image_path}")
        print(f"📏 Image size: {self.test_image_path.stat().st_size} bytes")
        
        # Test each provider
        tests = [
            self.test_gpt4_vision(),
            self.test_mathpix_ocr(),
            self.test_got_ocr2(),
            self.test_unimernet()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        diagnostic_results = {
            'test_image': str(self.test_image_path),
            'test_timestamp': time.time(),
            'providers': {}
        }
        
        working_providers = []
        failed_providers = []
        
        for result in results:
            if isinstance(result, Exception):
                provider_name = "Unknown"
                result = {
                    'provider': provider_name,
                    'status': 'exception',
                    'error': str(result)
                }
            
            provider_name = result['provider']
            diagnostic_results['providers'][provider_name] = result
            
            if result['status'] == 'success':
                working_providers.append(provider_name)
            else:
                failed_providers.append(provider_name)
        
        # Summary
        diagnostic_results['summary'] = {
            'total_providers_tested': len(results),
            'working_providers': working_providers,
            'failed_providers': failed_providers,
            'success_rate': len(working_providers) / len(results) if results else 0
        }
        
        return diagnostic_results
    
    def print_diagnostic_report(self, results: Dict[str, Any]):
        """Print a formatted diagnostic report."""
        print("\n" + "=" * 60)
        print("📊 OCR PROVIDER DIAGNOSTIC REPORT")
        print("=" * 60)
        
        summary = results.get('summary', {})
        print(f"📈 Success Rate: {summary.get('success_rate', 0):.1%}")
        print(f"✅ Working Providers: {len(summary.get('working_providers', []))}")
        print(f"❌ Failed Providers: {len(summary.get('failed_providers', []))}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        print("-" * 40)
        
        for provider_name, result in results.get('providers', {}).items():
            status = result['status']
            if status == 'success':
                print(f"✅ {provider_name}")
                print(f"   ⏱️  Processing Time: {result.get('processing_time', 0):.2f}s")
                if 'confidence' in result:
                    print(f"   🎯 Confidence: {result['confidence']:.2f}")
                if 'sample_response' in result:
                    print(f"   📝 Sample: {result['sample_response'][:100]}...")
                elif 'sample_content' in result:
                    print(f"   📝 Sample: {result['sample_content'][:100]}...")
            else:
                print(f"❌ {provider_name}")
                print(f"   🚫 Status: {status}")
                print(f"   ⚠️  Error: {result.get('error', 'Unknown error')}")
                if 'suggestion' in result:
                    print(f"   💡 Suggestion: {result['suggestion']}")
            print()
        
        # Recommendations
        working = summary.get('working_providers', [])
        if working:
            print("🚀 RECOMMENDATIONS:")
            print(f"   • Use {working[0]} as primary OCR provider")
            if len(working) > 1:
                print(f"   • Use {working[1]} as fallback provider")
            print(f"   • Focus benchmarking efforts on working providers")
        else:
            print("⚠️  CRITICAL: No OCR providers are working!")
            print("   • Check API credentials and environment variables")
            print("   • Verify module imports and dependencies")
            print("   • Ensure test images are available")

async def main():
    """Main function to run the diagnostic."""
    diagnostic = OCRProviderDiagnostic()
    results = await diagnostic.run_full_diagnostic()
    
    # Save results
    timestamp = int(time.time())
    results_file = f"ocr_diagnostic_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print report
    diagnostic.print_diagnostic_report(results)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Diagnostic interrupted by user")
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}") 