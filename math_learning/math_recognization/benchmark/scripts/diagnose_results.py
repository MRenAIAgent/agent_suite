#!/usr/bin/env python3
"""
Results Diagnostic Tool

Analyzes why API results are poor and shows what the APIs are actually seeing.
"""

import json
import os
import sys
from pathlib import Path
import glob

# Setup paths
current_dir = Path(__file__).parent
math_recognization_root = current_dir.parent.parent
sys.path.insert(0, str(math_recognization_root))

def load_latest_results():
    """Load the most recent API results"""
    result_files = glob.glob("real_api_results_*.json")
    
    if not result_files:
        print("❌ No API results found")
        return None
    
    latest_file = max(result_files, key=os.path.getctime)
    print(f"📊 Analyzing: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def load_expected_data():
    """Load expected annotations for comparison"""
    annotations_file = "../real_datasets/PGDP5K/processed/test_benchmark.json"
    
    try:
        with open(annotations_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading expected data: {e}")
        return []

def analyze_failure_reasons(results, expected_data):
    """Analyze why the APIs are failing"""
    
    print("\n🔍 FAILURE ANALYSIS")
    print("=" * 50)
    
    failure_reasons = {
        "image_quality": 0,
        "api_confusion": 0,
        "wrong_prompt": 0,
        "missing_content": 0
    }
    
    for i, (result, expected) in enumerate(zip(results['individual_results'], expected_data)):
        print(f"\n📝 Sample {result['id']}:")
        print(f"   Expected shapes: {expected.get('expected_shapes', [])}")
        print(f"   Expected text: '{expected.get('expected_text', '')}'")
        print(f"   Detected shapes: {result.get('detected_shapes', [])}")
        print(f"   API Response: '{result['extracted_text'][:100]}...'")
        
        # Analyze failure type
        api_response = result['extracted_text'].lower()
        
        if "unable to analyze" in api_response or "can't" in api_response:
            failure_reasons["api_confusion"] += 1
            print("   🔴 ISSUE: API says it can't analyze the image")
            
        elif "unclear" in api_response or "low quality" in api_response:
            failure_reasons["image_quality"] += 1
            print("   🔴 ISSUE: API reports image quality problems")
            
        elif "no visible text" in api_response or "no math test" in api_response:
            failure_reasons["missing_content"] += 1
            print("   🔴 ISSUE: API doesn't see expected content")
            
        else:
            failure_reasons["wrong_prompt"] += 1
            print("   🔴 ISSUE: API prompt may be wrong for geometry")
    
    print(f"\n📊 FAILURE BREAKDOWN:")
    for reason, count in failure_reasons.items():
        if count > 0:
            print(f"   {reason.replace('_', ' ').title()}: {count} cases")
    
    return failure_reasons

def show_image_info():
    """Show information about the test images"""
    
    print(f"\n🖼️  IMAGE ANALYSIS")
    print("=" * 30)
    
    image_dir = "../real_datasets/PGDP5K/images/"
    
    if not os.path.exists(image_dir):
        print("❌ Image directory not found")
        return
    
    images = glob.glob(f"{image_dir}sample_*.jpg")
    
    for image_path in sorted(images):
        filename = os.path.basename(image_path)
        size = os.path.getsize(image_path)
        
        print(f"📄 {filename}: {size:,} bytes")
        
        # Try to get image dimensions
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                print(f"   Dimensions: {img.size[0]}x{img.size[1]}")
                print(f"   Mode: {img.mode}")
        except Exception as e:
            print(f"   Error reading: {e}")

def suggest_fixes(failure_reasons):
    """Suggest fixes based on failure analysis"""
    
    print(f"\n💡 SUGGESTED FIXES")
    print("=" * 25)
    
    if failure_reasons["api_confusion"] > 0:
        print("1. 🔧 Fix API Prompt:")
        print("   - Current prompt expects 'math test with student answers'")
        print("   - Our images are 'geometry diagrams with labels'") 
        print("   - Need geometry-specific prompt")
        
    if failure_reasons["missing_content"] > 0:
        print("2. 🖼️  Image Content Issue:")
        print("   - API expects complex math problems")
        print("   - Our sample images are simple geometric shapes")
        print("   - Need more complex geometry problems")
        
    if failure_reasons["image_quality"] > 0:
        print("3. 📸 Image Quality:")
        print("   - Images may be too simple/synthetic looking")
        print("   - Try with real textbook scans")
        
    print("\n🚀 IMMEDIATE ACTIONS:")
    print("1. Update GPT-4 Vision prompt for geometry recognition")
    print("2. Test with more complex geometry images")
    print("3. Try GOT-OCR2.0 instead of GPT-4 Vision for geometry")

def main():
    """Main diagnostic function"""
    
    print("🔍 API RESULTS DIAGNOSTIC")
    print("=" * 40)
    
    # Load data
    results = load_latest_results()
    if not results:
        return
        
    expected_data = load_expected_data()
    if not expected_data:
        return
    
    print(f"📊 Results Summary:")
    print(f"   Overall Accuracy: {results['metrics']['overall_accuracy']}%")
    print(f"   Success Rate: {results['metrics']['success_rate']}%")
    print(f"   Shape Detection: {results['metrics']['shape_detection_accuracy']}%")
    print(f"   Text Recognition: {results['metrics']['text_recognition_accuracy']}%")
    
    # Analyze failures
    failure_reasons = analyze_failure_reasons(results, expected_data)
    
    # Show image info
    show_image_info()
    
    # Suggest fixes
    suggest_fixes(failure_reasons)
    
    print(f"\n🎯 CONCLUSION:")
    print("The low accuracy is due to prompt mismatch - GPT-4 Vision")
    print("expects 'math test analysis' but we're giving it 'geometry diagrams'")

if __name__ == "__main__":
    main() 