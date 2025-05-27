#!/usr/bin/env python3
"""
DETAILED FAILURE ANALYSIS
Analyzes exactly why each test case is failing to help prioritize fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_system_level_test import ComprehensiveSystemTest

def analyze_test_failures():
    """Run tests and provide detailed failure analysis"""
    
    print("🔍 DETAILED FAILURE ANALYSIS")
    print("=" * 60)
    
    # Initialize test system
    test_system = ComprehensiveSystemTest()
    
    # Run all tests
    for i, test_case in enumerate(test_system.test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST {i}: {test_case.test_id}")
        print(f"{'='*50}")
        
        # Run the test
        result = test_system.run_single_test(test_case)
        
        # Show detailed results
        print(f"📝 Description: {test_case.description}")
        print(f"👤 Student Profile: {test_case.student_profile_type}")
        print(f"❓ Question: {test_case.question}")
        print(f"✅ Correct Answer: {test_case.correct_answer}")
        print(f"❌ Student Answer: {test_case.student_answer}")
        
        print(f"\n🎯 EXPECTED RESULTS:")
        print(f"   • Concept: {test_case.expected_missing_concept}")
        print(f"   • Confidence Range: {test_case.expected_confidence_range[0]:.1%} - {test_case.expected_confidence_range[1]:.1%}")
        print(f"   • Remediation Level: {test_case.expected_remediation_level}")
        
        print(f"\n📊 ACTUAL RESULTS:")
        print(f"   • Concept: {result.diagnostic_result.missing_concept}")
        print(f"   • Confidence: {result.diagnostic_result.confidence_score:.1%}")
        print(f"   • Remediation Level: {result.diagnostic_result.remediation_level}")
        
        print(f"\n✅ VALIDATION RESULTS:")
        print(f"   • Concept Match: {'✅' if result.concept_match else '❌'}")
        print(f"   • Confidence in Range: {'✅' if result.confidence_in_range else '❌'}")
        print(f"   • Remediation Appropriate: {'✅' if result.remediation_appropriate else '❌'}")
        print(f"   • Overall Success: {'✅' if result.overall_success else '❌'}")
        
        # Detailed failure analysis
        if not result.overall_success:
            print(f"\n🚨 FAILURE ANALYSIS:")
            
            if not result.concept_match:
                print(f"   ❌ CONCEPT MISMATCH:")
                print(f"      Expected: '{test_case.expected_missing_concept}'")
                print(f"      Got: '{result.diagnostic_result.missing_concept}'")
                print(f"      Fix: Update error pattern mapping or test expectation")
            
            if not result.confidence_in_range:
                expected_min, expected_max = test_case.expected_confidence_range
                actual = result.diagnostic_result.confidence_score
                print(f"   ❌ CONFIDENCE OUT OF RANGE:")
                print(f"      Expected: {expected_min:.1%} - {expected_max:.1%}")
                print(f"      Got: {actual:.1%}")
                if actual < expected_min:
                    print(f"      Fix: Increase confidence calculation (too low by {expected_min - actual:.1%})")
                else:
                    print(f"      Fix: Decrease confidence calculation (too high by {actual - expected_max:.1%})")
            
            if not result.remediation_appropriate:
                print(f"   ❌ REMEDIATION LEVEL MISMATCH:")
                print(f"      Expected: '{test_case.expected_remediation_level}'")
                print(f"      Got: '{result.diagnostic_result.remediation_level}'")
                print(f"      Fix: Adjust remediation level assignment logic")
        
        # Show error pattern if available
        if result.diagnostic_result.error_pattern:
            print(f"\n🔍 ERROR PATTERN DETAILS:")
            print(f"   • Type: {result.diagnostic_result.error_pattern.error_type}")
            print(f"   • Frequency: {result.diagnostic_result.error_pattern.frequency:.1%}")
            print(f"   • Explanation: {result.diagnostic_result.error_pattern.explanation}")

if __name__ == "__main__":
    analyze_test_failures() 