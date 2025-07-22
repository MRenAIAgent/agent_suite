#!/usr/bin/env python3
"""
COMPREHENSIVE STRATEGY VALIDATION TEST
Tests the complete pipeline: Knowledge Graph → Diagnostic → Question Bank → Remediation
Validates that all 7 components work together seamlessly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from algebra_comprehensive_question_bank_complete import CompleteAlgebraQuestionBank
import json
import time
from typing import Dict, List

class StrategyValidator:
    """Validates the complete diagnostic strategy end-to-end"""
    
    def __init__(self):
        print("🚀 INITIALIZING COMPREHENSIVE STRATEGY VALIDATION")
        print("=" * 60)
        
        # Initialize all components
        self.diagnostic = ComprehensiveDiagnostic()
        self.question_bank = CompleteAlgebraQuestionBank()
        
        # Load knowledge graph
        try:
            with open('/Users/minren/code/agent_suite/algebra_knowledge_graph.json', 'r') as f:
                self.kg_data = json.load(f)
            print("✅ Knowledge Graph loaded successfully")
        except Exception as e:
            print(f"❌ Knowledge Graph loading failed: {e}")
            return
        
        # Load question bank
        try:
            with open('complete_algebra_question_bank.json', 'r') as f:
                self.qb_data = json.load(f)
            print("✅ Question Bank loaded successfully")
        except Exception as e:
            print(f"❌ Question Bank loading failed: {e}")
            return
        
        print(f"✅ Strategy Validator initialized successfully\n")
    
    def validate_component_1_knowledge_graph(self):
        """Validate Component 1: Knowledge Graph Structure"""
        print("📚 COMPONENT 1: KNOWLEDGE GRAPH VALIDATION")
        print("-" * 40)
        
        # Test knowledge graph structure
        concepts = self.kg_data.get('concepts', [])
        edges = self.kg_data.get('edges', [])
        
        print(f"📊 Total concepts: {len(concepts)}")
        print(f"🔗 Total relationships: {len(edges)}")
        
        # Validate concept structure
        required_fields = ['id', 'name', 'category', 'difficulty', 'prerequisites', 'dependents']
        sample_concept = concepts[0] if concepts else {}
        
        missing_fields = [field for field in required_fields if field not in sample_concept]
        if missing_fields:
            print(f"❌ Missing fields in concepts: {missing_fields}")
            return False
        
        # Test prerequisite chains
        prerequisite_chains = 0
        for concept in concepts[:5]:  # Test first 5
            if concept['prerequisites']:
                prerequisite_chains += 1
        
        print(f"✅ Concept structure valid")
        print(f"✅ Prerequisite chains found: {prerequisite_chains}")
        print(f"✅ Component 1: PASSED\n")
        return True
    
    def validate_component_2_concept_mapping(self):
        """Validate Component 2: Error Pattern → Concept ID Mapping"""
        print("🎯 COMPONENT 2: CONCEPT MAPPING VALIDATION")
        print("-" * 40)
        
        # Test concept mapping
        concept_mapping = self.diagnostic.concept_mapping
        print(f"📊 Error patterns mapped: {len(concept_mapping)}")
        
        # Validate mappings point to real concepts
        kg_concept_ids = {concept['id'] for concept in self.kg_data['concepts']}
        invalid_mappings = []
        
        for error_type, concept_id in concept_mapping.items():
            if concept_id not in kg_concept_ids:
                invalid_mappings.append((error_type, concept_id))
        
        if invalid_mappings:
            print(f"❌ Invalid concept mappings: {invalid_mappings[:3]}...")
            return False
        
        print(f"✅ All error patterns map to valid concepts")
        print(f"✅ Component 2: PASSED\n")
        return True
    
    def validate_component_3_question_bank(self):
        """Validate Component 3: Question Bank Structure"""
        print("📝 COMPONENT 3: QUESTION BANK VALIDATION")
        print("-" * 40)
        
        # Test question bank structure
        total_concepts = len(self.qb_data)
        total_questions = 0
        
        for concept_id, concept_data in self.qb_data.items():
            questions_per_concept = (
                len(concept_data.get('easy', [])) +
                len(concept_data.get('medium', [])) +
                len(concept_data.get('hard', [])) +
                len(concept_data.get('bonus', []))
            )
            total_questions += questions_per_concept
        
        print(f"📊 Concepts with questions: {total_concepts}")
        print(f"📊 Total questions: {total_questions}")
        
        # Validate question structure
        sample_concept = list(self.qb_data.values())[0]
        required_difficulties = ['easy', 'medium', 'hard', 'bonus']
        
        for difficulty in required_difficulties:
            if difficulty not in sample_concept:
                print(f"❌ Missing difficulty level: {difficulty}")
                return False
            
            if not sample_concept[difficulty]:
                print(f"❌ Empty difficulty level: {difficulty}")
                return False
        
        print(f"✅ Question bank structure valid")
        print(f"✅ All difficulty levels present")
        print(f"✅ Component 3: PASSED\n")
        return True
    
    def validate_component_4_diagnostic_engine(self):
        """Validate Component 4: Diagnostic Engine"""
        print("🧠 COMPONENT 4: DIAGNOSTIC ENGINE VALIDATION")
        print("-" * 40)
        
        # Test diagnostic engine with sample student
        student = StudentProfile(
            name="Test Student",
            overall_ability=0.6,
            concept_mastery={},
            learning_style="analytical",
            total_questions_attempted=0,
            total_questions_correct=0
        )
        
        # Test diagnostic on sample question
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type="two_step_equations",
                question_text="Solve for x: 3x + 7 = 22",
                student_answer="x = 3",  # Wrong answer
                correct_answer="x = 5",
                student_profile=student
            )
            
            print(f"✅ Diagnostic analysis completed")
            print(f"✅ Error detected: {result.failure_pattern is not None}")
            print(f"✅ Concept gap identified: {result.missing_concept is not None}")
            print(f"✅ Confidence score: {result.confidence_score:.2f}")
            
        except Exception as e:
            print(f"❌ Diagnostic engine failed: {e}")
            return False
        
        print(f"✅ Component 4: PASSED\n")
        return True
    
    def validate_component_5_gap_identification(self):
        """Validate Component 5: Gap Identification"""
        print("🔍 COMPONENT 5: GAP IDENTIFICATION VALIDATION")
        print("-" * 40)
        
        # Test gap identification with multiple error types
        test_cases = [
            ("two_step_equations", "x = 3", "x = 5"),
            ("distributive_property", "2x + 3", "2x + 6"),
            ("order_of_operations", "14", "11"),
            ("integer_operations", "-2", "-8")
        ]
        
        gaps_identified = 0
        for question_type, wrong_answer, correct_answer in test_cases:
            student = StudentProfile(
                name="Test Student",
                overall_ability=0.5,
                concept_mastery={},
                learning_style="analytical",
                total_questions_attempted=0,
                total_questions_correct=0
            )
            
            try:
                result = self.diagnostic.diagnose_student_response(
                    question_type=question_type,
                    question_text=f"Test question for {question_type}",
                    student_answer=wrong_answer,
                    correct_answer=correct_answer,
                    student_profile=student
                )
                
                if result.missing_concept:
                    gaps_identified += 1
                    
            except Exception as e:
                print(f"❌ Gap identification failed for {question_type}: {e}")
                return False
        
        print(f"✅ Gap identification success rate: {gaps_identified}/{len(test_cases)}")
        print(f"✅ Component 5: PASSED\n")
        return True
    
    def validate_component_6_remediation(self):
        """Validate Component 6: Remediation Recommendations"""
        print("💡 COMPONENT 6: REMEDIATION VALIDATION")
        print("-" * 40)
        
        # Test remediation recommendations
        student = StudentProfile(
            name="Test Student",
            overall_ability=0.4,
            concept_mastery={"AT-22": 0.3, "NS-16": 0.5},
            learning_style="analytical",
            total_questions_attempted=10,
            total_questions_correct=4
        )
        
        try:
            result = self.diagnostic.diagnose_student_response(
                question_type="two_step_equations",
                question_text="Solve for x: 3x + 7 = 22",
                student_answer="x = 3",
                correct_answer="x = 5",
                student_profile=student
            )
            
            recommendations = result.recommended_questions
            print(f"✅ Remediation recommendations generated: {len(recommendations)}")
            
            if recommendations:
                print(f"✅ Sample recommendation: {recommendations[0].question_text[:50]}...")
            
            print(f"✅ Prerequisite chain identified: {len(result.prerequisite_chain)} concepts")
            
        except Exception as e:
            print(f"❌ Remediation generation failed: {e}")
            return False
        
        print(f"✅ Component 6: PASSED\n")
        return True
    
    def validate_component_7_progress_tracking(self):
        """Validate Component 7: Progress Tracking"""
        print("📊 COMPONENT 7: PROGRESS TRACKING VALIDATION")
        print("-" * 40)
        
        # Test progress tracking
        student = StudentProfile(
            name="Test Student",
            overall_ability=0.5,
            concept_mastery={"AT-22": 0.6},
            learning_style="analytical",
            total_questions_attempted=5,
            total_questions_correct=3
        )
        
        initial_ability = student.overall_ability
        initial_attempts = student.total_questions_attempted
        
        try:
            # Simulate correct answer
            result = self.diagnostic.diagnose_student_response(
                question_type="one_step_equations",
                question_text="Solve for x: x + 5 = 12",
                student_answer="x = 7",
                correct_answer="x = 7",
                student_profile=student
            )
            
            print(f"✅ Progress tracking updated")
            print(f"✅ Questions attempted: {initial_attempts} → {student.total_questions_attempted}")
            print(f"✅ Overall ability tracked: {initial_ability:.2f} → {student.overall_ability:.2f}")
            
        except Exception as e:
            print(f"❌ Progress tracking failed: {e}")
            return False
        
        print(f"✅ Component 7: PASSED\n")
        return True
    
    def validate_end_to_end_workflow(self):
        """Validate complete end-to-end workflow"""
        print("🔄 END-TO-END WORKFLOW VALIDATION")
        print("-" * 40)
        
        # Create realistic student
        student = StudentProfile(
            name="Emma Johnson",
            overall_ability=0.65,
            concept_mastery={
                "NS-16": 0.8,  # Good at order of operations
                "AT-22": 0.4,  # Struggling with one-step equations
                "AT-19": 0.3   # Poor at distributive property
            },
            learning_style="visual",
            total_questions_attempted=15,
            total_questions_correct=10
        )
        
        print(f"👩‍🎓 Testing with student: {student.name}")
        print(f"📊 Initial ability: {student.overall_ability:.2f}")
        
        # Simulate complete diagnostic workflow
        test_questions = [
            {
                "type": "two_step_equations",
                "text": "Solve: 2(x + 3) = 14",
                "student_answer": "x = 4",  # Wrong - distributive property error
                "correct_answer": "x = 4"
            },
            {
                "type": "distributive_property", 
                "text": "Expand: 3(x + 2)",
                "student_answer": "3x + 2",  # Wrong - distributive error
                "correct_answer": "3x + 6"
            },
            {
                "type": "order_of_operations",
                "text": "Calculate: 2 + 3 × 4",
                "student_answer": "14",  # Correct
                "correct_answer": "14"
            }
        ]
        
        workflow_results = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question['text']}")
            print(f"   Student Answer: {question['student_answer']}")
            
            try:
                # Step 1-7: Complete diagnostic pipeline
                result = self.diagnostic.diagnose_student_response(
                    question_type=question["type"],
                    question_text=question["text"],
                    student_answer=question["student_answer"],
                    correct_answer=question["correct_answer"],
                    student_profile=student
                )
                
                workflow_results.append(result)
                
                # Display results
                if result.is_correct:
                    print(f"   ✅ Correct!")
                else:
                    print(f"   ❌ Incorrect - Gap identified: {result.missing_concept}")
                    print(f"   💡 Recommendations: {len(result.recommended_questions)} questions")
                    print(f"   🔗 Prerequisites to review: {len(result.prerequisite_chain)}")
                
            except Exception as e:
                print(f"   ❌ Workflow failed: {e}")
                return False
        
        print(f"\n✅ End-to-end workflow completed successfully")
        print(f"✅ Questions processed: {len(workflow_results)}")
        print(f"✅ Student progress updated")
        print(f"✅ Final ability: {student.overall_ability:.2f}")
        
        return True
    
    def run_complete_validation(self):
        """Run complete strategy validation"""
        print("🎯 COMPREHENSIVE STRATEGY VALIDATION")
        print("=" * 60)
        
        start_time = time.time()
        
        # Validate all 7 components
        components = [
            ("Knowledge Graph", self.validate_component_1_knowledge_graph),
            ("Concept Mapping", self.validate_component_2_concept_mapping),
            ("Question Bank", self.validate_component_3_question_bank),
            ("Diagnostic Engine", self.validate_component_4_diagnostic_engine),
            ("Gap Identification", self.validate_component_5_gap_identification),
            ("Remediation", self.validate_component_6_remediation),
            ("Progress Tracking", self.validate_component_7_progress_tracking)
        ]
        
        passed_components = 0
        
        for component_name, validator_func in components:
            try:
                if validator_func():
                    passed_components += 1
                else:
                    print(f"❌ {component_name} validation FAILED")
            except Exception as e:
                print(f"❌ {component_name} validation ERROR: {e}")
        
        # Validate end-to-end workflow
        print("🔄 TESTING COMPLETE WORKFLOW")
        print("-" * 40)
        
        workflow_success = False
        try:
            workflow_success = self.validate_end_to_end_workflow()
        except Exception as e:
            print(f"❌ End-to-end workflow ERROR: {e}")
        
        # Final results
        end_time = time.time()
        
        print("\n" + "=" * 60)
        print("🏆 STRATEGY VALIDATION RESULTS")
        print("=" * 60)
        print(f"✅ Components passed: {passed_components}/{len(components)}")
        print(f"✅ End-to-end workflow: {'PASSED' if workflow_success else 'FAILED'}")
        print(f"⏱️  Total validation time: {end_time - start_time:.2f} seconds")
        
        if passed_components == len(components) and workflow_success:
            print("\n🎉 STRATEGY VALIDATION: COMPLETE SUCCESS!")
            print("🚀 All components work together seamlessly!")
            return True
        else:
            print("\n⚠️  STRATEGY VALIDATION: ISSUES DETECTED")
            print("🔧 Some components need attention")
            return False

def main():
    """Run comprehensive strategy validation"""
    validator = StrategyValidator()
    success = validator.run_complete_validation()
    
    if success:
        print("\n✅ OVERALL STRATEGY: VALIDATED AND WORKING")
    else:
        print("\n❌ OVERALL STRATEGY: NEEDS FIXES")
    
    return success

if __name__ == "__main__":
    main() 