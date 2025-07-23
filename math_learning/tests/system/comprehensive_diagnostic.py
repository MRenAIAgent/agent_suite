"""
Comprehensive Diagnostic System
Complete pipeline: Test → Failure → Gap Detection → Remediation Recommendation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random

from failure_simulation import FailureSimulator, FailurePattern
from remediation_questions import RemediationQuestionBank, RemediationQuestion
from knowledge_graph import KnowledgeGraph
from knowledge_graph.algebra_graph import build_algebra_knowledge_graph

@dataclass
class DiagnosticResult:
    """Complete diagnostic result with gap analysis and recommendations"""
    question_text: str
    student_answer: str
    correct_answer: str
    is_correct: bool
    failure_pattern: Optional[FailurePattern]
    missing_concept: Optional[str]
    prerequisite_chain: List[str]
    recommended_questions: List[RemediationQuestion]
    confidence_score: float
    explanation: str

@dataclass
class StudentProfile:
    """Student learning profile with ability levels and progress tracking"""
    name: str
    overall_ability: float  # 0.0 to 1.0
    concept_mastery: Dict[str, float]  # concept -> mastery level (0.0 to 1.0)
    learning_style: str  # "visual", "analytical", "hands_on"
    total_questions_attempted: int
    total_questions_correct: int

class ComprehensiveDiagnostic:
    """Main diagnostic system integrating all components"""
    
    def __init__(self):
        self.failure_simulator = FailureSimulator()
        self.question_bank = RemediationQuestionBank()
        self.knowledge_graph = build_algebra_knowledge_graph()  # Load full algebra graph
        self.diagnostic_history: List[DiagnosticResult] = []
        
        # Map error concept names to knowledge graph concept IDs
        self.concept_mapping = {
            "ONE_STEP_EQUATIONS": "AT-20",  # One-Step Linear Equations
            "ORDER_OF_OPERATIONS": "NS-16",  # Order of Operations
            "DISTRIBUTIVE_PROPERTY": "AT-21",  # Distributive Property
            "COMBINE_LIKE_TERMS": "AT-19",  # Combine Like Terms
            "INTEGER_OPERATIONS": "NS-13",  # Integer Addition/Subtraction
            "MULTIPLICATION_FACTS": "NS-06",  # Multiplication Facts 0–12
            "FRACTIONS_AS_PART_WHOLE": "NS-08",  # Fractions as Part–Whole
            "FUNCTION_EVALUATION": "FN-62",  # Function Evaluation (fixed)
            "FUNCTION_COMPOSITION": "FN-63",  # Function Composition (fixed)
            "ADDITION_FACTS": "NS-01",  # Basic addition (using counting as proxy)
        }
    
    def diagnose_student_response(
        self, 
        question_type: str,
        question_text: str,
        student_answer: str,
        correct_answer: str,
        student_profile: StudentProfile
    ) -> DiagnosticResult:
        """
        Complete diagnostic analysis of a student's response
        
        Args:
            question_type: Type of question (e.g., "two_step_equations")
            question_text: The actual question asked
            student_answer: What the student wrote
            correct_answer: The correct answer
            student_profile: Student's learning profile
            
        Returns:
            Complete diagnostic result with recommendations
        """
        
        # Check if answer is correct
        is_correct = student_answer.strip() == correct_answer.strip()
        
        if is_correct:
            return DiagnosticResult(
                question_text=question_text,
                student_answer=student_answer,
                correct_answer=correct_answer,
                is_correct=True,
                failure_pattern=None,
                missing_concept=None,
                prerequisite_chain=[],
                recommended_questions=[],
                confidence_score=1.0,
                explanation="✅ Correct answer! Student demonstrates mastery of this concept."
            )
        
        # Analyze the failure
        failure_pattern = self.failure_simulator.analyze_student_answer(
            question_type, student_answer, correct_answer
        )
        
        if not failure_pattern:
            return DiagnosticResult(
                question_text=question_text,
                student_answer=student_answer,
                correct_answer=correct_answer,
                is_correct=False,
                failure_pattern=None,
                missing_concept=None,
                prerequisite_chain=[],
                recommended_questions=[],
                confidence_score=0.3,
                explanation="❌ Incorrect answer, but error pattern not recognized."
            )
        
        # Identify missing concept and prerequisite chain
        missing_concept = failure_pattern.missing_concept
        prerequisite_chain = self._get_prerequisite_chain(missing_concept)
        
        # Get remediation recommendations
        recommended_questions = self._get_remediation_recommendations(
            missing_concept, student_profile, prerequisite_chain
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            failure_pattern, student_profile, missing_concept
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            failure_pattern, missing_concept, prerequisite_chain
        )
        
        result = DiagnosticResult(
            question_text=question_text,
            student_answer=student_answer,
            correct_answer=correct_answer,
            is_correct=False,
            failure_pattern=failure_pattern,
            missing_concept=missing_concept,
            prerequisite_chain=prerequisite_chain,
            recommended_questions=recommended_questions,
            confidence_score=confidence_score,
            explanation=explanation
        )
        
        # Update student profile
        self._update_student_profile(student_profile, result)
        
        # Add to diagnostic history
        self.diagnostic_history.append(result)
        
        return result
    
    def _get_prerequisite_chain(self, concept: str) -> List[str]:
        """Get the prerequisite chain for a concept"""
        try:
            # Get direct prerequisites
            prerequisites = []
            for edge in self.knowledge_graph.edges:
                if edge.to_concept == concept and edge.relationship_type == "prerequisite":
                    prerequisites.append(edge.from_concept)
            
            # Build chain by following prerequisites
            chain = []
            visited = set()
            
            def build_chain(current_concept):
                if current_concept in visited:
                    return
                visited.add(current_concept)
                
                # Find prerequisites of current concept
                concept_prereqs = []
                for edge in self.knowledge_graph.edges:
                    if edge.to_concept == current_concept and edge.relationship_type == "prerequisite":
                        concept_prereqs.append(edge.from_concept)
                
                # Add prerequisites to chain first (depth-first)
                for prereq in concept_prereqs:
                    build_chain(prereq)
                
                # Add current concept to chain
                if current_concept != concept:  # Don't include the target concept itself
                    chain.append(current_concept)
            
            build_chain(concept)
            return chain
            
        except Exception:
            # Fallback if knowledge graph not available
            return []
    
    def _get_remediation_recommendations(
        self, 
        missing_concept: str, 
        student_profile: StudentProfile,
        prerequisite_chain: List[str]
    ) -> List[RemediationQuestion]:
        """Get personalized remediation question recommendations"""
        
        recommendations = []
        
        # Start with the most fundamental missing prerequisite
        concepts_to_address = prerequisite_chain + [missing_concept]
        
        for concept in concepts_to_address:
            # Get student's mastery level for this concept
            mastery_level = student_profile.concept_mastery.get(concept, 0.0)
            
            # If student has low mastery, recommend practice
            if mastery_level < 0.7:
                best_question = self.question_bank.get_best_question_for_gap(
                    concept, student_profile.overall_ability
                )
                if best_question:
                    recommendations.append(best_question)
        
        # Limit to top 3 recommendations
        return recommendations[:3]
    
    def _calculate_confidence_score(
        self, 
        failure_pattern: FailurePattern, 
        student_profile: StudentProfile,
        missing_concept: str
    ) -> float:
        """Calculate confidence in the diagnostic analysis"""
        
        base_confidence = failure_pattern.frequency  # Higher for common errors
        
        # Adjust based on student's historical performance
        if missing_concept in student_profile.concept_mastery:
            mastery = student_profile.concept_mastery[missing_concept]
            if mastery < 0.3:
                base_confidence += 0.2  # More confident if student consistently struggles
        
        # Adjust based on error pattern recognition
        if failure_pattern.error_type != "unknown_error":
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _generate_explanation(
        self, 
        failure_pattern: FailurePattern,
        missing_concept: str,
        prerequisite_chain: List[str]
    ) -> str:
        """Generate a comprehensive explanation of the diagnostic result"""
        
        explanation = f"🚨 **Error Detected**: {failure_pattern.explanation}\n\n"
        
        # Get detailed concept information from knowledge graph
        concept_id = self.concept_mapping.get(missing_concept)
        if concept_id:
            concept_node = self.knowledge_graph.get_concept(concept_id)
            if concept_node:
                explanation += f"🎯 **Missing Concept**: {concept_node.name}\n"
                explanation += f"📖 **Description**: {concept_node.description}\n"
                explanation += f"📊 **Difficulty Level**: {concept_node.difficulty}/5\n"
                explanation += f"⏱️ **Time to Master**: {concept_node.time_to_master} minutes\n"
                explanation += f"📚 **Category**: {concept_node.category}\n"
                
                if concept_node.examples:
                    explanation += f"💡 **Examples**:\n"
                    for example in concept_node.examples[:2]:  # Show first 2 examples
                        explanation += f"   • {example}\n"
                explanation += "\n"
            else:
                explanation += f"🎯 **Missing Concept**: {missing_concept}\n\n"
        else:
            explanation += f"🎯 **Missing Concept**: {missing_concept}\n\n"
        
        if prerequisite_chain:
            explanation += f"📚 **Prerequisite Chain**: {' → '.join(prerequisite_chain)} → {missing_concept}\n\n"
        
        explanation += f"💡 **Recommendation**: Practice {missing_concept} concepts to build understanding."
        
        return explanation
    
    def _update_student_profile(self, student_profile: StudentProfile, result: DiagnosticResult):
        """Update student profile based on diagnostic result"""
        
        student_profile.total_questions_attempted += 1
        
        if result.is_correct:
            student_profile.total_questions_correct += 1
        
        # Update overall ability
        student_profile.overall_ability = (
            student_profile.total_questions_correct / student_profile.total_questions_attempted
        )
        
        # Update concept mastery
        if result.missing_concept:
            current_mastery = student_profile.concept_mastery.get(result.missing_concept, 0.5)
            # Decrease mastery for failed concept
            new_mastery = max(0.0, current_mastery - 0.1)
            student_profile.concept_mastery[result.missing_concept] = new_mastery
    
    def run_comprehensive_test(self, student_profile: StudentProfile) -> List[DiagnosticResult]:
        """Run a comprehensive diagnostic test with multiple question types"""
        
        test_questions = [
            {
                "type": "two_step_equations",
                "text": "Solve for x: 3x + 7 = 22",
                "correct_answer": "x = 5"
            },
            {
                "type": "distributive_property", 
                "text": "Simplify: 2(x + 3) + 4x",
                "correct_answer": "6x + 6"
            },
            {
                "type": "function_composition",
                "text": "Find f(g(1)) where f(x) = x + 2, g(x) = 3x",
                "correct_answer": "5"
            },
            {
                "type": "order_of_operations",
                "text": "Calculate: 3 + 4 × 2",
                "correct_answer": "11"
            },
            {
                "type": "integer_operations",
                "text": "Calculate: -5 + (-3)",
                "correct_answer": "-8"
            }
        ]
        
        results = []
        
        print(f"🧪 COMPREHENSIVE DIAGNOSTIC TEST FOR {student_profile.name}")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question['text']}")
            
            # Simulate student response based on their ability
            failure = self.failure_simulator.simulate_failure(
                question["type"], student_profile.overall_ability
            )
            
            if failure:
                student_answer = failure.wrong_answer
                print(f"   Student Answer: {student_answer}")
            else:
                student_answer = question["correct_answer"]
                print(f"   Student Answer: {student_answer} ✅")
            
            # Diagnose the response
            result = self.diagnose_student_response(
                question["type"],
                question["text"],
                student_answer,
                question["correct_answer"],
                student_profile
            )
            
            results.append(result)
            
            # Display diagnostic result
            if not result.is_correct:
                print(f"\n   🚨 DIAGNOSTIC ANALYSIS:")
                print(f"      Missing Concept: {result.missing_concept}")
                print(f"      Confidence: {result.confidence_score:.1%}")
                print(f"      Error: {result.failure_pattern.explanation}")
                
                if result.recommended_questions:
                    print(f"\n   📚 RECOMMENDED PRACTICE:")
                    for j, rec_q in enumerate(result.recommended_questions[:2], 1):
                        print(f"      {j}. {rec_q.question_text}")
                        print(f"         (Difficulty: {rec_q.difficulty_level})")
        
        return results
    
    def generate_learning_plan(self, student_profile: StudentProfile) -> Dict:
        """Generate a personalized learning plan based on diagnostic results"""
        
        # Analyze all diagnostic results for this student
        student_results = [r for r in self.diagnostic_history 
                          if not r.is_correct]  # Focus on areas needing improvement
        
        # Count concept gaps
        concept_gaps = {}
        for result in student_results:
            if result.missing_concept:
                concept_gaps[result.missing_concept] = concept_gaps.get(result.missing_concept, 0) + 1
        
        # Sort by frequency (most problematic concepts first)
        priority_concepts = sorted(concept_gaps.items(), key=lambda x: x[1], reverse=True)
        
        # Generate learning plan
        learning_plan = {
            "student_name": student_profile.name,
            "overall_ability": student_profile.overall_ability,
            "total_questions": student_profile.total_questions_attempted,
            "accuracy": student_profile.overall_ability,
            "priority_concepts": [],
            "recommended_study_time": 0,
            "learning_path": []
        }
        
        total_study_time = 0
        
        for concept, frequency in priority_concepts[:5]:  # Top 5 priority concepts
            # Get remediation questions for this concept
            questions = self.question_bank.get_progressive_questions(concept, 3)
            
            # Get detailed concept information from knowledge graph
            concept_id = self.concept_mapping.get(concept)
            concept_node = None
            if concept_id:
                concept_node = self.knowledge_graph.get_concept(concept_id)
            
            if concept_node:
                base_time = concept_node.time_to_master / 60.0  # Convert minutes to hours
                # Adjust for student ability
                adjusted_time = base_time * (2.0 - student_profile.overall_ability)
                total_study_time += adjusted_time
                
                learning_plan["priority_concepts"].append({
                    "concept": concept,
                    "concept_name": concept_node.name,
                    "description": concept_node.description,
                    "category": concept_node.category,
                    "frequency": frequency,
                    "estimated_hours": adjusted_time,
                    "practice_questions": len(questions),
                    "difficulty": concept_node.difficulty,
                    "examples": concept_node.examples[:2] if concept_node.examples else []
                })
            else:
                # Fallback if concept not found in knowledge graph
                estimated_time = 2.0 * (2.0 - student_profile.overall_ability)
                total_study_time += estimated_time
                
                learning_plan["priority_concepts"].append({
                    "concept": concept,
                    "concept_name": concept,
                    "description": "Concept details not available",
                    "category": "Unknown",
                    "frequency": frequency,
                    "estimated_hours": estimated_time,
                    "practice_questions": len(questions),
                    "difficulty": 2,  # Default difficulty
                    "examples": []
                })
        
        learning_plan["recommended_study_time"] = total_study_time
        
        return learning_plan

def demonstrate_comprehensive_diagnostic():
    """Demonstrate the complete diagnostic system"""
    
    diagnostic = ComprehensiveDiagnostic()
    
    # Create test student profiles
    students = [
        StudentProfile(
            name="Emma",
            overall_ability=0.75,
            concept_mastery={
                "ADDITION_FACTS": 0.9,
                "MULTIPLICATION_FACTS": 0.8,
                "ONE_STEP_EQUATIONS": 0.6
            },
            learning_style="analytical",
            total_questions_attempted=0,
            total_questions_correct=0
        ),
        StudentProfile(
            name="Jake", 
            overall_ability=0.25,
            concept_mastery={
                "ADDITION_FACTS": 0.4,
                "MULTIPLICATION_FACTS": 0.3,
                "ORDER_OF_OPERATIONS": 0.2
            },
            learning_style="hands_on",
            total_questions_attempted=0,
            total_questions_correct=0
        )
    ]
    
    # Run comprehensive tests
    for student in students:
        print(f"\n{'='*80}")
        results = diagnostic.run_comprehensive_test(student)
        
        # Generate learning plan
        print(f"\n📋 PERSONALIZED LEARNING PLAN FOR {student.name}")
        print("-" * 50)
        
        learning_plan = diagnostic.generate_learning_plan(student)
        
        print(f"Overall Ability: {learning_plan['overall_ability']:.1%}")
        print(f"Test Accuracy: {learning_plan['accuracy']:.1%}")
        print(f"Recommended Study Time: {learning_plan['recommended_study_time']:.1f} hours")
        
        print(f"\n🎯 Priority Concepts to Master:")
        for i, concept_info in enumerate(learning_plan['priority_concepts'], 1):
            print(f"   {i}. {concept_info['concept_name']} ({concept_info['concept']})")
            print(f"      📖 {concept_info['description']}")
            print(f"      📚 Category: {concept_info['category']}")
            print(f"      📊 Difficulty: {concept_info['difficulty']}/5")
            print(f"      🚨 Errors: {concept_info['frequency']}")
            print(f"      ⏱️ Study Time: {concept_info['estimated_hours']:.1f} hours")
            print(f"      📝 Practice Questions: {concept_info['practice_questions']}")
            if concept_info['examples']:
                print(f"      💡 Examples:")
                for example in concept_info['examples']:
                    print(f"         • {example}")
            print()

if __name__ == "__main__":
    demonstrate_comprehensive_diagnostic() 