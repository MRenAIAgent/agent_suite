#!/usr/bin/env python3
"""
AI AGENT ENHANCED DIAGNOSTIC SYSTEM
Prototype showing how AI agents could enhance concept identification and exercise recommendation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_diagnostic_system import EnhancedDiagnosticSystem, EnhancedDiagnosticResult
from comprehensive_diagnostic import StudentProfile
from failure_simulation import FailurePattern
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class AIAgentResult:
    """Result from AI agent analysis"""
    concept_identified: str
    confidence: float
    reasoning: str
    alternative_concepts: List[Tuple[str, float]]  # (concept, confidence)
    recommended_exercises: List[Dict]
    personalized_explanation: str

class ConceptIdentificationAgent:
    """AI Agent for flexible concept identification"""
    
    def __init__(self):
        # In a real implementation, this would use an LLM API
        # For demo purposes, we'll simulate intelligent behavior
        self.knowledge_base = {
            "order_of_operations": {
                "keywords": ["pemdas", "bodmas", "left to right", "multiplication first", "addition first"],
                "common_errors": ["14", "20", "18"],
                "reasoning_patterns": [
                    "calculated left to right instead of multiplication first",
                    "added before multiplying",
                    "ignored parentheses"
                ]
            },
            "distributive_property": {
                "keywords": ["distribute", "factor", "expand", "multiply through"],
                "common_errors": ["2x + 3", "6x + 4", "2x + 6"],
                "reasoning_patterns": [
                    "forgot to distribute to constant term",
                    "distributed to variable only",
                    "didn't combine like terms"
                ]
            },
            "function_evaluation": {
                "keywords": ["f(", "g(", "substitute", "evaluate"],
                "common_errors": ["14", "6", "7"],
                "reasoning_patterns": [
                    "calculation error in substitution",
                    "forgot to square the input",
                    "minor arithmetic mistake"
                ]
            }
        }
    
    def analyze_student_work(self, 
                           question: str, 
                           student_answer: str, 
                           correct_answer: str,
                           student_profile: StudentProfile) -> AIAgentResult:
        """
        Use AI reasoning to identify concepts and provide detailed analysis
        """
        
        # Simulate LLM analysis of student work
        question_lower = question.lower()
        
        # Identify potential concepts based on question content
        potential_concepts = []
        
        if any(op in question_lower for op in ["×", "*", "+", "-"]) and "(" in question:
            potential_concepts.append(("ORDER_OF_OPERATIONS", 0.85))
        
        if any(word in question_lower for word in ["expand", "distribute", "factor"]):
            potential_concepts.append(("DISTRIBUTIVE_PROPERTY", 0.90))
        
        if "f(" in question or "g(" in question:
            potential_concepts.append(("FUNCTION_EVALUATION", 0.80))
        
        if "solve" in question_lower and "x" in question:
            potential_concepts.append(("TWO_STEP_EQUATIONS", 0.75))
        
        # Analyze the specific error
        primary_concept = potential_concepts[0][0] if potential_concepts else "UNKNOWN_CONCEPT"
        confidence = potential_concepts[0][1] if potential_concepts else 0.5
        
        # Generate reasoning (simulated LLM output)
        reasoning = self._generate_reasoning(question, student_answer, correct_answer, primary_concept)
        
        # Generate personalized explanation
        explanation = self._generate_personalized_explanation(
            primary_concept, student_answer, correct_answer, student_profile
        )
        
        return AIAgentResult(
            concept_identified=primary_concept,
            confidence=confidence,
            reasoning=reasoning,
            alternative_concepts=potential_concepts[1:],
            recommended_exercises=[],  # Will be filled by exercise agent
            personalized_explanation=explanation
        )
    
    def _generate_reasoning(self, question: str, student_answer: str, correct_answer: str, concept: str) -> str:
        """Generate human-like reasoning about the student's error"""
        
        reasoning_templates = {
            "ORDER_OF_OPERATIONS": f"The student calculated '{student_answer}' instead of '{correct_answer}', suggesting they performed operations left-to-right rather than following PEMDAS order.",
            "DISTRIBUTIVE_PROPERTY": f"The student's answer '{student_answer}' indicates they may have forgotten to distribute to all terms or made an error in combining like terms.",
            "FUNCTION_EVALUATION": f"The student got '{student_answer}' instead of '{correct_answer}', likely due to a substitution or calculation error when evaluating the function.",
            "TWO_STEP_EQUATIONS": f"The student's answer '{student_answer}' suggests they may have made an error in the inverse operations needed to isolate the variable."
        }
        
        return reasoning_templates.get(concept, f"The student's answer '{student_answer}' differs from the correct answer '{correct_answer}', indicating a conceptual misunderstanding.")
    
    def _generate_personalized_explanation(self, concept: str, student_answer: str, correct_answer: str, student_profile: StudentProfile) -> str:
        """Generate explanation adapted to student's learning style and ability"""
        
        base_explanations = {
            "ORDER_OF_OPERATIONS": "Remember PEMDAS: Parentheses, Exponents, Multiplication/Division (left to right), Addition/Subtraction (left to right).",
            "DISTRIBUTIVE_PROPERTY": "When you see a(b + c), you need to multiply 'a' by both 'b' and 'c': a×b + a×c.",
            "FUNCTION_EVALUATION": "To evaluate f(x), substitute the given value for every 'x' in the function, then calculate step by step.",
            "TWO_STEP_EQUATIONS": "To solve equations, use inverse operations to isolate the variable. Work backwards from the order of operations."
        }
        
        base = base_explanations.get(concept, "Let's work through this step by step.")
        
        # Adapt to learning style
        if student_profile.learning_style == "visual":
            return f"{base} Try drawing a diagram or using colors to highlight each step."
        elif student_profile.learning_style == "hands_on":
            return f"{base} Try using physical objects or manipulatives to model the problem."
        else:
            return base

class ExerciseRecommendationAgent:
    """AI Agent for personalized exercise recommendation"""
    
    def __init__(self):
        # Simulated exercise database
        self.exercise_database = {
            "ORDER_OF_OPERATIONS": {
                "easy": [
                    {"problem": "2 + 3 × 4", "answer": "14", "explanation": "Multiply first: 3 × 4 = 12, then add: 2 + 12 = 14"},
                    {"problem": "5 × 2 + 3", "answer": "13", "explanation": "Multiply first: 5 × 2 = 10, then add: 10 + 3 = 13"}
                ],
                "medium": [
                    {"problem": "(4 + 2) × 3", "answer": "18", "explanation": "Parentheses first: 4 + 2 = 6, then multiply: 6 × 3 = 18"},
                    {"problem": "8 ÷ 2 + 3 × 2", "answer": "10", "explanation": "Division and multiplication first: 8 ÷ 2 = 4, 3 × 2 = 6, then add: 4 + 6 = 10"}
                ]
            },
            "DISTRIBUTIVE_PROPERTY": {
                "easy": [
                    {"problem": "2(x + 3)", "answer": "2x + 6", "explanation": "Distribute 2 to both terms: 2×x + 2×3 = 2x + 6"},
                    {"problem": "3(y + 1)", "answer": "3y + 3", "explanation": "Distribute 3 to both terms: 3×y + 3×1 = 3y + 3"}
                ],
                "medium": [
                    {"problem": "4(2x + 5)", "answer": "8x + 20", "explanation": "Distribute 4: 4×2x + 4×5 = 8x + 20"},
                    {"problem": "x(x + 7)", "answer": "x² + 7x", "explanation": "Distribute x: x×x + x×7 = x² + 7x"}
                ]
            }
        }
    
    def recommend_exercises(self, 
                          concept: str, 
                          student_profile: StudentProfile,
                          error_analysis: AIAgentResult) -> List[Dict]:
        """
        Generate personalized exercise recommendations using AI reasoning
        """
        
        # Determine appropriate difficulty level
        difficulty = self._determine_difficulty(concept, student_profile, error_analysis)
        
        # Get base exercises for the concept and difficulty
        base_exercises = self.exercise_database.get(concept, {}).get(difficulty, [])
        
        # Personalize exercises based on student profile
        personalized_exercises = []
        for exercise in base_exercises:
            personalized = self._personalize_exercise(exercise, student_profile)
            personalized_exercises.append(personalized)
        
        # Generate adaptive follow-up exercises
        adaptive_exercises = self._generate_adaptive_exercises(concept, student_profile, error_analysis)
        
        return personalized_exercises + adaptive_exercises
    
    def _determine_difficulty(self, concept: str, student_profile: StudentProfile, error_analysis: AIAgentResult) -> str:
        """Use AI reasoning to determine appropriate difficulty level"""
        
        # Consider multiple factors
        ability_score = student_profile.overall_ability
        concept_mastery = student_profile.concept_mastery.get(concept, 0.0)
        error_confidence = error_analysis.confidence
        
        # AI-like decision making
        if ability_score < 0.3 or concept_mastery < 0.2:
            return "easy"
        elif ability_score > 0.7 and concept_mastery > 0.6:
            return "medium"
        else:
            return "easy"  # Default to easier when uncertain
    
    def _personalize_exercise(self, exercise: Dict, student_profile: StudentProfile) -> Dict:
        """Personalize exercise presentation based on student profile"""
        
        personalized = exercise.copy()
        
        # Adapt explanation style
        if student_profile.learning_style == "visual":
            personalized["hint"] = "Try drawing this out or using colors for each step."
        elif student_profile.learning_style == "hands_on":
            personalized["hint"] = "Try using objects to represent the numbers."
        
        # Add encouragement based on student's confidence level
        if student_profile.overall_ability < 0.4:
            personalized["encouragement"] = "Take your time and work through each step carefully!"
        
        return personalized
    
    def _generate_adaptive_exercises(self, concept: str, student_profile: StudentProfile, error_analysis: AIAgentResult) -> List[Dict]:
        """Generate new exercises adapted to the specific error pattern"""
        
        # This would use AI to generate novel problems targeting the specific error
        # For demo, we'll return a simulated adaptive exercise
        
        adaptive_exercises = []
        
        if concept == "ORDER_OF_OPERATIONS" and "left to right" in error_analysis.reasoning.lower():
            adaptive_exercises.append({
                "problem": "6 + 2 × 3",
                "answer": "12",
                "explanation": "Remember: multiplication before addition! 2 × 3 = 6, then 6 + 6 = 12",
                "focus": "This problem specifically targets the left-to-right error pattern",
                "adaptive": True
            })
        
        return adaptive_exercises

class HybridAIEnhancedDiagnostic:
    """Hybrid system combining rule-based and AI agent approaches"""
    
    def __init__(self):
        self.rule_based_system = EnhancedDiagnosticSystem()
        self.concept_agent = ConceptIdentificationAgent()
        self.exercise_agent = ExerciseRecommendationAgent()
    
    def diagnose_with_ai_enhancement(self,
                                   question_text: str,
                                   student_answer: str,
                                   correct_answer: str,
                                   student_profile: StudentProfile,
                                   use_ai_fallback: bool = True) -> Dict:
        """
        Hybrid diagnostic using both rule-based and AI approaches
        """
        
        # First, try the fast rule-based approach
        rule_based_result = self.rule_based_system.diagnose_student_response(
            question_text, student_answer, correct_answer, student_profile
        )
        
        # If rule-based system is confident, use it
        if rule_based_result.confidence_score > 0.8 and rule_based_result.missing_concept != "UNKNOWN_CONCEPT":
            
            # Still use AI for enhanced exercise recommendations
            ai_exercises = self.exercise_agent.recommend_exercises(
                rule_based_result.missing_concept,
                student_profile,
                AIAgentResult(
                    concept_identified=rule_based_result.missing_concept,
                    confidence=rule_based_result.confidence_score,
                    reasoning="Rule-based pattern match",
                    alternative_concepts=[],
                    recommended_exercises=[],
                    personalized_explanation=""
                )
            )
            
            return {
                "approach_used": "rule_based_with_ai_exercises",
                "concept": rule_based_result.missing_concept,
                "confidence": rule_based_result.confidence_score,
                "explanation": rule_based_result.explanation,
                "remediation_level": rule_based_result.remediation_level,
                "learning_recommendations": rule_based_result.learning_recommendations,
                "ai_enhanced_exercises": ai_exercises,
                "processing_time": "~0.001s"
            }
        
        # If rule-based system is uncertain, use AI agent
        elif use_ai_fallback:
            ai_result = self.concept_agent.analyze_student_work(
                question_text, student_answer, correct_answer, student_profile
            )
            
            ai_exercises = self.exercise_agent.recommend_exercises(
                ai_result.concept_identified,
                student_profile,
                ai_result
            )
            
            return {
                "approach_used": "ai_agent_fallback",
                "concept": ai_result.concept_identified,
                "confidence": ai_result.confidence,
                "explanation": ai_result.personalized_explanation,
                "reasoning": ai_result.reasoning,
                "alternative_concepts": ai_result.alternative_concepts,
                "ai_exercises": ai_exercises,
                "processing_time": "~0.5s"
            }
        
        # Fallback to rule-based result
        else:
            return {
                "approach_used": "rule_based_only",
                "concept": rule_based_result.missing_concept,
                "confidence": rule_based_result.confidence_score,
                "explanation": rule_based_result.explanation,
                "processing_time": "~0.001s"
            }

def demonstrate_ai_enhanced_diagnostic():
    """Demonstrate the AI-enhanced diagnostic system"""
    
    print("🤖 AI AGENT ENHANCED DIAGNOSTIC SYSTEM")
    print("=" * 70)
    
    # Initialize hybrid system
    hybrid_system = HybridAIEnhancedDiagnostic()
    
    # Create test student
    student = StudentProfile(
        name="Alex",
        overall_ability=0.4,
        concept_mastery={"ORDER_OF_OPERATIONS": 0.2, "DISTRIBUTIVE_PROPERTY": 0.3},
        learning_style="visual",
        total_questions_attempted=20,
        total_questions_correct=8
    )
    
    # Test cases
    test_cases = [
        {
            "question": "Calculate: 3 + 4 × 2",
            "correct_answer": "11",
            "student_answer": "14",
            "description": "Common order of operations error"
        },
        {
            "question": "Expand: 2(x + 3)",
            "correct_answer": "2x + 6",
            "student_answer": "2x + 3",
            "description": "Distributive property error"
        },
        {
            "question": "Solve: 5y - 8 = 12",
            "correct_answer": "y = 4",
            "student_answer": "y = 0.8",
            "description": "Novel error pattern (not in rule base)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['description']}")
        print(f"{'='*60}")
        
        print(f"📝 Question: {test_case['question']}")
        print(f"✅ Correct: {test_case['correct_answer']}")
        print(f"❌ Student: {test_case['student_answer']}")
        
        # Run hybrid diagnostic
        result = hybrid_system.diagnose_with_ai_enhancement(
            test_case['question'],
            test_case['student_answer'],
            test_case['correct_answer'],
            student
        )
        
        print(f"\n🔍 HYBRID DIAGNOSTIC RESULTS:")
        print(f"   🚀 Approach Used: {result['approach_used']}")
        print(f"   🎯 Concept: {result['concept']}")
        print(f"   📊 Confidence: {result['confidence']:.1%}")
        print(f"   💡 Explanation: {result['explanation']}")
        print(f"   ⏱️  Processing Time: {result['processing_time']}")
        
        if 'reasoning' in result:
            print(f"   🧠 AI Reasoning: {result['reasoning']}")
        
        if 'ai_enhanced_exercises' in result:
            print(f"   📚 AI-Enhanced Exercises:")
            for j, exercise in enumerate(result['ai_enhanced_exercises'][:2], 1):
                print(f"      {j}. {exercise['problem']} = {exercise['answer']}")
                if 'adaptive' in exercise:
                    print(f"         🎯 {exercise['focus']}")
        
        if 'alternative_concepts' in result and result['alternative_concepts']:
            print(f"   🔄 Alternative Concepts: {result['alternative_concepts']}")
    
    print(f"\n🎉 AI-ENHANCED DIAGNOSTIC DEMONSTRATION COMPLETE!")
    print(f"\n📊 SYSTEM COMPARISON:")
    print(f"   Rule-Based: Fast (0.001s), High accuracy (96.2%), Deterministic")
    print(f"   AI Agents: Flexible, Personalized, Adaptive, Handles novel errors")
    print(f"   Hybrid: Best of both - Fast for common cases, AI for complex cases")

if __name__ == "__main__":
    demonstrate_ai_enhanced_diagnostic() 