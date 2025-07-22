"""
Diagnostic Test Engine for identifying concept misunderstandings.
Simulates student errors and maps them to knowledge graph concepts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random
from collections import defaultdict

from math_learning.testing.test_question import TestQuestion, ErrorMapping, DifficultyLevel
from math_learning.testing.algebra_test_bank import AlgebraTestBank
from math_learning.knowledge_graph.concept import Concept
from math_learning.knowledge_graph.graph import KnowledgeGraph

class StudentProfile(Enum):
    """Different types of student learning profiles for simulation."""
    STRONG_STUDENT = "strong"           # 85-95% accuracy
    AVERAGE_STUDENT = "average"         # 70-80% accuracy  
    STRUGGLING_STUDENT = "struggling"   # 50-65% accuracy
    CONCEPT_GAPS = "concept_gaps"       # Strong in some areas, weak in others

@dataclass
class StudentResponse:
    """Represents a student's response to a test question."""
    question_id: str
    student_answer: str
    is_correct: bool
    error_mapping: Optional[ErrorMapping]
    time_taken_seconds: int
    confidence_level: int  # 1-5 scale

@dataclass
class ConceptGap:
    """Represents an identified gap in student understanding."""
    concept_id: str
    concept_name: str
    evidence_count: int  # Number of questions showing this gap
    severity: float  # 0.0-1.0, higher = more severe
    related_errors: List[ErrorMapping]
    prerequisite_gaps: List[str]  # Prerequisite concepts also showing gaps

@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic report for a student."""
    student_id: str
    total_questions: int
    correct_answers: int
    accuracy_percentage: float
    
    # Concept analysis
    concept_gaps: List[ConceptGap]
    mastered_concepts: List[str]
    prerequisite_issues: List[str]
    
    # Recommendations
    priority_concepts: List[str]  # Most important to address first
    suggested_practice: List[str]  # Question IDs for practice
    estimated_study_time: int  # Minutes needed for remediation

class DiagnosticEngine:
    """Engine for running diagnostic tests and analyzing concept gaps."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, test_bank: AlgebraTestBank):
        self.knowledge_graph = knowledge_graph
        self.test_bank = test_bank
        
    def simulate_student_test(self, 
                            student_profile: StudentProfile,
                            num_questions: int = 10,
                            difficulty_range: List[DifficultyLevel] = None) -> List[StudentResponse]:
        """
        Simulate a student taking a diagnostic test with realistic error patterns.
        """
        if difficulty_range is None:
            difficulty_range = [DifficultyLevel.BASIC, DifficultyLevel.INTERMEDIATE]
        
        # Select questions based on difficulty range
        available_questions = []
        for difficulty in difficulty_range:
            available_questions.extend(self.test_bank.get_questions_by_difficulty(difficulty))
        
        # Randomly select questions
        selected_questions = random.sample(available_questions, min(num_questions, len(available_questions)))
        
        responses = []
        for question in selected_questions:
            response = self._simulate_student_response(question, student_profile)
            responses.append(response)
        
        return responses
    
    def _simulate_student_response(self, question: TestQuestion, profile: StudentProfile) -> StudentResponse:
        """Simulate how a student with given profile would answer a question."""
        
        # Determine if student gets it right based on profile
        accuracy_rates = {
            StudentProfile.STRONG_STUDENT: 0.90,
            StudentProfile.AVERAGE_STUDENT: 0.75,
            StudentProfile.STRUGGLING_STUDENT: 0.58,
            StudentProfile.CONCEPT_GAPS: 0.70  # Varies by concept
        }
        
        base_accuracy = accuracy_rates[profile]
        
        # Adjust accuracy based on question difficulty
        difficulty_modifiers = {
            DifficultyLevel.BASIC: 0.1,
            DifficultyLevel.INTERMEDIATE: 0.0,
            DifficultyLevel.ADVANCED: -0.15,
            DifficultyLevel.EXPERT: -0.25
        }
        
        adjusted_accuracy = base_accuracy + difficulty_modifiers[question.difficulty]
        adjusted_accuracy = max(0.1, min(0.95, adjusted_accuracy))  # Clamp between 10-95%
        
        # Determine if correct
        is_correct = random.random() < adjusted_accuracy
        
        if is_correct:
            student_answer = question.correct_answer
            error_mapping = None
        else:
            # Select a realistic wrong answer from common errors
            if question.common_errors:
                # Weight errors by how common they are (first errors are more common)
                weights = [1.0 / (i + 1) for i in range(len(question.common_errors))]
                error = random.choices(question.common_errors, weights=weights)[0]
                student_answer = error.wrong_answer
                error_mapping = error
            else:
                # Generate a random wrong answer if no common errors defined
                student_answer = "unknown_error"
                error_mapping = ErrorMapping(
                    wrong_answer="unknown_error",
                    concept_id=question.primary_concepts[0] if question.primary_concepts else "UNKNOWN",
                    concept_name="Unknown Concept",
                    error_description="Simulated error - no specific pattern",
                    remediation_hint="Review the concept"
                )
        
        # Simulate time taken (varies by difficulty and correctness)
        base_time = question.time_limit_minutes * 60  # Convert to seconds
        time_modifier = 1.0
        if not is_correct:
            time_modifier = random.uniform(0.8, 1.3)  # Wrong answers take variable time
        if question.difficulty == DifficultyLevel.ADVANCED:
            time_modifier *= 1.2
        
        time_taken = int(base_time * time_modifier)
        
        # Simulate confidence (correct answers generally higher confidence)
        if is_correct:
            confidence = random.randint(3, 5)
        else:
            confidence = random.randint(1, 3)
        
        return StudentResponse(
            question_id=question.id,
            student_answer=student_answer,
            is_correct=is_correct,
            error_mapping=error_mapping,
            time_taken_seconds=time_taken,
            confidence_level=confidence
        )
    
    def analyze_responses(self, responses: List[StudentResponse], student_id: str = "test_student") -> DiagnosticReport:
        """
        Analyze student responses to identify concept gaps and generate recommendations.
        """
        total_questions = len(responses)
        correct_answers = sum(1 for r in responses if r.is_correct)
        accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
        
        # Track concept performance
        concept_errors = defaultdict(list)  # concept_id -> list of errors
        concept_attempts = defaultdict(int)  # concept_id -> number of attempts
        concept_successes = defaultdict(int)  # concept_id -> number of successes
        
        # Analyze each response
        for response in responses:
            question = self.test_bank.get_question_by_id(response.question_id)
            
            # Track all concepts involved in this question
            all_concepts = question.get_concept_dependencies()
            
            for concept_id in all_concepts:
                concept_attempts[concept_id] += 1
                
                if response.is_correct:
                    concept_successes[concept_id] += 1
                elif response.error_mapping:
                    concept_errors[concept_id].append(response.error_mapping)
        
        # Identify concept gaps
        concept_gaps = []
        for concept_id, errors in concept_errors.items():
            attempts = concept_attempts[concept_id]
            successes = concept_successes[concept_id]
            
            if attempts > 0:
                success_rate = successes / attempts
                severity = 1.0 - success_rate  # Higher severity = lower success rate
                
                # Only consider it a gap if severity is significant
                if severity >= 0.3:  # 70% or lower success rate
                    
                    # Find prerequisite gaps
                    prerequisite_gaps = []
                    if concept_id in self.knowledge_graph.concepts:
                        prerequisites = self.knowledge_graph.get_prerequisites(concept_id)
                        for prereq in prerequisites:
                            if prereq in concept_errors:
                                prerequisite_gaps.append(prereq)
                    
                    concept_name = "Unknown Concept"
                    if concept_id in self.knowledge_graph.concepts:
                        concept_name = self.knowledge_graph.concepts[concept_id].name
                    
                    gap = ConceptGap(
                        concept_id=concept_id,
                        concept_name=concept_name,
                        evidence_count=len(errors),
                        severity=severity,
                        related_errors=errors,
                        prerequisite_gaps=prerequisite_gaps
                    )
                    concept_gaps.append(gap)
        
        # Sort gaps by severity (most severe first)
        concept_gaps.sort(key=lambda g: g.severity, reverse=True)
        
        # Identify mastered concepts (high success rate)
        mastered_concepts = []
        for concept_id, attempts in concept_attempts.items():
            if attempts > 0:
                success_rate = concept_successes[concept_id] / attempts
                if success_rate >= 0.8:  # 80% or higher success rate
                    mastered_concepts.append(concept_id)
        
        # Identify prerequisite issues (gaps in foundational concepts)
        prerequisite_issues = []
        for gap in concept_gaps:
            prerequisite_issues.extend(gap.prerequisite_gaps)
        prerequisite_issues = list(set(prerequisite_issues))  # Remove duplicates
        
        # Generate priority concepts (most important to address first)
        priority_concepts = []
        for gap in concept_gaps[:5]:  # Top 5 most severe gaps
            priority_concepts.append(gap.concept_id)
        
        # Suggest practice questions
        suggested_practice = []
        for concept_id in priority_concepts:
            practice_questions = self.test_bank.get_questions_for_concept(concept_id)
            for q in practice_questions[:2]:  # Up to 2 questions per concept
                suggested_practice.append(q.id)
        
        # Estimate study time (based on number and severity of gaps)
        total_severity = sum(gap.severity for gap in concept_gaps)
        estimated_minutes = int(total_severity * 30)  # 30 minutes per severity point
        
        return DiagnosticReport(
            student_id=student_id,
            total_questions=total_questions,
            correct_answers=correct_answers,
            accuracy_percentage=accuracy * 100,
            concept_gaps=concept_gaps,
            mastered_concepts=mastered_concepts,
            prerequisite_issues=prerequisite_issues,
            priority_concepts=priority_concepts,
            suggested_practice=suggested_practice,
            estimated_study_time=estimated_minutes
        )
    
    def run_comprehensive_diagnostic(self, 
                                   student_profile: StudentProfile,
                                   student_id: str = "test_student") -> DiagnosticReport:
        """
        Run a comprehensive diagnostic test covering multiple difficulty levels.
        """
        # Test across different difficulty levels
        all_responses = []
        
        # Basic level questions (foundation)
        basic_responses = self.simulate_student_test(
            student_profile, 
            num_questions=4, 
            difficulty_range=[DifficultyLevel.BASIC]
        )
        all_responses.extend(basic_responses)
        
        # Intermediate level questions (core skills)
        intermediate_responses = self.simulate_student_test(
            student_profile, 
            num_questions=6, 
            difficulty_range=[DifficultyLevel.INTERMEDIATE]
        )
        all_responses.extend(intermediate_responses)
        
        # Advanced level questions (if student is doing well)
        basic_accuracy = sum(1 for r in basic_responses if r.is_correct) / len(basic_responses)
        if basic_accuracy >= 0.7:  # Only give advanced if doing well on basics
            advanced_responses = self.simulate_student_test(
                student_profile, 
                num_questions=3, 
                difficulty_range=[DifficultyLevel.ADVANCED]
            )
            all_responses.extend(advanced_responses)
        
        return self.analyze_responses(all_responses, student_id)
    
    def print_diagnostic_report(self, report: DiagnosticReport):
        """Print a formatted diagnostic report."""
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT FOR STUDENT: {report.student_id}")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Questions Answered: {report.total_questions}")
        print(f"   Correct Answers: {report.correct_answers}")
        print(f"   Accuracy: {report.accuracy_percentage:.1f}%")
        
        if report.concept_gaps:
            print(f"\n🚨 CONCEPT GAPS IDENTIFIED ({len(report.concept_gaps)}):")
            for i, gap in enumerate(report.concept_gaps[:5], 1):  # Show top 5
                print(f"   {i}. {gap.concept_name} ({gap.concept_id})")
                print(f"      Severity: {gap.severity:.2f} | Evidence: {gap.evidence_count} errors")
                if gap.related_errors:
                    print(f"      Common Error: {gap.related_errors[0].error_description}")
                print()
        
        if report.mastered_concepts:
            print(f"✅ MASTERED CONCEPTS ({len(report.mastered_concepts)}):")
            for concept_id in report.mastered_concepts[:5]:
                if concept_id in self.knowledge_graph.concepts:
                    concept_name = self.knowledge_graph.concepts[concept_id].name
                    print(f"   • {concept_name} ({concept_id})")
        
        if report.prerequisite_issues:
            print(f"\n⚠️  PREREQUISITE ISSUES:")
            for concept_id in report.prerequisite_issues[:3]:
                if concept_id in self.knowledge_graph.concepts:
                    concept_name = self.knowledge_graph.concepts[concept_id].name
                    print(f"   • {concept_name} ({concept_id})")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   Priority Concepts to Study: {len(report.priority_concepts)}")
        for concept_id in report.priority_concepts[:3]:
            if concept_id in self.knowledge_graph.concepts:
                concept_name = self.knowledge_graph.concepts[concept_id].name
                print(f"   • {concept_name}")
        
        print(f"\n   Suggested Practice Questions: {len(report.suggested_practice)}")
        print(f"   Estimated Study Time: {report.estimated_study_time} minutes")
        
        print(f"\n{'='*60}") 