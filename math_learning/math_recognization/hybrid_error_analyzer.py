"""
Hybrid Error Analyzer

This module combines rule-based error analysis with AI/LLM analysis to provide
comprehensive error understanding and diagnosis.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json

from math_learning.math_recognization.error_analysis_engine import ErrorAnalysisEngine, ErrorAnalysisResult, ErrorType
from math_learning.math_recognization.root_cause_analyzer import RootCauseAnalyzer, RootCauseAnalysis
from math_learning.math_recognization.knowledge_gap_identifier import KnowledgeGapIdentifier, GapAnalysisResult
from math_learning.math_recognization.answer_correctness_engine import AnswerCorrectnessEngine, ValidationResult

@dataclass
class HybridAnalysisResult:
    """Comprehensive result combining all analysis approaches."""
    # Core analysis results
    correctness_result: ValidationResult
    error_analysis: ErrorAnalysisResult
    root_cause_analysis: RootCauseAnalysis
    gap_analysis: GapAnalysisResult
    
    # AI insights
    ai_insights: Dict[str, Any]
    ai_confidence: float
    
    # Synthesis
    primary_diagnosis: str
    confidence_score: float
    recommended_actions: List[str]
    learning_priority: str  # "immediate", "short_term", "long_term"
    
    # Additional context
    student_level_assessment: str
    problem_difficulty_match: str
    metacognitive_insights: List[str] = field(default_factory=list)

class AIErrorAnalyzer:
    """Uses AI/LLM to analyze errors with natural language understanding."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Could be OpenAI, Anthropic, etc.
    
    async def analyze_error_with_ai(self, question: str, student_answer: str, 
                                  correct_answer: str, work_shown: str = "") -> Dict[str, Any]:
        """Use AI to analyze the error with natural language understanding."""
        
        # Create analysis prompt
        prompt = self._create_analysis_prompt(question, student_answer, correct_answer, work_shown)
        
        # Get AI analysis (this would use your preferred LLM)
        ai_response = await self._get_ai_analysis(prompt)
        
        # Parse and structure the AI response
        structured_analysis = self._parse_ai_response(ai_response)
        
        return structured_analysis
    
    def _create_analysis_prompt(self, question: str, student_answer: str, 
                              correct_answer: str, work_shown: str) -> str:
        """Create a comprehensive prompt for AI error analysis."""
        prompt = f"""
You are an expert mathematics educator analyzing a student's error. Please provide a comprehensive analysis.

PROBLEM: {question}
STUDENT ANSWER: {student_answer}
CORRECT ANSWER: {correct_answer}
STUDENT WORK SHOWN: {work_shown if work_shown else "No work shown"}

Please analyze this error and provide insights in the following areas:

1. ERROR TYPE CLASSIFICATION:
   - Primary error type (conceptual, procedural, arithmetic, careless, notation)
   - Specific error subtype
   - Confidence in classification (0-1)

2. ROOT CAUSE ANALYSIS:
   - Most likely root cause
   - Contributing factors
   - Missing prerequisite knowledge
   - Misconceptions identified

3. STUDENT UNDERSTANDING ASSESSMENT:
   - What does the student understand correctly?
   - What specific concepts are missing or confused?
   - Student's problem-solving approach evaluation
   - Metacognitive awareness level

4. EDUCATIONAL INSIGHTS:
   - Appropriate grade/skill level for this student
   - Is this problem appropriate for the student's level?
   - Learning style considerations
   - Motivation and confidence factors

5. REMEDIATION RECOMMENDATIONS:
   - Immediate next steps
   - Specific concepts to review
   - Teaching strategies that would help
   - Practice problem types needed

6. DIAGNOSTIC QUESTIONS:
   - 3-5 specific questions to ask the student to confirm your analysis
   - What these questions would reveal about their understanding

Please provide specific, actionable insights based on the mathematical content and error patterns you observe.
"""
        return prompt
    
    async def _get_ai_analysis(self, prompt: str) -> str:
        """Get analysis from AI/LLM service."""
        if not self.llm_client:
            # Return a mock response for now
            return self._mock_ai_response()
        
        # Use the mock client if provided
        if hasattr(self.llm_client, 'generate'):
            response = await self.llm_client.generate(prompt)
            return response.content
        
        # This would integrate with your preferred LLM service
        # Example for OpenAI:
        # response = await self.llm_client.chat.completions.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.3
        # )
        # return response.choices[0].message.content
        
        return self._mock_ai_response()
    
    def _mock_ai_response(self) -> str:
        """Mock AI response for demonstration purposes."""
        return """
1. ERROR TYPE CLASSIFICATION:
   - Primary error type: Conceptual
   - Specific error subtype: Fraction addition misconception
   - Confidence: 0.9

2. ROOT CAUSE ANALYSIS:
   - Most likely root cause: Student doesn't understand common denominators
   - Contributing factors: Procedural confusion, lack of fraction visualization
   - Missing prerequisite: Understanding equivalent fractions
   - Misconceptions: Adding numerators and denominators separately

3. STUDENT UNDERSTANDING ASSESSMENT:
   - Understands: Basic fraction notation, arithmetic operations
   - Missing: Common denominator concept, fraction equivalence
   - Problem-solving approach: Applies addition rules incorrectly to fractions
   - Metacognitive awareness: Low - doesn't recognize the error

4. EDUCATIONAL INSIGHTS:
   - Appropriate level: Elementary to middle school
   - Problem appropriateness: Slightly above current understanding
   - Learning style: Would benefit from visual fraction models
   - Confidence: Likely frustrated with fraction operations

5. REMEDIATION RECOMMENDATIONS:
   - Immediate: Review equivalent fractions with visual models
   - Concepts to review: Common denominators, fraction visualization
   - Teaching strategies: Use pie charts, fraction bars, hands-on activities
   - Practice: Start with same-denominator addition, then progress

6. DIAGNOSTIC QUESTIONS:
   - "What does 1/2 + 1/3 mean in words?"
   - "Can you draw pictures to show these fractions?"
   - "Are 2/4 and 1/2 the same amount?"
   - "How would you add 1/4 + 1/4?"
   - "What makes adding fractions different from adding whole numbers?"
"""
    
    def _parse_ai_response(self, response: str) -> Dict[str, Any]:
        """Parse the AI response into structured data."""
        # This is a simplified parser - in practice, you'd use more robust parsing
        # or request structured output from the AI
        
        lines = response.strip().split('\n')
        parsed = {
            'error_classification': {},
            'root_cause': {},
            'student_assessment': {},
            'educational_insights': {},
            'remediation': {},
            'diagnostic_questions': [],
            'confidence': 0.8  # Default confidence
        }
        
        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Identify sections
            if 'ERROR TYPE' in line:
                current_section = 'error_classification'
            elif 'ROOT CAUSE' in line:
                current_section = 'root_cause'
            elif 'STUDENT UNDERSTANDING' in line:
                current_section = 'student_assessment'
            elif 'EDUCATIONAL INSIGHTS' in line:
                current_section = 'educational_insights'
            elif 'REMEDIATION' in line:
                current_section = 'remediation'
            elif 'DIAGNOSTIC QUESTIONS' in line:
                current_section = 'diagnostic_questions'
            elif line.startswith('-') and current_section:
                # Parse content
                content = line[1:].strip()
                if current_section == 'diagnostic_questions':
                    if content.startswith('"') and content.endswith('"'):
                        parsed[current_section].append(content[1:-1])
                else:
                    # Store as text for now - could be more sophisticated
                    if current_section not in parsed:
                        parsed[current_section] = []
                    if isinstance(parsed[current_section], list):
                        parsed[current_section].append(content)
                    else:
                        parsed[current_section] = content
        
        return parsed

class AnalysisSynthesizer:
    """Synthesizes results from multiple analysis approaches."""
    
    def __init__(self):
        pass
    
    def synthesize_analyses(self, correctness_result: ValidationResult,
                          error_analysis: ErrorAnalysisResult,
                          root_cause_analysis: RootCauseAnalysis,
                          gap_analysis: GapAnalysisResult,
                          ai_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize multiple analysis results into unified insights."""
        
        synthesis = {
            'primary_diagnosis': self._determine_primary_diagnosis(
                error_analysis, root_cause_analysis, ai_insights
            ),
            'confidence_score': self._calculate_synthesis_confidence(
                correctness_result, error_analysis, root_cause_analysis, gap_analysis, ai_insights
            ),
            'recommended_actions': self._synthesize_recommendations(
                error_analysis, root_cause_analysis, gap_analysis, ai_insights
            ),
            'learning_priority': self._determine_learning_priority(
                gap_analysis, root_cause_analysis
            ),
            'student_level_assessment': self._assess_student_level(
                error_analysis, gap_analysis, ai_insights
            ),
            'problem_difficulty_match': self._assess_problem_difficulty(
                error_analysis, ai_insights
            ),
            'metacognitive_insights': self._extract_metacognitive_insights(ai_insights)
        }
        
        return synthesis
    
    def _determine_primary_diagnosis(self, error_analysis: ErrorAnalysisResult,
                                   root_cause_analysis: RootCauseAnalysis,
                                   ai_insights: Dict[str, Any]) -> str:
        """Determine the primary diagnosis by synthesizing all analyses."""
        
        # Weight different sources
        rule_based_diagnosis = root_cause_analysis.primary_cause
        
        # Handle case where ai_insights might be a string
        if isinstance(ai_insights, str):
            return rule_based_diagnosis
        
        ai_diagnosis = ai_insights.get('root_cause', {})
        
        # If AI and rule-based agree, use that
        if isinstance(ai_diagnosis, dict) and 'primary_cause' in ai_diagnosis:
            ai_primary = ai_diagnosis['primary_cause']
            if any(word in rule_based_diagnosis.lower() for word in ai_primary.lower().split()):
                return f"Confirmed: {rule_based_diagnosis}"
        
        # If they disagree, provide both perspectives
        return f"Primary: {rule_based_diagnosis} (AI suggests: {str(ai_diagnosis)[:100]}...)"
    
    def _calculate_synthesis_confidence(self, correctness_result: ValidationResult,
                                      error_analysis: ErrorAnalysisResult,
                                      root_cause_analysis: RootCauseAnalysis,
                                      gap_analysis: GapAnalysisResult,
                                      ai_insights: Dict[str, Any]) -> float:
        """Calculate overall confidence in the synthesized analysis."""
        
        # Handle case where ai_insights might be a string
        ai_confidence = 0.5
        if isinstance(ai_insights, dict):
            ai_confidence = ai_insights.get('confidence', 0.5)
        
        confidences = [
            correctness_result.confidence,
            error_analysis.confidence,
            root_cause_analysis.confidence,
            gap_analysis.confidence_score,
            ai_confidence
        ]
        
        # Weighted average with higher weight on agreement
        base_confidence = sum(confidences) / len(confidences)
        
        # Boost confidence if multiple approaches agree
        agreement_boost = self._calculate_agreement_boost(
            error_analysis, root_cause_analysis, ai_insights
        )
        
        return min(base_confidence + agreement_boost, 1.0)
    
    def _calculate_agreement_boost(self, error_analysis: ErrorAnalysisResult,
                                 root_cause_analysis: RootCauseAnalysis,
                                 ai_insights: Dict[str, Any]) -> float:
        """Calculate confidence boost based on agreement between analyses."""
        
        # Handle case where ai_insights might be a string instead of dict
        if isinstance(ai_insights, str):
            return 0.0  # No agreement calculation possible
        
        # Check if error types agree
        rule_based_type = error_analysis.error_type.value
        error_classification = ai_insights.get('error_classification', {})
        if isinstance(error_classification, dict):
            ai_type = error_classification.get('primary_type', '')
        else:
            ai_type = ''
        
        agreement_score = 0.0
        
        if rule_based_type in ai_type.lower() or ai_type.lower() in rule_based_type:
            agreement_score += 0.1
        
        # Check if root causes are similar
        rule_cause = root_cause_analysis.primary_cause.lower()
        ai_cause = str(ai_insights.get('root_cause', '')).lower()
        
        if any(word in ai_cause for word in rule_cause.split() if len(word) > 3):
            agreement_score += 0.1
        
        return agreement_score
    
    def _synthesize_recommendations(self, error_analysis: ErrorAnalysisResult,
                                  root_cause_analysis: RootCauseAnalysis,
                                  gap_analysis: GapAnalysisResult,
                                  ai_insights: Dict[str, Any]) -> List[str]:
        """Synthesize recommendations from all analysis sources."""
        
        recommendations = []
        
        # Add rule-based recommendations
        recommendations.append(error_analysis.suggested_remediation)
        
        # Add root cause recommendations
        if root_cause_analysis.learning_pathway:
            recommendations.append(f"Follow learning pathway: {' → '.join(root_cause_analysis.learning_pathway[:3])}")
        
        # Add gap-based recommendations
        if gap_analysis.priority_gaps:
            recommendations.append(f"Address priority gaps: {', '.join(gap_analysis.priority_gaps[:2])}")
        
        # Add AI recommendations
        ai_remediation = ai_insights.get('remediation', {})
        if isinstance(ai_remediation, list):
            recommendations.extend(ai_remediation[:2])
        elif isinstance(ai_remediation, str):
            recommendations.append(ai_remediation)
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))[:5]
        return [rec for rec in unique_recommendations if rec and len(rec.strip()) > 10]
    
    def _determine_learning_priority(self, gap_analysis: GapAnalysisResult,
                                   root_cause_analysis: RootCauseAnalysis) -> str:
        """Determine the learning priority level."""
        
        # Check gap analysis priority
        if gap_analysis.priority_gaps:
            return "immediate"
        
        # Check root cause priority
        if root_cause_analysis.remediation_priority == "immediate":
            return "immediate"
        elif root_cause_analysis.remediation_priority == "short_term":
            return "short_term"
        else:
            return "long_term"
    
    def _assess_student_level(self, error_analysis: ErrorAnalysisResult,
                            gap_analysis: GapAnalysisResult,
                            ai_insights: Dict[str, Any]) -> str:
        """Assess the student's current level based on error patterns."""
        
        # Use AI insights if available
        ai_level = ai_insights.get('educational_insights', {}).get('student_level', '')
        if ai_level:
            return ai_level
        
        # Use gap analysis
        if gap_analysis.total_remediation_time > 20:
            return "Significant gaps - needs foundational review"
        elif gap_analysis.total_remediation_time > 10:
            return "Moderate gaps - targeted remediation needed"
        else:
            return "Minor gaps - ready for advancement"
    
    def _assess_problem_difficulty(self, error_analysis: ErrorAnalysisResult,
                                 ai_insights: Dict[str, Any]) -> str:
        """Assess if the problem difficulty matches student level."""
        
        ai_assessment = ai_insights.get('educational_insights', {}).get('problem_appropriateness', '')
        if ai_assessment:
            return ai_assessment
        
        # Use error severity as proxy
        if error_analysis.severity.value in ['high', 'critical']:
            return "Problem may be too difficult for current level"
        else:
            return "Problem difficulty appears appropriate"
    
    def _extract_metacognitive_insights(self, ai_insights: Dict[str, Any]) -> List[str]:
        """Extract insights about student's metacognitive awareness."""
        
        insights = []
        
        student_assessment = ai_insights.get('student_assessment', {})
        if isinstance(student_assessment, dict):
            metacognitive = student_assessment.get('metacognitive_awareness', '')
            if metacognitive:
                insights.append(f"Metacognitive awareness: {metacognitive}")
        
        # Add general insights
        if 'confidence' in str(ai_insights).lower():
            insights.append("Student confidence factors identified")
        
        return insights

class HybridErrorAnalyzer:
    """Main class that orchestrates hybrid error analysis."""
    
    def __init__(self, llm_client=None):
        self.correctness_engine = AnswerCorrectnessEngine()
        self.error_analyzer = ErrorAnalysisEngine()
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.gap_identifier = KnowledgeGapIdentifier()
        self.ai_analyzer = AIErrorAnalyzer(llm_client)
        self.synthesizer = AnalysisSynthesizer()
    
    async def analyze_comprehensive(self, question: str, student_answer: str, 
                                  correct_answer: str, work_shown: str = "",
                                  student_history: Dict[str, Any] = None) -> HybridAnalysisResult:
        """
        Perform comprehensive hybrid analysis of a mathematical error.
        
        Args:
            question: The mathematical question/problem
            student_answer: Student's submitted answer
            correct_answer: The expected correct answer
            work_shown: Any work/steps shown by student
            student_history: Historical performance data
            
        Returns:
            HybridAnalysisResult with comprehensive analysis
        """
        
        # Step 1: Check answer correctness
        correctness_result = self.correctness_engine.check_correctness(
            student_answer, correct_answer, question
        )
        
        # If correct, no further analysis needed
        if correctness_result.is_correct:
            return self._create_correct_answer_result(correctness_result)
        
        # Step 2: Rule-based error analysis
        error_analysis = self.error_analyzer.analyze_error(
            question, student_answer, correct_answer, work_shown
        )
        
        # Step 3: Root cause analysis
        root_cause_analysis = self.root_cause_analyzer.analyze_root_cause(
            error_analysis, work_shown, student_history
        )
        
        # Step 4: Knowledge gap identification
        gap_analysis = self.gap_identifier.identify_gaps(
            error_analysis, root_cause_analysis, student_history
        )
        
        # Step 5: AI-powered analysis (parallel to rule-based)
        ai_insights = await self.ai_analyzer.analyze_error_with_ai(
            question, student_answer, correct_answer, work_shown
        )
        
        # Step 6: Synthesize all analyses
        synthesis = self.synthesizer.synthesize_analyses(
            correctness_result, error_analysis, root_cause_analysis, 
            gap_analysis, ai_insights
        )
        
        # Step 7: Create comprehensive result
        return HybridAnalysisResult(
            correctness_result=correctness_result,
            error_analysis=error_analysis,
            root_cause_analysis=root_cause_analysis,
            gap_analysis=gap_analysis,
            ai_insights=ai_insights,
            ai_confidence=ai_insights.get('confidence', 0.5),
            primary_diagnosis=synthesis['primary_diagnosis'],
            confidence_score=synthesis['confidence_score'],
            recommended_actions=synthesis['recommended_actions'],
            learning_priority=synthesis['learning_priority'],
            student_level_assessment=synthesis['student_level_assessment'],
            problem_difficulty_match=synthesis['problem_difficulty_match'],
            metacognitive_insights=synthesis['metacognitive_insights']
        )
    
    def _create_correct_answer_result(self, correctness_result: ValidationResult) -> HybridAnalysisResult:
        """Create a result for correct answers."""
        return HybridAnalysisResult(
            correctness_result=correctness_result,
            error_analysis=None,
            root_cause_analysis=None,
            gap_analysis=None,
            ai_insights={'message': 'Answer is correct'},
            ai_confidence=1.0,
            primary_diagnosis="Answer is correct - no errors to analyze",
            confidence_score=1.0,
            recommended_actions=["Continue with next problem", "Consider more challenging problems"],
            learning_priority="advancement",
            student_level_assessment="Demonstrates understanding of this concept",
            problem_difficulty_match="Appropriate or potentially too easy",
            metacognitive_insights=["Student successfully solved the problem"]
        )
    
    def export_analysis_report(self, result: HybridAnalysisResult, 
                             format: str = "detailed") -> Dict[str, Any]:
        """
        Export the analysis result in various formats.
        
        Args:
            result: The hybrid analysis result
            format: "detailed", "summary", or "student_friendly"
            
        Returns:
            Formatted analysis report
        """
        
        if format == "summary":
            return self._create_summary_report(result)
        elif format == "student_friendly":
            return self._create_student_friendly_report(result)
        else:
            return self._create_detailed_report(result)
    
    def _create_summary_report(self, result: HybridAnalysisResult) -> Dict[str, Any]:
        """Create a summary report for educators."""
        return {
            'student_answer_status': 'correct' if result.correctness_result.is_correct else 'incorrect',
            'primary_issue': result.primary_diagnosis,
            'confidence': result.confidence_score,
            'immediate_actions': result.recommended_actions[:2],
            'learning_priority': result.learning_priority,
            'key_gaps': result.gap_analysis.priority_gaps if result.gap_analysis else [],
            'estimated_remediation_time': result.gap_analysis.total_remediation_time if result.gap_analysis else 0
        }
    
    def _create_student_friendly_report(self, result: HybridAnalysisResult) -> Dict[str, Any]:
        """Create a student-friendly report."""
        if result.correctness_result.is_correct:
            return {
                'message': "Great job! Your answer is correct.",
                'next_steps': ["Try a more challenging problem", "Help a classmate with this topic"]
            }
        
        return {
            'message': "Let's work on understanding this problem better.",
            'what_to_focus_on': result.recommended_actions[0] if result.recommended_actions else "Review the concepts",
            'helpful_questions': result.root_cause_analysis.diagnostic_questions[:2] if result.root_cause_analysis else [],
            'encouragement': "Everyone makes mistakes - that's how we learn!"
        }
    
    def _create_detailed_report(self, result: HybridAnalysisResult) -> Dict[str, Any]:
        """Create a detailed report with all analysis components."""
        return {
            'correctness_analysis': {
                'is_correct': result.correctness_result.is_correct,
                'correctness_level': result.correctness_result.correctness_level.value,
                'partial_credit': result.correctness_result.partial_credit,
                'validation_method': result.correctness_result.validation_method
            },
            'error_analysis': {
                'error_type': result.error_analysis.error_type.value if result.error_analysis else None,
                'error_subtype': result.error_analysis.error_subtype if result.error_analysis else None,
                'root_cause': result.error_analysis.root_cause if result.error_analysis else None,
                'severity': result.error_analysis.severity.value if result.error_analysis else None
            },
            'root_cause_analysis': {
                'primary_cause': result.root_cause_analysis.primary_cause if result.root_cause_analysis else None,
                'contributing_factors': result.root_cause_analysis.contributing_factors if result.root_cause_analysis else [],
                'prerequisite_gaps': result.root_cause_analysis.prerequisite_gaps if result.root_cause_analysis else [],
                'learning_pathway': result.root_cause_analysis.learning_pathway if result.root_cause_analysis else []
            },
            'knowledge_gaps': {
                'identified_gaps': [gap.concept_name for gap in result.gap_analysis.identified_gaps] if result.gap_analysis else [],
                'priority_gaps': result.gap_analysis.priority_gaps if result.gap_analysis else [],
                'total_remediation_time': result.gap_analysis.total_remediation_time if result.gap_analysis else 0,
                'learning_sequence': result.gap_analysis.learning_sequence if result.gap_analysis else []
            },
            'ai_insights': result.ai_insights,
            'synthesis': {
                'primary_diagnosis': result.primary_diagnosis,
                'confidence_score': result.confidence_score,
                'recommended_actions': result.recommended_actions,
                'learning_priority': result.learning_priority,
                'student_level_assessment': result.student_level_assessment,
                'problem_difficulty_match': result.problem_difficulty_match,
                'metacognitive_insights': result.metacognitive_insights
            }
        } 