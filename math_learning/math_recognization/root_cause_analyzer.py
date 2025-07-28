"""
Root Cause Analyzer

This module provides diagnostic analysis to identify the root causes of
mathematical errors and suggest targeted remediation strategies.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re

from .error_analysis_engine import ErrorAnalysisResult, ErrorType

class DiagnosticCategory(Enum):
    """Categories of diagnostic analysis."""
    PREREQUISITE_KNOWLEDGE = "prerequisite_knowledge"
    PROCEDURAL_UNDERSTANDING = "procedural_understanding"
    CONCEPTUAL_UNDERSTANDING = "conceptual_understanding"
    APPLICATION_SKILLS = "application_skills"
    ATTENTION_FOCUS = "attention_focus"

@dataclass
class DiagnosticQuestion:
    """Represents a diagnostic question to probe understanding."""
    question: str
    category: DiagnosticCategory
    expected_correct_response: str
    indicates_gap: str
    follow_up_questions: List[str] = field(default_factory=list)

@dataclass
class RootCauseAnalysis:
    """Result of root cause analysis."""
    primary_cause: str
    contributing_factors: List[str]
    prerequisite_gaps: List[str]
    diagnostic_questions: List[DiagnosticQuestion]
    confidence: float
    remediation_priority: str  # "immediate", "short_term", "long_term"
    learning_pathway: List[str]

class PrerequisiteAnalyzer:
    """Analyzes missing prerequisite knowledge."""
    
    def __init__(self):
        self.prerequisite_map = {
            # Arithmetic prerequisites
            'multiplication': ['addition', 'counting', 'number_recognition'],
            'division': ['multiplication', 'subtraction', 'basic_facts'],
            'fractions': ['division', 'multiplication', 'part_whole_concepts'],
            
            # Algebra prerequisites
            'linear_equations': ['arithmetic_operations', 'inverse_operations', 'variable_concepts'],
            'quadratic_equations': ['linear_equations', 'factoring', 'algebraic_manipulation'],
            'factoring': ['multiplication', 'distributive_property', 'algebraic_expressions'],
            
            # Advanced topics
            'exponent_rules': ['multiplication', 'repeated_multiplication', 'pattern_recognition'],
            'logarithms': ['exponent_rules', 'inverse_functions', 'algebraic_manipulation'],
            
            # Geometry prerequisites
            'area_perimeter': ['multiplication', 'addition', 'measurement_concepts'],
            'pythagorean_theorem': ['exponent_rules', 'square_roots', 'right_triangles']
        }
    
    def identify_prerequisite_gaps(self, missing_concepts: List[str]) -> List[str]:
        """Identify prerequisite concepts that may be missing."""
        all_prerequisites = set()
        
        for concept in missing_concepts:
            prerequisites = self.prerequisite_map.get(concept, [])
            all_prerequisites.update(prerequisites)
            
            # Recursively find prerequisites of prerequisites
            for prereq in prerequisites:
                deeper_prereqs = self.prerequisite_map.get(prereq, [])
                all_prerequisites.update(deeper_prereqs)
        
        return list(all_prerequisites)
    
    def generate_prerequisite_diagnostics(self, prerequisites: List[str]) -> List[DiagnosticQuestion]:
        """Generate diagnostic questions for prerequisite concepts."""
        diagnostic_questions = []
        
        for prereq in prerequisites:
            questions = self._get_prerequisite_questions(prereq)
            diagnostic_questions.extend(questions)
        
        return diagnostic_questions
    
    def _get_prerequisite_questions(self, concept: str) -> List[DiagnosticQuestion]:
        """Get diagnostic questions for a specific prerequisite concept."""
        question_bank = {
            'addition': [
                DiagnosticQuestion(
                    question="What is 7 + 8?",
                    category=DiagnosticCategory.PREREQUISITE_KNOWLEDGE,
                    expected_correct_response="15",
                    indicates_gap="basic_addition_facts",
                    follow_up_questions=["Can you count on your fingers to check?", "What about 6 + 9?"]
                )
            ],
            'multiplication': [
                DiagnosticQuestion(
                    question="What is 6 × 7?",
                    category=DiagnosticCategory.PREREQUISITE_KNOWLEDGE,
                    expected_correct_response="42",
                    indicates_gap="multiplication_tables",
                    follow_up_questions=["Do you know your times tables?", "What is 6 × 8?"]
                )
            ],
            'variable_concepts': [
                DiagnosticQuestion(
                    question="If x = 5, what is x + 3?",
                    category=DiagnosticCategory.CONCEPTUAL_UNDERSTANDING,
                    expected_correct_response="8",
                    indicates_gap="variable_substitution",
                    follow_up_questions=["What does the letter x represent?", "Can x represent different numbers?"]
                )
            ],
            'inverse_operations': [
                DiagnosticQuestion(
                    question="If 5 + ? = 12, what goes in the blank?",
                    category=DiagnosticCategory.PROCEDURAL_UNDERSTANDING,
                    expected_correct_response="7",
                    indicates_gap="inverse_relationship_addition_subtraction",
                    follow_up_questions=["How did you figure that out?", "What operation undoes addition?"]
                )
            ]
        }
        
        return question_bank.get(concept, [])

class ProceduralAnalyzer:
    """Analyzes procedural understanding and method application."""
    
    def __init__(self):
        self.procedure_steps = {
            'equation_solving': [
                "Identify the variable to solve for",
                "Isolate terms with the variable on one side",
                "Apply inverse operations to both sides",
                "Simplify to get the variable alone",
                "Check the solution by substitution"
            ],
            'fraction_addition': [
                "Check if denominators are the same",
                "Find common denominator if different",
                "Convert fractions to equivalent fractions",
                "Add numerators, keep denominator",
                "Simplify the result if possible"
            ],
            'quadratic_factoring': [
                "Identify the quadratic form ax² + bx + c",
                "Look for common factors first",
                "Find two numbers that multiply to ac and add to b",
                "Rewrite the middle term using these numbers",
                "Factor by grouping"
            ]
        }
    
    def analyze_procedural_error(self, error_analysis: ErrorAnalysisResult, 
                               work_shown: str = "") -> Dict[str, Any]:
        """Analyze where the student went wrong in their procedure."""
        analysis = {
            'procedure_identified': None,
            'step_analysis': [],
            'failed_at_step': None,
            'missing_steps': [],
            'incorrect_steps': []
        }
        
        # Identify which procedure the student was attempting
        procedure = self._identify_procedure(error_analysis, work_shown)
        analysis['procedure_identified'] = procedure
        
        if procedure and work_shown:
            # Analyze each step of the student's work
            student_steps = self._parse_student_steps(work_shown)
            correct_steps = self.procedure_steps.get(procedure, [])
            
            analysis['step_analysis'] = self._compare_steps(student_steps, correct_steps)
            analysis['failed_at_step'] = self._identify_failure_point(analysis['step_analysis'])
        
        return analysis
    
    def _identify_procedure(self, error_analysis: ErrorAnalysisResult, work_shown: str) -> Optional[str]:
        """Identify which mathematical procedure the student was attempting."""
        # Look for keywords and patterns to identify the procedure
        combined_text = f"{error_analysis.description} {work_shown}".lower()
        
        if any(word in combined_text for word in ['solve', 'equation', 'x =']):
            return 'equation_solving'
        elif any(word in combined_text for word in ['fraction', 'add', 'denominator']):
            return 'fraction_addition'
        elif any(word in combined_text for word in ['factor', 'quadratic', 'x²']):
            return 'quadratic_factoring'
        
        return None
    
    def _parse_student_steps(self, work_shown: str) -> List[str]:
        """Parse the student's work into individual steps."""
        if not work_shown:
            return []
        
        # Split by lines and clean up
        steps = [step.strip() for step in work_shown.split('\n') if step.strip()]
        return steps
    
    def _compare_steps(self, student_steps: List[str], correct_steps: List[str]) -> List[Dict[str, Any]]:
        """Compare student steps with correct procedure steps."""
        comparison = []
        
        for i, correct_step in enumerate(correct_steps):
            step_analysis = {
                'step_number': i + 1,
                'correct_step': correct_step,
                'student_step': student_steps[i] if i < len(student_steps) else None,
                'status': 'missing'
            }
            
            if i < len(student_steps):
                # Analyze if the student step is correct
                if self._step_is_correct(student_steps[i], correct_step):
                    step_analysis['status'] = 'correct'
                else:
                    step_analysis['status'] = 'incorrect'
                    step_analysis['error_description'] = self._describe_step_error(student_steps[i], correct_step)
            
            comparison.append(step_analysis)
        
        return comparison
    
    def _step_is_correct(self, student_step: str, correct_step_description: str) -> bool:
        """Determine if a student's step aligns with the correct procedure."""
        # This is a simplified check - in practice, you'd need more sophisticated analysis
        # For now, we'll use keyword matching
        keywords = correct_step_description.lower().split()
        student_lower = student_step.lower()
        
        # Check if key concepts are present
        key_matches = sum(1 for keyword in keywords if keyword in student_lower)
        return key_matches >= len(keywords) * 0.6  # At least 60% of keywords match
    
    def _describe_step_error(self, student_step: str, correct_step: str) -> str:
        """Describe what went wrong in a specific step."""
        return f"Student wrote '{student_step}' but should have focused on: {correct_step}"
    
    def _identify_failure_point(self, step_analysis: List[Dict[str, Any]]) -> Optional[int]:
        """Identify at which step the student first went wrong."""
        for step in step_analysis:
            if step['status'] in ['incorrect', 'missing']:
                return step['step_number']
        return None

class ConceptualAnalyzer:
    """Analyzes conceptual understanding gaps."""
    
    def __init__(self):
        self.concept_relationships = {
            'fractions': {
                'part_whole_relationship': "Understanding that fractions represent parts of a whole",
                'equivalent_fractions': "Knowing that fractions can look different but have same value",
                'fraction_operations': "Understanding how to add, subtract, multiply, divide fractions"
            },
            'negative_numbers': {
                'number_line_understanding': "Visualizing negative numbers on a number line",
                'operations_with_negatives': "Rules for adding, subtracting negative numbers",
                'negative_in_context': "Understanding negative numbers in real-world contexts"
            },
            'algebraic_variables': {
                'variable_as_unknown': "Understanding variables represent unknown quantities",
                'variable_manipulation': "Knowing how to work with variables in expressions",
                'equation_balance': "Understanding that equations represent balanced relationships"
            }
        }
    
    def analyze_conceptual_gaps(self, error_analysis: ErrorAnalysisResult) -> Dict[str, Any]:
        """Analyze conceptual understanding gaps."""
        analysis = {
            'primary_concept': None,
            'related_concepts': [],
            'misconceptions_identified': [],
            'conceptual_diagnostics': []
        }
        
        # Identify the primary concept involved
        primary_concept = self._identify_primary_concept(error_analysis)
        analysis['primary_concept'] = primary_concept
        
        if primary_concept:
            # Get related concepts
            analysis['related_concepts'] = list(self.concept_relationships.get(primary_concept, {}).keys())
            
            # Identify specific misconceptions
            analysis['misconceptions_identified'] = self._identify_misconceptions(error_analysis, primary_concept)
            
            # Generate conceptual diagnostic questions
            analysis['conceptual_diagnostics'] = self._generate_conceptual_diagnostics(primary_concept)
        
        return analysis
    
    def _identify_primary_concept(self, error_analysis: ErrorAnalysisResult) -> Optional[str]:
        """Identify the primary mathematical concept involved in the error."""
        missing_concepts = error_analysis.missing_concepts
        
        # Map specific concepts to broader conceptual categories
        concept_mapping = {
            'fraction_addition': 'fractions',
            'fraction_multiplication': 'fractions',
            'common_denominators': 'fractions',
            'negative_exponents': 'negative_numbers',
            'variable_concepts': 'algebraic_variables',
            'equation_solving': 'algebraic_variables'
        }
        
        for concept in missing_concepts:
            if concept in concept_mapping:
                return concept_mapping[concept]
        
        return None
    
    def _identify_misconceptions(self, error_analysis: ErrorAnalysisResult, concept: str) -> List[str]:
        """Identify specific misconceptions based on the error pattern."""
        misconceptions = []
        
        if concept == 'fractions':
            if 'addition' in error_analysis.error_subtype:
                misconceptions.append("Adding fractions by adding numerators and denominators separately")
            if 'multiplication' in error_analysis.error_subtype:
                misconceptions.append("Confusing fraction multiplication with addition rules")
        
        elif concept == 'negative_numbers':
            if 'exponent' in error_analysis.error_subtype:
                misconceptions.append("Treating negative exponents as making the result negative")
        
        elif concept == 'algebraic_variables':
            if 'equation' in error_analysis.error_subtype:
                misconceptions.append("Not understanding that operations must be applied to both sides")
        
        return misconceptions
    
    def _generate_conceptual_diagnostics(self, concept: str) -> List[DiagnosticQuestion]:
        """Generate diagnostic questions for conceptual understanding."""
        diagnostics = []
        
        if concept == 'fractions':
            diagnostics.extend([
                DiagnosticQuestion(
                    question="What does the fraction 3/4 mean?",
                    category=DiagnosticCategory.CONCEPTUAL_UNDERSTANDING,
                    expected_correct_response="3 parts out of 4 equal parts",
                    indicates_gap="part_whole_understanding"
                ),
                DiagnosticQuestion(
                    question="Are 2/4 and 1/2 the same? Why?",
                    category=DiagnosticCategory.CONCEPTUAL_UNDERSTANDING,
                    expected_correct_response="Yes, they represent the same amount",
                    indicates_gap="equivalent_fractions"
                )
            ])
        
        elif concept == 'algebraic_variables':
            diagnostics.extend([
                DiagnosticQuestion(
                    question="What does the letter x represent in math?",
                    category=DiagnosticCategory.CONCEPTUAL_UNDERSTANDING,
                    expected_correct_response="An unknown number or variable",
                    indicates_gap="variable_concept"
                ),
                DiagnosticQuestion(
                    question="If x + 3 = 10, what is x?",
                    category=DiagnosticCategory.APPLICATION_SKILLS,
                    expected_correct_response="7",
                    indicates_gap="basic_equation_solving"
                )
            ])
        
        return diagnostics

class RootCauseAnalyzer:
    """Main analyzer for identifying root causes of mathematical errors."""
    
    def __init__(self):
        self.prerequisite_analyzer = PrerequisiteAnalyzer()
        self.procedural_analyzer = ProceduralAnalyzer()
        self.conceptual_analyzer = ConceptualAnalyzer()
    
    def analyze_root_cause(self, error_analysis: ErrorAnalysisResult, 
                          work_shown: str = "", student_history: Dict[str, Any] = None) -> RootCauseAnalysis:
        """
        Perform comprehensive root cause analysis of a mathematical error.
        
        Args:
            error_analysis: Result from error analysis engine
            work_shown: Student's work/steps shown
            student_history: Historical performance data for the student
            
        Returns:
            RootCauseAnalysis with detailed diagnostic information
        """
        # Identify prerequisite gaps
        prerequisite_gaps = self.prerequisite_analyzer.identify_prerequisite_gaps(
            error_analysis.missing_concepts
        )
        
        # Analyze procedural issues
        procedural_analysis = self.procedural_analyzer.analyze_procedural_error(
            error_analysis, work_shown
        )
        
        # Analyze conceptual gaps
        conceptual_analysis = self.conceptual_analyzer.analyze_conceptual_gaps(error_analysis)
        
        # Synthesize findings into primary cause
        primary_cause = self._determine_primary_cause(
            error_analysis, procedural_analysis, conceptual_analysis, prerequisite_gaps
        )
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            error_analysis, procedural_analysis, conceptual_analysis
        )
        
        # Generate comprehensive diagnostic questions
        diagnostic_questions = self._generate_comprehensive_diagnostics(
            error_analysis, prerequisite_gaps, conceptual_analysis
        )
        
        # Determine remediation priority
        priority = self._determine_remediation_priority(error_analysis, prerequisite_gaps)
        
        # Create learning pathway
        learning_pathway = self._create_learning_pathway(
            prerequisite_gaps, error_analysis.missing_concepts
        )
        
        return RootCauseAnalysis(
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            prerequisite_gaps=prerequisite_gaps,
            diagnostic_questions=diagnostic_questions,
            confidence=self._calculate_analysis_confidence(error_analysis, procedural_analysis),
            remediation_priority=priority,
            learning_pathway=learning_pathway
        )
    
    def _determine_primary_cause(self, error_analysis: ErrorAnalysisResult,
                               procedural_analysis: Dict[str, Any],
                               conceptual_analysis: Dict[str, Any],
                               prerequisite_gaps: List[str]) -> str:
        """Determine the primary root cause of the error."""
        # Priority order: Prerequisites -> Conceptual -> Procedural -> Careless
        
        if prerequisite_gaps:
            return f"Missing prerequisite knowledge: {', '.join(prerequisite_gaps[:2])}"
        
        elif conceptual_analysis.get('misconceptions_identified'):
            return f"Conceptual misconception: {conceptual_analysis['misconceptions_identified'][0]}"
        
        elif procedural_analysis.get('failed_at_step'):
            return f"Procedural error at step {procedural_analysis['failed_at_step']}"
        
        elif error_analysis.error_type == ErrorType.ARITHMETIC:
            return "Arithmetic calculation error"
        
        else:
            return error_analysis.root_cause
    
    def _identify_contributing_factors(self, error_analysis: ErrorAnalysisResult,
                                     procedural_analysis: Dict[str, Any],
                                     conceptual_analysis: Dict[str, Any]) -> List[str]:
        """Identify factors that contributed to the error."""
        factors = []
        
        # Add error-specific factors
        if error_analysis.error_type == ErrorType.CARELESS:
            factors.append("Attention to detail")
            factors.append("Checking work")
        
        # Add procedural factors
        if procedural_analysis.get('missing_steps'):
            factors.append("Incomplete procedure knowledge")
        
        # Add conceptual factors
        if conceptual_analysis.get('misconceptions_identified'):
            factors.append("Conceptual misconceptions")
        
        return factors
    
    def _generate_comprehensive_diagnostics(self, error_analysis: ErrorAnalysisResult,
                                          prerequisite_gaps: List[str],
                                          conceptual_analysis: Dict[str, Any]) -> List[DiagnosticQuestion]:
        """Generate comprehensive diagnostic questions."""
        diagnostics = []
        
        # Add prerequisite diagnostics
        if prerequisite_gaps:
            diagnostics.extend(
                self.prerequisite_analyzer.generate_prerequisite_diagnostics(prerequisite_gaps[:3])
            )
        
        # Add conceptual diagnostics
        diagnostics.extend(conceptual_analysis.get('conceptual_diagnostics', []))
        
        # Add error-specific diagnostics
        diagnostics.extend(error_analysis.diagnostic_questions)
        
        return diagnostics
    
    def _determine_remediation_priority(self, error_analysis: ErrorAnalysisResult,
                                      prerequisite_gaps: List[str]) -> str:
        """Determine the priority level for remediation."""
        if prerequisite_gaps:
            return "immediate"  # Prerequisites must be addressed first
        
        elif error_analysis.severity.value in ['high', 'critical']:
            return "immediate"
        
        elif error_analysis.error_type == ErrorType.CONCEPTUAL:
            return "short_term"
        
        else:
            return "long_term"
    
    def _create_learning_pathway(self, prerequisite_gaps: List[str],
                               missing_concepts: List[str]) -> List[str]:
        """Create a recommended learning pathway."""
        pathway = []
        
        # Start with prerequisites (in dependency order)
        if prerequisite_gaps:
            # Sort prerequisites by dependency (simplified)
            sorted_prereqs = sorted(prerequisite_gaps)
            pathway.extend(sorted_prereqs)
        
        # Add main concepts
        pathway.extend(missing_concepts)
        
        # Add practice and application
        pathway.append("guided_practice")
        pathway.append("independent_practice")
        pathway.append("application_problems")
        
        return pathway
    
    def _calculate_analysis_confidence(self, error_analysis: ErrorAnalysisResult,
                                     procedural_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in the root cause analysis."""
        base_confidence = error_analysis.confidence
        
        # Adjust based on available information
        if procedural_analysis.get('step_analysis'):
            base_confidence += 0.1  # More confidence with work shown
        
        if error_analysis.error_type in [ErrorType.ARITHMETIC, ErrorType.PROCEDURAL]:
            base_confidence += 0.1  # More confidence with clear error types
        
        return min(base_confidence, 1.0) 