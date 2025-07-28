"""
Complete Test Analyzer

This module provides the complete workflow for analyzing student math tests,
from image processing through comprehensive error analysis and remediation.
"""

import asyncio
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import base64
from datetime import datetime

from math_learning.math_recognization.hybrid_error_analyzer import HybridErrorAnalyzer, HybridAnalysisResult
from math_learning.math_recognization.hybrid_image_processor import HybridImageProcessor, ProcessingStrategy

class TestAnalysisStatus(Enum):
    """Status of test analysis process."""
    PENDING = "pending"
    PROCESSING_IMAGE = "processing_image"
    EXTRACTING_CONTENT = "extracting_content"
    ANALYZING_ANSWERS = "analyzing_answers"
    GENERATING_INSIGHTS = "generating_insights"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair from the test."""
    question_id: str
    question_text: str
    student_answer: str
    expected_answer: Optional[str] = None
    work_shown: str = ""
    confidence: float = 0.0
    bounding_box: Optional[Dict[str, int]] = None

@dataclass
class TestImageAnalysis:
    """Result of test image processing."""
    image_quality_score: float
    layout_detected: bool
    total_questions_found: int
    question_answer_pairs: List[QuestionAnswerPair]
    processing_confidence: float
    ocr_method_used: str
    processing_time_ms: float

@dataclass
class CompleteTestAnalysis:
    """Complete analysis result for a math test."""
    # Test metadata
    test_id: str
    student_id: str
    timestamp: datetime
    
    # Image processing results
    image_analysis: TestImageAnalysis
    
    # Answer analyses
    answer_analyses: List[HybridAnalysisResult]
    
    # Overall assessment
    overall_score_percentage: float
    total_questions: int
    correct_answers: int
    partially_correct: int
    incorrect_answers: int
    
    # Learning insights
    concept_gaps_identified: List[str]
    prerequisite_gaps: List[str]
    learning_priority_areas: List[str]
    estimated_remediation_time: float
    
    # Recommendations
    immediate_remediation: List[str]
    practice_recommendations: List[str]
    next_topics_ready: List[str]
    
    # Confidence and metadata
    analysis_confidence: float
    processing_status: TestAnalysisStatus
    error_messages: List[str] = field(default_factory=list)

class ProductionImageProcessor:
    """Production image processor using Mathpix OCR + GPT-4 Vision."""
    
    def __init__(self, mathpix_app_id: str = None, mathpix_app_key: str = None,
                 openai_api_key: str = None):
        """
        Initialize production image processor.
        
        Args:
            mathpix_app_id: Mathpix application ID
            mathpix_app_key: Mathpix application key
            openai_api_key: OpenAI API key for GPT-4 Vision
        """
        self.hybrid_processor = HybridImageProcessor(
            mathpix_app_id=mathpix_app_id,
            mathpix_app_key=mathpix_app_key,
            openai_api_key=openai_api_key,
            default_strategy=ProcessingStrategy.PARALLEL
        )
        self.supported_formats = ['jpg', 'jpeg', 'png', 'pdf']
    
    async def process_test_image(self, image_data: bytes, 
                               image_format: str = "jpg") -> TestImageAnalysis:
        """
        Process a test image to extract questions and answers using
        Mathpix OCR + GPT-4 Vision hybrid approach.
        
        Args:
            image_data: Raw image data as bytes
            image_format: Format of the image
            
        Returns:
            TestImageAnalysis with extracted question-answer pairs
        """
        
        # Process with hybrid approach
        result = await self.hybrid_processor.process_test_image(image_data)
        
        if not result.success:
            # Return empty result if processing failed
            return TestImageAnalysis(
                image_quality_score=0.0,
                layout_detected=False,
                total_questions_found=0,
                question_answer_pairs=[],
                processing_confidence=0.0,
                ocr_method_used="hybrid_failed",
                processing_time_ms=result.total_processing_time_ms
            )
        
        # Convert hybrid result to TestImageAnalysis format
        question_pairs = []
        for qa_dict in result.question_answer_pairs:
            pair = QuestionAnswerPair(
                question_id=qa_dict.get('question_id', 'Q1'),
                question_text=qa_dict.get('question_text', ''),
                student_answer=qa_dict.get('student_answer', ''),
                expected_answer=qa_dict.get('expected_answer'),
                work_shown=qa_dict.get('work_shown', ''),
                confidence=qa_dict.get('confidence', 0.0),
                bounding_box=qa_dict.get('bounding_box')
            )
            question_pairs.append(pair)
        
        # Determine image quality and layout detection
        metadata = result.metadata
        layout_detected = bool(metadata.get('layout_analysis', {}).get('total_questions', 0) > 0)
        
        # Calculate image quality score based on confidence and processing success
        image_quality = "high"
        if result.vision_result and result.vision_result.layout_analysis:
            image_quality = result.vision_result.layout_analysis.get('image_quality', 'medium')
        
        quality_score = {
            'high': 0.9,
            'medium': 0.7,
            'low': 0.5
        }.get(image_quality, 0.7)
        
        # Determine OCR method used
        ocr_method = "hybrid_mathpix_gpt4v"
        if result.processing_strategy == ProcessingStrategy.OCR_ONLY:
            ocr_method = "mathpix_only"
        elif result.processing_strategy == ProcessingStrategy.VISION_ONLY:
            ocr_method = "gpt4v_only"
        
        return TestImageAnalysis(
            image_quality_score=quality_score,
            layout_detected=layout_detected,
            total_questions_found=len(question_pairs),
            question_answer_pairs=question_pairs,
            processing_confidence=result.confidence_score,
            ocr_method_used=ocr_method,
            processing_time_ms=result.total_processing_time_ms
        )

# Legacy ImageProcessor for backward compatibility
class ImageProcessor(ProductionImageProcessor):
    """Legacy ImageProcessor - redirects to ProductionImageProcessor."""
    pass

class AnswerKeyMatcher:
    """Matches extracted questions with answer keys."""
    
    def __init__(self):
        self.matching_strategies = ['exact_match', 'fuzzy_match', 'semantic_match']
    
    def match_with_answer_key(self, question_pairs: List[QuestionAnswerPair],
                            answer_key: Dict[str, str]) -> List[QuestionAnswerPair]:
        """Match extracted questions with provided answer key."""
        
        matched_pairs = []
        
        for pair in question_pairs:
            # Try to find matching answer in answer key
            expected_answer = self._find_expected_answer(pair, answer_key)
            
            pair.expected_answer = expected_answer
            matched_pairs.append(pair)
        
        return matched_pairs
    
    def _find_expected_answer(self, pair: QuestionAnswerPair, 
                            answer_key: Dict[str, str]) -> Optional[str]:
        """Find the expected answer for a question."""
        
        # Direct question ID match
        if pair.question_id in answer_key:
            return answer_key[pair.question_id]
        
        # Try to match by question content
        for key, answer in answer_key.items():
            if self._questions_match(pair.question_text, key):
                return answer
        
        return None
    
    def _questions_match(self, question1: str, question2: str) -> bool:
        """Check if two questions are the same (simplified)."""
        # Normalize and compare
        q1_normalized = question1.lower().replace(" ", "").replace(":", "")
        q2_normalized = question2.lower().replace(" ", "").replace(":", "")
        
        return q1_normalized in q2_normalized or q2_normalized in q1_normalized

class TestReportGenerator:
    """Generates comprehensive test analysis reports."""
    
    def __init__(self):
        pass
    
    def generate_student_report(self, analysis: CompleteTestAnalysis) -> Dict[str, Any]:
        """Generate a student-friendly report."""
        
        score_percentage = analysis.overall_score_percentage
        
        # Determine encouragement message
        if score_percentage >= 90:
            encouragement = "Excellent work! You have a strong understanding of these concepts."
        elif score_percentage >= 80:
            encouragement = "Great job! You're doing well with most concepts."
        elif score_percentage >= 70:
            encouragement = "Good effort! Let's work on a few areas to improve your understanding."
        elif score_percentage >= 60:
            encouragement = "You're making progress! Let's focus on building stronger foundations."
        else:
            encouragement = "Don't worry - everyone learns at their own pace. Let's work together to strengthen your understanding."
        
        return {
            'student_name': f"Student {analysis.student_id}",
            'test_date': analysis.timestamp.strftime("%B %d, %Y"),
            'overall_score': f"{score_percentage:.1f}%",
            'questions_correct': f"{analysis.correct_answers}/{analysis.total_questions}",
            'encouragement_message': encouragement,
            'areas_of_strength': self._identify_strengths(analysis),
            'areas_to_improve': analysis.concept_gaps_identified[:3],
            'next_steps': analysis.immediate_remediation[:2],
            'practice_suggestions': analysis.practice_recommendations[:3],
            'estimated_study_time': f"{analysis.estimated_remediation_time:.1f} hours"
        }
    
    def generate_teacher_report(self, analysis: CompleteTestAnalysis) -> Dict[str, Any]:
        """Generate a detailed report for teachers."""
        
        return {
            'student_id': analysis.student_id,
            'test_analysis_summary': {
                'overall_score': analysis.overall_score_percentage,
                'total_questions': analysis.total_questions,
                'correct': analysis.correct_answers,
                'partial': analysis.partially_correct,
                'incorrect': analysis.incorrect_answers,
                'analysis_confidence': analysis.analysis_confidence
            },
            'detailed_question_analysis': [
                {
                    'question_id': pair.question_id,
                    'question': pair.question_text,
                    'student_answer': pair.student_answer,
                    'expected_answer': pair.expected_answer,
                    'analysis': self._summarize_question_analysis(analysis.answer_analyses[i]) if i < len(analysis.answer_analyses) else None
                }
                for i, pair in enumerate(analysis.image_analysis.question_answer_pairs)
            ],
            'learning_gaps_analysis': {
                'concept_gaps': analysis.concept_gaps_identified,
                'prerequisite_gaps': analysis.prerequisite_gaps,
                'priority_areas': analysis.learning_priority_areas,
                'estimated_remediation_time': analysis.estimated_remediation_time
            },
            'instructional_recommendations': {
                'immediate_interventions': analysis.immediate_remediation,
                'practice_recommendations': analysis.practice_recommendations,
                'ready_for_advancement': analysis.next_topics_ready
            },
            'technical_details': {
                'image_processing_confidence': analysis.image_analysis.processing_confidence,
                'ocr_method': analysis.image_analysis.ocr_method_used,
                'processing_time_ms': analysis.image_analysis.processing_time_ms
            }
        }
    
    def _identify_strengths(self, analysis: CompleteTestAnalysis) -> List[str]:
        """Identify areas where the student performed well."""
        strengths = []
        
        if analysis.correct_answers > 0:
            strengths.append("Demonstrated understanding in some areas")
        
        if analysis.partially_correct > 0:
            strengths.append("Shows partial understanding and problem-solving approach")
        
        # Analyze specific strengths from individual question analyses
        for answer_analysis in analysis.answer_analyses:
            if answer_analysis.correctness_result.is_correct:
                strengths.append("Strong performance on this type of problem")
                break
        
        return strengths[:3]  # Limit to top 3
    
    def _summarize_question_analysis(self, analysis: HybridAnalysisResult) -> Dict[str, Any]:
        """Summarize analysis for a single question."""
        
        if analysis.correctness_result.is_correct:
            return {
                'status': 'correct',
                'feedback': 'Well done!',
                'confidence': analysis.confidence_score
            }
        
        return {
            'status': 'incorrect',
            'primary_issue': analysis.primary_diagnosis,
            'recommended_action': analysis.recommended_actions[0] if analysis.recommended_actions else "Review concepts",
            'learning_priority': analysis.learning_priority,
            'confidence': analysis.confidence_score
        }

class CompleteTestAnalyzer:
    """Main class that orchestrates the complete test analysis workflow."""
    
    def __init__(self, llm_client=None, mathpix_app_id: str = None, 
                 mathpix_app_key: str = None, openai_api_key: str = None):
        """
        Initialize the complete test analyzer.
        
        Args:
            llm_client: LLM client for hybrid analysis
            mathpix_app_id: Mathpix application ID for OCR
            mathpix_app_key: Mathpix application key for OCR
            openai_api_key: OpenAI API key for GPT-4 Vision
        """
        self.image_processor = ProductionImageProcessor(
            mathpix_app_id=mathpix_app_id,
            mathpix_app_key=mathpix_app_key,
            openai_api_key=openai_api_key
        )
        self.answer_matcher = AnswerKeyMatcher()
        self.hybrid_analyzer = HybridErrorAnalyzer(llm_client)
        self.report_generator = TestReportGenerator()
    
    async def analyze_test(self, image_data: bytes, test_id: str, student_id: str,
                         answer_key: Dict[str, str] = None,
                         image_format: str = "jpg") -> CompleteTestAnalysis:
        """
        Perform complete analysis of a student's math test.
        
        Args:
            image_data: The test image as bytes
            test_id: Unique identifier for the test
            student_id: Unique identifier for the student
            answer_key: Dictionary mapping questions to correct answers
            image_format: Format of the image (jpg, png, etc.)
            
        Returns:
            CompleteTestAnalysis with comprehensive results
        """
        
        analysis = CompleteTestAnalysis(
            test_id=test_id,
            student_id=student_id,
            timestamp=datetime.now(),
            image_analysis=None,
            answer_analyses=[],
            overall_score_percentage=0.0,
            total_questions=0,
            correct_answers=0,
            partially_correct=0,
            incorrect_answers=0,
            concept_gaps_identified=[],
            prerequisite_gaps=[],
            learning_priority_areas=[],
            estimated_remediation_time=0.0,
            immediate_remediation=[],
            practice_recommendations=[],
            next_topics_ready=[],
            analysis_confidence=0.0,
            processing_status=TestAnalysisStatus.PENDING
        )
        
        try:
            # Step 1: Process the image
            analysis.processing_status = TestAnalysisStatus.PROCESSING_IMAGE
            image_analysis = await self.image_processor.process_test_image(
                image_data, image_format
            )
            analysis.image_analysis = image_analysis
            
            # Step 2: Match with answer key if provided
            analysis.processing_status = TestAnalysisStatus.EXTRACTING_CONTENT
            if answer_key:
                matched_pairs = self.answer_matcher.match_with_answer_key(
                    image_analysis.question_answer_pairs, answer_key
                )
                analysis.image_analysis.question_answer_pairs = matched_pairs
            
            # Step 3: Analyze each question-answer pair
            analysis.processing_status = TestAnalysisStatus.ANALYZING_ANSWERS
            answer_analyses = []
            
            for pair in analysis.image_analysis.question_answer_pairs:
                if pair.expected_answer:  # Only analyze if we have expected answer
                    question_analysis = await self.hybrid_analyzer.analyze_comprehensive(
                        question=pair.question_text,
                        student_answer=pair.student_answer,
                        correct_answer=pair.expected_answer,
                        work_shown=pair.work_shown
                    )
                    answer_analyses.append(question_analysis)
            
            analysis.answer_analyses = answer_analyses
            
            # Step 4: Generate overall insights
            analysis.processing_status = TestAnalysisStatus.GENERATING_INSIGHTS
            self._generate_overall_insights(analysis)
            
            analysis.processing_status = TestAnalysisStatus.COMPLETED
            
        except Exception as e:
            analysis.processing_status = TestAnalysisStatus.FAILED
            analysis.error_messages.append(f"Analysis failed: {str(e)}")
        
        return analysis
    
    def _generate_overall_insights(self, analysis: CompleteTestAnalysis):
        """Generate overall insights from individual question analyses."""
        
        if not analysis.answer_analyses:
            return
        
        # Calculate scores
        total_questions = len(analysis.answer_analyses)
        correct_count = sum(1 for a in analysis.answer_analyses if a.correctness_result.is_correct)
        partial_count = sum(1 for a in analysis.answer_analyses 
                          if a.correctness_result.correctness_level.value == 'partially_correct')
        incorrect_count = total_questions - correct_count - partial_count
        
        analysis.total_questions = total_questions
        analysis.correct_answers = correct_count
        analysis.partially_correct = partial_count
        analysis.incorrect_answers = incorrect_count
        
        # Calculate percentage score (with partial credit)
        total_points = sum(a.correctness_result.partial_credit for a in analysis.answer_analyses)
        analysis.overall_score_percentage = (total_points / total_questions) * 100 if total_questions > 0 else 0
        
        # Aggregate concept gaps
        all_gaps = []
        all_prerequisites = []
        all_remediation = []
        all_practice = []
        total_remediation_time = 0
        
        for answer_analysis in analysis.answer_analyses:
            if answer_analysis.gap_analysis:
                all_gaps.extend([gap.concept_name for gap in answer_analysis.gap_analysis.identified_gaps])
                total_remediation_time += answer_analysis.gap_analysis.total_remediation_time
            
            if answer_analysis.root_cause_analysis:
                all_prerequisites.extend(answer_analysis.root_cause_analysis.prerequisite_gaps)
            
            all_remediation.extend(answer_analysis.recommended_actions)
        
        # Remove duplicates and prioritize
        analysis.concept_gaps_identified = list(dict.fromkeys(all_gaps))[:5]
        analysis.prerequisite_gaps = list(dict.fromkeys(all_prerequisites))[:5]
        analysis.estimated_remediation_time = total_remediation_time
        
        # Prioritize recommendations
        analysis.immediate_remediation = list(dict.fromkeys(all_remediation))[:3]
        
        # Identify learning priority areas
        priority_areas = []
        for answer_analysis in analysis.answer_analyses:
            if answer_analysis.learning_priority == "immediate":
                if answer_analysis.gap_analysis and answer_analysis.gap_analysis.priority_gaps:
                    priority_areas.extend(answer_analysis.gap_analysis.priority_gaps)
        
        analysis.learning_priority_areas = list(dict.fromkeys(priority_areas))[:3]
        
        # Generate practice recommendations
        analysis.practice_recommendations = [
            "Practice similar problems with step-by-step solutions",
            "Review fundamental concepts identified in gaps",
            "Work with a tutor on priority areas"
        ]
        
        # Identify topics ready for advancement
        if analysis.overall_score_percentage >= 80:
            analysis.next_topics_ready = ["Ready for more advanced topics in strong areas"]
        
        # Calculate overall confidence
        confidences = [a.confidence_score for a in analysis.answer_analyses]
        analysis.analysis_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    def export_analysis_report(self, analysis: CompleteTestAnalysis, 
                             format: str = "teacher") -> Dict[str, Any]:
        """
        Export the complete analysis in various formats.
        
        Args:
            analysis: The complete test analysis
            format: "teacher", "student", or "json"
            
        Returns:
            Formatted report
        """
        
        if format == "student":
            return self.report_generator.generate_student_report(analysis)
        elif format == "teacher":
            return self.report_generator.generate_teacher_report(analysis)
        elif format == "json":
            return self._export_json_report(analysis)
        else:
            return self.report_generator.generate_teacher_report(analysis)
    
    def _export_json_report(self, analysis: CompleteTestAnalysis) -> Dict[str, Any]:
        """Export complete analysis as JSON-serializable dictionary."""
        
        return {
            'test_metadata': {
                'test_id': analysis.test_id,
                'student_id': analysis.student_id,
                'timestamp': analysis.timestamp.isoformat(),
                'processing_status': analysis.processing_status.value
            },
            'image_processing': {
                'quality_score': analysis.image_analysis.image_quality_score if analysis.image_analysis else 0,
                'questions_found': analysis.image_analysis.total_questions_found if analysis.image_analysis else 0,
                'processing_confidence': analysis.image_analysis.processing_confidence if analysis.image_analysis else 0,
                'ocr_method': analysis.image_analysis.ocr_method_used if analysis.image_analysis else "none"
            },
            'overall_performance': {
                'score_percentage': analysis.overall_score_percentage,
                'correct_answers': analysis.correct_answers,
                'partially_correct': analysis.partially_correct,
                'incorrect_answers': analysis.incorrect_answers,
                'total_questions': analysis.total_questions
            },
            'learning_analysis': {
                'concept_gaps': analysis.concept_gaps_identified,
                'prerequisite_gaps': analysis.prerequisite_gaps,
                'priority_areas': analysis.learning_priority_areas,
                'estimated_remediation_hours': analysis.estimated_remediation_time
            },
            'recommendations': {
                'immediate_actions': analysis.immediate_remediation,
                'practice_suggestions': analysis.practice_recommendations,
                'advancement_ready': analysis.next_topics_ready
            },
            'confidence_metrics': {
                'overall_confidence': analysis.analysis_confidence,
                'image_processing_confidence': analysis.image_analysis.processing_confidence if analysis.image_analysis else 0
            },
            'detailed_question_analyses': [
                {
                    'question_id': pair.question_id,
                    'question_text': pair.question_text,
                    'student_answer': pair.student_answer,
                    'expected_answer': pair.expected_answer,
                    'is_correct': analysis.answer_analyses[i].correctness_result.is_correct if i < len(analysis.answer_analyses) else False,
                    'primary_diagnosis': analysis.answer_analyses[i].primary_diagnosis if i < len(analysis.answer_analyses) else "Not analyzed",
                    'confidence': analysis.answer_analyses[i].confidence_score if i < len(analysis.answer_analyses) else 0.0
                }
                for i, pair in enumerate(analysis.image_analysis.question_answer_pairs if analysis.image_analysis else [])
            ]
        } 