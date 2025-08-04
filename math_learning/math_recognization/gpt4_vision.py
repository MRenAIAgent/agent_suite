"""
GPT-4 Vision Integration for Math Test Analysis

This module provides integration with OpenAI's GPT-4 Vision API for analyzing
math test images, understanding layout, and parsing question-answer pairs.
"""

import base64
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import os

try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

class VisionAnalysisType(Enum):
    """Types of vision analysis."""
    LAYOUT_DETECTION = "layout_detection"
    QUESTION_PARSING = "question_parsing"
    ANSWER_EXTRACTION = "answer_extraction"
    COMPREHENSIVE = "comprehensive"

@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair extracted from test image."""
    question_id: str
    question_text: str
    student_answer: str
    expected_answer: Optional[str] = None
    work_shown: str = ""
    confidence: float = 0.0
    bounding_box: Optional[Dict[str, int]] = None
    problem_type: str = "general_math"

@dataclass
class VisionAnalysisResult:
    """Result from GPT-4 Vision analysis."""
    success: bool
    question_answer_pairs: List[QuestionAnswerPair]
    layout_analysis: Dict[str, Any]
    processing_confidence: float
    analysis_type: VisionAnalysisType
    processing_time_ms: float
    total_questions_detected: int
    error_message: Optional[str] = None
    raw_response: Optional[str] = None

class GPT4VisionClient:
    """Client for GPT-4 Vision API integration."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize GPT-4 Vision client.
        
        Args:
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
            model: Model to use (gpt-4o supports vision capabilities)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not available. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Use gpt-4o which supports vision (replaces deprecated gpt-4-vision-preview)
        self.model = model
        self.client = AsyncOpenAI(api_key=self.api_key)
    
    async def analyze_test_image(self, 
                               image_data: Union[bytes, str],
                               ocr_text: str = "",
                               analysis_type: VisionAnalysisType = VisionAnalysisType.COMPREHENSIVE,
                               **kwargs) -> VisionAnalysisResult:
        """
        Analyze a math test image using GPT-4 Vision.
        
        Args:
            image_data: Image as bytes or base64 string
            ocr_text: OCR-extracted text to provide context
            analysis_type: Type of analysis to perform
            **kwargs: Additional options
            
        Returns:
            VisionAnalysisResult with parsed question-answer pairs
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Prepare image data
            if isinstance(image_data, bytes):
                image_b64 = base64.b64encode(image_data).decode('utf-8')
            else:
                image_b64 = image_data
            
            # Create analysis prompt based on type
            prompt = self._create_analysis_prompt(analysis_type, ocr_text)
            
            # Prepare messages for GPT-4 Vision
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call GPT-4 Vision API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.1
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Parse response
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                return self._parse_vision_response(
                    content, analysis_type, processing_time
                )
            else:
                return VisionAnalysisResult(
                    success=False,
                    question_answer_pairs=[],
                    layout_analysis={},
                    processing_confidence=0.0,
                    analysis_type=analysis_type,
                    processing_time_ms=processing_time,
                    total_questions_detected=0,
                    error_message="Empty response from GPT-4 Vision"
                )
                
        except Exception as e:
            return VisionAnalysisResult(
                success=False,
                question_answer_pairs=[],
                layout_analysis={},
                processing_confidence=0.0,
                analysis_type=analysis_type,
                processing_time_ms=0,
                total_questions_detected=0,
                error_message=str(e)
            )
    
    def _create_analysis_prompt(self, analysis_type: VisionAnalysisType, 
                              ocr_text: str = "") -> str:
        """Create analysis prompt based on the requested analysis type."""
        
        base_context = f"""
You are an expert at analyzing student math tests. Analyze this math test image and extract structured information.

{f"OCR Context (mathematical text extracted from image): {ocr_text}" if ocr_text else ""}

Please analyze the image carefully and provide a JSON response with the following structure:
"""
        
        if analysis_type == VisionAnalysisType.COMPREHENSIVE:
            return base_context + """
{
  "questions": [
    {
      "question_id": "Q1",
      "question_text": "Complete original problem statement",
      "student_answer": "Student's written answer",
      "work_shown": "Any work/steps shown by student",
      "confidence": 0.95,
      "problem_type": "algebra|arithmetic|fractions|geometry|etc",
      "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 50}
    }
  ],
  "layout_analysis": {
    "total_questions": 3,
    "layout_type": "vertical|grid|mixed",
    "image_quality": "high|medium|low",
    "handwriting_quality": "clear|moderate|poor",
    "has_printed_text": true,
    "has_handwritten_text": true
  },
  "processing_confidence": 0.9,
  "notes": "Any additional observations"
}

IMPORTANT INSTRUCTIONS:
1. Extract the COMPLETE original problem text, not just the equation
2. Identify the student's final answer, not intermediate steps
3. Include all visible work/steps the student showed
4. Be precise about mathematical notation and symbols
5. If text is unclear, indicate lower confidence
6. Identify the type of math problem (algebra, arithmetic, etc.)
7. Provide bounding box coordinates if possible (approximate is fine)
"""
        
        elif analysis_type == VisionAnalysisType.QUESTION_PARSING:
            return base_context + """
Focus specifically on identifying and extracting the original math problems/questions:

{
  "questions": [
    {
      "question_id": "Q1", 
      "question_text": "Complete problem statement as written",
      "problem_type": "algebra|arithmetic|fractions|etc",
      "confidence": 0.95
    }
  ]
}
"""
        
        elif analysis_type == VisionAnalysisType.ANSWER_EXTRACTION:
            return base_context + """
Focus specifically on extracting student answers and work:

{
  "answers": [
    {
      "question_id": "Q1",
      "student_answer": "Final answer student provided", 
      "work_shown": "Steps/work shown by student",
      "confidence": 0.95
    }
  ]
}
"""
        
        else:  # LAYOUT_DETECTION
            return base_context + """
Focus on analyzing the overall layout and structure:

{
  "layout_analysis": {
    "total_questions": 3,
    "layout_type": "vertical|grid|mixed",
    "question_regions": [
      {"question_id": "Q1", "bounding_box": {"x": 0, "y": 0, "width": 100, "height": 50}}
    ],
    "image_quality": "high|medium|low",
    "text_types": ["printed", "handwritten"]
  }
}
"""
    
    def _parse_vision_response(self, response_text: str, 
                             analysis_type: VisionAnalysisType,
                             processing_time: float) -> VisionAnalysisResult:
        """Parse GPT-4 Vision response into structured result."""
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
            else:
                # Fallback: try to parse the entire response as JSON
                parsed_data = json.loads(response_text)
            
            # Extract question-answer pairs
            question_pairs = []
            questions_data = parsed_data.get('questions', [])
            
            for i, q_data in enumerate(questions_data):
                question_pair = QuestionAnswerPair(
                    question_id=q_data.get('question_id', f'Q{i+1}'),
                    question_text=q_data.get('question_text', ''),
                    student_answer=q_data.get('student_answer', ''),
                    work_shown=q_data.get('work_shown', ''),
                    confidence=q_data.get('confidence', 0.8),
                    bounding_box=q_data.get('bounding_box'),
                    problem_type=q_data.get('problem_type', 'general_math')
                )
                question_pairs.append(question_pair)
            
            # Extract layout analysis
            layout_analysis = parsed_data.get('layout_analysis', {})
            
            # Calculate overall confidence
            processing_confidence = parsed_data.get('processing_confidence', 0.8)
            if question_pairs:
                avg_confidence = sum(q.confidence for q in question_pairs) / len(question_pairs)
                processing_confidence = (processing_confidence + avg_confidence) / 2
            
            return VisionAnalysisResult(
                success=True,
                question_answer_pairs=question_pairs,
                layout_analysis=layout_analysis,
                processing_confidence=processing_confidence,
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                total_questions_detected=len(question_pairs),
                raw_response=response_text
            )
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract information with regex
            return self._fallback_parse_response(response_text, analysis_type, processing_time)
        except Exception as e:
            return VisionAnalysisResult(
                success=False,
                question_answer_pairs=[],
                layout_analysis={},
                processing_confidence=0.0,
                analysis_type=analysis_type,
                processing_time_ms=processing_time,
                total_questions_detected=0,
                error_message=f"Parsing error: {str(e)}",
                raw_response=response_text
            )
    
    def _fallback_parse_response(self, response_text: str,
                               analysis_type: VisionAnalysisType,
                               processing_time: float) -> VisionAnalysisResult:
        """Fallback parsing when JSON parsing fails."""
        
        question_pairs = []
        
        # Try to extract question-answer pairs with regex
        # Look for patterns like "Question: ... Answer: ..."
        qa_pattern = r'(?:Question|Problem)\s*\d*:?\s*([^?!.]*[?!.]?)\s*(?:Answer|Solution)\s*:?\s*([^\n]+)'
        matches = re.findall(qa_pattern, response_text, re.IGNORECASE | re.MULTILINE)
        
        for i, (question, answer) in enumerate(matches):
            question_pair = QuestionAnswerPair(
                question_id=f'Q{i+1}',
                question_text=question.strip(),
                student_answer=answer.strip(),
                confidence=0.6,  # Lower confidence for regex parsing
                problem_type='general_math'
            )
            question_pairs.append(question_pair)
        
        return VisionAnalysisResult(
            success=len(question_pairs) > 0,
            question_answer_pairs=question_pairs,
            layout_analysis={'parsing_method': 'regex_fallback'},
            processing_confidence=0.6,
            analysis_type=analysis_type,
            processing_time_ms=processing_time,
            total_questions_detected=len(question_pairs),
            raw_response=response_text
        )

class VisionTextProcessor:
    """Processes GPT-4 Vision results for mathematical content."""
    
    @staticmethod
    def enhance_with_ocr_context(vision_result: VisionAnalysisResult,
                               ocr_text: str) -> VisionAnalysisResult:
        """Enhance vision results with OCR context for better accuracy."""
        
        if not ocr_text or not vision_result.success:
            return vision_result
        
        # Extract mathematical expressions from OCR
        ocr_equations = re.findall(r'[^.!?]*=[^.!?]*', ocr_text)
        
        # Try to match OCR equations with vision-detected questions
        enhanced_pairs = []
        for pair in vision_result.question_answer_pairs:
            enhanced_pair = pair
            
            # If question text is vague, try to enhance with OCR
            if len(pair.question_text) < 10 and ocr_equations:
                # Use the first unmatched equation as the question
                for eq in ocr_equations:
                    if eq not in [p.question_text for p in enhanced_pairs]:
                        enhanced_pair.question_text = eq.strip()
                        enhanced_pair.confidence = min(pair.confidence + 0.1, 1.0)
                        break
            
            enhanced_pairs.append(enhanced_pair)
        
        # Update the result
        vision_result.question_answer_pairs = enhanced_pairs
        return vision_result
    
    @staticmethod
    def validate_mathematical_content(pairs: List[QuestionAnswerPair]) -> List[QuestionAnswerPair]:
        """Validate and clean mathematical content in question-answer pairs."""
        
        validated_pairs = []
        
        for pair in pairs:
            # Clean and validate question text
            question = pair.question_text.strip()
            if not question or len(question) < 3:
                continue  # Skip invalid questions
            
            # Clean and validate answer
            answer = pair.student_answer.strip()
            
            # Check if this looks like a math problem
            has_math_content = any(char in question for char in '=+-*/()[]{}x²³√∞')
            has_math_words = any(word in question.lower() for word in 
                               ['solve', 'factor', 'simplify', 'calculate', 'find', 'what is'])
            
            if has_math_content or has_math_words:
                validated_pairs.append(pair)
        
        return validated_pairs

# Convenience functions
async def analyze_math_test_image(image_data: Union[bytes, str],
                                ocr_text: str = "",
                                api_key: str = None) -> VisionAnalysisResult:
    """
    Convenience function to analyze a math test image with GPT-4 Vision.
    
    Args:
        image_data: Image as bytes or base64 string
        ocr_text: OCR-extracted text for context
        api_key: OpenAI API key
        
    Returns:
        VisionAnalysisResult with parsed question-answer pairs
    """
    client = GPT4VisionClient(api_key=api_key)
    result = await client.analyze_test_image(
        image_data=image_data,
        ocr_text=ocr_text,
        analysis_type=VisionAnalysisType.COMPREHENSIVE
    )
    
    # Enhance with OCR context
    result = VisionTextProcessor.enhance_with_ocr_context(result, ocr_text)
    
    # Validate content
    result.question_answer_pairs = VisionTextProcessor.validate_mathematical_content(
        result.question_answer_pairs
    )
    result.total_questions_detected = len(result.question_answer_pairs)
    
    return result 