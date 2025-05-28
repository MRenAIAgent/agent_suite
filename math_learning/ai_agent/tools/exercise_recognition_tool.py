"""
Exercise Recognition Tool for the Math Tutoring Agent.

This tool recognizes and classifies math exercises from various sources
including images, text, and structured content.
"""

import re
import base64
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# Optional imports for image processing
try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from llm.llm import LLMBase


@dataclass
class RecognizedExercise:
    """Data class for a recognized exercise."""
    id: str
    title: str
    problem_statement: str
    concept: str
    difficulty: str
    exercise_type: str
    expected_answer: Optional[str] = None
    hints: List[str] = None
    confidence: float = 0.0
    source: str = "unknown"
    
    def __post_init__(self):
        if self.hints is None:
            self.hints = []


class ExerciseRecognitionTool:
    """
    Tool for recognizing and classifying math exercises.
    
    This tool can:
    - Extract exercises from images using OCR
    - Parse text-based exercises
    - Classify exercise types and concepts
    - Determine difficulty levels
    - Extract problem statements and expected answers
    """
    
    def __init__(self, llm: LLMBase):
        """
        Initialize the Exercise Recognition Tool.
        
        Args:
            llm: Language model for exercise analysis
        """
        self.llm = llm
        self.name = "exercise_recognition"
        self.description = "Recognize and classify math exercises from images or text"
        
        # Exercise type patterns
        self.exercise_patterns = {
            "equation_solving": [
                r"solve\s+for\s+[a-z]",
                r"find\s+[a-z]",
                r"[a-z]\s*=",
                r"solve\s*:",
                r"\w+\s*=\s*\d+"
            ],
            "simplification": [
                r"simplify",
                r"reduce",
                r"factor",
                r"expand"
            ],
            "word_problem": [
                r"a\s+\w+\s+has",
                r"if\s+\w+",
                r"how\s+many",
                r"what\s+is\s+the",
                r"find\s+the\s+\w+\s+of"
            ],
            "computation": [
                r"\d+\s*[\+\-\*\/]\s*\d+",
                r"calculate",
                r"compute",
                r"evaluate"
            ],
            "proof": [
                r"prove\s+that",
                r"show\s+that",
                r"demonstrate",
                r"verify"
            ]
        }
        
        # Concept patterns
        self.concept_patterns = {
            "algebra": [
                r"[a-z]\s*[\+\-\*\/]\s*\d+",
                r"\d*[a-z][\²³⁴]?",
                r"quadratic",
                r"linear\s+equation"
            ],
            "fractions": [
                r"\d+\/\d+",
                r"fraction",
                r"numerator",
                r"denominator"
            ],
            "geometry": [
                r"triangle",
                r"circle",
                r"area",
                r"perimeter",
                r"volume",
                r"angle",
                r"degree"
            ],
            "calculus": [
                r"derivative",
                r"integral",
                r"limit",
                r"d\/dx",
                r"∫",
                r"∂"
            ],
            "trigonometry": [
                r"sin|cos|tan",
                r"sine|cosine|tangent",
                r"θ|phi|alpha|beta"
            ],
            "statistics": [
                r"mean",
                r"median",
                r"mode",
                r"standard\s+deviation",
                r"probability"
            ]
        }
    
    async def execute(self, content: str, source_type: str = "text", **kwargs) -> Dict[str, Any]:
        """
        Execute exercise recognition on the provided content.
        
        Args:
            content: The content to analyze (text, base64 image, or file path)
            source_type: Type of source ("text", "image", "base64")
            **kwargs: Additional parameters
            
        Returns:
            Dict containing recognized exercises and analysis
        """
        try:
            # Extract text content based on source type
            if source_type == "image":
                text_content = await self._extract_text_from_image_file(content)
            elif source_type == "base64":
                text_content = await self._extract_text_from_base64(content)
            else:
                text_content = content
            
            if not text_content or text_content.strip() == "":
                return {
                    "success": False,
                    "error": "No text content could be extracted",
                    "exercises": []
                }
            
            # Recognize exercises from the text
            exercises = await self._recognize_exercises_from_text(text_content)
            
            # Enhance exercises with LLM analysis
            enhanced_exercises = await self._enhance_exercises_with_llm(exercises, text_content)
            
            return {
                "success": True,
                "source_type": source_type,
                "extracted_text": text_content,
                "total_exercises": len(enhanced_exercises),
                "exercises": [self._exercise_to_dict(ex) for ex in enhanced_exercises],
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exercise recognition failed: {str(e)}",
                "exercises": []
            }
    
    async def _extract_text_from_image_file(self, image_path: str) -> str:
        """Extract text from an image file using OCR."""
        if not TESSERACT_AVAILABLE:
            return await self._llm_extract_text_from_image(image_path)
        
        try:
            image = Image.open(image_path)
            # Configure OCR for mathematical text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :,?'
            text = pytesseract.image_to_string(image, config=custom_config)
            return self._clean_ocr_text(text)
        except Exception:
            return await self._llm_extract_text_from_image(image_path)
    
    async def _extract_text_from_base64(self, base64_content: str) -> str:
        """Extract text from base64 encoded content."""
        try:
            # Try to decode as text first
            decoded_text = base64.b64decode(base64_content).decode('utf-8')
            return decoded_text
        except:
            # If not text, try as image
            if TESSERACT_AVAILABLE:
                try:
                    import io
                    image_data = base64.b64decode(base64_content)
                    image = Image.open(io.BytesIO(image_data))
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(image, config=custom_config)
                    return self._clean_ocr_text(text)
                except Exception:
                    pass
            
            # Fallback to LLM analysis
            return await self._llm_extract_text_from_base64(base64_content)
    
    async def _llm_extract_text_from_image(self, image_path: str) -> str:
        """Use LLM to extract text from image when OCR is not available."""
        prompt = f"""
        Extract all mathematical content from this image: {image_path}
        
        Focus on:
        - Mathematical problems and equations
        - Exercise instructions
        - Numbers and mathematical symbols
        - Any text that appears to be part of a math exercise
        
        Return only the extracted text content.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
        except Exception:
            pass
        
        return ""
    
    async def _llm_extract_text_from_base64(self, base64_content: str) -> str:
        """Use LLM to extract text from base64 content."""
        prompt = f"""
        This is base64 encoded content that may contain mathematical exercises.
        Extract all mathematical content including:
        - Mathematical problems and equations
        - Exercise instructions
        - Numbers and mathematical symbols
        
        Base64 content: {base64_content[:100]}...
        
        Return only the extracted mathematical text content.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
        except Exception:
            pass
        
        return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors for mathematical symbols
        replacements = {
            'x': '×',  # multiplication
            'X': '×',
            '÷': '/',  # division
            '—': '-',  # em dash to minus
            '–': '-',  # en dash to minus
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    async def _recognize_exercises_from_text(self, text: str) -> List[RecognizedExercise]:
        """Recognize individual exercises from text content."""
        exercises = []
        
        # Split text into potential exercise segments
        segments = self._split_into_exercise_segments(text)
        
        for i, segment in enumerate(segments):
            if self._is_likely_exercise(segment):
                exercise = RecognizedExercise(
                    id=f"recognized_ex_{i+1}",
                    title=f"Exercise {i+1}",
                    problem_statement=segment.strip(),
                    concept=self._classify_concept(segment),
                    difficulty=self._estimate_difficulty(segment),
                    exercise_type=self._classify_exercise_type(segment),
                    confidence=self._calculate_confidence(segment),
                    source="text_recognition"
                )
                exercises.append(exercise)
        
        return exercises
    
    def _split_into_exercise_segments(self, text: str) -> List[str]:
        """Split text into individual exercise segments."""
        # Common exercise delimiters
        delimiters = [
            r'\n\d+[\.\)]\s*',  # Numbered exercises (1. or 1))
            r'\n[a-z][\.\)]\s*',  # Lettered exercises (a. or a))
            r'\nExercise\s+\d+',  # "Exercise 1"
            r'\nProblem\s+\d+',   # "Problem 1"
            r'\n\n+',            # Double newlines
        ]
        
        segments = [text]
        
        for delimiter in delimiters:
            new_segments = []
            for segment in segments:
                parts = re.split(delimiter, segment)
                new_segments.extend([part.strip() for part in parts if part.strip()])
            segments = new_segments
        
        return segments
    
    def _is_likely_exercise(self, text: str) -> bool:
        """Determine if a text segment is likely a math exercise."""
        if len(text.strip()) < 10:  # Too short
            return False
        
        # Check for mathematical content
        math_indicators = [
            r'\d+',  # Contains numbers
            r'[a-z]\s*=',  # Variable assignment
            r'solve|find|calculate|simplify|factor|expand',  # Action words
            r'[\+\-\*\/\=\(\)]',  # Mathematical symbols
            r'x|y|z',  # Common variables
        ]
        
        score = 0
        for pattern in math_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        
        return score >= 2
    
    def _classify_concept(self, text: str) -> str:
        """Classify the mathematical concept of an exercise."""
        text_lower = text.lower()
        
        for concept, patterns in self.concept_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return concept
        
        return "general_math"
    
    def _classify_exercise_type(self, text: str) -> str:
        """Classify the type of exercise."""
        text_lower = text.lower()
        
        for ex_type, patterns in self.exercise_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return ex_type
        
        return "problem_solving"
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate the difficulty level of an exercise."""
        # Simple heuristics for difficulty estimation
        complexity_indicators = {
            "easy": [r'^\d+\s*[\+\-]\s*\d+$', r'simple', r'basic'],
            "medium": [r'[a-z]', r'fraction', r'\d+\/\d+'],
            "hard": [r'[a-z][\²³⁴]', r'quadratic', r'derivative', r'integral', r'complex']
        }
        
        text_lower = text.lower()
        
        for difficulty, patterns in complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return difficulty
        
        # Default based on length and complexity
        if len(text) > 100:
            return "hard"
        elif len(text) > 50:
            return "medium"
        else:
            return "easy"
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score for exercise recognition."""
        score = 0.0
        
        # Mathematical symbols increase confidence
        if re.search(r'[\+\-\*\/\=]', text):
            score += 0.3
        
        # Variables increase confidence
        if re.search(r'[a-z]', text):
            score += 0.2
        
        # Numbers increase confidence
        if re.search(r'\d+', text):
            score += 0.2
        
        # Action words increase confidence
        if re.search(r'solve|find|calculate|simplify', text, re.IGNORECASE):
            score += 0.3
        
        return min(score, 1.0)
    
    async def _enhance_exercises_with_llm(self, exercises: List[RecognizedExercise], original_text: str) -> List[RecognizedExercise]:
        """Enhance recognized exercises using LLM analysis."""
        if not exercises:
            return exercises
        
        try:
            # Create a prompt for LLM enhancement
            exercise_texts = [ex.problem_statement for ex in exercises]
            prompt = f"""
            Analyze these mathematical exercises and provide enhanced information:
            
            Original text: {original_text}
            
            Exercises:
            {chr(10).join([f"{i+1}. {text}" for i, text in enumerate(exercise_texts)])}
            
            For each exercise, provide:
            1. A clear title
            2. The exact mathematical concept (algebra, geometry, calculus, etc.)
            3. Difficulty level (easy, medium, hard)
            4. Exercise type (equation_solving, computation, word_problem, etc.)
            5. Expected answer if determinable
            6. 2-3 helpful hints
            
            Format as JSON with this structure:
            {{
                "exercises": [
                    {{
                        "title": "...",
                        "concept": "...",
                        "difficulty": "...",
                        "exercise_type": "...",
                        "expected_answer": "...",
                        "hints": ["...", "..."]
                    }}
                ]
            }}
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Try to parse JSON response
                import json
                try:
                    llm_data = json.loads(content)
                    if "exercises" in llm_data:
                        for i, ex_data in enumerate(llm_data["exercises"]):
                            if i < len(exercises):
                                # Update exercise with LLM enhancements
                                if "title" in ex_data:
                                    exercises[i].title = ex_data["title"]
                                if "concept" in ex_data:
                                    exercises[i].concept = ex_data["concept"]
                                if "difficulty" in ex_data:
                                    exercises[i].difficulty = ex_data["difficulty"]
                                if "exercise_type" in ex_data:
                                    exercises[i].exercise_type = ex_data["exercise_type"]
                                if "expected_answer" in ex_data:
                                    exercises[i].expected_answer = ex_data["expected_answer"]
                                if "hints" in ex_data:
                                    exercises[i].hints = ex_data["hints"]
                                exercises[i].confidence = min(exercises[i].confidence + 0.3, 1.0)
                except json.JSONDecodeError:
                    pass  # Keep original exercises if JSON parsing fails
        
        except Exception:
            pass  # Keep original exercises if LLM enhancement fails
        
        return exercises
    
    def _exercise_to_dict(self, exercise: RecognizedExercise) -> Dict[str, Any]:
        """Convert RecognizedExercise to dictionary."""
        return {
            "id": exercise.id,
            "title": exercise.title,
            "problem_statement": exercise.problem_statement,
            "concept": exercise.concept,
            "difficulty": exercise.difficulty,
            "exercise_type": exercise.exercise_type,
            "expected_answer": exercise.expected_answer,
            "hints": exercise.hints,
            "confidence": exercise.confidence,
            "source": exercise.source
        } 