"""
Enhanced Image Recognition Agent for Math Learning

This agent specializes in recognizing images and identifying questions/user answers
from various image sources including handwritten work, printed exercises, and digital content.
"""

import re
import base64
import io
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Optional imports for image processing
try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import pytesseract
    PIL_AVAILABLE = True
    TESSERACT_AVAILABLE = True
    ImageType = Image.Image
except ImportError:
    PIL_AVAILABLE = False
    TESSERACT_AVAILABLE = False
    ImageType = Any

# Try to import OpenCV for advanced image processing
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from .base_classes.base_agent import BaseAgent


@dataclass
class RecognizedContent:
    """Data class for recognized content from images."""
    questions: List[str]
    user_answers: List[str]
    correct_answers: List[str]
    mathematical_expressions: List[str]
    text_content: str
    confidence_score: float
    processing_method: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageAnalysisResult:
    """Comprehensive result from image analysis."""
    success: bool
    recognized_content: Optional[RecognizedContent]
    error_message: Optional[str]
    processing_details: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.recognized_content:
            result['recognized_content'] = self.recognized_content.to_dict()
        return result


class EnhancedImageRecognitionAgent(BaseAgent):
    """
    Enhanced agent for recognizing images and identifying questions/user answers.
    
    Key capabilities:
    - Multi-method text extraction (OCR + LLM)
    - Handwriting recognition
    - Mathematical expression parsing
    - Question/answer separation
    - Confidence scoring
    - Error detection and correction
    """
    
    def __init__(self, llm, prompt_manager, tools=None, thinking_pattern=None, memory_manager=None, log_manager=None, **kwargs):
        """Initialize the Enhanced Image Recognition Agent."""
        # Call parent constructor (even though it's abstract, we need to satisfy the signature)
        # Initialize base class attributes
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.tools = tools or []
        self.thinking_pattern = thinking_pattern
        self.memory_manager = memory_manager
        self.log_manager = log_manager
        
        self.name = "enhanced_image_recognition"
        self.description = "Advanced image recognition for math exercises and student work"
        
        # Check available dependencies
        self._check_dependencies()
        
        # Configure OCR settings
        self.ocr_config = {
            'math_focused': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=().abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ :,?!',
            'general': r'--oem 3 --psm 6',
            'handwriting': r'--oem 3 --psm 8',
            'single_line': r'--oem 3 --psm 7'
        }
        
        # Patterns for identifying different content types
        self.question_patterns = [
            r'(?i)(?:question|problem|exercise)\s*\d*[:\.]?\s*(.+?)(?=(?:answer|solution|student|your)|\n\n|\Z)',
            r'(?i)(?:solve|find|calculate|determine|what is|how many)\s+(.+?)(?=\n|$)',
            r'(?i)(?:\d+[\.\)]\s*)(.+?)(?=\n|$)',
            r'(?i)(?:[a-z][\.\)]\s*)(.+?)(?=\n|$)'
        ]
        
        self.answer_patterns = [
            r'(?i)(?:answer|solution|result)[:\s]*(.+?)(?=\n|$)',
            r'(?i)(?:student|your)\s*(?:answer|work|solution)[:\s]*(.+?)(?=\n|$)',
            r'(?:=\s*)([^=\n]+?)(?=\n|$)',
            r'(?i)(?:final|correct)\s*(?:answer|result)[:\s]*(.+?)(?=\n|$)'
        ]
        
        self.math_expression_patterns = [
            r'[a-zA-Z]*\s*[=]\s*[^=\n]+',
            r'\d+\s*[+\-*/÷×]\s*\d+(?:\s*[+\-*/÷×]\s*\d+)*',
            r'[a-zA-Z]+\s*[+\-*/÷×]\s*\d+(?:\s*[+\-*/÷×]\s*[a-zA-Z\d]+)*',
            r'\([^)]+\)\s*[+\-*/÷×]\s*\([^)]+\)',
            r'\d+\/\d+(?:\s*[+\-*/÷×]\s*\d+\/\d+)*'
        ]
    
    # Implement required abstract methods from BaseAgent
    
    async def arun(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input and generate a response asynchronously."""
        try:
            # Extract image data from user input or kwargs
            image_data = kwargs.get('image_data', user_input)
            image_format = kwargs.get('image_format', 'base64')
            processing_options = kwargs.get('processing_options', {})
            
            # Analyze the image
            result = await self.recognize_image(image_data, image_format, processing_options)
            
            # Format response
            if result.success and result.recognized_content:
                content = result.recognized_content
                response = f"Image analysis completed with {content.confidence_score:.2f} confidence.\n\n"
                
                if content.questions:
                    response += f"Questions found ({len(content.questions)}):\n"
                    for i, q in enumerate(content.questions, 1):
                        response += f"{i}. {q}\n"
                    response += "\n"
                
                if content.user_answers:
                    response += f"User answers found ({len(content.user_answers)}):\n"
                    for i, a in enumerate(content.user_answers, 1):
                        response += f"{i}. {a}\n"
                    response += "\n"
                
                if content.mathematical_expressions:
                    response += f"Mathematical expressions ({len(content.mathematical_expressions)}):\n"
                    for i, expr in enumerate(content.mathematical_expressions[:5], 1):
                        response += f"{i}. {expr}\n"
                    if len(content.mathematical_expressions) > 5:
                        response += f"... and {len(content.mathematical_expressions) - 5} more\n"
                
                if result.recommendations:
                    response += f"\nRecommendations:\n"
                    for i, rec in enumerate(result.recommendations, 1):
                        response += f"{i}. {rec}\n"
                
                return response
            else:
                return f"Image analysis failed: {result.error_message}"
                
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
    def run(self, user_input: str, model: str, **kwargs) -> str:
        """Process user input and generate a response synchronously."""
        return asyncio.run(self.arun(user_input, model, **kwargs))
    
    async def think(self, user_input: str, model: str) -> Dict[str, Any]:
        """Process user input using the agent's thinking process."""
        return {
            "agent_type": "enhanced_image_recognition",
            "input_analysis": {
                "has_image_data": bool(user_input and len(user_input) > 100),
                "input_length": len(user_input) if user_input else 0
            },
            "capabilities": self.capabilities,
            "processing_approach": "multi_method_extraction",
            "confidence_factors": [
                "text_extraction_quality",
                "pattern_matching_success",
                "mathematical_content_detection"
            ]
        }
    
    async def handle_single_tool_call(self, tool_name: str, tool_input: Any) -> Any:
        """Handle a single tool call."""
        if tool_name == "recognize_image":
            return await self.recognize_image(**tool_input)
        elif tool_name == "extract_questions":
            return self._extract_questions(tool_input)
        elif tool_name == "extract_answers":
            return self._extract_answers(tool_input)
        elif tool_name == "extract_math_expressions":
            return self._extract_mathematical_expressions(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def add_tool(self, tool) -> None:
        """Add a tool to the agent's toolset."""
        if tool not in self.tools:
            self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """Remove a tool from the agent's toolset."""
        for i, tool in enumerate(self.tools):
            if hasattr(tool, 'name') and tool.name == tool_name:
                del self.tools[i]
                return True
        return False
    
    async def save_memory(self, context: str) -> None:
        """Save the agent's current state to a file."""
        if self.memory_manager:
            await self.memory_manager.save_memory(context)
    
    async def load_memory(self) -> None:
        """Load the agent's state from a file."""
        if self.memory_manager:
            await self.memory_manager.load_memory()
    
    # Original image recognition methods continue below...
    
    def _check_dependencies(self):
        """Check and report available dependencies."""
        self.capabilities = {
            'pil_available': PIL_AVAILABLE,
            'tesseract_available': TESSERACT_AVAILABLE,
            'opencv_available': OPENCV_AVAILABLE,
            'llm_available': self.llm is not None
        }
        
        if not PIL_AVAILABLE:
            print("⚠️  Warning: PIL not available. Image processing will be limited.")
        if not TESSERACT_AVAILABLE:
            print("⚠️  Warning: pytesseract not available. OCR will use LLM fallback.")
        if not OPENCV_AVAILABLE:
            print("ℹ️  Info: OpenCV not available. Advanced image preprocessing disabled.")
    
    async def recognize_image(self, image_data: Union[str, bytes], 
                            image_format: str = "base64",
                            processing_options: Optional[Dict[str, Any]] = None) -> ImageAnalysisResult:
        """
        Main method to recognize and analyze image content.
        
        Args:
            image_data: Image data (base64, file path, or bytes)
            image_format: Format of image data ("base64", "file_path", "bytes", "url")
            processing_options: Optional processing configuration
            
        Returns:
            ImageAnalysisResult with comprehensive analysis
        """
        try:
            # Set default processing options
            options = processing_options or {}
            
            # Step 1: Load and preprocess image
            processed_image = await self._load_and_preprocess_image(image_data, image_format, options)
            
            # Step 2: Extract text using multiple methods
            extracted_text = await self._extract_text_multi_method(processed_image, image_data, image_format, options)
            
            # Step 3: Parse and identify content
            recognized_content = await self._parse_and_identify_content(extracted_text, options)
            
            # Step 4: Validate and enhance results
            enhanced_content = await self._validate_and_enhance_content(recognized_content, extracted_text, options)
            
            # Step 5: Generate recommendations
            recommendations = await self._generate_recommendations(enhanced_content, options)
            
            return ImageAnalysisResult(
                success=True,
                recognized_content=enhanced_content,
                error_message=None,
                processing_details={
                    'image_format': image_format,
                    'processing_method': enhanced_content.processing_method if enhanced_content else 'unknown',
                    'capabilities_used': self.capabilities,
                    'text_length': len(extracted_text),
                    'processing_timestamp': datetime.now().isoformat()
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            return ImageAnalysisResult(
                success=False,
                recognized_content=None,
                error_message=f"Image recognition failed: {str(e)}",
                processing_details={
                    'error_type': type(e).__name__,
                    'capabilities_used': self.capabilities,
                    'processing_timestamp': datetime.now().isoformat()
                },
                recommendations=["Please check image quality and try again"]
            )
    
    async def _load_and_preprocess_image(self, image_data: Union[str, bytes], 
                                       image_format: str, 
                                       options: Dict[str, Any]) -> Optional[ImageType]:
        """Load and preprocess image for optimal text extraction."""
        try:
            image = None
            
            # Load image based on format
            if image_format == "base64":
                if isinstance(image_data, str):
                    # Handle data URL format
                    if image_data.startswith('data:'):
                        image_data = image_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                else:
                    image_bytes = image_data
                
                if PIL_AVAILABLE:
                    image = Image.open(io.BytesIO(image_bytes))
                    
            elif image_format == "file_path":
                if PIL_AVAILABLE:
                    image = Image.open(image_data)
                    
            elif image_format == "bytes":
                if PIL_AVAILABLE:
                    image = Image.open(io.BytesIO(image_data))
            
            if image is None:
                return None
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing based on options
            preprocessing_level = options.get('preprocessing_level', 'standard')
            
            if preprocessing_level == 'aggressive':
                image = await self._aggressive_preprocessing(image)
            elif preprocessing_level == 'handwriting':
                image = await self._handwriting_preprocessing(image)
            else:
                image = await self._standard_preprocessing(image)
            
            return image
            
        except Exception as e:
            print(f"Error loading/preprocessing image: {e}")
            return None
    
    async def _standard_preprocessing(self, image: ImageType) -> ImageType:
        """Apply standard image preprocessing for better OCR."""
        try:
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Adjust brightness if needed
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception:
            return image
    
    async def _aggressive_preprocessing(self, image: ImageType) -> ImageType:
        """Apply aggressive preprocessing for difficult images."""
        try:
            if not OPENCV_AVAILABLE:
                return await self._standard_preprocessing(image)
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Convert back to PIL
            processed_image = Image.fromarray(thresh)
            processed_image = processed_image.convert('RGB')
            
            return processed_image
            
        except Exception:
            return await self._standard_preprocessing(image)
    
    async def _handwriting_preprocessing(self, image: ImageType) -> ImageType:
        """Specialized preprocessing for handwritten content."""
        try:
            # Enhance contrast more aggressively for handwriting
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Increase sharpness significantly
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.5)
            
            # Apply median filter to reduce noise while preserving edges
            if PIL_AVAILABLE:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
        except Exception:
            return image
    
    async def _extract_text_multi_method(self, image: Optional[ImageType], 
                                       image_data: Union[str, bytes], 
                                       image_format: str,
                                       options: Dict[str, Any]) -> str:
        """Extract text using multiple methods and combine results."""
        extracted_texts = []
        methods_used = []
        
        # Method 1: OCR with different configurations
        if image is not None and TESSERACT_AVAILABLE:
            for config_name, config in self.ocr_config.items():
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if text.strip():
                        extracted_texts.append(self._clean_ocr_text(text))
                        methods_used.append(f"ocr_{config_name}")
                except Exception:
                    continue
        
        # Method 2: LLM-based image analysis
        if self.llm:
            try:
                llm_text = await self._llm_image_analysis(image_data, image_format)
                if llm_text.strip():
                    extracted_texts.append(llm_text)
                    methods_used.append("llm_analysis")
            except Exception:
                pass
        
        # Method 3: Try to decode as text if base64
        if image_format == "base64":
            try:
                if isinstance(image_data, str):
                    decoded_text = base64.b64decode(image_data).decode('utf-8')
                    if decoded_text.strip():
                        extracted_texts.append(decoded_text)
                        methods_used.append("base64_decode")
            except Exception:
                pass
        
        # Combine and select best result
        if extracted_texts:
            # Choose the longest meaningful text
            best_text = max(extracted_texts, key=lambda x: len(x.strip()))
            return best_text
        
        return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR-extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors for mathematical symbols
        replacements = {
            'x': '×',  # multiplication (context-dependent)
            'X': '×',
            '÷': '/',
            '—': '-',  # em dash to minus
            '–': '-',  # en dash to minus
            'l': '1',  # lowercase l to 1 in math context
            'I': '1',  # uppercase I to 1 in math context
            'O': '0',  # uppercase O to 0 in math context
            'o': '0',  # lowercase o to 0 in math context
        }
        
        # Apply replacements carefully (only in mathematical contexts)
        for old, new in replacements.items():
            # Only replace if surrounded by numbers or math symbols
            if old in ['l', 'I', 'O', 'o']:
                text = re.sub(rf'\b{old}\b(?=\s*[+\-*/=\d])', new, text)
            else:
                text = text.replace(old, new)
        
        return text
    
    async def _llm_image_analysis(self, image_data: Union[str, bytes], image_format: str) -> str:
        """Use LLM to analyze image content."""
        if not self.llm:
            return ""
        
        prompt = """
        Analyze this image and extract all visible text content, paying special attention to:
        
        1. Mathematical problems and questions
        2. Student answers and work shown
        3. Correct answers or solutions
        4. Any handwritten or printed text
        5. Mathematical expressions and equations
        
        Please provide the extracted text exactly as it appears, maintaining the structure and organization.
        Separate different sections (questions, answers, work) clearly.
        
        If you can identify what are questions vs. answers vs. student work, please indicate that.
        """
        
        try:
            # This would need to be adapted based on your LLM interface
            # For now, providing a placeholder implementation
            messages = [{"role": "user", "content": prompt}]
            
            if hasattr(self.llm, 'chat_completion'):
                response = self.llm.chat_completion("gpt-4", messages)
                if hasattr(response, 'choices') and response.choices:
                    return response.choices[0].message.content.strip()
            elif hasattr(self.llm, 'generate'):
                response = await self.llm.generate(prompt)
                return response.strip()
            
        except Exception as e:
            print(f"LLM analysis error: {e}")
        
        return ""
    
    async def _parse_and_identify_content(self, extracted_text: str, options: Dict[str, Any]) -> RecognizedContent:
        """Parse extracted text and identify questions, answers, and mathematical content."""
        if not extracted_text.strip():
            return RecognizedContent(
                questions=[],
                user_answers=[],
                correct_answers=[],
                mathematical_expressions=[],
                text_content="",
                confidence_score=0.0,
                processing_method="none",
                timestamp=datetime.now().isoformat()
            )
        
        # Extract questions
        questions = self._extract_questions(extracted_text)
        
        # Extract answers (both user and correct)
        user_answers, correct_answers = self._extract_answers(extracted_text)
        
        # Extract mathematical expressions
        math_expressions = self._extract_mathematical_expressions(extracted_text)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(extracted_text, questions, user_answers, math_expressions)
        
        return RecognizedContent(
            questions=questions,
            user_answers=user_answers,
            correct_answers=correct_answers,
            mathematical_expressions=math_expressions,
            text_content=extracted_text,
            confidence_score=confidence,
            processing_method="multi_method",
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_questions(self, text: str) -> List[str]:
        """Extract questions from the text."""
        questions = []
        
        for pattern in self.question_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                question = match.group(1).strip() if match.groups() else match.group(0).strip()
                if len(question) > 5:  # Filter out very short matches
                    questions.append(question)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_questions = []
        for q in questions:
            if q not in seen:
                seen.add(q)
                unique_questions.append(q)
        
        return unique_questions
    
    def _extract_answers(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract user answers and correct answers from the text."""
        user_answers = []
        correct_answers = []
        
        # Extract answers using patterns
        for pattern in self.answer_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                answer = match.group(1).strip() if match.groups() else match.group(0).strip()
                if len(answer) > 1:
                    # Try to determine if it's a user answer or correct answer based on context
                    context = text[max(0, match.start()-50):match.end()+50].lower()
                    if any(word in context for word in ['student', 'your', 'incorrect', 'wrong']):
                        user_answers.append(answer)
                    elif any(word in context for word in ['correct', 'right', 'solution', 'final']):
                        correct_answers.append(answer)
                    else:
                        user_answers.append(answer)  # Default to user answer
        
        # Also look for standalone mathematical results
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if '=' in line and len(line) < 50:  # Likely a mathematical result
                parts = line.split('=')
                if len(parts) == 2:
                    result = parts[1].strip()
                    if result and result not in user_answers and result not in correct_answers:
                        user_answers.append(result)
        
        return user_answers, correct_answers
    
    def _extract_mathematical_expressions(self, text: str) -> List[str]:
        """Extract mathematical expressions from the text."""
        expressions = []
        
        for pattern in self.math_expression_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                expr = match.group(0).strip()
                if len(expr) > 2:  # Filter out very short matches
                    expressions.append(expr)
        
        # Remove duplicates
        return list(set(expressions))
    
    def _calculate_confidence_score(self, text: str, questions: List[str], 
                                  answers: List[str], expressions: List[str]) -> float:
        """Calculate confidence score for the recognition results."""
        score = 0.0
        
        # Base score from text length and quality
        if len(text.strip()) > 10:
            score += 0.2
        
        # Score from identified questions
        if questions:
            score += min(0.3, len(questions) * 0.1)
        
        # Score from identified answers
        if answers:
            score += min(0.2, len(answers) * 0.1)
        
        # Score from mathematical expressions
        if expressions:
            score += min(0.2, len(expressions) * 0.05)
        
        # Bonus for mathematical content
        math_indicators = len(re.findall(r'[+\-*/=()0-9]', text))
        if math_indicators > 5:
            score += 0.1
        
        return min(1.0, score)
    
    async def _validate_and_enhance_content(self, content: RecognizedContent, 
                                          original_text: str, 
                                          options: Dict[str, Any]) -> RecognizedContent:
        """Validate and enhance the recognized content using LLM if available."""
        if not self.llm or not content.text_content:
            return content
        
        try:
            # Use LLM to validate and enhance the extraction
            enhancement_prompt = f"""
            Please analyze this extracted text and improve the identification of:
            1. Questions/problems
            2. Student answers
            3. Correct answers
            4. Mathematical expressions
            
            Original text:
            {original_text}
            
            Currently identified:
            Questions: {content.questions}
            User answers: {content.user_answers}
            Correct answers: {content.correct_answers}
            Math expressions: {content.mathematical_expressions}
            
            Please provide improved identification in JSON format:
            {{
                "questions": ["list of questions"],
                "user_answers": ["list of user answers"],
                "correct_answers": ["list of correct answers"],
                "mathematical_expressions": ["list of math expressions"],
                "confidence_adjustment": 0.1
            }}
            """
            
            if hasattr(self.llm, 'chat_completion'):
                messages = [{"role": "user", "content": enhancement_prompt}]
                response = self.llm.chat_completion("gpt-4", messages)
                if hasattr(response, 'choices') and response.choices:
                    enhanced_data = json.loads(response.choices[0].message.content)
                    
                    # Update content with enhanced data
                    content.questions = enhanced_data.get('questions', content.questions)
                    content.user_answers = enhanced_data.get('user_answers', content.user_answers)
                    content.correct_answers = enhanced_data.get('correct_answers', content.correct_answers)
                    content.mathematical_expressions = enhanced_data.get('mathematical_expressions', content.mathematical_expressions)
                    
                    # Adjust confidence score
                    adjustment = enhanced_data.get('confidence_adjustment', 0)
                    content.confidence_score = min(1.0, content.confidence_score + adjustment)
                    content.processing_method = "enhanced_with_llm"
            
        except Exception as e:
            print(f"Enhancement failed: {e}")
        
        return content
    
    async def _generate_recommendations(self, content: Optional[RecognizedContent], 
                                      options: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the recognition results."""
        recommendations = []
        
        if not content:
            recommendations.append("Unable to extract content. Please ensure image quality is good.")
            return recommendations
        
        if content.confidence_score < 0.3:
            recommendations.append("Low confidence in text extraction. Consider improving image quality.")
        
        if not content.questions:
            recommendations.append("No questions detected. Ensure the image contains clear mathematical problems.")
        
        if not content.user_answers and not content.mathematical_expressions:
            recommendations.append("No answers or mathematical work detected. Check if student work is visible.")
        
        if content.questions and not content.user_answers:
            recommendations.append("Questions found but no student answers detected. Student may need to show their work.")
        
        if len(content.mathematical_expressions) > len(content.questions):
            recommendations.append("Multiple mathematical expressions found. Consider breaking down into individual problems.")
        
        if content.confidence_score > 0.8:
            recommendations.append("High quality recognition achieved. Content ready for analysis.")
        
        return recommendations
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the image recognition agent."""
        image_data = kwargs.get('image_data')
        image_format = kwargs.get('image_format', 'base64')
        processing_options = kwargs.get('processing_options', {})
        
        if not image_data:
            return {
                "success": False,
                "error": "No image data provided",
                "timestamp": datetime.now().isoformat()
            }
        
        result = await self.recognize_image(image_data, image_format, processing_options)
        return result.to_dict()


# Convenience function for easy usage
async def recognize_math_image(image_data: Union[str, bytes], 
                             image_format: str = "base64",
                             llm=None,
                             processing_options: Optional[Dict[str, Any]] = None) -> ImageAnalysisResult:
    """
    Convenience function to recognize math content from images.
    
    Args:
        image_data: Image data (base64, file path, or bytes)
        image_format: Format of image data
        llm: Language model for enhanced analysis
        processing_options: Optional processing configuration
        
    Returns:
        ImageAnalysisResult with comprehensive analysis
    """
    agent = EnhancedImageRecognitionAgent(llm=llm)
    return await agent.recognize_image(image_data, image_format, processing_options) 