"""
Image Analysis Tool for Math Exercise Analysis

This tool analyzes uploaded images of math exercises to identify errors,
missing concepts, and provide targeted recommendations.
"""

import base64
import io
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

# Image processing imports (will be optional dependencies)
try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
    ImageType = Image.Image
except ImportError:
    PIL_AVAILABLE = False
    # Create a dummy type for when PIL is not available
    ImageType = Any

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from ..error_analysis import MathErrorAnalyzer, ErrorAnalysis

class ImageAnalysisTool:
    """Tool for analyzing math exercise images and identifying learning gaps."""
    
    def __init__(self, learning_graph, llm, knowledge_graph=None):
        self.learning_graph = learning_graph
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.error_analyzer = MathErrorAnalyzer(knowledge_graph)
        
        # Check for required dependencies
        if not PIL_AVAILABLE:
            print("⚠️  Warning: PIL not available. Image processing will be limited.")
        if not TESSERACT_AVAILABLE:
            print("⚠️  Warning: pytesseract not available. OCR will use LLM fallback.")
    
    async def execute(self, student_id: str, image_data: str, image_format: str = "base64", 
                     **kwargs) -> Dict[str, Any]:
        """
        Analyze an uploaded image of a math exercise.
        
        Args:
            student_id: ID of the student
            image_data: Image data (base64 encoded or file path)
            image_format: Format of image_data ("base64", "file_path", or "url")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        try:
            # Step 1: Process the image
            processed_image = await self._process_image(image_data, image_format)
            
            # Step 2: Extract text from image using OCR
            extracted_text = await self._extract_text_from_image(processed_image, image_data, image_format)
            
            # Step 3: Parse the mathematical content
            parsed_content = await self._parse_mathematical_content(extracted_text)
            
            # Step 4: Analyze the error
            error_analysis = await self._analyze_mathematical_error(parsed_content)
            
            # Step 5: Update learning graph
            learning_updates = await self._update_learning_graph(student_id, error_analysis)
            
            # Step 6: Generate exercise recommendation
            recommendation = await self._generate_exercise_recommendation(student_id, error_analysis)
            
            # Step 7: Compile comprehensive response
            response = {
                "analysis_timestamp": datetime.now().isoformat(),
                "student_id": student_id,
                "extracted_content": {
                    "raw_text": extracted_text,
                    "parsed_problem": parsed_content.get("problem", ""),
                    "student_answer": parsed_content.get("student_answer", ""),
                    "correct_answer": parsed_content.get("correct_answer", "")
                },
                "error_analysis": {
                    "error_type": error_analysis.error_type.value if error_analysis else "unknown",
                    "missing_concepts": error_analysis.missing_concepts if error_analysis else [],
                    "confidence": error_analysis.confidence if error_analysis else 0.0,
                    "description": error_analysis.description if error_analysis else "",
                    "suggested_remediation": error_analysis.suggested_remediation if error_analysis else ""
                },
                "learning_graph_updates": learning_updates,
                "exercise_recommendation": recommendation,
                "success": True
            }
            
            return response
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat(),
                "student_id": student_id
            }
    
    async def _process_image(self, image_data: str, image_format: str) -> Optional[ImageType]:
        """Process and enhance the uploaded image for better OCR."""
        try:
            if image_format == "base64":
                # Decode base64 image
                image_bytes = base64.b64decode(image_data)
                if PIL_AVAILABLE:
                    image = Image.open(io.BytesIO(image_bytes))
                else:
                    return None
            elif image_format == "file_path":
                # Load from file path
                if PIL_AVAILABLE:
                    image = Image.open(image_data)
                else:
                    return None
            else:
                raise ValueError(f"Unsupported image format: {image_format}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image for better OCR
            # Increase contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Increase sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    async def _extract_text_from_image(self, image: Optional[ImageType], image_data: str = None, image_format: str = "base64") -> str:
        """Extract text from the processed image using OCR."""
        if image is None or not TESSERACT_AVAILABLE:
            # Fallback to LLM-based image analysis
            return await self._llm_image_analysis(image_data, image_format)
        
        try:
            # Use pytesseract for OCR
            # Configure for mathematical text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=().x÷×%/'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Clean up the extracted text
            text = self._clean_ocr_text(text)
            
            return text
            
        except Exception as e:
            print(f"OCR error: {e}")
            # Fallback to LLM analysis
            return await self._llm_image_analysis(image_data, image_format)
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR-extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors for mathematical symbols
        text = text.replace('x', '*')  # Often x is misread as multiplication
        text = text.replace('X', '*')
        text = text.replace('÷', '/')
        text = text.replace('×', '*')
        
        # Fix common number/letter confusions
        text = re.sub(r'\b[oO]\b', '0', text)  # O -> 0
        text = re.sub(r'\b[lI]\b', '1', text)  # l,I -> 1
        
        return text
    
    async def _llm_image_analysis(self, image_data: str = None, image_format: str = "base64") -> str:
        """Fallback method using LLM to analyze image content."""
        # If we have base64 text content, decode it directly
        if image_format == "base64" and image_data:
            try:
                # Try to decode as text first
                decoded_text = base64.b64decode(image_data).decode('utf-8')
                return decoded_text
            except:
                pass
        
        # Use LLM to analyze the image
        prompt = """
        Analyze this math exercise image and extract all visible text content including:
        - Mathematical problems
        - Student work and answers
        - Any corrections or notes
        
        Please provide the extracted text exactly as it appears.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content
            return ""
        except Exception as e:
            print(f"LLM analysis error: {e}")
            return ""
    
    async def _parse_mathematical_content(self, extracted_text: str) -> Dict[str, str]:
        """Parse the extracted text to identify problem, student answer, and correct answer."""
        
        if not extracted_text or extracted_text == "LLM_ANALYSIS_NEEDED":
            # Use LLM to parse the content
            return await self._llm_parse_content(extracted_text)
        
        # Use regex patterns to identify different parts
        content = {
            "problem": "",
            "student_answer": "",
            "correct_answer": ""
        }
        
        # Split text into lines for analysis
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        # Look for mathematical expressions
        math_expressions = []
        for line in lines:
            # Find expressions with equals signs
            if '=' in line:
                math_expressions.append(line)
        
        if math_expressions:
            # First expression is likely the problem
            if len(math_expressions) >= 1:
                content["problem"] = math_expressions[0]
            
            # Look for student's work (often has incorrect answer)
            if len(math_expressions) >= 2:
                content["student_answer"] = self._extract_answer_from_expression(math_expressions[-1])
        
        # Try to determine correct answer using LLM
        if content["problem"]:
            content["correct_answer"] = await self._calculate_correct_answer(content["problem"])
        
        return content
    
    async def _llm_parse_content(self, extracted_text: str = "") -> Dict[str, str]:
        """Use LLM to parse mathematical content from image."""
        prompt = f"""
        Analyze this math exercise content and extract:
        1. The original problem
        2. The student's answer
        3. The correct answer
        
        Content: {extracted_text}
        
        Return in JSON format:
        {{
            "problem": "original math problem",
            "student_answer": "student's incorrect answer", 
            "correct_answer": "correct answer"
        }}
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Try to parse JSON response
                import json
                parsed = json.loads(content)
                return parsed
        except Exception as e:
            print(f"LLM parsing error: {e}")
            
        # Fallback if LLM parsing fails
        return {
            "problem": "Unable to parse problem",
            "student_answer": "Unknown",
            "correct_answer": "Unknown"
        }
    
    def _extract_answer_from_expression(self, expression: str) -> str:
        """Extract the answer (right side of equals) from a mathematical expression."""
        if '=' in expression:
            parts = expression.split('=')
            if len(parts) >= 2:
                return parts[-1].strip()
        return expression.strip()
    
    async def _calculate_correct_answer(self, problem: str) -> str:
        """Calculate the correct answer for a mathematical problem."""
        prompt = f"""
        Solve this math problem and provide only the numerical answer:
        {problem}
        
        Answer:
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Extract numerical answer
                answer = re.findall(r'-?\d+\.?\d*', content)
                if answer:
                    return answer[0]
        except:
            pass
        
        return "Unknown"
    
    async def _analyze_mathematical_error(self, parsed_content: Dict[str, str]) -> Optional[ErrorAnalysis]:
        """Analyze the mathematical error using the error analyzer."""
        problem = parsed_content.get("problem", "")
        student_answer = parsed_content.get("student_answer", "")
        correct_answer = parsed_content.get("correct_answer", "")
        
        if not all([problem, student_answer, correct_answer]):
            return None
        
        # Use the error analyzer to identify the error
        error_analysis = self.error_analyzer.analyze_error(
            problem_text=problem,
            student_answer=student_answer,
            correct_answer=correct_answer
        )
        
        return error_analysis
    
    async def _update_learning_graph(self, student_id: str, error_analysis: Optional[ErrorAnalysis]) -> Dict[str, Any]:
        """Update the learning graph based on the error analysis."""
        if not error_analysis:
            return {"updated": False, "reason": "No error analysis available"}
        
        updates = {
            "concepts_updated": [],
            "new_gaps_identified": [],
            "confidence_changes": {}
        }
        
        try:
            # Update mastery levels for missing concepts
            for concept in error_analysis.missing_concepts:
                # Decrease confidence for missing concepts
                current_mastery = self.learning_graph.get_mastery(concept)
                
                # Use the update_mastery method with proper parameters
                # Since this is an error, we pass False for exercise_result
                # Use medium difficulty and full concept weight
                new_mastery = self.learning_graph.update_mastery(
                    concept_id=concept,
                    exercise_result=False,  # This was an error
                    exercise_difficulty=0.5,  # Medium difficulty
                    concept_weight=1.0  # Full weight for this concept
                )
                
                updates["concepts_updated"].append(concept)
                updates["confidence_changes"][concept] = {
                    "old": current_mastery,
                    "new": new_mastery
                }
            
            # Identify new learning gaps
            prerequisites = self.error_analyzer.get_prerequisite_concepts(error_analysis.missing_concepts)
            for prereq in prerequisites:
                if self.learning_graph.get_mastery(prereq) < 0.7:  # Below mastery threshold
                    updates["new_gaps_identified"].append(prereq)
            
            # Record the error in learning history
            # Note: Using record_exercise_attempt to track this analysis
            exercise_id = f"image_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if error_analysis.missing_concepts:
                primary_concept = error_analysis.missing_concepts[0]
                self.learning_graph.record_exercise_attempt(
                    exercise_id=exercise_id,
                    concept_id=primary_concept,
                    result=False,  # This was an error
                    difficulty=0.5,  # Medium difficulty
                    concept_weight=1.0
                )
            
            updates["updated"] = True
            
        except Exception as e:
            updates["updated"] = False
            updates["error"] = str(e)
        
        return updates
    
    async def _generate_exercise_recommendation(self, student_id: str, error_analysis: Optional[ErrorAnalysis]) -> Dict[str, Any]:
        """Generate a targeted exercise recommendation based on the error analysis."""
        if not error_analysis:
            return {"recommended": False, "reason": "No error analysis available"}
        
        try:
            # Find the most critical missing concept
            primary_concept = error_analysis.missing_concepts[0] if error_analysis.missing_concepts else "basic_arithmetic"
            
            # Generate exercise recommendation using LLM
            prompt = f"""
            Create a targeted math exercise to help a student master the concept: {primary_concept}
            
            The student made an error of type: {error_analysis.error_type.value}
            Error description: {error_analysis.description}
            Difficulty level needed: {error_analysis.difficulty_level}
            
            Provide a specific exercise recommendation.
            """
            
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.chat_completion("gpt-4", messages)
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                
                # Try to parse structured response or create fallback
                recommendation = {
                    "title": f"{primary_concept.replace('_', ' ').title()} Practice",
                    "problem": f"Practice {primary_concept} problems",
                    "concept_focus": primary_concept,
                    "difficulty": error_analysis.difficulty_level,
                    "reason": f"Based on {error_analysis.error_type.value} error in {primary_concept}",
                    "target_concept": primary_concept,
                    "recommended": True,
                    "llm_response": content
                }
                
                return recommendation
            
        except Exception as e:
            print(f"Exercise recommendation error: {e}")
            
        return {
            "recommended": False,
            "error": "Failed to generate recommendation",
            "fallback_suggestion": "Review basic arithmetic concepts"
        } 