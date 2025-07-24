"""
Concept Identification System for Math Learning

This module provides a standalone system for identifying mathematical concepts, 
courses, and topics from exercise content (text or images). It uses LLM analysis
combined with pattern matching to provide comprehensive concept identification.
"""

import re
import json
import base64
import io
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

# Optional imports for image processing
try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from llm.llm import LLMBase
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMBase = None


@dataclass
class ConceptIdentification:
    """Result of concept identification analysis."""
    course: str
    topic: str
    concepts: List[Dict[str, Any]]
    confidence: float
    exercise_type: str
    difficulty_level: str
    source_content: str
    analysis_method: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class IdentifiedConcept:
    """Individual concept identified from exercise."""
    concept_id: str
    concept_name: str
    confidence: float
    relationship_type: str  # "primary", "secondary", "prerequisite"
    evidence: List[str]  # Text patterns or reasoning that led to identification
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ConceptIdentificationSystem:
    """
    Standalone system for identifying mathematical concepts from exercises.
    
    This system can:
    1. Analyze text (markdown) or image content
    2. Identify the course/subject area (Algebra, Geometry, Calculus, etc.)
    3. Determine the specific topic within that course
    4. List all relevant mathematical concepts being tested
    5. Provide confidence scores and evidence for each identification
    6. Handle multiple exercises in a single input
    """
    
    def __init__(self, llm: Optional[LLMBase] = None):
        """
        Initialize the Concept Identification System.
        
        Args:
            llm: Language model for advanced analysis (optional)
        """
        self.llm = llm
        self.name = "concept_identification_system"
        
        # Course classification patterns
        self.course_patterns = {
            "Elementary Arithmetic": [
                r"addition|subtraction|multiplication|division",
                r"\d+\s*[\+\-\*\/]\s*\d+",
                r"basic.*operations?",
                r"counting|number.*sense"
            ],
            "Pre-Algebra": [
                r"integers?|fractions?|decimals?",
                r"order.*operations?|PEMDAS",
                r"basic.*equation",
                r"ratio|proportion|percent"
            ],
            "Algebra I": [
                r"linear.*equation|solve.*for.*[a-z]",
                r"[a-z]\s*=|variable",
                r"slope|y-intercept",
                r"system.*equation",
                r"quadratic.*equation",
                r"factor.*polynomial"
            ],
            "Algebra II": [
                r"exponential|logarithm",
                r"complex.*number",
                r"rational.*function",
                r"conic.*section",
                r"sequence|series"
            ],
            "Geometry": [
                r"triangle|circle|polygon",
                r"angle|degree|radius|diameter",
                r"area|perimeter|volume|surface.*area",
                r"congruent|similar",
                r"theorem|proof",
                r"coordinate.*geometry"
            ],
            "Trigonometry": [
                r"sin|cos|tan|sine|cosine|tangent",
                r"radian|degree.*measure",
                r"unit.*circle|trigonometric.*identity",
                r"law.*of.*sines?|law.*of.*cosines?"
            ],
            "Pre-Calculus": [
                r"polynomial.*function",
                r"inverse.*function",
                r"exponential.*growth|decay",
                r"matrix|matrices",
                r"parametric.*equation"
            ],
            "Calculus": [
                r"derivative|differentiat",
                r"integral|integrat",
                r"limit|continuity",
                r"d/dx|∫|∂",
                r"chain.*rule|product.*rule"
            ],
            "Statistics": [
                r"mean|median|mode|average",
                r"standard.*deviation|variance",
                r"probability|distribution",
                r"hypothesis.*test|confidence.*interval",
                r"regression|correlation"
            ]
        }
        
        # Topic patterns within courses
        self.topic_patterns = {
            "Algebra I": {
                "Linear Equations": [r"linear.*equation", r"solve.*for.*[a-z]", r"ax.*\+.*b"],
                "Graphing": [r"slope", r"y-intercept", r"graph.*line"],
                "Systems of Equations": [r"system.*equation", r"elimination|substitution"],
                "Quadratic Equations": [r"quadratic", r"x\^2|x²", r"factor.*trinomial"],
                "Polynomials": [r"polynomial", r"factor", r"expand.*expression"]
            },
            "Geometry": {
                "Triangles": [r"triangle", r"pythagorean", r"right.*triangle"],
                "Circles": [r"circle", r"radius|diameter", r"circumference|area.*circle"],
                "Area and Perimeter": [r"area|perimeter", r"rectangle|square"],
                "Angles": [r"angle", r"degree", r"complementary|supplementary"],
                "Proofs": [r"proof|prove", r"theorem", r"given.*prove"]
            },
            "Calculus": {
                "Limits": [r"limit", r"approach", r"lim.*x.*approaches"],
                "Derivatives": [r"derivative", r"d/dx", r"differentiat", r"slope.*tangent"],
                "Integrals": [r"integral", r"∫", r"area.*under.*curve", r"antiderivative"],
                "Applications": [r"related.*rates?", r"optimization", r"volume.*revolution"]
            }
        }
        
        # Enhanced detailed concept patterns with better matching
        self.concept_patterns = {
            # Algebra concepts - improved patterns
            "solving_linear_equations": [
                r"solve.*for.*[a-z]", r"find.*[a-z].*=", r"linear.*equation",
                r"\d*[a-z]\s*[+\-]\s*\d+\s*=\s*\d+", r"isolate.*variable",
                r"equation.*solving", r"one.*variable", r"[a-z]\s*=\s*\d+",
                r"solve.*equation", r"find.*value.*of.*[a-z]"
            ],
            "factoring_polynomials": [
                r"factor(?:ing|ize)?", r"factor.*polynomial", r"factor.*trinomial",
                r"difference.*squares", r"[a-z]²\s*[+\-]\s*\d*[a-z]\s*[+\-]\s*\d+",
                r"greatest.*common.*factor", r"gcf", r"quadratic.*factor",
                r"[a-z]².*\-.*\d+", r"factor.*expression"
            ],
            "quadratic_equations": [
                r"quadratic.*equation", r"[a-z]².*=.*0", r"quadratic.*formula",
                r"completing.*square", r"parabola", r"vertex.*form", r"discriminant"
            ],
            "graphing_lines": [
                r"graph.*line", r"plot.*line", r"y\s*=.*x", r"slope.*intercept",
                r"linear.*function", r"coordinate.*plane", r"x.*intercept", r"y.*intercept"
            ],
            "systems_equations": [
                r"system.*equations?", r"simultaneous.*equation", r"solve.*system",
                r"elimination.*method", r"substitution.*method"
            ],
            "polynomial_operations": [
                r"polynomial", r"expand.*expression", r"simplify.*polynomial",
                r"degree.*polynomial", r"leading.*coefficient"
            ],
            
            # Geometry concepts - improved patterns
            "pythagorean_theorem": [
                r"pythagorean", r"a².*\+.*b².*=.*c²", r"right.*triangle.*hypotenuse",
                r"leg.*hypotenuse", r"right.*triangle", r"√.*sum.*squares"
            ],
            "area_calculations": [
                r"area.*(?:rectangle|square|triangle|circle|parallelogram)",
                r"circumference", r"perimeter", r"surface.*area", r"base.*height",
                r"radius.*area", r"π.*r²", r"A\s*=", r"length.*width",
                r"area.*of.*(?:a\s+)?(?:circle|triangle|rectangle|square)"
            ],
            "volume_calculations": [
                r"volume.*(?:cube|sphere|cylinder|cone|prism)", r"cubic.*units",
                r"capacity", r"³", r"cubed", r"V\s*="
            ],
            "angle_relationships": [
                r"angle.*triangle", r"sum.*angles", r"complementary.*angle",
                r"supplementary.*angle", r"vertical.*angles", r"°", r"degrees?"
            ],
            "triangle_properties": [
                r"triangle.*properties", r"isosceles", r"equilateral", r"scalene",
                r"triangle.*congruent", r"similar.*triangles", r"triangle.*inequality"
            ],
            "circle_properties": [
                r"circle.*properties", r"radius", r"diameter", r"chord",
                r"tangent.*circle", r"arc.*length", r"sector.*area"
            ],
            
            # Calculus concepts - improved patterns
            "limit_definition": [
                r"limit", r"lim.*→", r"approaches", r"infinity", r"continuous",
                r"epsilon.*delta", r"limit.*definition"
            ],
            "derivative_rules": [
                r"derivative", r"d/dx", r"differentiate", r"f'", r"rate.*change",
                r"power.*rule", r"product.*rule", r"chain.*rule", r"quotient.*rule",
                r"implicit.*differentiation", r"find.*derivative"
            ],
            "integration_techniques": [
                r"integral", r"∫", r"integrate", r"antiderivative",
                r"substitution.*integration", r"integration.*by.*parts",
                r"partial.*fractions", r"area.*under.*curve"
            ],
            
            # Statistics concepts - improved patterns
            "descriptive_statistics": [
                r"mean", r"median", r"mode", r"average", r"standard.*deviation",
                r"variance", r"range", r"quartile", r"percentile", r"summary.*statistics",
                r"calculate.*mean", r"find.*average"
            ],
            "probability_basics": [
                r"probability", r"chance", r"odds", r"likely", r"dice", r"coin",
                r"sample.*space", r"event", r"outcome", r"favorable.*outcomes"
            ],
            "hypothesis_testing": [
                r"hypothesis.*test", r"null.*hypothesis", r"p-value", r"significance",
                r"confidence.*interval", r"t-test", r"z-test", r"alpha.*level"
            ],
            
            # Additional concepts
            "exponential_functions": [
                r"exponential", r"e\^", r"growth.*decay", r"exponential.*function",
                r"compound.*interest", r"natural.*exponential"
            ],
            "logarithmic_functions": [
                r"logarithm", r"log", r"natural.*log", r"ln", r"log.*base",
                r"logarithmic.*equation", r"properties.*logarithms"
            ],
            "trigonometric_functions": [
                r"sin", r"cos", r"tan", r"trigonometric", r"sine", r"cosine", r"tangent",
                r"unit.*circle", r"radian", r"degree", r"trig.*identity"
            ]
        }
    
    async def identify_concepts(self, 
                              content: str, 
                              content_type: str = "text",
                              use_llm: bool = True) -> ConceptIdentification:
        """
        Main method to identify concepts from exercise content.
        
        Args:
            content: The content to analyze (text, markdown, or base64 image)
            content_type: Type of content ("text", "markdown", "image", "base64")
            use_llm: Whether to use LLM for enhanced analysis
            
        Returns:
            ConceptIdentification object with comprehensive analysis
        """
        try:
            # Extract text content if needed
            if content_type in ["image", "base64"]:
                text_content = await self._extract_text_from_image(content, content_type)
            else:
                text_content = self._clean_markdown(content) if content_type == "markdown" else content
            
            if not text_content or text_content.strip() == "":
                return self._create_empty_result("No text content could be extracted")
            
            # Step 1: Pattern-based analysis (fast, always available)
            pattern_result = self._analyze_with_patterns(text_content)
            
            # Step 2: LLM-enhanced analysis (if available and requested)
            if use_llm and self.llm and LLM_AVAILABLE:
                llm_result = await self._analyze_with_llm(text_content, pattern_result)
                final_result = self._merge_analysis_results(pattern_result, llm_result)
            else:
                final_result = pattern_result
            
            # Step 3: Post-process and validate results
            validated_result = self._validate_and_enhance_result(final_result, text_content)
            
            return ConceptIdentification(
                course=validated_result["course"],
                topic=validated_result["topic"],
                concepts=validated_result["concepts"],
                confidence=validated_result["confidence"],
                exercise_type=validated_result["exercise_type"],
                difficulty_level=validated_result["difficulty_level"],
                source_content=text_content[:500] + "..." if len(text_content) > 500 else text_content,
                analysis_method=validated_result["analysis_method"],
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return self._create_empty_result(f"Analysis failed: {str(e)}")
    
    async def batch_identify_concepts(self, 
                                    contents: List[Dict[str, str]], 
                                    use_llm: bool = True) -> List[ConceptIdentification]:
        """
        Identify concepts from multiple exercises at once.
        
        Args:
            contents: List of dicts with 'content' and 'content_type' keys
            use_llm: Whether to use LLM for enhanced analysis
            
        Returns:
            List of ConceptIdentification results
        """
        results = []
        for item in contents:
            content = item.get("content", "")
            content_type = item.get("content_type", "text")
            
            result = await self.identify_concepts(content, content_type, use_llm)
            results.append(result)
        
        return results
    
    def _clean_markdown(self, markdown_text: str) -> str:
        """Clean markdown formatting to extract plain text."""
        # Remove markdown headers
        text = re.sub(r'^#+\s*', '', markdown_text, flags=re.MULTILINE)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    async def _extract_text_from_image(self, content: str, content_type: str) -> str:
        """Extract text from image using OCR or LLM."""
        if content_type == "base64":
            return await self._extract_from_base64(content)
        elif content_type == "image":
            return await self._extract_from_image_file(content)
        return ""
    
    async def _extract_from_base64(self, base64_content: str) -> str:
        """Extract text from base64 encoded image."""
        if TESSERACT_AVAILABLE:
            try:
                # Decode base64 to image
                image_data = base64.b64decode(base64_content)
                image = Image.open(io.BytesIO(image_data))
                
                # Use OCR to extract text
                text = pytesseract.image_to_string(image)
                return self._clean_ocr_text(text)
            except Exception:
                pass
        
        # Fallback to LLM if OCR fails
        if self.llm and LLM_AVAILABLE:
            return await self._llm_extract_from_base64(base64_content)
        
        return ""
    
    async def _extract_from_image_file(self, image_path: str) -> str:
        """Extract text from image file."""
        if TESSERACT_AVAILABLE:
            try:
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return self._clean_ocr_text(text)
            except Exception:
                pass
        
        # Fallback to LLM if OCR fails
        if self.llm and LLM_AVAILABLE:
            return await self._llm_extract_from_image_file(image_path)
        
        return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean and normalize OCR extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors for mathematical symbols
        replacements = {
            'x': '×',  # multiplication (context-dependent)
            '÷': '/',  # division
            '—': '-',  # em dash to minus
            '–': '-',  # en dash to minus
            'α': 'alpha',
            'β': 'beta',
            'θ': 'theta',
            '∞': 'infinity'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def _analyze_with_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze text using predefined patterns."""
        text_lower = text.lower()
        
        # Identify course
        course_scores = {}
        for course, patterns in self.course_patterns.items():
            score = 0
            matched_patterns = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    score += len(matches)
                    matched_patterns.append(pattern)
            course_scores[course] = {"score": score, "patterns": matched_patterns}
        
        # Get best course match
        best_course = max(course_scores.keys(), key=lambda x: course_scores[x]["score"])
        max_score = course_scores[best_course]["score"]
        
        # Improved confidence calculation
        if max_score == 0:
            # Try to infer course from keywords if no patterns matched
            inferred_course = self._infer_course_from_keywords(text_lower)
            if inferred_course:
                best_course = inferred_course
                course_confidence = 0.4
            else:
                best_course = "General Mathematics"
                course_confidence = 0.25
        else:
            # Higher confidence for clear matches, normalized to 0.4-1.0 range
            course_confidence = min(1.0, 0.4 + (max_score / 5.0) * 0.6)
        
        # Identify topic within course
        topic = "General"
        topic_confidence = 0.5
        if best_course in self.topic_patterns:
            topic_scores = {}
            for topic_name, patterns in self.topic_patterns[best_course].items():
                score = sum(1 for pattern in patterns if re.search(pattern, text_lower))
                topic_scores[topic_name] = score
            
            if topic_scores and max(topic_scores.values()) > 0:
                topic = max(topic_scores.keys(), key=lambda x: topic_scores[x])
                topic_confidence = min(1.0, topic_scores[topic] / 2.0)
        
        # Identify specific concepts with improved scoring
        concepts = []
        for concept_id, patterns in self.concept_patterns.items():
            matches = []
            confidence = 0
            matched_patterns = 0
            
            for pattern in patterns:
                pattern_matches = re.findall(pattern, text_lower)
                if pattern_matches:
                    matches.extend(pattern_matches)
                    confidence += len(pattern_matches)
                    matched_patterns += 1
            
            if matches:
                concept_name = concept_id.replace("_", " ").title()
                
                # Improved confidence calculation
                # Base confidence from number of pattern matches
                base_confidence = min(0.9, matched_patterns / len(patterns))
                
                # Bonus for multiple occurrences of same pattern
                occurrence_bonus = min(0.3, (confidence - matched_patterns) * 0.1)
                
                # Final confidence with minimum threshold
                final_confidence = max(0.6, base_confidence + occurrence_bonus)
                
                concepts.append({
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "confidence": final_confidence,
                    "relationship_type": "primary" if matched_patterns >= 2 else "secondary",
                    "evidence": list(set(matches))[:3]  # Top 3 unique matches
                })
        
        # Determine exercise type and difficulty
        exercise_type = self._classify_exercise_type(text_lower)
        difficulty_level = self._estimate_difficulty(text)
        
        # Ensure we have at least one specific concept
        if not concepts or all(c["concept_id"] == "general_problem_solving" for c in concepts):
            # Try to infer concept from course and exercise type
            specific_concept = self._infer_concept_from_context(best_course, exercise_type, text_lower)
            if specific_concept:
                concepts = [specific_concept]
        
        # Sort concepts by confidence
        concepts.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "course": best_course,
            "topic": topic,
            "concepts": concepts[:5],  # Top 5 concepts
            "confidence": (course_confidence + topic_confidence) / 2,
            "exercise_type": exercise_type,
            "difficulty_level": difficulty_level,
            "analysis_method": "pattern_based",
            "course_evidence": course_scores[best_course]["patterns"][:3]
        }
    
    async def _analyze_with_llm(self, text: str, pattern_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis using LLM."""
        prompt = f"""
You are an expert mathematics educator. Analyze this mathematical exercise and provide detailed concept identification.

EXERCISE TO ANALYZE:
{text}

INITIAL PATTERN ANALYSIS (for reference):
- Course: {pattern_result['course']}
- Topic: {pattern_result['topic']}
- Concepts: {[c['concept_name'] for c in pattern_result['concepts'][:3]]}

TASK: Provide comprehensive analysis focusing on SPECIFIC mathematical concepts being tested.

AVAILABLE COURSES:
- Elementary Arithmetic (basic operations, counting)
- Pre-Algebra (integers, fractions, basic equations)
- Algebra I (linear equations, graphing, factoring)
- Algebra II (quadratics, exponentials, logarithms)
- Geometry (shapes, area, volume, proofs)
- Trigonometry (trig functions, identities)
- Pre-Calculus (advanced functions, limits)
- Calculus (derivatives, integrals, applications)
- Statistics (data analysis, probability)

IMPORTANT: Be SPECIFIC about concepts. Instead of generic terms, identify exact mathematical skills:
- "solving_linear_equations" NOT "general problem solving"
- "area_calculations" NOT "geometry concepts"
- "derivative_rules" NOT "calculus operations"
- "factoring_polynomials" NOT "algebraic manipulation"

RESPONSE FORMAT (valid JSON only):
{{
    "course": "exact course name from list above",
    "topic": "specific topic within course",
    "concepts": [
        {{
            "concept_id": "specific_snake_case_concept_id",
            "concept_name": "Specific Mathematical Concept Name",
            "confidence": 0.85,
            "relationship_type": "primary",
            "evidence": ["specific mathematical term found", "operation type", "problem structure"]
        }}
    ],
    "exercise_type": "equation_solving|computation|word_problem|proof|graphing|free_response",
    "difficulty_level": "elementary|middle_school|high_school_basic|high_school_advanced|college",
    "reasoning": "Brief explanation of why these specific concepts were identified"
}}

Focus on identifying 1-3 SPECIFIC concepts rather than many generic ones. High confidence (0.8+) for clear matches."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.chat_completion("gpt-4", messages)
            
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                
                # Try to parse JSON response
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    parsed["analysis_method"] = "llm_enhanced"
                    return parsed
        
        except Exception as e:
            print(f"LLM analysis error: {e}")
        
        # Return pattern result if LLM fails
        return pattern_result
    
    def _merge_analysis_results(self, pattern_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge pattern-based and LLM analysis results."""
        # Use LLM result as base, but validate against patterns
        merged = llm_result.copy()
        
        # Adjust confidence based on agreement between methods
        pattern_course = pattern_result["course"]
        llm_course = llm_result.get("course", pattern_course)
        
        if pattern_course == llm_course:
            merged["confidence"] = min(1.0, merged.get("confidence", 0.8) + 0.1)
        else:
            merged["confidence"] = max(0.3, merged.get("confidence", 0.8) - 0.2)
        
        # Merge concept lists (combine unique concepts)
        pattern_concepts = {c["concept_id"]: c for c in pattern_result["concepts"]}
        llm_concepts = {c["concept_id"]: c for c in llm_result.get("concepts", [])}
        
        all_concepts = {}
        all_concepts.update(pattern_concepts)
        
        # Update with LLM concepts, boosting confidence for agreements
        for concept_id, llm_concept in llm_concepts.items():
            if concept_id in all_concepts:
                # Boost confidence for agreement
                all_concepts[concept_id]["confidence"] = min(1.0, 
                    (all_concepts[concept_id]["confidence"] + llm_concept["confidence"]) / 2 + 0.1)
                all_concepts[concept_id]["evidence"].extend(llm_concept.get("evidence", []))
            else:
                all_concepts[concept_id] = llm_concept
        
        merged["concepts"] = list(all_concepts.values())
        merged["analysis_method"] = "pattern_and_llm"
        
        return merged
    
    def _validate_and_enhance_result(self, result: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Validate and enhance the final analysis result."""
        # Ensure minimum confidence thresholds
        if result["confidence"] < 0.3:
            result["course"] = "General Mathematics"
            result["topic"] = "Mixed Topics"
        
        # Sort concepts by confidence
        result["concepts"] = sorted(result["concepts"], 
                                  key=lambda x: x["confidence"], reverse=True)
        
        # Limit to top 5 concepts
        result["concepts"] = result["concepts"][:5]
        
        # Ensure we have at least one concept
        if not result["concepts"]:
            result["concepts"] = [{
                "concept_id": "general_problem_solving",
                "concept_name": "General Problem Solving",
                "confidence": 0.5,
                "relationship_type": "primary",
                "evidence": ["mathematical content detected"]
            }]
        
        return result
    
    def _infer_concept_from_context(self, course: str, exercise_type: str, text: str) -> Optional[Dict[str, Any]]:
        """Infer specific concept from course and exercise type context."""
        # Course-specific concept mapping
        course_concept_map = {
            "Algebra I": {
                "equation_solving": "solving_linear_equations",
                "computation": "polynomial_operations",
                "word_problem": "solving_linear_equations"
            },
            "Geometry": {
                "equation_solving": "area_calculations",
                "computation": "area_calculations",
                "word_problem": "area_calculations"
            },
            "Calculus": {
                "equation_solving": "derivative_rules",
                "computation": "derivative_rules",
                "word_problem": "derivative_rules"
            },
            "Statistics": {
                "computation": "descriptive_statistics",
                "word_problem": "probability_basics"
            }
        }
        
        # Try to find specific concept based on course and exercise type
        if course in course_concept_map and exercise_type in course_concept_map[course]:
            concept_id = course_concept_map[course][exercise_type]
            concept_name = concept_id.replace("_", " ").title()
            
            return {
                "concept_id": concept_id,
                "concept_name": concept_name,
                "confidence": 0.7,  # Medium confidence for inferred concepts
                "relationship_type": "primary",
                "evidence": [f"inferred from {course} + {exercise_type}"]
            }
        
        # Additional keyword-based inference
        keyword_concept_map = {
            "derivative": "derivative_rules",
            "factor": "factoring_polynomials", 
            "area": "area_calculations",
            "mean": "descriptive_statistics",
            "solve": "solving_linear_equations"
        }
        
        for keyword, concept_id in keyword_concept_map.items():
            if keyword in text:
                concept_name = concept_id.replace("_", " ").title()
                return {
                    "concept_id": concept_id,
                    "concept_name": concept_name,
                    "confidence": 0.8,
                    "relationship_type": "primary",
                    "evidence": [f"keyword: {keyword}"]
                }
        
        return None

    def _infer_course_from_keywords(self, text: str) -> Optional[str]:
        """Infer course from specific mathematical keywords."""
        keyword_course_map = {
            # Algebra keywords
            "factor": "Algebra I",
            "solve": "Algebra I", 
            "equation": "Algebra I",
            "polynomial": "Algebra I",
            "quadratic": "Algebra II",
            "exponential": "Algebra II",
            "logarithm": "Algebra II",
            
            # Geometry keywords
            "area": "Geometry",
            "volume": "Geometry",
            "triangle": "Geometry",
            "circle": "Geometry",
            "radius": "Geometry",
            "angle": "Geometry",
            
            # Calculus keywords
            "derivative": "Calculus",
            "integral": "Calculus",
            "limit": "Calculus",
            "d/dx": "Calculus",
            
            # Statistics keywords
            "mean": "Statistics",
            "median": "Statistics",
            "probability": "Statistics",
            "standard deviation": "Statistics",
            
            # Trigonometry keywords
            "sin": "Trigonometry",
            "cos": "Trigonometry",
            "tan": "Trigonometry"
        }
        
        for keyword, course in keyword_course_map.items():
            if keyword in text:
                return course
        
        return None

    def _classify_exercise_type(self, text: str) -> str:
        """Classify the type of exercise based on content."""
        text_lower = text.lower()
        
        type_patterns = {
            "word_problem": [r"a.*has.*", r"if.*then", r"how many", r"what is the"],
            "equation_solving": [r"solve.*for", r"find.*[a-z]", r"[a-z]\s*="],
            "computation": [r"\d+\s*[\+\-\*\/]\s*\d+", r"calculate", r"compute"],
            "proof": [r"prove", r"show that", r"demonstrate"],
            "graphing": [r"graph", r"plot", r"sketch"],
            "multiple_choice": [r"\(a\)|\(b\)|\(c\)|\(d\)", r"choose.*correct"],
            "fill_in_blank": [r"_+", r"complete.*following"],
            "true_false": [r"true.*false", r"circle.*correct"]
        }
        
        for exercise_type, patterns in type_patterns.items():
            if any(re.search(pattern, text_lower) for pattern in patterns):
                return exercise_type
        
        return "free_response"
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate difficulty level based on content complexity."""
        text_lower = text.lower()
        
        # Count complexity indicators
        complexity_score = 0
        
        # Basic operations (low complexity)
        if re.search(r'\d+\s*[\+\-]\s*\d+', text):
            complexity_score += 1
        
        # Variables and algebra (medium complexity)
        if re.search(r'[a-z]\s*=|solve.*for', text_lower):
            complexity_score += 2
        
        # Advanced concepts (high complexity)
        advanced_patterns = [r'derivative|integral', r'theorem|proof', r'matrix', r'logarithm']
        for pattern in advanced_patterns:
            if re.search(pattern, text_lower):
                complexity_score += 3
        
        # Multiple steps or complex formatting
        if len(text.split('\n')) > 3 or len(text) > 200:
            complexity_score += 1
        
        # Map score to difficulty level
        if complexity_score <= 2:
            return "elementary"
        elif complexity_score <= 4:
            return "middle_school"
        elif complexity_score <= 6:
            return "high_school_basic"
        elif complexity_score <= 8:
            return "high_school_advanced"
        else:
            return "college"
    
    async def _llm_extract_from_base64(self, base64_content: str) -> str:
        """Use LLM to extract text from base64 image."""
        prompt = f"""
        Extract all mathematical content from this base64 encoded image.
        Focus on:
        - Mathematical problems and exercises
        - Equations and expressions
        - Numbers and mathematical symbols
        - Problem instructions and questions
        
        Base64 content: {base64_content[:100]}...
        
        Return only the extracted mathematical text content.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
        except Exception:
            pass
        
        return ""
    
    async def _llm_extract_from_image_file(self, image_path: str) -> str:
        """Use LLM to extract text from image file."""
        prompt = f"""
        Extract all mathematical content from this image: {image_path}
        
        Focus on:
        - Mathematical problems and exercises
        - Equations and expressions
        - Numbers and mathematical symbols
        - Problem instructions and questions
        
        Return only the extracted mathematical text content.
        """
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm.chat_completion("gpt-4", messages)
            if hasattr(response, 'choices') and response.choices:
                return response.choices[0].message.content.strip()
        except Exception:
            pass
        
        return ""
    
    def _create_empty_result(self, error_message: str) -> ConceptIdentification:
        """Create an empty result for error cases."""
        return ConceptIdentification(
            course="Unknown",
            topic="Unknown",
            concepts=[],
            confidence=0.0,
            exercise_type="unknown",
            difficulty_level="unknown",
            source_content=error_message,
            analysis_method="error",
            timestamp=datetime.now().isoformat()
        )
    
    def get_supported_courses(self) -> List[str]:
        """Get list of supported course names."""
        return list(self.course_patterns.keys())
    
    def get_supported_concepts(self) -> List[str]:
        """Get list of supported concept IDs."""
        return list(self.concept_patterns.keys())
    
    def export_analysis_report(self, results: List[ConceptIdentification], 
                             output_path: str = None) -> Dict[str, Any]:
        """
        Export analysis results to a comprehensive report.
        
        Args:
            results: List of analysis results
            output_path: Optional path to save JSON report
            
        Returns:
            Dictionary containing the comprehensive report
        """
        report = {
            "analysis_summary": {
                "total_exercises": len(results),
                "analysis_timestamp": datetime.now().isoformat(),
                "courses_identified": list(set(r.course for r in results)),
                "average_confidence": sum(r.confidence for r in results) / len(results) if results else 0,
                "analysis_methods_used": list(set(r.analysis_method for r in results))
            },
            "course_distribution": {},
            "concept_frequency": {},
            "difficulty_distribution": {},
            "detailed_results": [r.to_dict() for r in results]
        }
        
        # Calculate distributions
        for result in results:
            # Course distribution
            course = result.course
            report["course_distribution"][course] = report["course_distribution"].get(course, 0) + 1
            
            # Concept frequency
            for concept in result.concepts:
                concept_name = concept["concept_name"]
                report["concept_frequency"][concept_name] = report["concept_frequency"].get(concept_name, 0) + 1
            
            # Difficulty distribution
            difficulty = result.difficulty_level
            report["difficulty_distribution"][difficulty] = report["difficulty_distribution"].get(difficulty, 0) + 1
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


# Example usage and testing functions
async def demo_concept_identification():
    """Demonstrate the concept identification system."""
    
    # Initialize system (with mock LLM for demo)
    system = ConceptIdentificationSystem()
    
    # Test cases
    test_exercises = [
        {
            "content": "Solve for x: 2x + 5 = 13",
            "content_type": "text"
        },
        {
            "content": """
            # Geometry Problem
            
            Find the area of a triangle with base 8 cm and height 6 cm.
            
            **Formula**: A = (1/2) × base × height
            """,
            "content_type": "markdown"
        },
        {
            "content": "What is the derivative of f(x) = x² + 3x - 2?",
            "content_type": "text"
        }
    ]
    
    print("🔍 Concept Identification System Demo")
    print("=" * 50)
    
    for i, exercise in enumerate(test_exercises, 1):
        print(f"\n📝 Exercise {i}:")
        print(f"Content: {exercise['content'][:100]}...")
        
        result = await system.identify_concepts(
            exercise["content"], 
            exercise["content_type"],
            use_llm=False  # Set to True if LLM is available
        )
        
        print(f"Course: {result.course}")
        print(f"Topic: {result.topic}")
        print(f"Concepts: {[c['concept_name'] for c in result.concepts]}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Exercise Type: {result.exercise_type}")
        print(f"Difficulty: {result.difficulty_level}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_concept_identification()) 