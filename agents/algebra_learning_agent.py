"""
Algebra Learning Agent

A comprehensive learning agent that combines ReAct reasoning with image recognition
and personalized math learning capabilities. This agent can:

1. Recognize math problems from uploaded images
2. Analyze student answers and identify misconceptions
3. Build and update personal learning graphs
4. Recommend personalized exercises
5. Track learning progress and provide feedback

The agent uses the ReAct (Reasoning and Acting) pattern for intelligent
decision-making and tool usage.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Add math_learning to path for imports
sys.path.append(str(Path(__file__).parent.parent / "math_learning"))

from agents.react_agent import ReActAgent
from agents.enhanced_image_recognition_agent import EnhancedImageRecognitionAgent
from tools.tool import Tool

# Math learning system imports
try:
    from math_learning.learning_graph import LearningGraph, PersonalizedLearningSystem
    from math_learning.recommendation import GapAnalyzer, Recommender
    from math_learning.knowledge_graph import KnowledgeGraph
    MATH_LEARNING_AVAILABLE = True
except ImportError:
    MATH_LEARNING_AVAILABLE = False
    print("⚠️  Math learning system not available. Using mock implementations.")


class ImageRecognitionTool(Tool):
    """Tool that wraps the Enhanced Image Recognition Agent for use in ReAct agent."""
    
    def __init__(self, image_agent: EnhancedImageRecognitionAgent):
        self.image_agent = image_agent
        self.name = "image_recognition"
        self.description = """
        Recognize and analyze math problems from images. Can extract:
        - Questions and problems from images
        - Student answers and work
        - Mathematical expressions
        - Identify misconceptions and errors
        
        Input: image_data (base64 string), image_format (optional), processing_options (optional)
        Output: Detailed analysis of the image content including questions, answers, and recommendations
        """
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute image recognition and analysis."""
        try:
            image_data = kwargs.get('image_data')
            image_format = kwargs.get('image_format', 'base64')
            processing_options = kwargs.get('processing_options', {})
            
            if not image_data:
                return {"error": "No image data provided"}
            
            # Use the enhanced image recognition agent
            result = await self.image_agent.recognize_image(
                image_data=image_data,
                image_format=image_format,
                processing_options=processing_options
            )
            
            return result.to_dict()
            
        except Exception as e:
            return {"error": f"Image recognition failed: {str(e)}"}
    
    def convert_to_function_call(self) -> Dict[str, Any]:
        """Convert tool to function call format for LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_data": {
                            "type": "string",
                            "description": "Base64 encoded image data or file path"
                        },
                        "image_format": {
                            "type": "string",
                            "description": "Format of image data (base64, file_path, bytes)",
                            "default": "base64"
                        },
                        "processing_options": {
                            "type": "object",
                            "description": "Optional processing configuration",
                            "properties": {
                                "preprocessing_type": {
                                    "type": "string",
                                    "enum": ["standard", "aggressive", "handwriting"]
                                },
                                "focus_areas": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["image_data"]
                }
            }
        }


class LearningGraphTool(Tool):
    """Tool for managing student learning graphs and progress tracking."""
    
    def __init__(self, learning_graph: Optional[Any] = None):
        self.learning_graph = learning_graph
        self.name = "learning_graph"
        self.description = """
        Manage student learning progress and build personal learning graphs.
        Can update mastery levels, track concept understanding, and analyze progress.
        
        Actions: update_mastery, get_progress, analyze_gaps, get_recommendations
        """
        
        # Mock data for when learning graph is not available
        self.mock_data = {
            "students": {},
            "concepts": [
                "basic_algebra", "linear_equations", "quadratic_equations",
                "polynomials", "factoring", "graphing", "systems_of_equations"
            ]
        }
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute learning graph operations."""
        try:
            action = kwargs.get('action', 'get_progress')
            student_id = kwargs.get('student_id', 'default')
            
            if action == "update_mastery":
                return await self._update_mastery(student_id, kwargs)
            elif action == "get_progress":
                return await self._get_progress(student_id)
            elif action == "analyze_gaps":
                return await self._analyze_gaps(student_id)
            elif action == "get_recommendations":
                return await self._get_recommendations(student_id, kwargs)
            else:
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            return {"error": f"Learning graph operation failed: {str(e)}"}
    
    async def _update_mastery(self, student_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Update student's concept mastery."""
        concept = kwargs.get('concept')
        mastery_level = kwargs.get('mastery_level', 0.5)
        evidence = kwargs.get('evidence', {})
        
        if self.learning_graph and MATH_LEARNING_AVAILABLE:
            # Use real learning graph
            self.learning_graph.update_concept_mastery(student_id, concept, mastery_level, evidence)
            return {"success": True, "concept": concept, "new_mastery": mastery_level}
        else:
            # Use mock implementation
            if student_id not in self.mock_data["students"]:
                self.mock_data["students"][student_id] = {}
            
            self.mock_data["students"][student_id][concept] = {
                "mastery_level": mastery_level,
                "last_updated": datetime.now().isoformat(),
                "evidence": evidence
            }
            
            return {
                "success": True,
                "concept": concept,
                "new_mastery": mastery_level,
                "message": f"Updated {concept} mastery to {mastery_level:.2f}"
            }
    
    async def _get_progress(self, student_id: str) -> Dict[str, Any]:
        """Get student's learning progress."""
        if self.learning_graph and MATH_LEARNING_AVAILABLE:
            # Use real learning graph
            progress = self.learning_graph.get_student_progress(student_id)
            return {"student_id": student_id, "progress": progress}
        else:
            # Use mock implementation
            student_data = self.mock_data["students"].get(student_id, {})
            
            progress = {
                "total_concepts": len(self.mock_data["concepts"]),
                "mastered_concepts": len([c for c, data in student_data.items() 
                                        if data.get("mastery_level", 0) >= 0.8]),
                "in_progress_concepts": len([c for c, data in student_data.items() 
                                           if 0.3 <= data.get("mastery_level", 0) < 0.8]),
                "struggling_concepts": len([c for c, data in student_data.items() 
                                          if data.get("mastery_level", 0) < 0.3]),
                "concept_details": student_data
            }
            
            return {"student_id": student_id, "progress": progress}
    
    async def _analyze_gaps(self, student_id: str) -> Dict[str, Any]:
        """Analyze knowledge gaps for the student."""
        progress = await self._get_progress(student_id)
        student_data = progress["progress"]["concept_details"]
        
        gaps = []
        for concept in self.mock_data["concepts"]:
            mastery = student_data.get(concept, {}).get("mastery_level", 0)
            if mastery < 0.6:
                gaps.append({
                    "concept": concept,
                    "current_mastery": mastery,
                    "gap_severity": "high" if mastery < 0.3 else "medium",
                    "recommended_action": "practice" if mastery < 0.3 else "review"
                })
        
        return {
            "student_id": student_id,
            "knowledge_gaps": gaps,
            "total_gaps": len(gaps),
            "priority_concepts": [g["concept"] for g in gaps if g["gap_severity"] == "high"]
        }
    
    async def _get_recommendations(self, student_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized exercise recommendations."""
        gaps = await self._analyze_gaps(student_id)
        focus_area = kwargs.get('focus_area')
        difficulty = kwargs.get('difficulty', 'adaptive')
        
        recommendations = []
        
        if focus_area:
            # Focus on specific area
            recommendations.append({
                "type": "targeted_practice",
                "concept": focus_area,
                "exercises": [
                    f"{focus_area}_exercise_1",
                    f"{focus_area}_exercise_2",
                    f"{focus_area}_exercise_3"
                ],
                "difficulty": difficulty,
                "estimated_time": "15-20 minutes"
            })
        else:
            # General recommendations based on gaps
            for gap in gaps["knowledge_gaps"][:3]:  # Top 3 gaps
                recommendations.append({
                    "type": "gap_filling",
                    "concept": gap["concept"],
                    "exercises": [
                        f"{gap['concept']}_basic_1",
                        f"{gap['concept']}_basic_2"
                    ],
                    "difficulty": "easy" if gap["gap_severity"] == "high" else "medium",
                    "estimated_time": "10-15 minutes"
                })
        
        return {
            "student_id": student_id,
            "recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
    
    def convert_to_function_call(self) -> Dict[str, Any]:
        """Convert tool to function call format for LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["update_mastery", "get_progress", "analyze_gaps", "get_recommendations"],
                            "description": "Action to perform on learning graph"
                        },
                        "student_id": {
                            "type": "string",
                            "description": "Unique identifier for the student"
                        },
                        "concept": {
                            "type": "string",
                            "description": "Math concept name (for update_mastery)"
                        },
                        "mastery_level": {
                            "type": "number",
                            "description": "Mastery level between 0 and 1 (for update_mastery)"
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Evidence supporting the mastery level"
                        },
                        "focus_area": {
                            "type": "string",
                            "description": "Specific concept to focus on (for recommendations)"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard", "adaptive"],
                            "description": "Difficulty level for recommendations"
                        }
                    },
                    "required": ["action", "student_id"]
                }
            }
        }


class ConceptAnalysisTool(Tool):
    """Tool for analyzing mathematical concepts and misconceptions."""
    
    def __init__(self):
        self.name = "concept_analysis"
        self.description = """
        Analyze mathematical concepts, identify misconceptions, and provide detailed feedback.
        Can analyze student work, identify error patterns, and suggest remediation strategies.
        """
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute concept analysis."""
        try:
            analysis_type = kwargs.get('analysis_type', 'misconception')
            student_work = kwargs.get('student_work', '')
            correct_answer = kwargs.get('correct_answer', '')
            concept_area = kwargs.get('concept_area', 'algebra')
            
            if analysis_type == "misconception":
                return await self._analyze_misconceptions(student_work, correct_answer, concept_area)
            elif analysis_type == "concept_mastery":
                return await self._analyze_concept_mastery(student_work, concept_area)
            elif analysis_type == "error_pattern":
                return await self._analyze_error_patterns(student_work, concept_area)
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
                
        except Exception as e:
            return {"error": f"Concept analysis failed: {str(e)}"}
    
    async def _analyze_misconceptions(self, student_work: str, correct_answer: str, concept_area: str) -> Dict[str, Any]:
        """Analyze potential misconceptions in student work."""
        misconceptions = []
        
        # Common algebra misconceptions
        if concept_area == "algebra":
            if "x + x = x²" in student_work or "2x = x²" in student_work:
                misconceptions.append({
                    "type": "variable_multiplication",
                    "description": "Confusing addition and multiplication of variables",
                    "remediation": "Practice distinguishing between x + x = 2x and x × x = x²"
                })
            
            if "√(a + b) = √a + √b" in student_work:
                misconceptions.append({
                    "type": "radical_distribution",
                    "description": "Incorrectly distributing square root over addition",
                    "remediation": "Practice with specific examples showing √(9 + 16) ≠ √9 + √16"
                })
        
        # Analyze work for common patterns
        if student_work != correct_answer and student_work:
            misconceptions.append({
                "type": "general_error",
                "description": f"Student answer '{student_work}' differs from correct answer '{correct_answer}'",
                "remediation": "Review the problem step by step and identify where the error occurred"
            })
        
        return {
            "misconceptions": misconceptions,
            "concept_area": concept_area,
            "confidence": 0.8 if misconceptions else 0.3,
            "recommendations": [m["remediation"] for m in misconceptions]
        }
    
    async def _analyze_concept_mastery(self, student_work: str, concept_area: str) -> Dict[str, Any]:
        """Analyze student's mastery of specific concepts."""
        mastery_indicators = {
            "correct_notation": 0.2,
            "logical_steps": 0.3,
            "correct_result": 0.3,
            "explanation_quality": 0.2
        }
        
        # Simple heuristic analysis
        score = 0.0
        if student_work:
            if any(op in student_work for op in ['+', '-', '*', '/', '=']):
                score += mastery_indicators["correct_notation"]
            if len(student_work.split('\n')) > 1 or len(student_work.split()) > 3:
                score += mastery_indicators["logical_steps"]
            if any(char.isdigit() for char in student_work):
                score += mastery_indicators["correct_result"]
            if len(student_work) > 20:
                score += mastery_indicators["explanation_quality"]
        
        mastery_level = min(score, 1.0)
        
        return {
            "concept_area": concept_area,
            "mastery_level": mastery_level,
            "mastery_category": "high" if mastery_level >= 0.7 else "medium" if mastery_level >= 0.4 else "low",
            "strengths": ["Shows work"] if student_work else [],
            "areas_for_improvement": ["Provide more detailed explanations"] if mastery_level < 0.7 else []
        }
    
    async def _analyze_error_patterns(self, student_work: str, concept_area: str) -> Dict[str, Any]:
        """Identify patterns in student errors."""
        error_patterns = []
        
        if not student_work:
            error_patterns.append({
                "pattern": "no_work_shown",
                "description": "Student did not show their work",
                "frequency": 1,
                "impact": "high"
            })
        
        # Check for common error patterns
        if "=" in student_work:
            equals_count = student_work.count("=")
            if equals_count > 3:
                error_patterns.append({
                    "pattern": "excessive_equals",
                    "description": "Using too many equals signs in a single line",
                    "frequency": equals_count,
                    "impact": "medium"
                })
        
        return {
            "concept_area": concept_area,
            "error_patterns": error_patterns,
            "pattern_count": len(error_patterns),
            "severity": "high" if any(p["impact"] == "high" for p in error_patterns) else "medium"
        }
    
    def convert_to_function_call(self) -> Dict[str, Any]:
        """Convert tool to function call format for LLM."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "enum": ["misconception", "concept_mastery", "error_pattern"],
                            "description": "Type of analysis to perform"
                        },
                        "student_work": {
                            "type": "string",
                            "description": "Student's work or answer to analyze"
                        },
                        "correct_answer": {
                            "type": "string",
                            "description": "Correct answer for comparison"
                        },
                        "concept_area": {
                            "type": "string",
                            "description": "Mathematical concept area (e.g., algebra, geometry)"
                        }
                    },
                    "required": ["analysis_type", "student_work"]
                }
            }
        }


class AlgebraLearningAgent(ReActAgent):
    """
    Comprehensive Algebra Learning Agent that combines ReAct reasoning with
    image recognition and personalized math learning capabilities.
    
    This agent can:
    1. Process uploaded images of math problems
    2. Recognize questions and analyze student answers
    3. Identify misconceptions and mastery levels
    4. Update personal learning graphs
    5. Recommend personalized exercises
    6. Provide intelligent tutoring feedback
    """
    
    def __init__(self, llm, image_recognition_agent: EnhancedImageRecognitionAgent = None, 
                 learning_graph=None, max_iterations: int = 7):
        """
        Initialize the Algebra Learning Agent.
        
        Args:
            llm: Language model for reasoning and responses
            image_recognition_agent: Enhanced image recognition agent instance
            learning_graph: Learning graph for tracking student progress
            max_iterations: Maximum ReAct iterations
        """
        
        # Create image recognition agent if not provided
        if image_recognition_agent is None:
            # Create a mock LLM for the image agent if needed
            class MockLLM:
                async def agenerate(self, prompt, **kwargs):
                    return "Mock LLM response for image analysis"
            
            class MockPromptManager:
                def get_prompt(self, name, **kwargs):
                    return "Mock prompt"
            
            image_recognition_agent = EnhancedImageRecognitionAgent(
                llm=MockLLM(),
                prompt_manager=MockPromptManager()
            )
        
        # Initialize tools
        tools = [
            ImageRecognitionTool(image_recognition_agent),
            LearningGraphTool(learning_graph),
            ConceptAnalysisTool()
        ]
        
        # Define the agent's role and capabilities
        role = """You are an expert algebra tutor and learning assistant. You help students learn algebra by:
1. Analyzing uploaded images of math problems and student work
2. Identifying questions, student answers, and mathematical concepts
3. Detecting misconceptions and areas of mastery
4. Building personalized learning profiles
5. Recommending targeted practice exercises
6. Providing clear, encouraging feedback"""

        task = """Help students learn algebra through personalized, AI-powered tutoring. When a student uploads an image:
1. Use image_recognition to analyze the content
2. Use concept_analysis to identify misconceptions or mastery
3. Use learning_graph to update the student's progress
4. Provide helpful feedback and recommendations"""

        guide = """Always follow this process:
1. THINK about what the student needs
2. ACT by using appropriate tools to analyze their work
3. OBSERVE the results from your tools
4. THINK about what you learned and what to do next
5. Continue until you have a complete understanding
6. Provide a comprehensive, helpful response

Be encouraging, specific in your feedback, and always explain your reasoning."""

        examples = [
            """Student uploads image of algebra homework.

Thought: I need to analyze this image to understand what math problems are shown and how the student answered them.

Action: image_recognition
Action Input: {"image_data": "[base64_image_data]", "processing_options": {"preprocessing_type": "standard"}}

Observation: Found 3 algebra problems. Student answered 2x + 3 = 7 with x = 5, but correct answer is x = 2.

Thought: The student made an error. Let me analyze what misconception might be causing this.

Action: concept_analysis  
Action Input: {"analysis_type": "misconception", "student_work": "x = 5", "correct_answer": "x = 2", "concept_area": "linear_equations"}

Observation: Student may have subtracted 3 from both sides incorrectly, getting 2x = 4, then divided by 2 to get x = 5 instead of x = 2.

Thought: Now I should update their learning graph and provide feedback.

Action: learning_graph
Action Input: {"action": "update_mastery", "student_id": "student123", "concept": "linear_equations", "mastery_level": 0.4, "evidence": {"error_type": "arithmetic", "problem_type": "two_step_equation"}}

Observation: Updated student's mastery level for linear equations to 0.4.

Final Answer: I can see you're working on solving linear equations! You have the right approach by isolating x, but there's a small arithmetic error. When you have 2x + 3 = 7, you correctly subtract 3 from both sides to get 2x = 4. However, when dividing both sides by 2, you should get x = 2, not x = 5. Let's practice a few more similar problems to strengthen this skill!"""
        ]
        
        # Initialize the ReAct agent
        super().__init__(
            llm=llm,
            role=role,
            task=task,
            guide=guide,
            examples=examples,
            tools=tools,
            max_iterations=max_iterations
        )
        
        # Store additional components
        self.image_recognition_agent = image_recognition_agent
        self.learning_graph = learning_graph
        
        # Session management
        self.current_student_id = None
        self.session_data = {}
    
    def set_student(self, student_id: str) -> None:
        """Set the current student for the session."""
        self.current_student_id = student_id
        if student_id not in self.session_data:
            self.session_data[student_id] = {
                "session_start": datetime.now().isoformat(),
                "interactions": [],
                "progress_updates": []
            }
    
    async def process_image_upload(self, student_id: str, image_data: str, 
                                 image_format: str = "base64", **kwargs) -> str:
        """
        Process an uploaded image for a specific student.
        
        Args:
            student_id: Student identifier
            image_data: Image data (base64, file path, etc.)
            image_format: Format of the image data
            **kwargs: Additional parameters
            
        Returns:
            Comprehensive analysis and feedback
        """
        self.set_student(student_id)
        
        # Create a user input that includes the image context
        user_input = f"""A student (ID: {student_id}) has uploaded an image of their algebra work. 
Please analyze the image to identify:
1. What math problems are shown
2. How the student answered them  
3. Any errors or misconceptions
4. Update their learning progress
5. Provide helpful feedback and recommendations

Image data provided in the image_data parameter."""
        
        # Use the ReAct agent to process this
        response = await self.arun(
            user_input=user_input,
            model=kwargs.get('model', 'gpt-4'),
            image_data=image_data,
            image_format=image_format,
            student_id=student_id
        )
        
        # Log the interaction
        self.session_data[student_id]["interactions"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "image_upload",
            "response": response
        })
        
        return response
    
    async def get_student_dashboard(self, student_id: str) -> Dict[str, Any]:
        """Get a comprehensive dashboard for the student."""
        self.set_student(student_id)
        
        # Get progress from learning graph
        progress_tool = self.tools[1]  # LearningGraphTool
        progress = await progress_tool.execute(action="get_progress", student_id=student_id)
        gaps = await progress_tool.execute(action="analyze_gaps", student_id=student_id)
        recommendations = await progress_tool.execute(action="get_recommendations", student_id=student_id)
        
        return {
            "student_id": student_id,
            "progress": progress,
            "knowledge_gaps": gaps,
            "recommendations": recommendations,
            "session_data": self.session_data.get(student_id, {}),
            "generated_at": datetime.now().isoformat()
        }
    
    async def practice_session(self, student_id: str, focus_concept: str = None, 
                             difficulty: str = "adaptive") -> str:
        """Start a practice session for the student."""
        self.set_student(student_id)
        
        user_input = f"""Student {student_id} wants to start a practice session. 
Focus concept: {focus_concept or 'adaptive based on gaps'}
Difficulty: {difficulty}

Please analyze their current progress, identify the best concepts to practice, 
and provide personalized exercise recommendations."""
        
        return await self.arun(
            user_input=user_input,
            model='gpt-4',
            student_id=student_id,
            focus_concept=focus_concept,
            difficulty=difficulty
        )


# Convenience function for easy agent creation
async def create_algebra_learning_agent(llm, **kwargs) -> AlgebraLearningAgent:
    """
    Create a fully configured Algebra Learning Agent.
    
    Args:
        llm: Language model instance
        **kwargs: Additional configuration options
        
    Returns:
        Configured AlgebraLearningAgent
    """
    # Create enhanced image recognition agent
    class MockPromptManager:
        def get_prompt(self, name, **kwargs):
            return "You are an expert at analyzing mathematical content in images."
    
    image_agent = EnhancedImageRecognitionAgent(
        llm=llm,
        prompt_manager=MockPromptManager()
    )
    
    # Create the algebra learning agent
    agent = AlgebraLearningAgent(
        llm=llm,
        image_recognition_agent=image_agent,
        learning_graph=kwargs.get('learning_graph'),
        max_iterations=kwargs.get('max_iterations', 7)
    )
    
    return agent 