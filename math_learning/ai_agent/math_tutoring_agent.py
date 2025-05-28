"""
Math Tutoring Agent using React pattern for intelligent math tutoring.

This agent integrates with the math learning system to provide AI-powered
tutoring capabilities including concept analysis, exercise generation,
progress tracking, and personalized learning paths.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Fix import paths to match actual structure
from math_learning.learning_graph import LearningGraph
from math_learning.recommendation import GapAnalyzer, Recommender

# Import the tools
from .tools import (
    ConceptAnalysisTool, 
    ExerciseGenerationTool, 
    ProgressTrackingTool, 
    LearningPathTool, 
    ExplanationTool,
    ImageAnalysisTool,
    ExerciseRecognitionTool
)

# Use a simpler base for testing - avoid Redis dependency
class SimpleReActAgent:
    """Simplified React agent for testing without Redis dependencies."""
    
    def __init__(self, llm, tools: List[Any] = None, max_iterations: int = 5):
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.memory = {}  # Simple in-memory storage
        
    async def run(self, query: str, **kwargs) -> str:
        """Run the React reasoning loop."""
        thought = f"I need to help with: {query}"
        
        # Simple reasoning: analyze the query and use appropriate tools
        if "exercise" in query.lower() or "practice" in query.lower():
            # Use exercise generation tool
            if len(self.tools) > 1:
                result = await self.tools[1].execute(
                    student_id=kwargs.get('student_id', 'default'),
                    concept="general",
                    difficulty="medium",
                    count=3
                )
                return f"I've generated some practice exercises for you: {result.get('summary', 'Exercises created')}"
        
        elif "explain" in query.lower():
            # Use explanation tool
            if len(self.tools) > 4:
                result = await self.tools[4].execute(
                    explanation_type="concept",
                    content="general",
                    student_level="middle_school"
                )
                return result.get('explanation', {}).get('main_explanation', 'Here is an explanation of the concept.')
        
        # Default: use LLM to generate response
        return await self.llm.generate_response(query, **kwargs)

class MathTutoringAgent(SimpleReActAgent):
    """
    AI Math Tutoring Agent that provides personalized learning experiences.
    
    This agent integrates with the existing math learning system components:
    - LearningGraph for tracking student progress
    - Recommender for exercise suggestions
    - GapAnalyzer for identifying knowledge gaps
    - KnowledgeGraph for concept relationships
    """
    
    def __init__(self, llm, learning_graph, user_model, recommender, gap_analyzer, max_iterations: int = 5):
        """
        Initialize the Math Tutoring Agent.
        
        Args:
            llm: Language model for generating responses
            learning_graph: Student's learning progress graph
            user_model: User model for personalization
            recommender: Exercise recommendation system
            gap_analyzer: Knowledge gap analysis system
            max_iterations: Maximum reasoning iterations
        """
        # Initialize tools
        self.tools = [
            ConceptAnalysisTool(learning_graph, gap_analyzer),
            ExerciseGenerationTool(recommender, learning_graph),
            ProgressTrackingTool(learning_graph),
            LearningPathTool(learning_graph, gap_analyzer),
            ExplanationTool(learning_graph, llm),
            ImageAnalysisTool(learning_graph, llm, knowledge_graph=getattr(gap_analyzer, 'knowledge_graph', None)),
            ExerciseRecognitionTool(llm)
        ]
        
        super().__init__(llm, self.tools, max_iterations)
        
        self.learning_graph = learning_graph
        self.user_model = user_model
        self.recommender = recommender
        self.gap_analyzer = gap_analyzer
        
        # Session management
        self.current_session = {}
        self.session_history = []
        
    async def start_tutoring_session(self, student_id: str, context: Dict[str, Any]) -> None:
        """
        Start a new tutoring session for a student.
        
        Args:
            student_id: Unique identifier for the student
            context: Session context including topic, goals, etc.
        """
        self.current_session = {
            "student_id": student_id,
            "start_time": datetime.now().isoformat(),
            "context": context,
            "interactions": []
        }
        
        # Store in memory
        self.memory[f"session_{student_id}"] = self.current_session
        
    async def tutor(self, student_input: str, model: str = "gpt-4", **kwargs) -> str:
        """
        Main tutoring method that processes student input and provides intelligent responses.
        
        Args:
            student_input: The student's question or input
            model: LLM model to use
            **kwargs: Additional parameters
            
        Returns:
            AI tutor's response
        """
        # Add student input to session history
        if self.current_session:
            self.current_session["interactions"].append({
                "timestamp": datetime.now().isoformat(),
                "student_input": student_input,
                "type": "student_message"
            })
        
        # Use React reasoning to generate response
        response = await self.run(student_input, model=model, **kwargs)
        
        # Add response to session history
        if self.current_session:
            self.current_session["interactions"].append({
                "timestamp": datetime.now().isoformat(),
                "tutor_response": response,
                "type": "tutor_response"
            })
        
        return response
        
    async def generate_learning_plan(self, student_id: str, target_concepts: List[str], 
                                   timeframe_days: int = 30) -> Dict[str, Any]:
        """
        Generate a personalized learning plan for a student.
        
        Args:
            student_id: Student identifier
            target_concepts: List of concepts to master
            timeframe_days: Time frame for the learning plan
            
        Returns:
            Structured learning plan
        """
        # Use the learning path tool to generate the plan
        learning_path_tool = self.tools[3]  # LearningPathTool
        
        plan_steps = []
        for concept in target_concepts:
            path_result = await learning_path_tool.execute(
                student_id=student_id,
                target_concept=concept,
                path_type="adaptive"
            )
            
            if "learning_path" in path_result:
                plan_steps.extend(path_result["learning_path"])
        
        return {
            "student_id": student_id,
            "target_concepts": target_concepts,
            "timeframe_days": timeframe_days,
            "plan_steps": plan_steps,
            "generated_at": datetime.now().isoformat()
        }
        
    def get_student_progress_summary(self, student_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive progress summary for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Progress summary data
        """
        # Get session data
        session_key = f"session_{student_id}"
        session_data = self.memory.get(session_key, {})
        
        # Get mastery data from learning graph
        mastery_data = {}
        if hasattr(self.learning_graph, 'concept_mastery'):
            mastery_data = self.learning_graph.concept_mastery
        
        return {
            "student_id": student_id,
            "session_data": session_data,
            "mastery_summary": mastery_data,
            "generated_at": datetime.now().isoformat(),
            "total_concepts_practiced": len(mastery_data),
            "average_mastery": sum(data.get("mastery", 0) for data in mastery_data.values()) / max(len(mastery_data), 1)
        }
        
    async def analyze_student_needs(self, student_id: str) -> Dict[str, Any]:
        """
        Analyze student's current learning needs and gaps.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Analysis of student needs
        """
        # Use concept analysis tool
        concept_tool = self.tools[0]  # ConceptAnalysisTool
        
        analysis = await concept_tool.execute(
            student_id=student_id,
            concept="general_assessment",
            include_prerequisites=True
        )
        
        return analysis
        
    async def recommend_next_steps(self, student_id: str) -> Dict[str, Any]:
        """
        Recommend next learning steps for a student.
        
        Args:
            student_id: Student identifier
            
        Returns:
            Recommended next steps
        """
        # Use multiple tools to generate comprehensive recommendations
        
        # 1. Analyze current progress
        progress_tool = self.tools[2]  # ProgressTrackingTool
        progress = await progress_tool.execute(
            student_id=student_id,
            analysis_type="detailed"
        )
        
        # 2. Generate learning path
        path_tool = self.tools[3]  # LearningPathTool
        next_path = await path_tool.execute(
            student_id=student_id,
            target_concept="next_logical_step",
            path_type="sequential"
        )
        
        # 3. Suggest exercises
        exercise_tool = self.tools[1]  # ExerciseGenerationTool
        exercises = await exercise_tool.execute(
            student_id=student_id,
            concept="current_focus",
            difficulty="adaptive",
            count=5
        )
        
        return {
            "student_id": student_id,
            "progress_analysis": progress,
            "recommended_path": next_path,
            "suggested_exercises": exercises,
            "generated_at": datetime.now().isoformat()
        }
        
    async def analyze_exercise_image(self, student_id: str, image_data: str, 
                                   image_format: str = "base64") -> Dict[str, Any]:
        """
        Analyze an uploaded image of a math exercise with wrong answer.
        
        This method provides a direct interface for image analysis without
        going through the full React reasoning loop.
        
        Args:
            student_id: ID of the student
            image_data: Image data (base64 encoded or file path)
            image_format: Format of image_data ("base64", "file_path")
            
        Returns:
            Dictionary containing:
            - error_analysis: What went wrong and why
            - missing_concepts: Concepts the student needs to learn
            - learning_graph_updates: How the student's profile was updated
            - exercise_recommendation: Targeted exercise to practice
        """
        # Get the ImageAnalysisTool (should be the last tool in the list)
        image_tool = None
        for tool in self.tools:
            if isinstance(tool, ImageAnalysisTool):
                image_tool = tool
                break
        
        if not image_tool:
            return {
                "success": False,
                "error": "ImageAnalysisTool not available"
            }
        
        # Execute the image analysis
        result = await image_tool.execute(
            student_id=student_id,
            image_data=image_data,
            image_format=image_format
        )
        
        return result
    
    async def process_multiple_images(self, student_id: str, images: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process multiple exercise images to gradually build the learning graph.
        
        Args:
            student_id: ID of the student
            images: List of image dictionaries with 'data' and 'format' keys
            
        Returns:
            Dictionary containing aggregated analysis and learning progress
        """
        results = []
        learning_progression = []
        
        for i, image_info in enumerate(images):
            print(f"📸 Processing image {i+1}/{len(images)} for student {student_id}")
            
            # Analyze each image
            result = await self.analyze_exercise_image(
                student_id=student_id,
                image_data=image_info['data'],
                image_format=image_info.get('format', 'base64')
            )
            
            results.append(result)
            
            if result.get('success'):
                # Track learning progression
                learning_progression.append({
                    "image_number": i + 1,
                    "concepts_identified": result.get('error_analysis', {}).get('missing_concepts', []),
                    "confidence_changes": result.get('learning_graph_updates', {}).get('confidence_changes', {}),
                    "recommendation": result.get('exercise_recommendation', {}).get('target_concept', 'unknown')
                })
        
        # Generate summary of learning progression
        summary = self._generate_learning_progression_summary(student_id, learning_progression)
        
        return {
            "student_id": student_id,
            "total_images_processed": len(images),
            "individual_results": results,
            "learning_progression": learning_progression,
            "summary": summary,
            "success": True
        }
    
    def _generate_learning_progression_summary(self, student_id: str, progression: List[Dict]) -> Dict[str, Any]:
        """Generate a summary of the student's learning progression from multiple images."""
        if not progression:
            return {"no_data": True}
        
        # Aggregate concepts identified across all images
        all_concepts = []
        for entry in progression:
            all_concepts.extend(entry.get('concepts_identified', []))
        
        # Count concept frequency
        concept_frequency = {}
        for concept in all_concepts:
            concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
        
        # Identify most problematic concepts
        most_problematic = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Track confidence changes over time
        confidence_trends = {}
        for entry in progression:
            for concept, changes in entry.get('confidence_changes', {}).items():
                if concept not in confidence_trends:
                    confidence_trends[concept] = []
                confidence_trends[concept].append(changes.get('new', 0.0))
        
        return {
            "total_concepts_identified": len(set(all_concepts)),
            "most_problematic_concepts": [concept for concept, count in most_problematic],
            "concept_frequency": concept_frequency,
            "confidence_trends": confidence_trends,
            "learning_trajectory": "improving" if self._is_improving(confidence_trends) else "needs_attention",
            "recommendations": self._generate_progression_recommendations(most_problematic, confidence_trends)
        }
    
    def _is_improving(self, confidence_trends: Dict[str, List[float]]) -> bool:
        """Determine if the student is showing improvement over time."""
        improving_concepts = 0
        total_concepts = len(confidence_trends)
        
        for concept, values in confidence_trends.items():
            if len(values) >= 2:
                # Check if latest values show improvement
                if values[-1] > values[0]:
                    improving_concepts += 1
        
        return improving_concepts > total_concepts / 2 if total_concepts > 0 else False
    
    def _generate_progression_recommendations(self, most_problematic: List[Tuple[str, int]], 
                                           confidence_trends: Dict[str, List[float]]) -> List[str]:
        """Generate recommendations based on learning progression."""
        recommendations = []
        
        if most_problematic:
            primary_issue = most_problematic[0][0]
            recommendations.append(f"Focus on mastering {primary_issue} - appeared in {most_problematic[0][1]} exercises")
        
        # Check for concepts with declining confidence
        declining_concepts = []
        for concept, values in confidence_trends.items():
            if len(values) >= 2 and values[-1] < values[0]:
                declining_concepts.append(concept)
        
        if declining_concepts:
            recommendations.append(f"Review fundamentals in: {', '.join(declining_concepts[:2])}")
        
        if not recommendations:
            recommendations.append("Continue practicing with varied problem types")
        
        return recommendations
    
    async def recognize_exercises_from_text(self, text_content: str) -> Dict[str, Any]:
        """
        Recognize and classify exercises from text content.
        
        Args:
            text_content: The text content containing exercises
            
        Returns:
            Dict containing recognized exercises and analysis
        """
        try:
            # Find the exercise recognition tool
            exercise_recognition_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "exercise_recognition":
                    exercise_recognition_tool = tool
                    break
            
            if not exercise_recognition_tool:
                return {
                    "success": False,
                    "error": "Exercise recognition tool not available",
                    "exercises": []
                }
            
            # Execute exercise recognition
            result = await exercise_recognition_tool.execute(
                content=text_content,
                source_type="text"
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exercise recognition failed: {str(e)}",
                "exercises": []
            }
    
    async def recognize_exercises_from_image(self, image_data: str, image_format: str = "base64") -> Dict[str, Any]:
        """
        Recognize and classify exercises from image content.
        
        Args:
            image_data: The image data (base64 encoded or file path)
            image_format: Format of the image data ("base64" or "file")
            
        Returns:
            Dict containing recognized exercises and analysis
        """
        try:
            # Find the exercise recognition tool
            exercise_recognition_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "exercise_recognition":
                    exercise_recognition_tool = tool
                    break
            
            if not exercise_recognition_tool:
                return {
                    "success": False,
                    "error": "Exercise recognition tool not available",
                    "exercises": []
                }
            
            # Determine source type
            source_type = "base64" if image_format == "base64" else "image"
            
            # Execute exercise recognition
            result = await exercise_recognition_tool.execute(
                content=image_data,
                source_type=source_type
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exercise recognition from image failed: {str(e)}",
                "exercises": []
            }
    
    async def recognize_and_generate_practice(self, content: str, source_type: str = "text", student_id: str = "default") -> Dict[str, Any]:
        """
        Recognize exercises from content and generate additional practice exercises.
        
        Args:
            content: The content containing exercises (text or image data)
            source_type: Type of content ("text", "base64", "image")
            student_id: ID of the student for personalized practice
            
        Returns:
            Dict containing recognized exercises and generated practice
        """
        try:
            # First, recognize exercises from the content
            if source_type == "text":
                recognition_result = await self.recognize_exercises_from_text(content)
            else:
                recognition_result = await self.recognize_exercises_from_image(content, source_type)
            
            if not recognition_result.get("success"):
                return recognition_result
            
            recognized_exercises = recognition_result.get("exercises", [])
            
            # Generate additional practice exercises based on recognized concepts
            practice_exercises = []
            concepts_found = set()
            
            for exercise in recognized_exercises:
                concept = exercise.get("concept", "general_math")
                concepts_found.add(concept)
            
            # Generate practice exercises for each concept found
            exercise_generation_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == "exercise_generation":
                    exercise_generation_tool = tool
                    break
            
            if exercise_generation_tool:
                for concept in concepts_found:
                    try:
                        practice_result = await exercise_generation_tool.execute(
                            student_id=student_id,
                            concept=concept,
                            difficulty="adaptive",
                            count=3
                        )
                        
                        if "exercises" in practice_result:
                            practice_exercises.extend(practice_result["exercises"])
                    except Exception as e:
                        print(f"Failed to generate practice for {concept}: {e}")
            
            return {
                "success": True,
                "recognized_exercises": recognized_exercises,
                "practice_exercises": practice_exercises,
                "total_recognized": len(recognized_exercises),
                "total_practice": len(practice_exercises),
                "concepts_found": list(concepts_found),
                "analysis_timestamp": recognition_result.get("analysis_timestamp")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Exercise recognition and practice generation failed: {str(e)}",
                "recognized_exercises": [],
                "practice_exercises": []
            } 