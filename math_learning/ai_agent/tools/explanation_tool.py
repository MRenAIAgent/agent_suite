"""
Explanation Tool for the Math Tutoring Agent.

This tool provides natural language explanations for mathematical concepts,
problems, and solutions using LLM capabilities.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

# Fix import paths to match actual structure
from math_learning.learning_graph import LearningGraph


class ExplanationTool:
    """
    Tool for generating natural language explanations.
    
    This tool provides:
    - Concept explanations adapted to student level
    - Step-by-step problem solutions
    - Error analysis and correction guidance
    - Conceptual connections and analogies
    """
    
    def __init__(self, learning_graph: LearningGraph, llm):
        """
        Initialize the Explanation Tool.
        
        Args:
            learning_graph: Student's learning progress graph
            llm: Language model for generating explanations
        """
        self.learning_graph = learning_graph
        self.llm = llm
        self.name = "explanation"
        self.description = "Generate natural language explanations for mathematical concepts and problems"
        
    async def execute(self, explanation_type: str, content: str, 
                     student_level: str = "intermediate", context: Dict = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Execute explanation generation.
        
        Args:
            explanation_type: Type of explanation (concept, solution, error_analysis, connection)
            content: The mathematical content to explain
            student_level: Student's current level (beginner, intermediate, advanced)
            context: Additional context for the explanation
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the generated explanation and metadata
        """
        try:
            # Determine student's current understanding level
            current_level = self._determine_student_level()
            
            # Generate explanation based on type
            if explanation_type == "concept":
                explanation = await self._generate_concept_explanation(content, current_level, context)
            elif explanation_type == "solution":
                explanation = await self._generate_solution_explanation(content, current_level, context)
            elif explanation_type == "error_analysis":
                explanation = await self._generate_error_analysis(content, current_level, context)
            elif explanation_type == "connection":
                explanation = await self._generate_conceptual_connections(content, current_level, context)
            else:
                explanation = await self._generate_general_explanation(content, current_level, context)
            
            # Add explanation metadata
            explanation_metadata = self._generate_explanation_metadata(explanation, explanation_type)
            
            # Create comprehensive explanation response
            explanation_response = {
                "explanation_type": explanation_type,
                "content": content,
                "student_level": current_level,
                "explanation": explanation,
                "metadata": explanation_metadata,
                "follow_up_suggestions": self._generate_follow_up_suggestions(explanation_type, content),
                "related_concepts": self._identify_related_concepts(content),
                "difficulty_level": self._assess_explanation_difficulty(explanation),
                "generated_at": datetime.now().isoformat()
            }
            
            return explanation_response
            
        except Exception as e:
            return {
                "error": f"Failed to generate explanation: {str(e)}",
                "explanation_type": explanation_type,
                "content": content,
                "generated_at": datetime.now().isoformat()
            }
    
    def _determine_student_level(self) -> str:
        """Determine the student's current understanding level."""
        try:
            mastered_concepts = self.learning_graph.get_mastered_concepts(threshold=0.7)
            total_concepts = len(mastered_concepts) + 20  # Estimate total concepts
            mastery_rate = len(mastered_concepts) / max(total_concepts, 1)
            
            if mastery_rate < 0.3:
                return "beginner"
            elif mastery_rate < 0.7:
                return "intermediate"
            else:
                return "advanced"
        except Exception:
            return "intermediate"  # Default level
    
    async def _generate_concept_explanation(self, concept: str, level: str, context: Dict) -> Dict[str, Any]:
        """Generate a concept explanation adapted to student level."""
        # Create level-appropriate prompt
        if level == "beginner":
            prompt = f"Explain the mathematical concept '{concept}' in simple terms for a beginner. Use everyday examples and avoid complex terminology."
        elif level == "advanced":
            prompt = f"Provide a comprehensive explanation of '{concept}' including formal definitions, properties, and advanced applications."
        else:
            prompt = f"Explain the mathematical concept '{concept}' for an intermediate student. Include key properties and practical examples."
        
        # Add context if available
        if context:
            prompt += f" Context: {context.get('additional_info', '')}"
        
        # Generate explanation using LLM
        try:
            explanation_text = await self.llm.generate_response(prompt)
        except Exception:
            explanation_text = f"A mathematical concept that involves {concept}. This is an important topic in mathematics."
        
        return {
            "main_explanation": explanation_text,
            "key_points": self._extract_key_points(concept, level),
            "examples": self._generate_examples(concept, level),
            "visual_aids": self._suggest_visual_aids(concept),
            "common_misconceptions": self._identify_misconceptions(concept)
        }
    
    async def _generate_solution_explanation(self, problem: str, level: str, context: Dict) -> Dict[str, Any]:
        """Generate a step-by-step solution explanation."""
        # Create solution prompt
        prompt = f"Provide a step-by-step solution to this math problem: {problem}. Explain each step clearly for a {level} student."
        
        if context and context.get("show_work"):
            prompt += " Show all work and reasoning."
        
        # Generate solution using LLM
        try:
            solution_text = await self.llm.generate_response(prompt)
        except Exception:
            solution_text = f"To solve this problem: {problem}, we need to follow systematic steps."
        
        return {
            "step_by_step_solution": solution_text,
            "solution_steps": self._break_down_solution_steps(problem),
            "key_strategies": self._identify_solution_strategies(problem),
            "alternative_methods": self._suggest_alternative_methods(problem),
            "verification": self._suggest_verification_methods(problem)
        }
    
    async def _generate_error_analysis(self, error_content: str, level: str, context: Dict) -> Dict[str, Any]:
        """Generate error analysis and correction guidance."""
        # Create error analysis prompt
        prompt = f"Analyze this mathematical error and provide correction guidance: {error_content}. Explain what went wrong and how to fix it for a {level} student."
        
        # Generate analysis using LLM
        try:
            analysis_text = await self.llm.generate_response(prompt)
        except Exception:
            analysis_text = f"Let's analyze this error: {error_content} and find the correct approach."
        
        return {
            "error_analysis": analysis_text,
            "error_type": self._classify_error_type(error_content),
            "correction_steps": self._generate_correction_steps(error_content),
            "prevention_tips": self._generate_prevention_tips(error_content),
            "practice_suggestions": self._suggest_practice_problems(error_content)
        }
    
    async def _generate_conceptual_connections(self, concept: str, level: str, context: Dict) -> Dict[str, Any]:
        """Generate explanations of conceptual connections."""
        # Create connections prompt
        prompt = f"Explain how the mathematical concept '{concept}' connects to other mathematical ideas and real-world applications for a {level} student."
        
        # Generate connections using LLM
        try:
            connections_text = await self.llm.generate_response(prompt)
        except Exception:
            connections_text = f"The concept '{concept}' connects to many other mathematical ideas and has practical applications."
        
        return {
            "conceptual_connections": connections_text,
            "related_concepts": self._find_related_concepts(concept),
            "real_world_applications": self._identify_applications(concept),
            "prerequisite_concepts": self._identify_prerequisites(concept),
            "advanced_extensions": self._suggest_extensions(concept)
        }
    
    async def _generate_general_explanation(self, content: str, level: str, context: Dict) -> Dict[str, Any]:
        """Generate a general explanation for any mathematical content."""
        # Create general prompt
        prompt = f"Provide a clear mathematical explanation for: {content}. Adapt the explanation for a {level} student."
        
        # Generate explanation using LLM
        try:
            explanation_text = await self.llm.generate_response(prompt)
        except Exception:
            explanation_text = f"Here's an explanation of: {content}"
        
        return {
            "general_explanation": explanation_text,
            "key_insights": self._extract_key_insights(content),
            "practical_relevance": self._assess_practical_relevance(content),
            "learning_tips": self._generate_learning_tips(content)
        }
    
    def _extract_key_points(self, concept: str, level: str) -> List[str]:
        """Extract key points for a concept based on student level."""
        key_points_map = {
            "fractions": [
                "A fraction represents parts of a whole",
                "The numerator shows how many parts we have",
                "The denominator shows how many parts make the whole"
            ],
            "algebra": [
                "Variables represent unknown quantities",
                "Equations show relationships between quantities",
                "Solving means finding the value of the variable"
            ]
        }
        
        return key_points_map.get(concept, [f"Key concept: {concept}"])
    
    def _generate_examples(self, concept: str, level: str) -> List[Dict[str, str]]:
        """Generate examples for a concept based on student level."""
        if level == "beginner":
            return [
                {"example": "Simple example", "explanation": "Basic explanation"},
                {"example": "Visual example", "explanation": "Using pictures or diagrams"}
            ]
        else:
            return [
                {"example": "Standard example", "explanation": "Typical application"},
                {"example": "Advanced example", "explanation": "More complex scenario"}
            ]
    
    def _suggest_visual_aids(self, concept: str) -> List[str]:
        """Suggest visual aids for understanding a concept."""
        visual_aids_map = {
            "fractions": ["pie charts", "number lines", "visual fraction bars"],
            "geometry": ["diagrams", "3D models", "coordinate planes"],
            "algebra": ["graphs", "tables", "visual equation solving"]
        }
        
        return visual_aids_map.get(concept, ["diagrams", "charts", "visual representations"])
    
    def _identify_misconceptions(self, concept: str) -> List[str]:
        """Identify common misconceptions for a concept."""
        misconceptions_map = {
            "fractions": [
                "Thinking larger denominators mean larger fractions",
                "Adding numerators and denominators separately"
            ],
            "algebra": [
                "Thinking variables always equal 1",
                "Confusing solving with simplifying"
            ]
        }
        
        return misconceptions_map.get(concept, ["Common errors in understanding"])
    
    def _break_down_solution_steps(self, problem: str) -> List[Dict[str, str]]:
        """Break down a solution into clear steps."""
        # This would analyze the problem and create step-by-step breakdown
        return [
            {"step": 1, "action": "Identify what we're solving for", "explanation": "Understand the problem"},
            {"step": 2, "action": "Set up the equation or approach", "explanation": "Choose the method"},
            {"step": 3, "action": "Solve systematically", "explanation": "Apply mathematical operations"},
            {"step": 4, "action": "Check the answer", "explanation": "Verify the solution makes sense"}
        ]
    
    def _identify_solution_strategies(self, problem: str) -> List[str]:
        """Identify key strategies used in solving the problem."""
        return [
            "Break down complex problems into smaller parts",
            "Use systematic approach",
            "Check work at each step",
            "Verify final answer"
        ]
    
    def _suggest_alternative_methods(self, problem: str) -> List[str]:
        """Suggest alternative solution methods."""
        return [
            "Graphical approach",
            "Algebraic method",
            "Numerical estimation",
            "Visual representation"
        ]
    
    def _suggest_verification_methods(self, problem: str) -> List[str]:
        """Suggest ways to verify the solution."""
        return [
            "Substitute answer back into original problem",
            "Check if answer makes logical sense",
            "Use estimation to verify magnitude",
            "Try alternative solution method"
        ]
    
    def _classify_error_type(self, error_content: str) -> str:
        """Classify the type of mathematical error."""
        # This would analyze the error and classify it
        error_types = [
            "computational_error", "conceptual_misunderstanding", 
            "procedural_mistake", "notation_error"
        ]
        return "computational_error"  # Placeholder
    
    def _generate_correction_steps(self, error_content: str) -> List[str]:
        """Generate steps to correct the error."""
        return [
            "Identify where the error occurred",
            "Understand why it's incorrect",
            "Apply the correct method",
            "Practice similar problems"
        ]
    
    def _generate_prevention_tips(self, error_content: str) -> List[str]:
        """Generate tips to prevent similar errors."""
        return [
            "Double-check calculations",
            "Review the concept if unsure",
            "Practice more problems of this type",
            "Ask for help when confused"
        ]
    
    def _suggest_practice_problems(self, error_content: str) -> List[str]:
        """Suggest practice problems to reinforce correct understanding."""
        return [
            "Similar problems with step-by-step guidance",
            "Gradually increasing difficulty",
            "Mixed practice with related concepts"
        ]
    
    def _find_related_concepts(self, concept: str) -> List[str]:
        """Find concepts related to the given concept."""
        relations_map = {
            "fractions": ["decimals", "percentages", "ratios"],
            "algebra": ["equations", "variables", "functions"],
            "geometry": ["shapes", "area", "perimeter", "angles"]
        }
        
        return relations_map.get(concept, ["related mathematical concepts"])
    
    def _identify_applications(self, concept: str) -> List[str]:
        """Identify real-world applications of the concept."""
        applications_map = {
            "fractions": ["cooking recipes", "construction measurements", "financial calculations"],
            "algebra": ["business planning", "engineering design", "scientific research"],
            "geometry": ["architecture", "art and design", "navigation"]
        }
        
        return applications_map.get(concept, ["practical applications in daily life"])
    
    def _identify_prerequisites(self, concept: str) -> List[str]:
        """Identify prerequisite concepts."""
        prerequisites_map = {
            "fractions": ["basic arithmetic", "division"],
            "algebra": ["fractions", "basic arithmetic", "order of operations"],
            "geometry": ["basic shapes", "measurement"]
        }
        
        return prerequisites_map.get(concept, ["foundational mathematical concepts"])
    
    def _suggest_extensions(self, concept: str) -> List[str]:
        """Suggest advanced extensions of the concept."""
        extensions_map = {
            "fractions": ["complex fractions", "fraction operations", "rational expressions"],
            "algebra": ["systems of equations", "polynomials", "functions"],
            "geometry": ["trigonometry", "coordinate geometry", "3D geometry"]
        }
        
        return extensions_map.get(concept, ["advanced topics building on this concept"])
    
    def _extract_key_insights(self, content: str) -> List[str]:
        """Extract key insights from the content."""
        return [
            "Main mathematical principle involved",
            "Key relationships and patterns",
            "Important problem-solving strategies"
        ]
    
    def _assess_practical_relevance(self, content: str) -> str:
        """Assess the practical relevance of the content."""
        return "This concept has practical applications in various fields and daily life situations."
    
    def _generate_learning_tips(self, content: str) -> List[str]:
        """Generate learning tips for the content."""
        return [
            "Practice regularly with varied problems",
            "Connect to previously learned concepts",
            "Use visual aids when helpful",
            "Explain the concept to someone else"
        ]
    
    def _generate_explanation_metadata(self, explanation: Dict, explanation_type: str) -> Dict[str, Any]:
        """Generate metadata for the explanation."""
        return {
            "explanation_length": len(str(explanation)),
            "complexity_level": self._assess_complexity(explanation),
            "estimated_reading_time": self._estimate_reading_time(explanation),
            "key_concepts_covered": self._extract_concepts_covered(explanation),
            "explanation_quality": "high",  # Would assess based on various factors
            "personalization_level": "adapted"
        }
    
    def _generate_follow_up_suggestions(self, explanation_type: str, content: str) -> List[str]:
        """Generate follow-up suggestions based on the explanation."""
        suggestions = []
        
        if explanation_type == "concept":
            suggestions.extend([
                "Practice problems to reinforce understanding",
                "Explore related concepts",
                "Try real-world applications"
            ])
        elif explanation_type == "solution":
            suggestions.extend([
                "Try similar problems independently",
                "Practice the solution method",
                "Explore alternative approaches"
            ])
        elif explanation_type == "error_analysis":
            suggestions.extend([
                "Practice correct method multiple times",
                "Review related concepts",
                "Work on similar problems with guidance"
            ])
        
        return suggestions
    
    def _identify_related_concepts(self, content: str) -> List[str]:
        """Identify concepts related to the content."""
        # This would use concept mapping to find related topics
        return ["related_concept_1", "related_concept_2", "related_concept_3"]
    
    def _assess_explanation_difficulty(self, explanation: Dict) -> str:
        """Assess the difficulty level of the explanation."""
        # This would analyze the explanation complexity
        return "appropriate"  # Could be "easy", "appropriate", "challenging"
    
    def _assess_complexity(self, explanation: Dict) -> str:
        """Assess the complexity of the explanation."""
        # This would analyze various factors like vocabulary, concept depth, etc.
        return "moderate"
    
    def _estimate_reading_time(self, explanation: Dict) -> int:
        """Estimate reading time in minutes."""
        # Rough estimate based on content length
        total_text = str(explanation)
        words = len(total_text.split())
        return max(1, words // 200)  # Assuming 200 words per minute
    
    def _extract_concepts_covered(self, explanation: Dict) -> List[str]:
        """Extract the main concepts covered in the explanation."""
        # This would analyze the explanation content to identify key concepts
        return ["main_concept", "supporting_concept"] 