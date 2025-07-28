"""
Tests for HybridErrorAnalyzer

Tests the hybrid AI + rule-based error analysis system including:
- Integration of rule-based and AI analysis
- Analysis synthesis and confidence scoring
- AI error analysis with mock LLM
- Comprehensive error diagnosis
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from math_learning.math_recognization.hybrid_error_analyzer import (
    HybridErrorAnalyzer,
    HybridAnalysisResult, 
    AIErrorAnalyzer,
    AnalysisSynthesizer
)
from math_learning.math_recognization.tests.conftest import (
    assert_hybrid_analysis_result,
    MockLLMClient,
    SAMPLE_QUESTIONS,
    SAMPLE_STUDENT_HISTORY
)

class TestAIErrorAnalyzer:
    """Test the AI error analyzer component."""
    
    def setup_method(self):
        self.mock_client = MockLLMClient()
        self.analyzer = AIErrorAnalyzer(self.mock_client)
    
    @pytest.mark.asyncio
    async def test_ai_analysis_with_mock_client(self):
        """Test AI analysis with mock LLM client."""
        result = await self.analyzer.analyze_error_with_ai(
            question="Solve: 2x + 3 = 11",
            student_answer="x = 5",
            correct_answer="x = 4",
            work_shown="2x + 3 = 11\n2x = 8\nx = 5"
        )
        
        assert isinstance(result, dict)
        assert self.mock_client.call_count == 1
        assert len(self.mock_client.last_prompt) > 0
    
    @pytest.mark.asyncio
    async def test_ai_analysis_without_client(self):
        """Test AI analysis without LLM client (mock mode)."""
        analyzer = AIErrorAnalyzer(llm_client=None)
        
        result = await analyzer.analyze_error_with_ai(
            question="What is 5 + 3?",
            student_answer="7",
            correct_answer="8",
            work_shown="5 + 3 = 7"
        )
        
        assert isinstance(result, dict)
        assert "error_classification" in str(result).lower() or "mock" in str(result).lower()
    
    def test_prompt_creation(self):
        """Test that comprehensive prompts are created."""
        prompt = self.analyzer._create_analysis_prompt(
            question="Solve: x + 5 = 12",
            student_answer="x = 8", 
            correct_answer="x = 7",
            work_shown="x + 5 = 12\nx = 12 + 5\nx = 17"
        )
        
        assert len(prompt) > 100  # Should be comprehensive
        assert "PROBLEM:" in prompt
        assert "STUDENT ANSWER:" in prompt
        assert "CORRECT ANSWER:" in prompt
        assert "ERROR TYPE CLASSIFICATION:" in prompt
        assert "ROOT CAUSE ANALYSIS:" in prompt
        assert "REMEDIATION RECOMMENDATIONS:" in prompt
    
    def test_ai_response_parsing(self):
        """Test parsing of AI responses."""
        mock_response = """
        1. ERROR TYPE CLASSIFICATION:
           - Primary error type: Procedural
           - Confidence: 0.9
        
        2. ROOT CAUSE ANALYSIS:
           - Most likely root cause: Incorrect sign handling
        """
        
        parsed = self.analyzer._parse_ai_response(mock_response)
        
        assert isinstance(parsed, dict)
        assert "error_classification" in parsed or "root_cause" in parsed

class TestAnalysisSynthesizer:
    """Test the analysis synthesis component."""
    
    def setup_method(self):
        self.synthesizer = AnalysisSynthesizer()
    
    def test_synthesis_integration(self):
        """Test synthesis of multiple analysis results."""
        # This would require creating mock analysis results
        # For now, test that the synthesizer exists and has methods
        assert hasattr(self.synthesizer, 'synthesize_analyses')
        assert callable(self.synthesizer.synthesize_analyses)

class TestHybridErrorAnalyzer:
    """Test the main HybridErrorAnalyzer."""
    
    def setup_method(self):
        self.mock_client = MockLLMClient()
        self.analyzer = HybridErrorAnalyzer(self.mock_client)
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_correct_answer(self):
        """Test comprehensive analysis of correct answer."""
        result = await self.analyzer.analyze_comprehensive(
            question="What is 5 + 3?",
            student_answer="8",
            correct_answer="8",
            work_shown="5 + 3 = 8"
        )
        
        assert isinstance(result, HybridAnalysisResult)
        assert result.correctness_result.is_correct == True
        # For correct answers, analysis should be minimal
        assert result.primary_diagnosis is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis_incorrect_answer(self):
        """Test comprehensive analysis of incorrect answer."""
        result = await self.analyzer.analyze_comprehensive(
            question="Solve: 2x + 3 = 11",
            student_answer="x = 5",
            correct_answer="x = 4",
            work_shown="2x + 3 = 11\n2x = 8\nx = 5"
        )
        
        assert_hybrid_analysis_result(result)
        assert result.correctness_result.is_correct == False
        assert len(result.recommended_actions) > 0
        assert result.confidence_score > 0.0
        assert result.primary_diagnosis is not None
    
    @pytest.mark.asyncio
    async def test_arithmetic_error_analysis(self):
        """Test analysis of arithmetic errors."""
        result = await self.analyzer.analyze_comprehensive(
            question="What is 15 + 23?",
            student_answer="37",
            correct_answer="38",
            work_shown="15 + 23 = 37"
        )
        
        assert_hybrid_analysis_result(result)
        assert "arithmetic" in result.primary_diagnosis.lower() or "calculation" in result.primary_diagnosis.lower()
        assert result.learning_priority in ["immediate", "short_term", "long_term"]
    
    @pytest.mark.asyncio
    async def test_conceptual_error_analysis(self):
        """Test analysis of conceptual errors."""
        result = await self.analyzer.analyze_comprehensive(
            question="Add fractions: 1/3 + 1/4",
            student_answer="2/7",
            correct_answer="7/12",
            work_shown="1/3 + 1/4 = (1+1)/(3+4) = 2/7"
        )
        
        assert_hybrid_analysis_result(result)
        # Should identify this as a conceptual error
        assert len(result.recommended_actions) > 0
        assert result.confidence_score > 0.0
    
    @pytest.mark.asyncio
    async def test_procedural_error_analysis(self):
        """Test analysis of procedural errors."""
        result = await self.analyzer.analyze_comprehensive(
            question="Solve: 2x + 3 = 11",
            student_answer="x = 7", 
            correct_answer="x = 4",
            work_shown="2x + 3 = 11\n2x = 11 + 3\n2x = 14\nx = 7"
        )
        
        assert_hybrid_analysis_result(result)
        # Should identify procedural error in sign handling
        assert len(result.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_with_student_history(self):
        """Test analysis with student history context."""
        student_history = SAMPLE_STUDENT_HISTORY["alice"]
        
        result = await self.analyzer.analyze_comprehensive(
            question="Add fractions: 1/2 + 1/3",
            student_answer="2/5",
            correct_answer="5/6",
            work_shown="1/2 + 1/3 = (1+1)/(2+3) = 2/5",
            student_history=student_history
        )
        
        assert_hybrid_analysis_result(result)
        # Should consider student's known weakness with fractions
        assert result.student_level_assessment is not None
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence scoring across different scenarios."""
        # High confidence scenario - clear arithmetic error
        result1 = await self.analyzer.analyze_comprehensive(
            question="What is 10 + 5?",
            student_answer="14",
            correct_answer="15",
            work_shown="10 + 5 = 14"
        )
        
        # Lower confidence scenario - ambiguous error
        result2 = await self.analyzer.analyze_comprehensive(
            question="Solve complex equation",
            student_answer="x = 3",
            correct_answer="x = 4",
            work_shown=""  # No work shown
        )
        
        assert result1.confidence_score > 0.0
        assert result2.confidence_score > 0.0
        # First should generally have higher confidence
        # (though this depends on implementation details)
    
    @pytest.mark.asyncio
    async def test_learning_priority_assignment(self):
        """Test learning priority assignment."""
        # Minor error - should be immediate or short-term
        result1 = await self.analyzer.analyze_comprehensive(
            question="What is 7 + 8?",
            student_answer="14",
            correct_answer="15",
            work_shown="7 + 8 = 14"
        )
        
        # Major conceptual error - should be long-term
        result2 = await self.analyzer.analyze_comprehensive(
            question="Add fractions: 1/2 + 1/3",
            student_answer="2/5",
            correct_answer="5/6",
            work_shown="1/2 + 1/3 = (1+1)/(2+3) = 2/5"
        )
        
        assert result1.learning_priority in ["immediate", "short_term", "long_term"]
        assert result2.learning_priority in ["immediate", "short_term", "long_term"]
    
    @pytest.mark.asyncio
    async def test_ai_insights_integration(self):
        """Test that AI insights are properly integrated."""
        result = await self.analyzer.analyze_comprehensive(
            question="What is 12 × 7?",
            student_answer="74",
            correct_answer="84",
            work_shown="12 × 7 = 74"
        )
        
        assert_hybrid_analysis_result(result)
        assert result.ai_insights is not None
        assert isinstance(result.ai_insights, dict)
        assert result.ai_confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_metacognitive_insights(self):
        """Test extraction of metacognitive insights."""
        result = await self.analyzer.analyze_comprehensive(
            question="Solve: 3x - 2 = 10",
            student_answer="x = 6",
            correct_answer="x = 4",
            work_shown="3x - 2 = 10\n3x = 8\nx = 6"
        )
        
        assert_hybrid_analysis_result(result)
        assert isinstance(result.metacognitive_insights, list)
        # May or may not have insights depending on AI response
    
    @pytest.mark.asyncio
    async def test_problem_difficulty_assessment(self):
        """Test problem difficulty assessment."""
        # Easy problem for grade level
        result1 = await self.analyzer.analyze_comprehensive(
            question="What is 2 + 3?",
            student_answer="4",
            correct_answer="5",
            work_shown="2 + 3 = 4"
        )
        
        # Complex problem
        result2 = await self.analyzer.analyze_comprehensive(
            question="Solve the system: 2x + 3y = 7, x - y = 1",
            student_answer="x = 2, y = 1",
            correct_answer="x = 2, y = 1",
            work_shown=""
        )
        
        assert result1.problem_difficulty_match is not None
        assert result2.problem_difficulty_match is not None
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("question_data", SAMPLE_QUESTIONS[:2])  # Limit for async tests
    async def test_sample_questions_hybrid_analysis(self, question_data):
        """Test hybrid analysis with sample questions."""
        question = question_data["question"]
        correct_answer = question_data["correct_answer"]
        
        # Test one error type per question to keep test time reasonable
        error_type = "arithmetic_error"
        if error_type in question_data["student_answers"]:
            student_answer = question_data["student_answers"][error_type]
            work_shown = question_data.get("work_shown", {}).get(error_type, "")
            
            result = await self.analyzer.analyze_comprehensive(
                question=question,
                student_answer=student_answer,
                correct_answer=correct_answer,
                work_shown=work_shown
            )
            
            assert_hybrid_analysis_result(result)
    
    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty inputs
        result = await self.analyzer.analyze_comprehensive("", "", "", "")
        assert isinstance(result, HybridAnalysisResult)
        
        # None inputs (should handle gracefully)
        try:
            result = await self.analyzer.analyze_comprehensive(None, None, None, None)
            assert isinstance(result, HybridAnalysisResult)
        except Exception:
            # It's acceptable to raise an exception for None inputs
            pass
    
    @pytest.mark.asyncio
    async def test_performance(self):
        """Test performance with multiple analyses."""
        import time
        
        start_time = time.time()
        
        # Run 10 analyses (fewer due to async overhead)
        tasks = []
        for i in range(10):
            task = self.analyzer.analyze_comprehensive(
                f"What is {i} + 1?",
                f"{i + 2}",  # Wrong answer
                f"{i + 1}",  # Correct answer
                f"{i} + 1 = {i + 2}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 15.0, f"Performance test took {duration:.2f} seconds"
        assert len(results) == 10
        
        for result in results:
            assert_hybrid_analysis_result(result)
    
    def test_component_integration(self):
        """Test that all components are properly integrated."""
        analyzer = HybridErrorAnalyzer()
        
        # Should have all required components
        assert hasattr(analyzer, 'correctness_engine')
        assert hasattr(analyzer, 'error_analyzer')
        assert hasattr(analyzer, 'root_cause_analyzer')
        assert hasattr(analyzer, 'gap_identifier')
        assert hasattr(analyzer, 'ai_analyzer')
        assert hasattr(analyzer, 'synthesizer')
    
    @pytest.mark.asyncio
    async def test_analysis_without_work_shown(self):
        """Test analysis when no work is shown."""
        result = await self.analyzer.analyze_comprehensive(
            question="What is 25% of 80?",
            student_answer="15",
            correct_answer="20",
            work_shown=""  # No work shown
        )
        
        assert_hybrid_analysis_result(result)
        # Should still provide analysis even without work shown
        assert len(result.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_complex_mathematical_expressions(self):
        """Test with complex mathematical expressions."""
        result = await self.analyzer.analyze_comprehensive(
            question="Simplify: (x² - 4)/(x - 2)",
            student_answer="x - 2",
            correct_answer="x + 2",
            work_shown="(x² - 4)/(x - 2) = (x - 2)(x + 2)/(x - 2) = x - 2"
        )
        
        assert_hybrid_analysis_result(result)
        # Should handle complex algebraic expressions
    
    def test_analyzer_initialization_variants(self):
        """Test different ways to initialize the analyzer."""
        # Without LLM client
        analyzer1 = HybridErrorAnalyzer()
        assert analyzer1.ai_analyzer.llm_client is None
        
        # With mock LLM client
        mock_client = MockLLMClient()
        analyzer2 = HybridErrorAnalyzer(mock_client)
        assert analyzer2.ai_analyzer.llm_client is mock_client

if __name__ == "__main__":
    pytest.main([__file__]) 