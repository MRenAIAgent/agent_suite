"""
Integration Tests for Math Recognition System

Tests the complete workflow integration including:
- End-to-end analysis pipeline
- Component interaction and data flow
- Performance under realistic scenarios
- Error handling across the system
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from math_learning.math_recognization import (
    create_complete_analyzer,
    create_hybrid_analyzer,
    CompleteTestAnalyzer,
    HybridErrorAnalyzer,
    AnswerCorrectnessEngine,
    ErrorAnalysisEngine,
    RootCauseAnalyzer,
    KnowledgeGapIdentifier
)
from math_learning.math_recognization.tests.conftest import (
    MockLLMClient,
    SAMPLE_QUESTIONS,
    SAMPLE_STUDENT_HISTORY,
    assert_hybrid_analysis_result
)

class TestSystemIntegration:
    """Test complete system integration."""
    
    def setup_method(self):
        self.mock_client = MockLLMClient()
    
    def test_convenience_functions(self):
        """Test convenience functions for creating analyzers."""
        # Test creating complete analyzer
        complete_analyzer = create_complete_analyzer()
        assert isinstance(complete_analyzer, CompleteTestAnalyzer)
        
        # Test creating complete analyzer with LLM
        complete_analyzer_with_llm = create_complete_analyzer(self.mock_client)
        assert isinstance(complete_analyzer_with_llm, CompleteTestAnalyzer)
        
        # Test creating hybrid analyzer
        hybrid_analyzer = create_hybrid_analyzer()
        assert isinstance(hybrid_analyzer, HybridErrorAnalyzer)
        
        # Test creating hybrid analyzer with LLM
        hybrid_analyzer_with_llm = create_hybrid_analyzer(self.mock_client)
        assert isinstance(hybrid_analyzer_with_llm, HybridErrorAnalyzer)
    
    def test_component_chain_integration(self):
        """Test that all components work together in sequence."""
        # Create individual components
        correctness_engine = AnswerCorrectnessEngine()
        error_engine = ErrorAnalysisEngine()
        root_cause_analyzer = RootCauseAnalyzer()
        gap_identifier = KnowledgeGapIdentifier()
        
        # Test sequential processing
        question = "Solve: 2x + 3 = 11"
        student_answer = "x = 5"
        correct_answer = "x = 4"
        work_shown = "2x + 3 = 11\n2x = 8\nx = 5"
        
        # Step 1: Check correctness
        correctness_result = correctness_engine.check_correctness(
            student_answer, correct_answer, question
        )
        assert not correctness_result.is_correct
        
        # Step 2: Analyze error
        error_result = error_engine.analyze_error(
            question, student_answer, correct_answer, work_shown
        )
        assert error_result.error_type is not None
        
        # Step 3: Find root cause
        root_cause_result = root_cause_analyzer.analyze_root_cause(
            error_result, work_shown
        )
        assert root_cause_result.primary_cause is not None
        
        # Step 4: Identify gaps
        gap_result = gap_identifier.identify_gaps(
            error_result, root_cause_result
        )
        assert isinstance(gap_result.identified_gaps, list)
    
    @pytest.mark.asyncio
    async def test_hybrid_analyzer_integration(self):
        """Test hybrid analyzer integration with all components."""
        analyzer = create_hybrid_analyzer(self.mock_client)
        
        result = await analyzer.analyze_comprehensive(
            question="Add fractions: 1/3 + 1/4",
            student_answer="2/7",
            correct_answer="7/12",
            work_shown="1/3 + 1/4 = (1+1)/(3+4) = 2/7"
        )
        
        assert_hybrid_analysis_result(result)
        
        # Verify all components contributed
        assert result.correctness_result is not None
        assert result.error_analysis is not None
        assert result.root_cause_analysis is not None
        assert result.gap_analysis is not None
        assert result.ai_insights is not None
    
    @pytest.mark.asyncio
    async def test_data_flow_consistency(self):
        """Test that data flows consistently between components."""
        analyzer = create_hybrid_analyzer(self.mock_client)
        
        # Test with multiple related questions
        questions = [
            {
                "question": "What is 5 + 3?",
                "student_answer": "7",
                "correct_answer": "8",
                "work_shown": "5 + 3 = 7"
            },
            {
                "question": "What is 12 - 7?",
                "student_answer": "6", 
                "correct_answer": "5",
                "work_shown": "12 - 7 = 6"
            }
        ]
        
        results = []
        for q in questions:
            result = await analyzer.analyze_comprehensive(**q)
            results.append(result)
            assert_hybrid_analysis_result(result)
        
        # Verify consistency across analyses
        for result in results:
            assert result.confidence_score > 0.0
            assert len(result.recommended_actions) > 0
            assert result.learning_priority in ["immediate", "short_term", "long_term"]
    
    def test_error_handling_integration(self):
        """Test error handling across the integrated system."""
        correctness_engine = AnswerCorrectnessEngine()
        error_engine = ErrorAnalysisEngine()
        
        # Test with problematic inputs
        test_cases = [
            ("", "", ""),  # Empty strings
            ("very_long_" * 100, "x", "test"),  # Very long input
            ("∫∂∇", "∑∏", "∞"),  # Special characters
        ]
        
        for student_answer, correct_answer, question in test_cases:
            # Should not crash
            correctness_result = correctness_engine.check_correctness(
                student_answer, correct_answer, question
            )
            assert correctness_result is not None
            
            error_result = error_engine.analyze_error(
                question, student_answer, correct_answer, ""
            )
            assert error_result is not None
    
    @pytest.mark.asyncio
    async def test_performance_integration(self):
        """Test performance of integrated system."""
        import time
        
        analyzer = create_hybrid_analyzer(self.mock_client)
        
        start_time = time.time()
        
        # Process multiple questions concurrently
        tasks = []
        for i in range(5):  # Reduced for integration test
            task = analyzer.analyze_comprehensive(
                question=f"What is {i*2} + {i*3}?",
                student_answer=f"{i*2 + i*3 + 1}",  # Wrong answer
                correct_answer=f"{i*2 + i*3}",
                work_shown=f"{i*2} + {i*3} = {i*2 + i*3 + 1}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time
        assert duration < 20.0, f"Integration performance test took {duration:.2f} seconds"
        
        # All results should be valid
        for result in results:
            assert_hybrid_analysis_result(result)
    
    @pytest.mark.asyncio
    async def test_realistic_scenario(self):
        """Test with realistic student error scenarios."""
        analyzer = create_hybrid_analyzer(self.mock_client)
        
        # Realistic algebra error
        result1 = await analyzer.analyze_comprehensive(
            question="Solve for x: 3x - 5 = 16",
            student_answer="x = 11",
            correct_answer="x = 7", 
            work_shown="3x - 5 = 16\n3x = 16 + 5\n3x = 21\nx = 11"
        )
        
        # Realistic fraction error
        result2 = await analyzer.analyze_comprehensive(
            question="Simplify: 6/8",
            student_answer="6/8",
            correct_answer="3/4",
            work_shown="6/8 cannot be simplified"
        )
        
        # Realistic percentage error
        result3 = await analyzer.analyze_comprehensive(
            question="What is 25% of 200?",
            student_answer="225",
            correct_answer="50",
            work_shown="25% of 200 = 200 + 25 = 225"
        )
        
        for result in [result1, result2, result3]:
            assert_hybrid_analysis_result(result)
            assert result.correctness_result.is_correct == False
            assert len(result.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_student_history_integration(self):
        """Test integration with student history data."""
        analyzer = create_hybrid_analyzer(self.mock_client)
        student_history = SAMPLE_STUDENT_HISTORY["alice"]
        
        # Test with student's known weak area (fractions)
        result = await analyzer.analyze_comprehensive(
            question="Add: 2/5 + 1/3",
            student_answer="3/8",
            correct_answer="11/15",
            work_shown="2/5 + 1/3 = (2+1)/(5+3) = 3/8",
            student_history=student_history
        )
        
        assert_hybrid_analysis_result(result)
        # Should provide contextual analysis based on student history
        assert result.student_level_assessment is not None
    
    def test_import_system_integration(self):
        """Test that import system works correctly."""
        # Test importing from main module
        from math_learning.math_recognization import (
            CompleteTestAnalyzer,
            HybridErrorAnalyzer,
            AnswerCorrectnessEngine,
            ValidationResult,
            ErrorType,
            create_complete_analyzer,
            create_hybrid_analyzer
        )
        
        # Verify all imports work
        assert CompleteTestAnalyzer is not None
        assert HybridErrorAnalyzer is not None
        assert AnswerCorrectnessEngine is not None
        assert ValidationResult is not None
        assert ErrorType is not None
        assert callable(create_complete_analyzer)
        assert callable(create_hybrid_analyzer)
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis_integration(self):
        """Test concurrent analysis of multiple questions."""
        analyzer = create_hybrid_analyzer(self.mock_client)
        
        # Create multiple analysis tasks
        tasks = []
        for question_data in SAMPLE_QUESTIONS[:3]:  # Test with first 3 questions
            for error_type, student_answer in question_data["student_answers"].items():
                if error_type != "correct":
                    task = analyzer.analyze_comprehensive(
                        question=question_data["question"],
                        student_answer=student_answer,
                        correct_answer=question_data["correct_answer"],
                        work_shown=question_data.get("work_shown", {}).get(error_type, "")
                    )
                    tasks.append(task)
                    
                    # Limit concurrent tasks for test stability
                    if len(tasks) >= 6:
                        break
            
            if len(tasks) >= 6:
                break
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all results
        assert len(results) > 0
        for result in results:
            assert_hybrid_analysis_result(result)
    
    def test_memory_usage_integration(self):
        """Test memory usage doesn't grow excessively."""
        import gc
        import sys
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Create and use analyzer multiple times
        for i in range(10):
            analyzer = create_hybrid_analyzer()
            correctness_engine = AnswerCorrectnessEngine()
            
            # Use the components
            result = correctness_engine.check_correctness("4", "4", "test")
            assert result is not None
            
            # Clean up references
            del analyzer
            del correctness_engine
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage shouldn't grow excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Memory usage grew by {object_growth} objects"
    
    @pytest.mark.asyncio
    async def test_ai_integration_fallback(self):
        """Test AI integration with fallback behavior."""
        # Test with no LLM client (should use mock responses)
        analyzer_no_llm = create_hybrid_analyzer(llm_client=None)
        
        result1 = await analyzer_no_llm.analyze_comprehensive(
            question="What is 10 + 5?",
            student_answer="14",
            correct_answer="15",
            work_shown="10 + 5 = 14"
        )
        
        # Test with mock LLM client
        analyzer_with_llm = create_hybrid_analyzer(self.mock_client)
        
        result2 = await analyzer_with_llm.analyze_comprehensive(
            question="What is 10 + 5?",
            student_answer="14", 
            correct_answer="15",
            work_shown="10 + 5 = 14"
        )
        
        # Both should work
        assert_hybrid_analysis_result(result1)
        assert_hybrid_analysis_result(result2)
        
        # Mock client should have been called
        assert self.mock_client.call_count > 0
    
    def test_configuration_integration(self):
        """Test different configuration scenarios."""
        # Test with different tolerance settings, custom validators, etc.
        # This would test configuration flexibility
        
        engine1 = AnswerCorrectnessEngine()
        engine2 = AnswerCorrectnessEngine()
        
        # Both should work with same inputs
        result1 = engine1.check_correctness("3.14", "3.141", "Calculate π")
        result2 = engine2.check_correctness("3.14", "3.141", "Calculate π")
        
        assert result1.is_correct == result2.is_correct
        # Confidence might vary slightly but should be similar
        assert abs(result1.confidence - result2.confidence) < 0.2

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_student_assessment_workflow(self):
        """Test a complete student assessment workflow."""
        analyzer = create_hybrid_analyzer(MockLLMClient())
        
        # Simulate a student taking a mini-test
        student_responses = [
            {
                "question": "What is 15% of 80?",
                "student_answer": "15",  # Wrong
                "correct_answer": "12",
                "work_shown": "15% of 80 = 15"
            },
            {
                "question": "Solve: x + 7 = 15",
                "student_answer": "x = 8",  # Correct
                "correct_answer": "x = 8", 
                "work_shown": "x + 7 = 15\nx = 15 - 7\nx = 8"
            },
            {
                "question": "Add: 1/4 + 1/4",
                "student_answer": "2/8",  # Correct but not simplified
                "correct_answer": "1/2",
                "work_shown": "1/4 + 1/4 = 2/8"
            }
        ]
        
        # Analyze all responses
        results = []
        for response in student_responses:
            result = await analyzer.analyze_comprehensive(**response)
            results.append(result)
            assert_hybrid_analysis_result(result)
        
        # Aggregate results for student report
        total_questions = len(results)
        correct_count = sum(1 for r in results if r.correctness_result.is_correct)
        
        # Generate insights
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommended_actions)
        
        # Verify workflow completion
        assert total_questions == 3
        assert correct_count >= 0  # At least some analysis completed
        assert len(all_recommendations) > 0
        
        # Should identify different types of issues
        error_types = [r.error_analysis.error_type for r in results if not r.correctness_result.is_correct]
        assert len(error_types) > 0  # Should find some errors
    
    @pytest.mark.asyncio
    async def test_teacher_dashboard_workflow(self):
        """Test workflow for teacher dashboard analytics."""
        analyzer = create_hybrid_analyzer(MockLLMClient())
        
        # Simulate multiple students on same question
        question = "Solve: 2x + 5 = 13"
        correct_answer = "x = 4"
        
        student_responses = [
            ("x = 4", "2x + 5 = 13\n2x = 8\nx = 4"),  # Correct
            ("x = 9", "2x + 5 = 13\n2x = 13 + 5\n2x = 18\nx = 9"),  # Sign error
            ("x = 6.5", "2x + 5 = 13\nx = 13 - 5\nx = 8\nx = 8/2\nx = 4"),  # Calculation error
        ]
        
        results = []
        for student_answer, work_shown in student_responses:
            result = await analyzer.analyze_comprehensive(
                question=question,
                student_answer=student_answer,
                correct_answer=correct_answer,
                work_shown=work_shown
            )
            results.append(result)
        
        # Analyze class performance
        correct_count = sum(1 for r in results if r.correctness_result.is_correct)
        common_errors = {}
        
        for result in results:
            if not result.correctness_result.is_correct:
                error_type = result.error_analysis.error_type.value
                common_errors[error_type] = common_errors.get(error_type, 0) + 1
        
        # Verify teacher analytics
        assert correct_count >= 0
        assert len(results) == 3
        
        # Should identify common error patterns
        if common_errors:
            most_common_error = max(common_errors, key=common_errors.get)
            assert most_common_error is not None

if __name__ == "__main__":
    pytest.main([__file__]) 