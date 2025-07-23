"""
Test suite for Enhanced Image Recognition Agent

Tests the agent's ability to recognize images and identify questions/user answers
from various image sources including handwritten work, printed exercises, and digital content.
"""

import pytest
import asyncio
import base64
import io
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

# Import the agent
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from agents.enhanced_image_recognition_agent import (
    EnhancedImageRecognitionAgent,
    RecognizedContent,
    ImageAnalysisResult,
    recognize_math_image
)


class TestEnhancedImageRecognitionAgent:
    """Test cases for the Enhanced Image Recognition Agent."""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        llm.generate_response = AsyncMock(return_value="Mock LLM response")
        return llm
    
    @pytest.fixture
    def agent(self, mock_llm):
        """Create an agent instance for testing."""
        return EnhancedImageRecognitionAgent(llm=mock_llm)
    
    @pytest.fixture
    def sample_base64_image(self):
        """Create a sample base64 encoded image for testing."""
        # Create a simple test image
        try:
            from PIL import Image
            import io
            
            # Create a simple white image with text
            img = Image.new('RGB', (400, 200), color='white')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            # Return a minimal base64 string if PIL not available
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    def test_agent_initialization(self, mock_llm):
        """Test that the agent initializes correctly."""
        agent = EnhancedImageRecognitionAgent(llm=mock_llm)
        
        assert agent.name == "enhanced_image_recognition"
        assert agent.description == "Advanced image recognition for math exercises and student work"
        assert agent.llm == mock_llm
        assert hasattr(agent, 'capabilities')
        assert hasattr(agent, 'ocr_config')
        assert hasattr(agent, 'question_patterns')
        assert hasattr(agent, 'answer_patterns')
        assert hasattr(agent, 'math_expression_patterns')
    
    def test_dependency_checking(self, agent):
        """Test that dependency checking works correctly."""
        assert isinstance(agent.capabilities, dict)
        assert 'pil_available' in agent.capabilities
        assert 'llm_available' in agent.capabilities
        assert agent.capabilities['llm_available'] is True
    
    @pytest.mark.asyncio
    async def test_recognize_image_success(self, agent, sample_base64_image):
        """Test successful image recognition."""
        with patch.object(agent, '_load_and_preprocess_image', return_value=Mock()):
            with patch.object(agent, '_extract_text_multi_method', return_value="Question 1: What is 2+2? Answer: 4"):
                result = await agent.recognize_image(sample_base64_image, "base64")
                
                assert isinstance(result, ImageAnalysisResult)
                assert result.success is True
                assert result.recognized_content is not None
                assert result.error_message is None
                assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_recognize_image_failure(self, agent):
        """Test image recognition failure handling."""
        with patch.object(agent, '_load_and_preprocess_image', side_effect=Exception("Test error")):
            result = await agent.recognize_image("invalid_data", "base64")
            
            assert isinstance(result, ImageAnalysisResult)
            assert result.success is False
            assert result.recognized_content is None
            assert result.error_message is not None
            assert "Image recognition failed" in result.error_message
    
    def test_extract_questions(self, agent):
        """Test question extraction from text."""
        test_text = """
        Question 1: What is the value of x in 2x + 5 = 15?
        Problem 2: Solve for y: 3y - 7 = 14
        Find the area of a rectangle with length 8 and width 5.
        """
        
        questions = agent._extract_questions(test_text)
        assert len(questions) >= 2
        assert any("2x + 5 = 15" in q for q in questions)
    
    def test_extract_answers(self, agent):
        """Test answer extraction from text."""
        test_text = """
        Question: What is 2 + 2?
        Student answer: 4
        Correct answer: 4
        Your work: 2 + 2 = 4
        """
        
        user_answers, correct_answers = agent._extract_answers(test_text)
        assert len(user_answers) > 0 or len(correct_answers) > 0
    
    def test_extract_mathematical_expressions(self, agent):
        """Test mathematical expression extraction."""
        test_text = """
        x = 5
        2 + 3 = 5
        y = 2x + 3
        (a + b) * (c - d)
        3/4 + 1/2
        """
        
        expressions = agent._extract_mathematical_expressions(test_text)
        assert len(expressions) > 0
        assert any("=" in expr for expr in expressions)
    
    def test_calculate_confidence_score(self, agent):
        """Test confidence score calculation."""
        text = "Question: What is 2+2? Answer: 4"
        questions = ["What is 2+2?"]
        answers = ["4"]
        expressions = ["2+2", "= 4"]
        
        score = agent._calculate_confidence_score(text, questions, answers, expressions)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have reasonable confidence with clear content
    
    @pytest.mark.asyncio
    async def test_parse_and_identify_content(self, agent):
        """Test content parsing and identification."""
        test_text = """
        Question 1: What is 5 + 3?
        Student answer: 8
        Question 2: Solve x + 2 = 7
        Your answer: x = 5
        """
        
        content = await agent._parse_and_identify_content(test_text, {})
        
        assert isinstance(content, RecognizedContent)
        assert len(content.questions) > 0
        assert len(content.user_answers) > 0
        assert content.text_content == test_text
        assert 0.0 <= content.confidence_score <= 1.0
        assert content.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self, agent):
        """Test recommendation generation."""
        content = RecognizedContent(
            questions=["What is 2+2?"],
            user_answers=["4"],
            correct_answers=["4"],
            mathematical_expressions=["2+2=4"],
            text_content="Question: What is 2+2? Answer: 4",
            confidence_score=0.9,
            processing_method="ocr",
            timestamp="2024-01-01T00:00:00"
        )
        
        recommendations = await agent._generate_recommendations(content, {})
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_execute_method(self, agent, sample_base64_image):
        """Test the execute method."""
        with patch.object(agent, 'recognize_image') as mock_recognize:
            mock_result = ImageAnalysisResult(
                success=True,
                recognized_content=RecognizedContent(
                    questions=["Test question"],
                    user_answers=["Test answer"],
                    correct_answers=["Test correct"],
                    mathematical_expressions=["x=1"],
                    text_content="Test content",
                    confidence_score=0.8,
                    processing_method="test",
                    timestamp="2024-01-01T00:00:00"
                ),
                error_message=None,
                processing_details={},
                recommendations=["Test recommendation"]
            )
            mock_recognize.return_value = mock_result
            
            result = await agent.execute(
                image_data=sample_base64_image,
                image_format="base64"
            )
            
            assert isinstance(result, dict)
            assert result['success'] is True
            assert 'recognized_content' in result
    
    def test_recognized_content_to_dict(self):
        """Test RecognizedContent serialization."""
        content = RecognizedContent(
            questions=["Q1"],
            user_answers=["A1"],
            correct_answers=["CA1"],
            mathematical_expressions=["x=1"],
            text_content="Test",
            confidence_score=0.8,
            processing_method="test",
            timestamp="2024-01-01T00:00:00"
        )
        
        result = content.to_dict()
        assert isinstance(result, dict)
        assert result['questions'] == ["Q1"]
        assert result['confidence_score'] == 0.8
    
    def test_image_analysis_result_to_dict(self):
        """Test ImageAnalysisResult serialization."""
        content = RecognizedContent(
            questions=["Q1"],
            user_answers=["A1"],
            correct_answers=["CA1"],
            mathematical_expressions=["x=1"],
            text_content="Test",
            confidence_score=0.8,
            processing_method="test",
            timestamp="2024-01-01T00:00:00"
        )
        
        result = ImageAnalysisResult(
            success=True,
            recognized_content=content,
            error_message=None,
            processing_details={"test": "value"},
            recommendations=["Test rec"]
        )
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert 'recognized_content' in result_dict


class TestStandaloneFunction:
    """Test the standalone recognize_math_image function."""
    
    @pytest.mark.asyncio
    async def test_recognize_math_image_function(self):
        """Test the standalone function."""
        with patch('agents.enhanced_image_recognition_agent.EnhancedImageRecognitionAgent') as MockAgent:
            mock_agent = Mock()
            mock_result = ImageAnalysisResult(
                success=True,
                recognized_content=None,
                error_message=None,
                processing_details={},
                recommendations=[]
            )
            mock_agent.recognize_image = AsyncMock(return_value=mock_result)
            MockAgent.return_value = mock_agent
            
            result = await recognize_math_image("test_data", "base64")
            
            assert isinstance(result, ImageAnalysisResult)
            assert result.success is True


class TestIntegrationScenarios:
    """Integration test scenarios for real-world usage."""
    
    @pytest.fixture
    def agent_with_mock_llm(self):
        """Create agent with mock LLM for integration tests."""
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="""
        Based on the image, I can identify:
        Questions: What is 2+2?
        Student Answer: 4
        Mathematical expressions: 2+2=4
        """)
        return EnhancedImageRecognitionAgent(llm=mock_llm)
    
    @pytest.mark.asyncio
    async def test_math_homework_scenario(self, agent_with_mock_llm):
        """Test scenario: analyzing a math homework image."""
        # Simulate a homework image with questions and answers
        homework_text = """
        Math Homework - Chapter 5
        
        1. What is 15 + 27?
        Student answer: 42
        
        2. Solve for x: 3x - 9 = 15
        Your work: 3x = 24, x = 8
        
        3. Find the area of a circle with radius 5
        Answer: A = πr² = π(5)² = 25π
        """
        
        with patch.object(agent_with_mock_llm, '_extract_text_multi_method', return_value=homework_text):
            with patch.object(agent_with_mock_llm, '_load_and_preprocess_image', return_value=Mock()):
                result = await agent_with_mock_llm.recognize_image("mock_image", "base64")
                
                assert result.success is True
                assert result.recognized_content is not None
                assert len(result.recognized_content.questions) > 0
                assert len(result.recognized_content.user_answers) > 0
                assert result.recognized_content.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_handwritten_work_scenario(self, agent_with_mock_llm):
        """Test scenario: analyzing handwritten student work."""
        handwritten_text = """
        Problem: 2x + 5 = 13
        
        Step 1: 2x = 13 - 5
        Step 2: 2x = 8
        Step 3: x = 4
        
        Check: 2(4) + 5 = 8 + 5 = 13 ✓
        """
        
        with patch.object(agent_with_mock_llm, '_extract_text_multi_method', return_value=handwritten_text):
            with patch.object(agent_with_mock_llm, '_load_and_preprocess_image', return_value=Mock()):
                result = await agent_with_mock_llm.recognize_image(
                    "mock_image", 
                    "base64",
                    processing_options={'preprocessing_type': 'handwriting'}
                )
                
                assert result.success is True
                assert result.recognized_content is not None
                assert len(result.recognized_content.mathematical_expressions) > 0
    
    @pytest.mark.asyncio
    async def test_quiz_scenario(self, agent_with_mock_llm):
        """Test scenario: analyzing a quiz with multiple choice questions."""
        quiz_text = """
        Quiz: Algebra Basics
        
        1. What is the value of x in 2x + 3 = 11?
        a) 3  b) 4  c) 5  d) 6
        Student answer: b) 4
        
        2. Simplify: 3(x + 2) - 2x
        Student work: 3x + 6 - 2x = x + 6
        
        3. Solve: y/3 + 4 = 7
        Your answer: y = 9
        """
        
        with patch.object(agent_with_mock_llm, '_extract_text_multi_method', return_value=quiz_text):
            with patch.object(agent_with_mock_llm, '_load_and_preprocess_image', return_value=Mock()):
                result = await agent_with_mock_llm.recognize_image("mock_image", "base64")
                
                assert result.success is True
                assert result.recognized_content is not None
                assert len(result.recognized_content.questions) >= 3
                assert len(result.recognized_content.user_answers) >= 3


if __name__ == "__main__":
    # Run a simple test to verify the agent works
    async def simple_test():
        print("🧪 Testing Enhanced Image Recognition Agent...")
        
        # Create a mock LLM
        mock_llm = Mock()
        mock_llm.generate_response = AsyncMock(return_value="Test response")
        
        # Create agent
        agent = EnhancedImageRecognitionAgent(llm=mock_llm)
        print(f"✅ Agent created: {agent.name}")
        print(f"📋 Capabilities: {agent.capabilities}")
        
        # Test content parsing
        test_text = """
        Question 1: What is 5 + 3?
        Student answer: 8
        Question 2: Solve x + 2 = 7
        Your answer: x = 5
        """
        
        content = await agent._parse_and_identify_content(test_text, {})
        print(f"✅ Parsed content:")
        print(f"   Questions: {content.questions}")
        print(f"   User answers: {content.user_answers}")
        print(f"   Confidence: {content.confidence_score:.2f}")
        
        print("🎉 Enhanced Image Recognition Agent is working correctly!")
    
    # Run the test
    asyncio.run(simple_test()) 