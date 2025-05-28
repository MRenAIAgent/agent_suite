"""
Test script for RAG-Enhanced Math Learning System.

This module provides tests to verify the RAG integration works correctly
with different storage configurations.
"""

import asyncio
import pytest
from typing import Dict, Any

from math_learning.config.rag_config import RagConfig
from math_learning.knowledge_graph.rag_enhanced_algebra_graph import (
    create_rag_enhanced_algebra_graph,
    RagEnhancedAlgebraGraph
)
from math_learning.learning_graph.rag_enhanced_user_model import (
    RagEnhancedLearningGraph
)
from math_learning.examples.rag_integration_example import IntelligentMathTutor


class TestRagIntegration:
    """Test suite for RAG integration."""
    
    @pytest.fixture
    def test_config(self) -> RagConfig:
        """Create test configuration using memory storage."""
        return RagConfig(
            primary_storage_type="graph",
            vector_storage_type="memory",
            embedding_provider="dummy",
            graph_storage_type="memory",
            kv_storage_type="memory",
            similarity_threshold=0.5,
            max_results=3
        )
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_creation(self, test_config):
        """Test RAG-enhanced knowledge graph creation."""
        kg = None
        try:
            kg = await create_rag_enhanced_algebra_graph()
            assert kg is not None
            assert isinstance(kg, RagEnhancedAlgebraGraph)
            
            # Test basic functionality
            concepts = await kg.find_similar_concepts("number sense", top_k=2)
            assert isinstance(concepts, list)
            
        finally:
            if kg:
                await kg.close()
    
    @pytest.mark.asyncio
    async def test_learning_graph_creation(self, test_config):
        """Test RAG-enhanced learning graph creation."""
        lg = None
        try:
            lg = RagEnhancedLearningGraph("test_user", "Test Learning")
            assert lg is not None
            assert lg.user_id == "test_user"
            assert lg.name == "Test Learning"
            
            # Test session creation
            session_id = await lg.start_learning_session(["Test goal"])
            assert session_id is not None
            assert isinstance(session_id, str)
            
        finally:
            if lg:
                await lg.close()
    
    @pytest.mark.asyncio
    async def test_intelligent_tutor_initialization(self, test_config):
        """Test intelligent tutor initialization."""
        tutor = None
        try:
            tutor = IntelligentMathTutor("test_student", "Test Student")
            await tutor.initialize()
            
            assert tutor.knowledge_graph is not None
            assert tutor.learning_graph is not None
            assert tutor.student_id == "test_student"
            assert tutor.student_name == "Test Student"
            
        finally:
            if tutor:
                await tutor.close()
    
    @pytest.mark.asyncio
    async def test_learning_session_flow(self, test_config):
        """Test complete learning session flow."""
        tutor = None
        try:
            tutor = IntelligentMathTutor("flow_test", "Flow Test")
            await tutor.initialize()
            
            # Start session
            session_id = await tutor.start_learning_session([
                "Test counting", "Test basic math"
            ])
            assert session_id is not None
            
            # Get exercise
            exercise = await tutor.get_next_exercise()
            assert isinstance(exercise, dict)
            assert "exercise_id" in exercise
            assert "concept_id" in exercise
            assert "difficulty" in exercise
            
            # Submit result
            feedback = await tutor.submit_exercise_result(
                exercise, result=True, time_spent=60, confidence=4
            )
            assert isinstance(feedback, dict)
            assert "result" in feedback
            assert "feedback_message" in feedback
            
            # Get insights
            insights = await tutor.get_learning_insights()
            assert isinstance(insights, dict)
            assert "recommendations" in insights
            
            # End session
            summary = await tutor.end_session(mood_rating=4, difficulty_rating=3)
            assert isinstance(summary, dict)
            assert "duration_minutes" in summary
            assert "success_rate" in summary
            
        finally:
            if tutor:
                await tutor.close()
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, test_config):
        """Test semantic search functionality."""
        kg = None
        try:
            kg = await create_rag_enhanced_algebra_graph()
            
            # Test concept search
            concepts = await kg.find_similar_concepts("basic arithmetic", top_k=3)
            assert isinstance(concepts, list)
            assert len(concepts) <= 3
            
            # Test learning path suggestions
            path = await kg.get_learning_path_suggestions("NS-01", "algebra basics")
            assert isinstance(path, list)
            
        finally:
            if kg:
                await kg.close()
    
    @pytest.mark.asyncio
    async def test_personalized_recommendations(self, test_config):
        """Test personalized recommendation system."""
        kg = None
        try:
            kg = await create_rag_enhanced_algebra_graph()
            
            student_profile = {
                "learning_style": "visual",
                "difficulty_preference": "moderate",
                "interests": ["games"]
            }
            
            recommendations = await kg.get_teaching_recommendations(
                "NS-01", student_profile
            )
            assert isinstance(recommendations, dict)
            
        finally:
            if kg:
                await kg.close()
    
    @pytest.mark.asyncio
    async def test_learning_analytics(self, test_config):
        """Test learning analytics functionality."""
        lg = None
        try:
            lg = RagEnhancedLearningGraph("analytics_test", "Analytics Test")
            
            # Start session and record some attempts
            session_id = await lg.start_learning_session(["Test analytics"])
            
            await lg.record_enhanced_exercise_attempt(
                session_id=session_id,
                exercise_id="test_ex_1",
                concept_id="NS-01",
                result=True,
                difficulty=0.5,
                time_spent=60,
                hints_used=0,
                student_confidence=4
            )
            
            # Generate insights
            insights = await lg.generate_learning_insights()
            assert isinstance(insights, list)
            
            # Get recommendations
            recommendations = await lg.get_personalized_recommendations(session_id)
            assert isinstance(recommendations, dict)
            assert "focus_areas" in recommendations
            
            # Test adaptive difficulty
            difficulty = await lg.get_adaptive_difficulty("NS-01")
            assert isinstance(difficulty, float)
            assert 0.0 <= difficulty <= 1.0
            
        finally:
            if lg:
                await lg.close()


async def run_integration_test():
    """Run a simple integration test."""
    print("🧪 Running RAG Integration Test")
    print("=" * 35)
    
    try:
        # Test knowledge graph
        print("📚 Testing Knowledge Graph...")
        kg = await create_rag_enhanced_algebra_graph()
        concepts = await kg.find_similar_concepts("counting numbers", top_k=2)
        print(f"✅ Found {len(concepts)} similar concepts")
        await kg.close()
        
        # Test learning graph
        print("📊 Testing Learning Graph...")
        lg = RagEnhancedLearningGraph("test_user", "Test User")
        session_id = await lg.start_learning_session(["Test goal"])
        print(f"✅ Created session: {session_id[:8]}...")
        await lg.close()
        
        # Test intelligent tutor
        print("🎓 Testing Intelligent Tutor...")
        tutor = IntelligentMathTutor("test_student", "Test Student")
        await tutor.initialize()
        
        session_id = await tutor.start_learning_session(["Basic math"])
        exercise = await tutor.get_next_exercise()
        feedback = await tutor.submit_exercise_result(
            exercise, result=True, time_spent=45, confidence=3
        )
        summary = await tutor.end_session(mood_rating=4)
        
        print(f"✅ Exercise: {exercise['concept_id']}")
        print(f"✅ Feedback: {feedback['result']}")
        print(f"✅ Success rate: {summary['success_rate']:.1%}")
        
        await tutor.close()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run simple integration test
    success = asyncio.run(run_integration_test())
    exit(0 if success else 1) 