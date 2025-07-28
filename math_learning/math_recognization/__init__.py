"""
Math Recognition System

A comprehensive system for analyzing student math tests, providing:
- Answer correctness validation
- Error analysis and classification  
- Root cause identification
- Knowledge gap analysis
- Complete test workflow integration

Main Components:
- CompleteTestAnalyzer: End-to-end test analysis workflow
- HybridErrorAnalyzer: AI + rule-based error analysis
- AnswerCorrectnessEngine: Multi-method answer validation
- ErrorAnalysisEngine: Comprehensive error classification
- RootCauseAnalyzer: Diagnostic analysis
- KnowledgeGapIdentifier: Knowledge graph integration
"""

# Main integration classes
from .complete_test_analyzer import (
    CompleteTestAnalyzer,
    CompleteTestAnalysis,
    TestAnalysisStatus,
    QuestionAnswerPair,
    TestImageAnalysis
)

from .hybrid_error_analyzer import (
    HybridErrorAnalyzer,
    HybridAnalysisResult
)

# Individual component classes
from .answer_correctness_engine import (
    AnswerCorrectnessEngine,
    ValidationResult,
    AnswerType,
    CorrectnessLevel
)

from .error_analysis_engine import (
    ErrorAnalysisEngine,
    ErrorAnalysisResult,
    ErrorType,
    ErrorSeverity
)

from .root_cause_analyzer import (
    RootCauseAnalyzer,
    RootCauseAnalysis,
    DiagnosticQuestion,
    DiagnosticCategory
)

from .knowledge_gap_identifier import (
    KnowledgeGapIdentifier,
    GapAnalysisResult,
    KnowledgeGap,
    GapSeverity,
    GapType
)

# Image processing components
from .mathpix_ocr import (
    MathpixOCRClient,
    MathpixResult,
    MathpixTextProcessor,
    extract_math_from_image
)

from .gpt4_vision import (
    GPT4VisionClient,
    VisionAnalysisResult,
    VisionTextProcessor,
    analyze_math_test_image
)

from .hybrid_image_processor import (
    HybridImageProcessor,
    ProcessingResult,
    ProcessingStrategy,
    process_math_test_image
)

from .advanced_geometry_ocr import (
    GeometryOCRProcessor,
    GeometryOCRProvider,
    GeometryOCRResult,
    process_geometry_image_auto,
    get_geometry_ocr_recommendations
)

# Convenience functions
def create_complete_analyzer(llm_client=None, mathpix_app_id: str = None,
                           mathpix_app_key: str = None, openai_api_key: str = None):
    """
    Create a complete test analyzer with all components integrated.
    
    Args:
        llm_client: Optional LLM client for AI-powered analysis
        mathpix_app_id: Mathpix application ID for OCR
        mathpix_app_key: Mathpix application key for OCR
        openai_api_key: OpenAI API key for GPT-4 Vision
        
    Returns:
        CompleteTestAnalyzer ready for use
    """
    return CompleteTestAnalyzer(
        llm_client=llm_client,
        mathpix_app_id=mathpix_app_id,
        mathpix_app_key=mathpix_app_key,
        openai_api_key=openai_api_key
    )

def create_hybrid_analyzer(llm_client=None):
    """
    Create a hybrid error analyzer for individual question analysis.
    
    Args:
        llm_client: Optional LLM client for AI-powered analysis
        
    Returns:
        HybridErrorAnalyzer ready for use
    """
    return HybridErrorAnalyzer(llm_client=llm_client)

# Version info
__version__ = "1.0.0"
__author__ = "Math Learning System"

# Export all main classes for easy import
__all__ = [
    # Main integration
    'CompleteTestAnalyzer',
    'HybridErrorAnalyzer',
    'CompleteTestAnalysis',
    'HybridAnalysisResult',
    
    # Core engines
    'AnswerCorrectnessEngine',
    'ErrorAnalysisEngine', 
    'RootCauseAnalyzer',
    'KnowledgeGapIdentifier',
    
    # Data classes
    'ValidationResult',
    'ErrorAnalysisResult',
    'RootCauseAnalysis',
    'GapAnalysisResult',
    'KnowledgeGap',
    'DiagnosticQuestion',
    'QuestionAnswerPair',
    'TestImageAnalysis',
    
    # Enums
    'AnswerType',
    'CorrectnessLevel',
    'ErrorType',
    'ErrorSeverity',
    'GapSeverity',
    'GapType',
    'DiagnosticCategory',
    'TestAnalysisStatus',
    
    # Convenience functions
    'create_complete_analyzer',
    'create_hybrid_analyzer'
] 