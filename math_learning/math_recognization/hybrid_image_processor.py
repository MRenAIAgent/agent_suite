"""
Hybrid Image Processor for Math Test Analysis

This module combines Mathpix OCR and GPT-4 Vision to provide comprehensive
analysis of math test images, extracting both mathematical content and
question-answer structure.
"""

import asyncio
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime

from .mathpix_ocr import MathpixOCRClient, MathpixResult, MathpixTextProcessor
from .gpt4_vision import GPT4VisionClient, VisionAnalysisResult, VisionTextProcessor, VisionAnalysisType
from .advanced_geometry_ocr import GeometryOCRProcessor, GeometryOCRProvider, GeometryOCRResult

class ProcessingStrategy(Enum):
    """Image processing strategies."""
    OCR_FIRST = "ocr_first"  # OCR first, then Vision with context
    VISION_FIRST = "vision_first"  # Vision first, then OCR for validation
    PARALLEL = "parallel"  # Both simultaneously, then combine
    OCR_ONLY = "ocr_only"  # Only Mathpix OCR
    VISION_ONLY = "vision_only"  # Only GPT-4 Vision
    GEOMETRY_FIRST = "geometry_first"  # Geometry OCR first, then others
    GEOMETRY_ONLY = "geometry_only"  # Only advanced geometry OCR

@dataclass
class ProcessingResult:
    """Combined result from hybrid processing."""
    success: bool
    question_answer_pairs: List[Dict[str, Any]]
    ocr_result: Optional[MathpixResult] = None
    vision_result: Optional[VisionAnalysisResult] = None
    geometry_result: Optional[GeometryOCRResult] = None
    processing_strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL
    total_processing_time_ms: float = 0.0
    confidence_score: float = 0.0
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HybridImageProcessor:
    """
    Hybrid processor that combines Mathpix OCR and GPT-4 Vision
    for comprehensive math test image analysis.
    """
    
    def __init__(self, 
                 mathpix_app_id: str = None,
                 mathpix_app_key: str = None,
                 openai_api_key: str = None,
                 geometry_providers: List[GeometryOCRProvider] = None,
                 default_strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL):
        """
        Initialize the hybrid processor.
        
        Args:
            mathpix_app_id: Mathpix application ID
            mathpix_app_key: Mathpix application key  
            openai_api_key: OpenAI API key for GPT-4 Vision
            geometry_providers: List of geometry OCR providers to use
            default_strategy: Default processing strategy
        """
        self.mathpix_app_id = mathpix_app_id
        self.mathpix_app_key = mathpix_app_key
        self.openai_api_key = openai_api_key
        self.geometry_providers = geometry_providers or [GeometryOCRProvider.GOT_OCR2]
        self.default_strategy = default_strategy
        
        # Initialize clients
        self.mathpix_client = None
        self.vision_client = None
        self.geometry_processor = None
        
        # Performance tracking
        self.processing_stats = {
            'total_images_processed': 0,
            'average_processing_time': 0.0,
            'success_rate': 0.0,
            'strategy_usage': {}
        }
    
    async def process_test_image(self,
                               image_data: Union[bytes, str],
                               strategy: ProcessingStrategy = None,
                               **options) -> ProcessingResult:
        """
        Process a math test image using the hybrid approach.
        
        Args:
            image_data: Image as bytes or base64 string
            strategy: Processing strategy to use (defaults to instance default)
            **options: Additional processing options
            
        Returns:
            ProcessingResult with combined analysis
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        try:
            # Initialize clients if needed
            await self._ensure_clients_ready()
            
            # Process based on strategy
            if strategy == ProcessingStrategy.OCR_FIRST:
                result = await self._process_ocr_first(image_data, **options)
            elif strategy == ProcessingStrategy.VISION_FIRST:
                result = await self._process_vision_first(image_data, **options)
            elif strategy == ProcessingStrategy.PARALLEL:
                result = await self._process_parallel(image_data, **options)
            elif strategy == ProcessingStrategy.OCR_ONLY:
                result = await self._process_ocr_only(image_data, **options)
            elif strategy == ProcessingStrategy.VISION_ONLY:
                result = await self._process_vision_only(image_data, **options)
            elif strategy == ProcessingStrategy.GEOMETRY_FIRST:
                result = await self._process_geometry_first(image_data, **options)
            elif strategy == ProcessingStrategy.GEOMETRY_ONLY:
                result = await self._process_geometry_only(image_data, **options)
            else:
                raise ValueError(f"Unknown processing strategy: {strategy}")
            
            # Calculate total processing time
            total_time = (time.time() - start_time) * 1000
            result.total_processing_time_ms = total_time
            result.processing_strategy = strategy
            
            # Update statistics
            self._update_processing_stats(result, strategy)
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                question_answer_pairs=[],
                processing_strategy=strategy,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                error_messages=[str(e)]
            )
    
    async def _ensure_clients_ready(self):
        """Ensure clients are initialized and ready."""
        if not self.mathpix_client and (self.mathpix_app_id and self.mathpix_app_key):
            self.mathpix_client = MathpixOCRClient(
                app_id=self.mathpix_app_id,
                app_key=self.mathpix_app_key
            )
        
        if not self.vision_client and self.openai_api_key:
            self.vision_client = GPT4VisionClient(api_key=self.openai_api_key)
        
        if not self.geometry_processor:
            self.geometry_processor = GeometryOCRProcessor(providers=self.geometry_providers)
    
    async def _process_ocr_first(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with OCR first, then Vision with OCR context."""
        
        # Step 1: Extract text with Mathpix OCR
        ocr_result = None
        if self.mathpix_client:
            async with self.mathpix_client as client:
                ocr_result = await client.extract_text(image_data)
        
        # Step 2: Use OCR result as context for Vision analysis
        vision_result = None
        if self.vision_client:
            ocr_text = ocr_result.text if ocr_result and ocr_result.text else ""
            vision_result = await self.vision_client.analyze_test_image(
                image_data=image_data,
                ocr_text=ocr_text,
                analysis_type=VisionAnalysisType.COMPREHENSIVE
            )
        
        # Step 3: Combine results
        return self._combine_results(ocr_result, vision_result, None, "ocr_first")
    
    async def _process_vision_first(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with Vision first, then OCR for validation."""
        
        # Step 1: Analyze with GPT-4 Vision
        vision_result = None
        if self.vision_client:
            vision_result = await self.vision_client.analyze_test_image(
                image_data=image_data,
                analysis_type=VisionAnalysisType.COMPREHENSIVE
            )
        
        # Step 2: Extract mathematical text with OCR for validation
        ocr_result = None
        if self.mathpix_client:
            async with self.mathpix_client as client:
                ocr_result = await client.extract_text(image_data)
        
        # Step 3: Enhance Vision results with OCR data
        if vision_result and ocr_result:
            vision_result = VisionTextProcessor.enhance_with_ocr_context(
                vision_result, ocr_result.text
            )
        
        # Step 4: Combine results
        return self._combine_results(ocr_result, vision_result, None, "vision_first")
    
    async def _process_parallel(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with both OCR and Vision simultaneously."""
        
        # Run both processes in parallel
        tasks = []
        
        # OCR task
        if self.mathpix_client:
            async def ocr_task():
                async with self.mathpix_client as client:
                    return await client.extract_text(image_data)
            tasks.append(ocr_task())
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task
        
        # Vision task
        if self.vision_client:
            vision_task = self.vision_client.analyze_test_image(
                image_data=image_data,
                analysis_type=VisionAnalysisType.COMPREHENSIVE
            )
            tasks.append(vision_task)
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task
        
        # Wait for both to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Extract results
        ocr_result = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None
        vision_result = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        
        # If we have both, enhance Vision with OCR context
        if vision_result and ocr_result and hasattr(ocr_result, 'text'):
            vision_result = VisionTextProcessor.enhance_with_ocr_context(
                vision_result, ocr_result.text
            )
        
        # Combine results
        return self._combine_results(ocr_result, vision_result, None, "parallel")
    
    async def _process_ocr_only(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with only Mathpix OCR."""
        
        ocr_result = None
        if self.mathpix_client:
            async with self.mathpix_client as client:
                ocr_result = await client.extract_text(image_data)
        
        return self._combine_results(ocr_result, None, None, "ocr_only")
    
    async def _process_vision_only(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with only GPT-4 Vision."""
        
        vision_result = None
        if self.vision_client:
            vision_result = await self.vision_client.analyze_test_image(
                image_data=image_data,
                analysis_type=VisionAnalysisType.COMPREHENSIVE
            )
        
        return self._combine_results(None, vision_result, None, "vision_only")
    
    async def _process_geometry_first(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with geometry OCR first, then fallback to others if needed."""
        
        # Step 1: Try geometry OCR first
        geometry_result = None
        if self.geometry_processor:
            geometry_result = await self.geometry_processor.process_geometry_image(image_data)
        
        # Step 2: If geometry OCR succeeds with good confidence, use it
        if geometry_result and not geometry_result.error and geometry_result.confidence > 0.7:
            return self._combine_results(None, None, geometry_result, "geometry_first")
        
        # Step 3: Fallback to standard OCR + Vision if geometry OCR fails
        ocr_result = None
        if self.mathpix_client:
            async with self.mathpix_client as client:
                ocr_result = await client.extract_text(image_data)
        
        vision_result = None
        if self.vision_client:
            ocr_text = ocr_result.text if ocr_result and ocr_result.text else ""
            vision_result = await self.vision_client.analyze_test_image(
                image_data=image_data,
                ocr_text=ocr_text,
                analysis_type=VisionAnalysisType.COMPREHENSIVE
            )
        
        return self._combine_results(ocr_result, vision_result, geometry_result, "geometry_first_fallback")
    
    async def _process_geometry_only(self, image_data: Union[bytes, str], **options) -> ProcessingResult:
        """Process with only advanced geometry OCR."""
        
        geometry_result = None
        if self.geometry_processor:
            geometry_result = await self.geometry_processor.process_geometry_image(image_data)
        
        return self._combine_results(None, None, geometry_result, "geometry_only")
    
    def _combine_results(self, 
                        ocr_result: Optional[MathpixResult],
                        vision_result: Optional[VisionAnalysisResult],
                        geometry_result: Optional[GeometryOCRResult],
                        method: str) -> ProcessingResult:
        """Combine OCR and Vision results into a unified result."""
        
        question_answer_pairs = []
        error_messages = []
        confidence_scores = []
        
        # Process OCR results
        if ocr_result:
            if ocr_result.error:
                error_messages.append(f"OCR Error: {ocr_result.error}")
            else:
                confidence_scores.append(ocr_result.confidence)
                
                # Extract equations from OCR text
                equations = MathpixTextProcessor.extract_equations(ocr_result.text)
                problem_types = MathpixTextProcessor.identify_problem_types(
                    ocr_result.text, ocr_result.latex
                )
                
                # Create basic question-answer pairs from OCR
                for i, equation in enumerate(equations):
                    parts = equation.split('=')
                    if len(parts) >= 2:
                        question_answer_pairs.append({
                            'question_id': f'OCR_Q{i+1}',
                            'question_text': equation.strip(),
                            'student_answer': parts[-1].strip(),
                            'work_shown': '',
                            'confidence': ocr_result.confidence,
                            'problem_type': problem_types[0] if problem_types else 'general_math',
                            'source': 'ocr'
                        })
        
        # Process Geometry results (highest priority for geometry content)
        if geometry_result:
            if geometry_result.error:
                error_messages.append(f"Geometry Error: {geometry_result.error}")
            else:
                confidence_scores.append(geometry_result.confidence)
                
                # Create question-answer pairs from geometry result
                if geometry_result.geometric_elements:
                    for i, element in enumerate(geometry_result.geometric_elements):
                        qa_dict = {
                            'question_id': f'GEOM_Q{i+1}',
                            'question_text': geometry_result.text,
                            'student_answer': element.get('text', ''),
                            'work_shown': '',
                            'confidence': geometry_result.confidence,
                            'problem_type': f"geometry_{element.get('type', 'unknown')}",
                            'source': 'geometry',
                            'geometric_elements': geometry_result.geometric_elements,
                            'detected_shapes': geometry_result.detected_shapes,
                            'diagram_type': geometry_result.diagram_type
                        }
                        question_answer_pairs.append(qa_dict)
                elif geometry_result.text:
                    # If no specific elements, create general geometry question
                    qa_dict = {
                        'question_id': 'GEOM_Q1',
                        'question_text': geometry_result.text,
                        'student_answer': '',
                        'work_shown': '',
                        'confidence': geometry_result.confidence,
                        'problem_type': f"geometry_{geometry_result.diagram_type}",
                        'source': 'geometry',
                        'detected_shapes': geometry_result.detected_shapes,
                        'diagram_type': geometry_result.diagram_type
                    }
                    question_answer_pairs.append(qa_dict)

        # Process Vision results (prioritize over OCR if both exist)
        if vision_result:
            if vision_result.error_message:
                error_messages.append(f"Vision Error: {vision_result.error_message}")
            elif vision_result.success:
                confidence_scores.append(vision_result.processing_confidence)
                
                # Convert Vision question-answer pairs to dict format
                for pair in vision_result.question_answer_pairs:
                    qa_dict = {
                        'question_id': pair.question_id,
                        'question_text': pair.question_text,
                        'student_answer': pair.student_answer,
                        'expected_answer': pair.expected_answer,
                        'work_shown': pair.work_shown,
                        'confidence': pair.confidence,
                        'problem_type': pair.problem_type,
                        'bounding_box': pair.bounding_box,
                        'source': 'vision'
                    }
                    question_answer_pairs.append(qa_dict)
        
        # If we have both OCR and Vision results, merge intelligently
        if ocr_result and vision_result and vision_result.success:
            question_answer_pairs = self._merge_ocr_vision_pairs(
                question_answer_pairs, ocr_result, vision_result
            )
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Prepare metadata
        metadata = {
            'processing_method': method,
            'ocr_available': ocr_result is not None,
            'vision_available': vision_result is not None,
            'geometry_available': geometry_result is not None,
            'timestamp': datetime.now().isoformat()
        }
        
        if ocr_result:
            metadata['ocr_processing_time'] = ocr_result.processing_time_ms
            metadata['ocr_confidence'] = ocr_result.confidence
            metadata['detected_alphabets'] = ocr_result.detected_alphabets
            metadata['is_handwritten'] = ocr_result.is_handwritten
        
        if vision_result:
            metadata['vision_processing_time'] = vision_result.processing_time_ms
            metadata['vision_confidence'] = vision_result.processing_confidence
            metadata['layout_analysis'] = vision_result.layout_analysis
        
        if geometry_result:
            metadata['geometry_processing_time'] = geometry_result.processing_time_ms
            metadata['geometry_confidence'] = geometry_result.confidence
            metadata['geometry_provider'] = geometry_result.provider.value
            metadata['detected_shapes'] = geometry_result.detected_shapes
            metadata['diagram_type'] = geometry_result.diagram_type
            metadata['coordinate_info'] = geometry_result.coordinate_info
        
        return ProcessingResult(
            success=len(question_answer_pairs) > 0,
            question_answer_pairs=question_answer_pairs,
            ocr_result=ocr_result,
            vision_result=vision_result,
            geometry_result=geometry_result,
            confidence_score=avg_confidence,
            error_messages=error_messages,
            metadata=metadata
        )
    
    def _merge_ocr_vision_pairs(self, 
                              pairs: List[Dict[str, Any]],
                              ocr_result: MathpixResult,
                              vision_result: VisionAnalysisResult) -> List[Dict[str, Any]]:
        """Intelligently merge OCR and Vision question-answer pairs."""
        
        # Separate OCR and Vision pairs
        ocr_pairs = [p for p in pairs if p.get('source') == 'ocr']
        vision_pairs = [p for p in pairs if p.get('source') == 'vision']
        
        # If we have Vision pairs, prioritize them and enhance with OCR data
        if vision_pairs:
            enhanced_pairs = []
            for v_pair in vision_pairs:
                enhanced_pair = v_pair.copy()
                
                # Try to enhance question text with OCR if Vision text is short/unclear
                if len(v_pair['question_text']) < 10:
                    # Look for matching OCR equations
                    for o_pair in ocr_pairs:
                        if any(word in o_pair['question_text'].lower() 
                              for word in v_pair['question_text'].lower().split()):
                            enhanced_pair['question_text'] = o_pair['question_text']
                            enhanced_pair['confidence'] = min(
                                (v_pair['confidence'] + o_pair['confidence']) / 2, 1.0
                            )
                            break
                
                enhanced_pairs.append(enhanced_pair)
            
            return enhanced_pairs
        
        # If only OCR pairs, return them
        return ocr_pairs
    
    def _update_processing_stats(self, result: ProcessingResult, strategy: ProcessingStrategy):
        """Update processing statistics."""
        self.processing_stats['total_images_processed'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        total_processed = self.processing_stats['total_images_processed']
        new_avg = ((current_avg * (total_processed - 1)) + result.total_processing_time_ms) / total_processed
        self.processing_stats['average_processing_time'] = new_avg
        
        # Update success rate
        current_successes = self.processing_stats['success_rate'] * (total_processed - 1)
        new_successes = current_successes + (1 if result.success else 0)
        self.processing_stats['success_rate'] = new_successes / total_processed
        
        # Update strategy usage
        strategy_name = strategy.value
        if strategy_name not in self.processing_stats['strategy_usage']:
            self.processing_stats['strategy_usage'][strategy_name] = 0
        self.processing_stats['strategy_usage'][strategy_name] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    async def batch_process_images(self,
                                 images: List[Union[bytes, str]],
                                 strategy: ProcessingStrategy = None,
                                 max_concurrent: int = 3) -> List[ProcessingResult]:
        """
        Process multiple images concurrently.
        
        Args:
            images: List of images to process
            strategy: Processing strategy to use
            max_concurrent: Maximum concurrent processing tasks
            
        Returns:
            List of ProcessingResults
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(image_data):
            async with semaphore:
                return await self.process_test_image(image_data, strategy)
        
        tasks = [process_single(img) for img in images]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Convenience functions
async def process_math_test_image(image_data: Union[bytes, str],
                                mathpix_app_id: str = None,
                                mathpix_app_key: str = None,
                                openai_api_key: str = None,
                                geometry_providers: List[GeometryOCRProvider] = None,
                                strategy: ProcessingStrategy = ProcessingStrategy.PARALLEL) -> ProcessingResult:
    """
    Convenience function to process a math test image with hybrid approach.
    
    Args:
        image_data: Image as bytes or base64 string
        mathpix_app_id: Mathpix application ID
        mathpix_app_key: Mathpix application key
        openai_api_key: OpenAI API key
        geometry_providers: List of geometry OCR providers
        strategy: Processing strategy
        
    Returns:
        ProcessingResult with combined analysis
    """
    processor = HybridImageProcessor(
        mathpix_app_id=mathpix_app_id,
        mathpix_app_key=mathpix_app_key,
        openai_api_key=openai_api_key,
        geometry_providers=geometry_providers,
        default_strategy=strategy
    )
    
    return await processor.process_test_image(image_data, strategy) 