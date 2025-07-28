"""
Advanced Geometry OCR Integration

This module provides integration with state-of-the-art OCR systems that excel at
geometry recognition, including shapes, diagrams, and geometric constructions.
These are alternatives to Mathpix that handle geometry much better.
"""

import asyncio
import base64
import json
import aiohttp
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from pathlib import Path

try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    transformers = None

class GeometryOCRProvider(Enum):
    """Available geometry OCR providers."""
    GOT_OCR2 = "got_ocr2"  # General OCR Theory v2.0 - handles geometric shapes
    UNIMERNET = "unimernet"  # UniMERNet - excellent for math expressions
    MONKEY_OCR = "monkey_ocr"  # MonkeyOCR - advanced document parsing
    PP_FORMULA = "pp_formula"  # PP-FormulaNet - PaddleOCR's formula recognition
    TROCR_GEOMETRY = "trocr_geometry"  # TrOCR fine-tuned for geometry
    OPEN_OCR = "open_ocr"  # OpenOCR - general purpose with geometry support

@dataclass
class GeometryOCRResult:
    """Result from geometry OCR processing."""
    text: str
    latex: str
    geometric_elements: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: float
    provider: GeometryOCRProvider
    detected_shapes: List[str]
    coordinate_info: Dict[str, Any]
    diagram_type: str
    error: Optional[str] = None
    raw_output: Optional[str] = None

class GOTOCRClient:
    """Client for GOT-OCR2.0 - handles geometric shapes explicitly."""
    
    def __init__(self, model_path: str = "stepfun-ai/GOT-OCR2_0", device: str = "auto"):
        """
        Initialize GOT-OCR2.0 client.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for GOT-OCR")
        
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.processor = None
    
    async def load_model(self):
        """Load the GOT-OCR2.0 model."""
        if self.model is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True,
                    device_map=self.device
                )
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(f"Failed to load GOT-OCR2.0 model: {e}")
    
    async def extract_geometry(self, image_data: Union[bytes, str], 
                             ocr_type: str = "format") -> GeometryOCRResult:
        """
        Extract geometric content using GOT-OCR2.0.
        
        Args:
            image_data: Image as bytes or base64 string
            ocr_type: OCR type - 'format' for formatted output, 'plain' for plain text
            
        Returns:
            GeometryOCRResult with extracted geometric content
        """
        await self.load_model()
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare image
            if isinstance(image_data, bytes):
                # Save temporarily for processing
                temp_path = "/tmp/temp_geometry_image.jpg"
                with open(temp_path, "wb") as f:
                    f.write(image_data)
                image_path = temp_path
            else:
                # Assume it's a file path
                image_path = image_data
            
            # Process with GOT-OCR2.0
            # GOT-OCR2.0 supports geometric shapes via special prompts
            if ocr_type == "format":
                # For geometric diagrams, use formatted output
                result = self.model.chat(
                    self.tokenizer,
                    image_path,
                    ocr_type='format',
                    render=True  # Enable rendering for geometric shapes
                )
            else:
                result = self.model.chat(
                    self.tokenizer,
                    image_path,
                    ocr_type='plain'
                )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Parse result for geometric elements
            geometric_elements = self._parse_geometric_elements(result)
            detected_shapes = self._detect_shapes(result)
            diagram_type = self._classify_diagram_type(result, detected_shapes)
            
            # Clean up temp file
            if isinstance(image_data, bytes) and os.path.exists(temp_path):
                os.remove(temp_path)
            
            return GeometryOCRResult(
                text=result,
                latex=self._extract_latex(result),
                geometric_elements=geometric_elements,
                confidence=0.9,  # GOT-OCR2.0 generally has high confidence
                processing_time_ms=processing_time,
                provider=GeometryOCRProvider.GOT_OCR2,
                detected_shapes=detected_shapes,
                coordinate_info=self._extract_coordinates(result),
                diagram_type=diagram_type,
                raw_output=result
            )
            
        except Exception as e:
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            return GeometryOCRResult(
                text="",
                latex="",
                geometric_elements=[],
                confidence=0.0,
                processing_time_ms=processing_time,
                provider=GeometryOCRProvider.GOT_OCR2,
                detected_shapes=[],
                coordinate_info={},
                diagram_type="unknown",
                error=str(e)
            )
    
    def _parse_geometric_elements(self, text: str) -> List[Dict[str, Any]]:
        """Parse geometric elements from OCR result."""
        elements = []
        
        # Look for common geometric terms
        geometric_terms = [
            r'triangle|△|∆',
            r'circle|○|◯',
            r'square|□|■',
            r'rectangle|▭|▯',
            r'line|—|―',
            r'angle|∠|∟',
            r'point|·|•',
            r'polygon',
            r'parallelogram',
            r'rhombus',
            r'trapezoid',
            r'pentagon',
            r'hexagon',
            r'octagon'
        ]
        
        for i, term_pattern in enumerate(geometric_terms):
            matches = re.finditer(term_pattern, text, re.IGNORECASE)
            for match in matches:
                elements.append({
                    'type': term_pattern.split('|')[0],
                    'text': match.group(),
                    'position': match.span(),
                    'confidence': 0.8
                })
        
        return elements
    
    def _detect_shapes(self, text: str) -> List[str]:
        """Detect geometric shapes mentioned in the text."""
        shapes = []
        shape_patterns = {
            'triangle': r'triangle|△|∆',
            'circle': r'circle|○|◯',
            'square': r'square|□|■',
            'rectangle': r'rectangle|▭|▯',
            'line': r'line|—|―',
            'angle': r'angle|∠|∟',
            'polygon': r'polygon',
            'parallelogram': r'parallelogram',
            'rhombus': r'rhombus',
            'trapezoid': r'trapezoid'
        }
        
        for shape, pattern in shape_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                shapes.append(shape)
        
        return list(set(shapes))
    
    def _classify_diagram_type(self, text: str, shapes: List[str]) -> str:
        """Classify the type of geometric diagram."""
        if not shapes:
            return "text_only"
        
        if len(shapes) == 1:
            return f"single_{shapes[0]}"
        
        # Check for common combinations
        if 'triangle' in shapes and 'circle' in shapes:
            return "triangle_circle_construction"
        elif 'angle' in shapes and ('line' in shapes or 'triangle' in shapes):
            return "angle_measurement"
        elif len([s for s in shapes if s in ['square', 'rectangle', 'parallelogram']]) > 1:
            return "quadrilateral_comparison"
        else:
            return "complex_geometry"
    
    def _extract_latex(self, text: str) -> str:
        """Extract LaTeX expressions from the text."""
        # Look for LaTeX patterns
        latex_patterns = [
            r'\$[^$]+\$',  # Inline math
            r'\$\$[^$]+\$\$',  # Display math
            r'\\[a-zA-Z]+\{[^}]*\}',  # LaTeX commands
        ]
        
        latex_parts = []
        for pattern in latex_patterns:
            matches = re.findall(pattern, text)
            latex_parts.extend(matches)
        
        return ' '.join(latex_parts) if latex_parts else ""
    
    def _extract_coordinates(self, text: str) -> Dict[str, Any]:
        """Extract coordinate information from the text."""
        coordinates = {}
        
        # Look for coordinate patterns
        coord_patterns = [
            r'\((-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)',  # (x, y)
            r'x\s*=\s*(-?\d+(?:\.\d+)?)',  # x = value
            r'y\s*=\s*(-?\d+(?:\.\d+)?)',  # y = value
        ]
        
        points = []
        for pattern in coord_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) == 2:
                    points.append({
                        'x': float(match.group(1)),
                        'y': float(match.group(2)),
                        'text': match.group(0)
                    })
        
        if points:
            coordinates['points'] = points
            coordinates['count'] = len(points)
        
        return coordinates

class UniMERNetClient:
    """Client for UniMERNet - excellent for mathematical expressions including geometry."""
    
    def __init__(self, model_path: str = "wanderkid/unimernet_base"):
        """Initialize UniMERNet client."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for UniMERNet")
        
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
    
    async def load_model(self):
        """Load UniMERNet model."""
        if self.model is None:
            try:
                # UniMERNet loading would go here
                # This is a placeholder as the actual implementation
                # depends on the specific UniMERNet setup
                pass
            except Exception as e:
                raise RuntimeError(f"Failed to load UniMERNet: {e}")
    
    async def extract_geometry(self, image_data: Union[bytes, str]) -> GeometryOCRResult:
        """Extract geometric content using UniMERNet."""
        # Placeholder implementation
        # In practice, this would use the actual UniMERNet inference
        return GeometryOCRResult(
            text="UniMERNet processing placeholder",
            latex="",
            geometric_elements=[],
            confidence=0.0,
            processing_time_ms=0,
            provider=GeometryOCRProvider.UNIMERNET,
            detected_shapes=[],
            coordinate_info={},
            diagram_type="placeholder",
            error="Not implemented - placeholder"
        )

class GeometryOCRProcessor:
    """Main processor that combines multiple geometry OCR providers."""
    
    def __init__(self, providers: List[GeometryOCRProvider] = None,
                 fallback_order: List[GeometryOCRProvider] = None):
        """
        Initialize geometry OCR processor.
        
        Args:
            providers: List of providers to use
            fallback_order: Order to try providers if primary fails
        """
        self.providers = providers or [GeometryOCRProvider.GOT_OCR2]
        self.fallback_order = fallback_order or [
            GeometryOCRProvider.GOT_OCR2,
            GeometryOCRProvider.UNIMERNET,
            GeometryOCRProvider.MONKEY_OCR
        ]
        
        # Initialize clients
        self.clients = {}
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize available OCR clients."""
        try:
            self.clients[GeometryOCRProvider.GOT_OCR2] = GOTOCRClient()
        except Exception:
            pass
        
        try:
            self.clients[GeometryOCRProvider.UNIMERNET] = UniMERNetClient()
        except Exception:
            pass
    
    async def process_geometry_image(self, image_data: Union[bytes, str],
                                   preferred_provider: GeometryOCRProvider = None,
                                   **kwargs) -> GeometryOCRResult:
        """
        Process geometry image with the best available OCR.
        
        Args:
            image_data: Image as bytes or base64 string
            preferred_provider: Preferred OCR provider
            **kwargs: Additional options for specific providers
            
        Returns:
            GeometryOCRResult with extracted geometric content
        """
        # Determine provider order
        if preferred_provider and preferred_provider in self.clients:
            providers_to_try = [preferred_provider] + [
                p for p in self.fallback_order if p != preferred_provider and p in self.clients
            ]
        else:
            providers_to_try = [p for p in self.fallback_order if p in self.clients]
        
        last_error = None
        
        for provider in providers_to_try:
            try:
                client = self.clients[provider]
                result = await client.extract_geometry(image_data, **kwargs)
                
                if not result.error and result.confidence > 0.3:
                    return result
                else:
                    last_error = result.error
                    
            except Exception as e:
                last_error = str(e)
                continue
        
        # If all providers failed, return error result
        return GeometryOCRResult(
            text="",
            latex="",
            geometric_elements=[],
            confidence=0.0,
            processing_time_ms=0,
            provider=GeometryOCRProvider.GOT_OCR2,
            detected_shapes=[],
            coordinate_info={},
            diagram_type="failed",
            error=f"All providers failed. Last error: {last_error}"
        )
    
    async def batch_process_geometry(self, images: List[Union[bytes, str]],
                                   max_concurrent: int = 3) -> List[GeometryOCRResult]:
        """Process multiple geometry images concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(image_data):
            async with semaphore:
                return await self.process_geometry_image(image_data)
        
        tasks = [process_single(img) for img in images]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_available_providers(self) -> List[GeometryOCRProvider]:
        """Get list of available OCR providers."""
        return list(self.clients.keys())
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available providers."""
        info = {}
        
        provider_details = {
            GeometryOCRProvider.GOT_OCR2: {
                "name": "GOT-OCR2.0",
                "description": "General OCR Theory v2.0 - handles geometric shapes explicitly",
                "strengths": ["Geometric shapes", "Diagrams", "Charts", "Multi-modal content"],
                "best_for": "Complex geometric diagrams with mixed content"
            },
            GeometryOCRProvider.UNIMERNET: {
                "name": "UniMERNet",
                "description": "Universal Mathematical Expression Recognition Network",
                "strengths": ["Mathematical expressions", "Formulas", "Equations", "Geometry proofs"],
                "best_for": "Mathematical geometry problems with formulas"
            },
            GeometryOCRProvider.MONKEY_OCR: {
                "name": "MonkeyOCR",
                "description": "Advanced document parsing with structure recognition",
                "strengths": ["Document structure", "Layout analysis", "Multi-element recognition"],
                "best_for": "Structured geometry worksheets and textbooks"
            }
        }
        
        for provider in self.clients.keys():
            if provider in provider_details:
                info[provider.value] = provider_details[provider]
                info[provider.value]["available"] = True
            else:
                info[provider.value] = {"available": False}
        
        return info

# Convenience functions
async def extract_geometry_with_got_ocr(image_data: Union[bytes, str]) -> GeometryOCRResult:
    """Convenience function to extract geometry using GOT-OCR2.0."""
    client = GOTOCRClient()
    return await client.extract_geometry(image_data)

async def process_geometry_image_auto(image_data: Union[bytes, str],
                                    preferred_provider: str = "got_ocr2") -> GeometryOCRResult:
    """
    Convenience function to process geometry image with automatic provider selection.
    
    Args:
        image_data: Image as bytes or base64 string
        preferred_provider: Preferred provider name
        
    Returns:
        GeometryOCRResult with extracted content
    """
    processor = GeometryOCRProcessor()
    
    # Convert string to enum
    provider_map = {
        "got_ocr2": GeometryOCRProvider.GOT_OCR2,
        "unimernet": GeometryOCRProvider.UNIMERNET,
        "monkey_ocr": GeometryOCRProvider.MONKEY_OCR
    }
    
    preferred = provider_map.get(preferred_provider, GeometryOCRProvider.GOT_OCR2)
    return await processor.process_geometry_image(image_data, preferred_provider=preferred)

def get_geometry_ocr_recommendations() -> Dict[str, str]:
    """Get recommendations for which OCR to use for different geometry types."""
    return {
        "basic_shapes": "GOT-OCR2.0 - Best for recognizing basic geometric shapes",
        "construction_diagrams": "GOT-OCR2.0 - Excellent for complex construction diagrams",
        "geometry_proofs": "UniMERNet - Best for mathematical proofs with formulas",
        "coordinate_geometry": "GOT-OCR2.0 - Good at extracting coordinate information",
        "textbook_problems": "MonkeyOCR - Best for structured textbook layouts",
        "hand_drawn_diagrams": "GOT-OCR2.0 - Good generalization to hand-drawn content",
        "mixed_content": "GOT-OCR2.0 - Handles text, shapes, and formulas together"
    } 