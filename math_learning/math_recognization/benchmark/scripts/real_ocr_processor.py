#!/usr/bin/env python3
"""
Real OCR Processor

Replaces simulation with actual API calls to OCR providers.
Integrates with GPT-4V, Mathpix, and other real OCR services.
"""

import os
import time
import base64
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

@dataclass
class OCRResult:
    """Result from OCR processing."""
    success: bool
    text: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    provider: str = ""
    error: str = ""
    raw_response: Dict = None

class RealOCRProcessor:
    """Real OCR processor that makes actual API calls."""
    
    def __init__(self):
        """Initialize with API credentials."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.mathpix_app_id = os.getenv("MATHPIX_APP_ID")
        self.mathpix_app_key = os.getenv("MATHPIX_APP_KEY") or os.getenv("MATHPIX_API_KEY")
        
        # Validate API keys
        self.validate_credentials()
    
    def validate_credentials(self):
        """Validate that required API credentials are available."""
        missing_keys = []
        
        if not self.openai_api_key:
            missing_keys.append("OPENAI_API_KEY")
        if not self.mathpix_app_id:
            missing_keys.append("MATHPIX_APP_ID")  
        if not self.mathpix_app_key:
            missing_keys.append("MATHPIX_APP_KEY or MATHPIX_API_KEY")
        
        if missing_keys:
            print(f"⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
            print("   Some OCR providers will not be available")
        else:
            print("✅ All API credentials validated")
    
    async def process_image_gpt5(self, image_path: str, prompt: str = None) -> OCRResult:
        """Process image using GPT-5 Vision API."""
        
        if not self.openai_api_key:
            return OCRResult(success=False, error="OpenAI API key not available", provider="gpt5")
        
        try:
            start_time = time.time()
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Default prompt for math recognition
            if not prompt:
                prompt = """Analyze this mathematical image and extract all text, equations, geometric shapes, and coordinates.

For geometric content, identify:
1. All shapes (triangles, circles, lines, points, polygons)
2. Coordinates and measurements
3. Labels and text
4. Mathematical symbols and notation

For algebraic content, extract:
1. All equations and expressions
2. Variables and constants
3. Mathematical operations
4. Solution steps if present

Provide a detailed, structured response with all mathematical content."""

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }
            
            payload = {
                "model": "gpt-5",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_completion_tokens": 1000
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        if result.get("choices") and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            
                            return OCRResult(
                                success=True,
                                text=content,
                                confidence=0.85,  # GPT-4V typically high confidence
                                processing_time=processing_time,
                                provider="gpt5",
                                raw_response=result
                            )
                        else:
                            return OCRResult(
                                success=False,
                                error="No content in response",
                                processing_time=processing_time,
                                provider="gpt5"
                            )
                    else:
                        error_text = await response.text()
                        return OCRResult(
                            success=False,
                            error=f"API error {response.status}: {error_text}",
                            processing_time=processing_time,
                            provider="gpt5"
                        )
        
        except Exception as e:
            return OCRResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                provider="gpt5"
            )
    
    async def process_image_mathpix(self, image_path: str) -> OCRResult:
        """Process image using Mathpix OCR API."""
        
        if not self.mathpix_app_id or not self.mathpix_app_key:
            return OCRResult(success=False, error="Mathpix API credentials not available", provider="mathpix")
        
        try:
            start_time = time.time()
            
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            headers = {
                "app_id": self.mathpix_app_id,
                "app_key": self.mathpix_app_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "src": f"data:image/jpeg;base64,{image_data}",
                "formats": ["text", "latex_styled"],
                "data_options": {
                    "include_asciimath": True,
                    "include_latex": True
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.mathpix.com/v3/text",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    processing_time = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        text = result.get("text", "")
                        confidence = result.get("confidence", 0.0)
                        
                        return OCRResult(
                            success=True,
                            text=text,
                            confidence=confidence,
                            processing_time=processing_time,
                            provider="mathpix",
                            raw_response=result
                        )
                    else:
                        error_text = await response.text()
                        return OCRResult(
                            success=False,
                            error=f"Mathpix API error {response.status}: {error_text}",
                            processing_time=processing_time,
                            provider="mathpix"
                        )
        
        except Exception as e:
            return OCRResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                provider="mathpix"
            )
    
    async def process_image_hybrid(self, image_path: str) -> OCRResult:
        """Process image using hybrid approach (Mathpix + GPT-4V fallback)."""
        
        # Try Mathpix first
        mathpix_result = await self.process_image_mathpix(image_path)
        
        if mathpix_result.success and mathpix_result.confidence > 0.7:
            mathpix_result.provider = "mathpix_hybrid"
            return mathpix_result
        
        # Fallback to GPT-5
        print("   🔄 Mathpix failed or low confidence, falling back to GPT-5...")
        gpt5_result = await self.process_image_gpt5(image_path)
        
        if gpt5_result.success:
            gpt5_result.provider = "gpt5_hybrid_fallback"
            return gpt5_result
        
        # Return the better of the two results
        if mathpix_result.success:
            mathpix_result.provider = "mathpix_hybrid_degraded"
            return mathpix_result
        
        return OCRResult(
            success=False,
            error="Both Mathpix and GPT-4V failed",
            provider="hybrid_failed"
        )
    
    async def process_problem_with_config(self, image_path: str, config_name: str, problem_data: Dict = None) -> Dict[str, Any]:
        """Process a problem using the specified configuration."""
        
        print(f"   🔍 Processing {Path(image_path).name} with {config_name}")
        
        # Select processing method based on configuration
        if config_name == "gpt-5":
            result = await self.process_image_gpt5(image_path)
        elif config_name == "mathpix_gpt5_hybrid":
            result = await self.process_image_hybrid(image_path)
        elif config_name == "geometry_specialist":
            # Use GPT-5 with geometry-specific prompt
            geometry_prompt = """Analyze this geometric diagram and identify:

1. ALL GEOMETRIC SHAPES:
   - Points (with coordinates if visible)
   - Lines (with endpoints)
   - Circles (with center and radius)
   - Triangles (type: right, isosceles, equilateral, etc.)
   - Polygons (number of sides, regular/irregular)

2. MEASUREMENTS AND LABELS:
   - All numerical values
   - Variable names and labels
   - Units of measurement
   - Angle measurements

3. GEOMETRIC RELATIONSHIPS:
   - Parallel lines
   - Perpendicular lines
   - Congruent shapes
   - Similar shapes

Be extremely detailed and precise. List every shape, measurement, and label you can identify."""
            
            result = await self.process_image_gpt5(image_path, geometry_prompt)
        else:
            # Default to GPT-4V
            result = await self.process_image_gpt4v(image_path)
        
        # Convert OCR result to benchmark format
        return {
            "success": result.success,
            "extracted_text": result.text,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "provider": result.provider,
            "error": result.error,
            "shapes_detected": self.extract_shapes_from_text(result.text) if result.success else [],
            "coordinates_extracted": self.extract_coordinates_from_text(result.text) if result.success else [],
            "text_recognized": self.extract_mathematical_text(result.text) if result.success else []
        }
    
    def extract_shapes_from_text(self, text: str) -> List[str]:
        """Extract geometric shapes mentioned in the OCR text."""
        shapes = []
        shape_keywords = [
            "triangle", "circle", "square", "rectangle", "polygon", 
            "line", "point", "angle", "arc", "ellipse", "rhombus",
            "parallelogram", "trapezoid", "pentagon", "hexagon"
        ]
        
        text_lower = text.lower()
        for shape in shape_keywords:
            if shape in text_lower:
                shapes.append(shape)
        
        return list(set(shapes))  # Remove duplicates
    
    def extract_coordinates_from_text(self, text: str) -> List[Tuple[float, float]]:
        """Extract coordinate pairs from the OCR text."""
        import re
        
        # Look for coordinate patterns like (x, y), (x,y), etc.
        coordinate_pattern = r'\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)'
        matches = re.findall(coordinate_pattern, text)
        
        coordinates = []
        for match in matches:
            try:
                x, y = float(match[0]), float(match[1])
                coordinates.append((x, y))
            except ValueError:
                continue
        
        return coordinates
    
    def extract_mathematical_text(self, text: str) -> List[str]:
        """Extract mathematical expressions and symbols from text."""
        import re
        
        # Look for mathematical expressions
        math_patterns = [
            r'[a-zA-Z]\s*[+\-*/=]\s*[a-zA-Z0-9]+',  # Variables with operations
            r'\d+\s*[+\-*/=]\s*\d+',  # Numeric expressions
            r'[a-zA-Z]\^?\d*',  # Variables with exponents
            r'√\d+',  # Square roots
            r'∠[A-Z]+',  # Angles
            r'[≤≥≠≈∞∑∏∫]',  # Mathematical symbols
        ]
        
        mathematical_text = []
        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            mathematical_text.extend(matches)
        
        return list(set(mathematical_text))  # Remove duplicates

# Global instance for use in benchmark
real_ocr_processor = RealOCRProcessor()
