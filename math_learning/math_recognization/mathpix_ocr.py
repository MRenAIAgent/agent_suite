"""
Mathpix OCR Integration

This module provides integration with Mathpix OCR API for extracting
mathematical expressions, equations, and text from images.
"""

import base64
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import re
import os
from pathlib import Path

class MathpixFormat(Enum):
    """Supported Mathpix output formats."""
    TEXT = "text"
    LATEX = "latex_styled"
    MATHML = "mathml"
    ASCIIMATH = "asciimath"

@dataclass
class MathpixResult:
    """Result from Mathpix OCR processing."""
    text: str
    latex: str
    confidence: float
    processing_time_ms: float
    detected_alphabets: List[str]
    is_printed: bool
    is_handwritten: bool
    auto_rotate_confidence: float
    auto_rotate_degrees: int
    error: Optional[str] = None

class MathpixOCRClient:
    """Client for Mathpix OCR API."""
    
    def __init__(self, app_id: str = None, app_key: str = None):
        """
        Initialize Mathpix OCR client.
        
        Args:
            app_id: Mathpix application ID (or from MATHPIX_APP_ID env var)
            app_key: Mathpix application key (or from MATHPIX_APP_KEY env var)
        """
        self.app_id = app_id or os.getenv('MATHPIX_APP_ID')
        self.app_key = app_key or os.getenv('MATHPIX_APP_KEY')
        
        if not self.app_id or not self.app_key:
            raise ValueError(
                "Mathpix credentials required. Set MATHPIX_APP_ID and MATHPIX_APP_KEY "
                "environment variables or pass them as parameters."
            )
        
        self.base_url = "https://api.mathpix.com/v3"
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def extract_text(self, image_data: Union[bytes, str], 
                          formats: List[MathpixFormat] = None,
                          **options) -> MathpixResult:
        """
        Extract mathematical text from image using Mathpix OCR.
        
        Args:
            image_data: Image as bytes or base64 string
            formats: List of output formats to request
            **options: Additional Mathpix API options
            
        Returns:
            MathpixResult with extracted text and metadata
        """
        if not self.session:
            raise RuntimeError("MathpixOCRClient must be used as async context manager")
        
        # Prepare image data
        if isinstance(image_data, bytes):
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        else:
            image_b64 = image_data
        
        # Default formats
        if formats is None:
            formats = [MathpixFormat.TEXT, MathpixFormat.LATEX]
        
        # Prepare request
        headers = {
            'app_id': self.app_id,
            'app_key': self.app_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'src': f"data:image/jpeg;base64,{image_b64}",
            'formats': [fmt.value for fmt in formats],
            'data_options': {
                'include_asciimath': True,
                'include_latex': True,
                'include_mathml': False,
                'include_table_html': True,
                'include_svg': False
            },
            **options
        }
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.session.post(
                f"{self.base_url}/text",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                
                processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    return self._parse_mathpix_response(result, processing_time)
                else:
                    error_text = await response.text()
                    return MathpixResult(
                        text="",
                        latex="",
                        confidence=0.0,
                        processing_time_ms=processing_time,
                        detected_alphabets=[],
                        is_printed=False,
                        is_handwritten=False,
                        auto_rotate_confidence=0.0,
                        auto_rotate_degrees=0,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            return MathpixResult(
                text="",
                latex="",
                confidence=0.0,
                processing_time_ms=30000,
                detected_alphabets=[],
                is_printed=False,
                is_handwritten=False,
                auto_rotate_confidence=0.0,
                auto_rotate_degrees=0,
                error="Request timeout"
            )
        except Exception as e:
            return MathpixResult(
                text="",
                latex="",
                confidence=0.0,
                processing_time_ms=0,
                detected_alphabets=[],
                is_printed=False,
                is_handwritten=False,
                auto_rotate_confidence=0.0,
                auto_rotate_degrees=0,
                error=str(e)
            )
    
    def _parse_mathpix_response(self, response: Dict[str, Any], 
                               processing_time: float) -> MathpixResult:
        """Parse Mathpix API response into MathpixResult."""
        
        # Extract text and LaTeX
        text = response.get('text', '')
        latex = response.get('latex_styled', '')
        
        # Extract metadata
        confidence = response.get('confidence', 0.0)
        detected_alphabets = response.get('detected_alphabets', [])
        is_printed = response.get('is_printed', False)
        is_handwritten = response.get('is_handwritten', False)
        
        # Auto-rotation info
        auto_rotate_confidence = response.get('auto_rotate_confidence', 0.0)
        auto_rotate_degrees = response.get('auto_rotate_degrees', 0)
        
        return MathpixResult(
            text=text,
            latex=latex,
            confidence=confidence,
            processing_time_ms=processing_time,
            detected_alphabets=detected_alphabets,
            is_printed=is_printed,
            is_handwritten=is_handwritten,
            auto_rotate_confidence=auto_rotate_confidence,
            auto_rotate_degrees=auto_rotate_degrees
        )

class MathpixTextProcessor:
    """Processes Mathpix OCR results for mathematical content."""
    
    @staticmethod
    def clean_mathematical_text(text: str) -> str:
        """Clean and normalize mathematical text from Mathpix."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common mathematical symbols
        replacements = {
            r'\\cdot': '*',
            r'\\times': '*',
            r'\\div': '/',
            r'\\frac': '',  # Will be handled separately
            r'\\left(': '(',
            r'\\right)': ')',
            r'\\left[': '[',
            r'\\right]': ']',
            r'\\left\\{': '{',
            r'\\right\\}': '}',
        }
        
        for latex_sym, replacement in replacements.items():
            text = text.replace(latex_sym, replacement)
        
        return text
    
    @staticmethod
    def extract_equations(text: str) -> List[str]:
        """Extract mathematical equations from text."""
        equations = []
        
        # Find expressions with equals signs
        equation_pattern = r'[^.!?]*=[^.!?]*'
        matches = re.findall(equation_pattern, text)
        
        for match in matches:
            equation = match.strip()
            if equation and len(equation) > 3:  # Filter out very short matches
                equations.append(equation)
        
        return equations
    
    @staticmethod
    def parse_fractions(latex: str) -> List[Dict[str, str]]:
        """Parse fraction expressions from LaTeX."""
        fractions = []
        
        # Find \frac{numerator}{denominator} patterns
        frac_pattern = r'\\frac\{([^}]+)\}\{([^}]+)\}'
        matches = re.findall(frac_pattern, latex)
        
        for numerator, denominator in matches:
            fractions.append({
                'numerator': numerator.strip(),
                'denominator': denominator.strip(),
                'expression': f"{numerator}/{denominator}"
            })
        
        return fractions
    
    @staticmethod
    def identify_problem_types(text: str, latex: str) -> List[str]:
        """Identify types of mathematical problems in the text."""
        problem_types = []
        
        # Check for different problem types
        if 'solve' in text.lower() or '=' in text:
            problem_types.append('equation_solving')
        
        if 'factor' in text.lower():
            problem_types.append('factoring')
        
        if any(op in text for op in ['+', '-', '*', '/', '×', '÷']):
            problem_types.append('arithmetic')
        
        if '\\frac' in latex or '/' in text:
            problem_types.append('fractions')
        
        if any(func in text.lower() for func in ['sin', 'cos', 'tan', 'log']):
            problem_types.append('trigonometry')
        
        if re.search(r'x\^?\d+|x\*\*\d+', text):
            problem_types.append('algebra')
        
        if '%' in text or 'percent' in text.lower():
            problem_types.append('percentage')
        
        return problem_types if problem_types else ['general_math']

# Convenience functions
async def extract_math_from_image(image_data: Union[bytes, str], 
                                 app_id: str = None, 
                                 app_key: str = None) -> MathpixResult:
    """
    Convenience function to extract mathematical content from image.
    
    Args:
        image_data: Image as bytes or base64 string
        app_id: Mathpix application ID
        app_key: Mathpix application key
        
    Returns:
        MathpixResult with extracted mathematical content
    """
    async with MathpixOCRClient(app_id, app_key) as client:
        return await client.extract_text(image_data) 