#!/usr/bin/env python3
"""
OCR Configuration System for Math Recognition Benchmarking

This module provides a comprehensive configuration system for switching between
different OCR models and evaluation strategies in the math recognition system.
"""

import json
import os
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Silently continue if dotenv is not available

class OCRProvider(Enum):
    """Available OCR providers."""
    MATHPIX = "mathpix"
    GPT5_VISION = "gpt5_vision"
    GOT_OCR2 = "got_ocr2"
    UNIMERNET = "unimernet"
    MONKEY_OCR = "monkey_ocr"
    PP_FORMULA = "pp_formula"
    TROCR_GEOMETRY = "trocr_geometry"
    OPEN_OCR = "open_ocr"

class ProcessingStrategy(Enum):
    """Processing strategies for combining OCR models."""
    OCR_FIRST = "ocr_first"              # Mathpix first, then GPT-5
    VISION_FIRST = "vision_first"        # GPT-5 first, then Mathpix
    PARALLEL = "parallel"                # Both simultaneously
    OCR_ONLY = "ocr_only"               # Only Mathpix OCR
    VISION_ONLY = "vision_only"         # Only GPT-5 Vision
    GEOMETRY_FIRST = "geometry_first"    # Geometry OCR first, then others
    GEOMETRY_ONLY = "geometry_only"     # Only advanced geometry OCR
    HYBRID_BEST = "hybrid_best"         # Intelligent provider selection

@dataclass
class OCRModelConfig:
    """Configuration for a single OCR model."""
    provider: OCRProvider
    model_name: str
    api_key_env: Optional[str] = None
    model_path: Optional[str] = None
    device: str = "auto"
    max_retries: int = 3
    timeout_seconds: int = 30
    confidence_threshold: float = 0.5
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    name: str
    description: str
    primary_ocr: OCRModelConfig
    fallback_ocr: Optional[OCRModelConfig] = None
    processing_strategy: ProcessingStrategy = ProcessingStrategy.HYBRID_BEST
    evaluation_metrics: List[str] = None
    dataset_types: List[str] = None
    max_samples: Optional[int] = None
    parallel_processing: bool = True
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "ocr_accuracy", "question_detection", "answer_extraction", 
                "processing_time", "confidence_score"
            ]
        if self.dataset_types is None:
            self.dataset_types = ["math_expressions", "geometry", "test_papers"]

class OCRConfigManager:
    """Manager for OCR configurations and benchmark setups."""
    
    def __init__(self, config_dir: str = None):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent / "configs"
        self.config_dir.mkdir(exist_ok=True)
        
        # Load predefined configurations
        self._load_predefined_configs()
    
    def _load_predefined_configs(self):
        """Load predefined OCR model configurations."""
        self.ocr_models = {
            "mathpix_standard": OCRModelConfig(
                provider=OCRProvider.MATHPIX,
                model_name="mathpix-standard",
                api_key_env="MATHPIX_APP_KEY",
                custom_params={
                    "formats": ["text", "latex_simplified"],
                    "include_asciimath": True,
                    "include_table_html": True
                }
            ),
            
            "gpt5_vision": OCRModelConfig(
                provider=OCRProvider.GPT5_VISION,
                model_name="gpt-5",
                api_key_env="OPENAI_API_KEY",
                custom_params={
                    "max_completion_tokens": 4000,
                    "temperature": 0.1,
                    "detail": "high"
                }
            ),
            
            "gpt5_vision_fast": OCRModelConfig(
                provider=OCRProvider.GPT5_VISION,
                model_name="gpt-5",
                api_key_env="OPENAI_API_KEY",
                custom_params={
                    "max_completion_tokens": 1000,  # Reduced for speed
                    "temperature": 0.0,             # More deterministic
                    "detail": "low"                 # Faster processing
                }
            ),
            
            "gpt5_vision_faster": OCRModelConfig(
                provider=OCRProvider.GPT5_VISION,
                model_name="gpt-5",
                api_key_env="OPENAI_API_KEY",
                custom_params={
                    "max_completion_tokens": 500,   # Ultra reduced for max speed
                    "temperature": 0.0,             # Deterministic
                    "detail": "auto"                # Automatic detail level
                }
            ),
            
            "got_ocr2": OCRModelConfig(
                provider=OCRProvider.GOT_OCR2,
                model_name="stepfun-ai/GOT-OCR2_0",
                model_path="stepfun-ai/GOT-OCR2_0",
                device="auto",
                custom_params={
                    "ocr_type": "format",
                    "trust_remote_code": True
                }
            ),
            
            "unimernet": OCRModelConfig(
                provider=OCRProvider.UNIMERNET,
                model_name="unimernet",
                model_path="wanderkid/unimernet",
                device="auto",
                custom_params={
                    "precision": "fp16",
                    "batch_size": 1
                }
            ),
            
            "monkey_ocr": OCRModelConfig(
                provider=OCRProvider.MONKEY_OCR,
                model_name="monkey-ocr",
                model_path="echo840/Monkey-Chat",
                device="auto",
                custom_params={
                    "temperature": 0.1,
                    "max_new_tokens": 2048
                }
            )
        }
        
        # Predefined benchmark configurations
        self.benchmark_configs = {
            "mathpix_gpt5_hybrid": BenchmarkConfig(
                name="Mathpix + GPT-5 Hybrid",
                description="Industry standard: Mathpix OCR with GPT-5 Vision fallback",
                primary_ocr=self.ocr_models["mathpix_standard"],
                fallback_ocr=self.ocr_models["gpt5_vision"],
                processing_strategy=ProcessingStrategy.OCR_FIRST
            ),
            
            "gpt-5": BenchmarkConfig(
                name="GPT-5 Vision Only",
                description="Pure GPT-5 Vision processing for comprehensive analysis",
                primary_ocr=self.ocr_models["gpt5_vision"],
                processing_strategy=ProcessingStrategy.VISION_ONLY
            ),
            
            "gpt-5-fast": BenchmarkConfig(
                name="GPT-5 Vision Fast",
                description="Optimized GPT-5 Vision for speed with reduced detail processing",
                primary_ocr=self.ocr_models["gpt5_vision_fast"],
                processing_strategy=ProcessingStrategy.VISION_ONLY,
                evaluation_metrics=[
                    "ocr_accuracy", "processing_time"
                ]
            ),
            
            "gpt-5-faster": BenchmarkConfig(
                name="GPT-5 Vision Faster",
                description="Ultra-optimized GPT-5 Vision for maximum speed (500 tokens, auto detail)",
                primary_ocr=self.ocr_models["gpt5_vision_faster"],
                processing_strategy=ProcessingStrategy.VISION_ONLY,
                evaluation_metrics=[
                    "ocr_accuracy", "processing_time"
                ]
            ),
            
            "lightning_fast": BenchmarkConfig(
                name="Lightning Fast OCR",
                description="Ultra-fast OCR using GOT-OCR2.0 only for maximum speed",
                primary_ocr=self.ocr_models["got_ocr2"],
                processing_strategy=ProcessingStrategy.GEOMETRY_FIRST,
                evaluation_metrics=[
                    "ocr_accuracy", "processing_time"
                ]
            ),
            
            "geometry_specialist": BenchmarkConfig(
                name="Geometry OCR Specialist",
                description="GOT-OCR2.0 optimized for geometric content",
                primary_ocr=self.ocr_models["got_ocr2"],
                fallback_ocr=self.ocr_models["gpt5_vision"],
                processing_strategy=ProcessingStrategy.GEOMETRY_FIRST,
                dataset_types=["geometry", "diagrams", "constructions"]
            ),
            
            "math_expression_expert": BenchmarkConfig(
                name="Math Expression Expert",
                description="UniMERNet specialized for mathematical expressions",
                primary_ocr=self.ocr_models["unimernet"],
                fallback_ocr=self.ocr_models["mathpix_standard"],
                processing_strategy=ProcessingStrategy.OCR_FIRST,
                dataset_types=["math_expressions", "equations", "formulas"]
            ),
            
            "comprehensive_parallel": BenchmarkConfig(
                name="Comprehensive Parallel",
                description="Multiple OCR models running in parallel for maximum accuracy",
                primary_ocr=self.ocr_models["mathpix_standard"],
                fallback_ocr=self.ocr_models["gpt5_vision"],
                processing_strategy=ProcessingStrategy.PARALLEL,
                evaluation_metrics=[
                    "ocr_accuracy", "question_detection", "answer_extraction",
                    "processing_time", "confidence_score", "cost_efficiency"
                ]
            ),
            
            "cost_optimized": BenchmarkConfig(
                name="Cost Optimized",
                description="Open source models prioritizing cost efficiency",
                primary_ocr=self.ocr_models["got_ocr2"],
                fallback_ocr=self.ocr_models["unimernet"],
                processing_strategy=ProcessingStrategy.GEOMETRY_FIRST,
                evaluation_metrics=[
                    "ocr_accuracy", "processing_time", "cost_efficiency"
                ]
            )
        }
    
    def get_config(self, config_name: str) -> BenchmarkConfig:
        """Get a benchmark configuration by name."""
        if config_name not in self.benchmark_configs:
            available = list(self.benchmark_configs.keys())
            raise ValueError(f"Config '{config_name}' not found. Available: {available}")
        return self.benchmark_configs[config_name]
    
    def list_configs(self) -> Dict[str, str]:
        """List all available configurations with descriptions."""
        return {
            name: config.description 
            for name, config in self.benchmark_configs.items()
        }
    
    def create_custom_config(self, 
                           name: str,
                           primary_provider: str,
                           fallback_provider: str = None,
                           strategy: str = "hybrid_best",
                           **kwargs) -> BenchmarkConfig:
        """Create a custom benchmark configuration."""
        
        if primary_provider not in self.ocr_models:
            available = list(self.ocr_models.keys())
            raise ValueError(f"Primary provider '{primary_provider}' not found. Available: {available}")
        
        primary_ocr = self.ocr_models[primary_provider]
        fallback_ocr = self.ocr_models.get(fallback_provider) if fallback_provider else None
        
        try:
            processing_strategy = ProcessingStrategy(strategy)
        except ValueError:
            available = [s.value for s in ProcessingStrategy]
            raise ValueError(f"Strategy '{strategy}' not found. Available: {available}")
        
        config = BenchmarkConfig(
            name=name,
            description=kwargs.get("description", f"Custom config: {primary_provider}"),
            primary_ocr=primary_ocr,
            fallback_ocr=fallback_ocr,
            processing_strategy=processing_strategy,
            **{k: v for k, v in kwargs.items() if k != "description"}
        )
        
        self.benchmark_configs[name] = config
        return config
    
    def save_config(self, config_name: str, config: BenchmarkConfig):
        """Save a configuration to file."""
        config_file = self.config_dir / f"{config_name}.json"
        
        # Convert to serializable format
        config_dict = asdict(config)
        # Handle enums
        config_dict["primary_ocr"]["provider"] = config.primary_ocr.provider.value
        config_dict["processing_strategy"] = config.processing_strategy.value
        if config.fallback_ocr:
            config_dict["fallback_ocr"]["provider"] = config.fallback_ocr.provider.value
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_config(self, config_name: str) -> BenchmarkConfig:
        """Load a configuration from file."""
        config_file = self.config_dir / f"{config_name}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Handle enums
        config_dict["primary_ocr"]["provider"] = OCRProvider(config_dict["primary_ocr"]["provider"])
        config_dict["processing_strategy"] = ProcessingStrategy(config_dict["processing_strategy"])
        if config_dict.get("fallback_ocr"):
            config_dict["fallback_ocr"]["provider"] = OCRProvider(config_dict["fallback_ocr"]["provider"])
        
        # Reconstruct objects
        primary_ocr = OCRModelConfig(**config_dict["primary_ocr"])
        fallback_ocr = OCRModelConfig(**config_dict["fallback_ocr"]) if config_dict.get("fallback_ocr") else None
        
        config_dict["primary_ocr"] = primary_ocr
        config_dict["fallback_ocr"] = fallback_ocr
        
        return BenchmarkConfig(**config_dict)
    
    def validate_config(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Validate a configuration and return status."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "missing_requirements": []
        }
        
        # Check API keys
        if config.primary_ocr.api_key_env:
            if not os.getenv(config.primary_ocr.api_key_env):
                validation_result["missing_requirements"].append(
                    f"Environment variable {config.primary_ocr.api_key_env} not set"
                )
        
        if config.fallback_ocr and config.fallback_ocr.api_key_env:
            if not os.getenv(config.fallback_ocr.api_key_env):
                validation_result["warnings"].append(
                    f"Fallback OCR env var {config.fallback_ocr.api_key_env} not set"
                )
        
        # Check model paths for local models
        for ocr_config in [config.primary_ocr, config.fallback_ocr]:
            if ocr_config and ocr_config.model_path:
                # This would require checking if model is downloadable/accessible
                validation_result["warnings"].append(
                    f"Model path {ocr_config.model_path} - verify accessibility"
                )
        
        # Check strategy compatibility
        if (config.processing_strategy in [ProcessingStrategy.GEOMETRY_FIRST, ProcessingStrategy.GEOMETRY_ONLY] and
            config.primary_ocr.provider not in [OCRProvider.GOT_OCR2, OCRProvider.UNIMERNET]):
            validation_result["warnings"].append(
                "Geometry strategy selected but primary OCR is not geometry-optimized"
            )
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result
    
    def get_provider_recommendations(self, content_type: str) -> Dict[str, str]:
        """Get OCR provider recommendations for different content types."""
        recommendations = {
            "basic_math": {
                "primary": "mathpix_standard",
                "fallback": "gpt5_vision",
                "reason": "Mathpix excels at standard mathematical expressions"
            },
            "geometry": {
                "primary": "got_ocr2", 
                "fallback": "gpt5_vision",
                "reason": "GOT-OCR2.0 specialized for geometric shapes and diagrams"
            },
            "handwritten": {
                "primary": "gpt5_vision",
                "fallback": "got_ocr2",
                "reason": "GPT-5 Vision handles handwritten content well"
            },
            "complex_expressions": {
                "primary": "unimernet",
                "fallback": "mathpix_standard", 
                "reason": "UniMERNet designed for complex mathematical expressions"
            },
            "mixed_content": {
                "primary": "gpt5_vision",
                "fallback": "mathpix_standard",
                "reason": "GPT-5 Vision handles mixed text and math content"
            },
            "cost_sensitive": {
                "primary": "got_ocr2",
                "fallback": "unimernet",
                "reason": "Open source models reduce API costs"
            }
        }
        
        return recommendations.get(content_type, {
            "primary": "mathpix_standard",
            "fallback": "gpt5_vision", 
            "reason": "Default reliable combination"
        })

# Global configuration manager instance
config_manager = OCRConfigManager()

def get_config_manager() -> OCRConfigManager:
    """Get the global configuration manager instance."""
    return config_manager 