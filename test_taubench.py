#!/usr/bin/env python
"""
Test TauBench Implementation

This script tests our minimal TauBench implementation.
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our TauBench implementation
from agents.eval.taubench.taubench_minimal import load_dataset, Evaluator, calculate_metrics

async def main():
    """Test the minimal TauBench implementation."""
    logger.info("Testing minimal TauBench implementation...")
    
    # Test loading datasets
    categories = ["reasoning", "planning", "tool_use", "qa", "code_generation"]
    
    for category in categories:
        logger.info(f"Loading {category} dataset...")
        tasks = load_dataset(category)
        logger.info(f"Loaded {len(tasks)} tasks for {category}")
        
        if tasks:
            # Print sample task
            logger.info(f"Sample task: {json.dumps(tasks[0], indent=2)}")
            
            # Test evaluation
            evaluator = Evaluator()
            sample_response = "The answer is 42. This is a test response that contains some keywords from the reference."
            reference = tasks[0].get("reference", "")
            criteria = tasks[0].get("evaluation_criteria", {})
            
            logger.info(f"Reference: {reference}")
            metrics = evaluator.evaluate(sample_response, reference, criteria)
            logger.info(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")
    
    logger.info("TauBench minimal implementation test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 