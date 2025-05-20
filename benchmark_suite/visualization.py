"""
Benchmark Suite Visualization

This module provides visualization functions for benchmark results.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

from benchmark_suite.utils import generate_plots, generate_report


logger = logging.getLogger("benchmark_suite.visualization")


def visualize_benchmark_results(results_dir: str) -> Optional[str]:
    """
    Generate visualizations and report from benchmark results.
    
    Args:
        results_dir: Directory with benchmark result files
        
    Returns:
        Path to generated report
    """
    if not os.path.exists(results_dir):
        logger.error(f"Results directory does not exist: {results_dir}")
        return None
    
    # Create plots directory
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Check if there are result files to process
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "config.json"]
    
    if not result_files:
        logger.warning(f"No benchmark result files found in {results_dir}")
        return None
    
    # Generate plots
    logger.info("Generating benchmark plots...")
    plot_files = generate_plots(results_dir, plots_dir)
    
    if plot_files:
        logger.info(f"Generated {len(plot_files)} plots")
    else:
        logger.warning("No plots generated, possibly due to insufficient data")
    
    # Generate HTML report
    logger.info("Generating HTML report...")
    report_path = generate_report(results_dir)
    
    if report_path:
        logger.info(f"Report generated: {report_path}")
    else:
        logger.warning("Failed to generate report")
    
    return report_path 