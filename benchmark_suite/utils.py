"""
Benchmark Suite Utilities

This module provides utility functions for the benchmark suite.
"""

import os
import time
import json
import logging
import random
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with a consistent format.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handler if needed
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def get_process_memory_usage() -> float:
    """
    Get current process memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except ImportError:
        return 0.0


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics from a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with statistics
    """
    if not values:
        return {}
        
    stats = {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": sorted(values)[len(values) // 2],
    }
    
    # Standard deviation
    if len(values) > 1:
        mean = stats["mean"]
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        stats["stddev"] = variance ** 0.5
    else:
        stats["stddev"] = 0
    
    # Percentiles
    sorted_values = sorted(values)
    stats["p90"] = sorted_values[int(len(values) * 0.9)]
    stats["p95"] = sorted_values[int(len(values) * 0.95)]
    stats["p99"] = sorted_values[int(len(values) * 0.99)]
    
    return stats


def generate_plots(results_dir: str, output_dir: Optional[str] = None) -> List[str]:
    """
    Generate plots from benchmark results.
    
    Args:
        results_dir: Directory with benchmark result files
        output_dir: Optional directory to save plots
        
    Returns:
        List of saved plot file paths
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json") and f != "config.json"]
    
    if not result_files:
        return []
    
    results_by_type = {}
    
    for filename in result_files:
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'r') as f:
            result_data = json.load(f)
            
        benchmark_type = result_data.get("benchmark_type")
        
        if benchmark_type not in results_by_type:
            results_by_type[benchmark_type] = []
            
        results_by_type[benchmark_type].append(result_data)
    
    # Generate plots
    plot_files = []
    
    # 1. Latency comparison plot
    latency_plot = _generate_latency_comparison_plot(results_by_type, output_dir)
    if latency_plot:
        plot_files.append(latency_plot)
    
    # 2. Throughput comparison plot
    throughput_plot = _generate_throughput_comparison_plot(results_by_type, output_dir)
    if throughput_plot:
        plot_files.append(throughput_plot)
    
    # 3. Memory usage plot
    memory_plot = _generate_memory_usage_plot(results_by_type, output_dir)
    if memory_plot:
        plot_files.append(memory_plot)
    
    # 4. Hybrid system scaling plots
    hybrid_plots = _generate_hybrid_scaling_plots(results_by_type.get("hybrid", []), output_dir)
    plot_files.extend(hybrid_plots)
    
    return plot_files


def _generate_latency_comparison_plot(results_by_type: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Optional[str]:
    """
    Generate latency comparison plot.
    
    Args:
        results_by_type: Dictionary of results grouped by benchmark type
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot or None
    """
    plt.figure(figsize=(10, 6))
    
    benchmark_names = []
    latencies = []
    colors = []
    
    # Color mapping for benchmark types
    color_map = {
        "graph": "blue",
        "vector": "green",
        "hybrid": "red",
        "other": "gray"
    }
    
    # Extract latency data
    for benchmark_type, results in results_by_type.items():
        for result in results:
            name = result.get("benchmark_name", "Unknown")
            stats = result.get("stats", {}).get("latency", {})
            
            if stats and "mean_ms" in stats:
                benchmark_names.append(name)
                latencies.append(stats["mean_ms"])
                colors.append(color_map.get(benchmark_type, "gray"))
    
    if not benchmark_names:
        return None
    
    # Create bar chart
    plt.bar(benchmark_names, latencies, color=colors)
    plt.ylabel("Mean Latency (ms)")
    plt.title("Benchmark Latency Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def _generate_throughput_comparison_plot(results_by_type: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Optional[str]:
    """
    Generate throughput comparison plot.
    
    Args:
        results_by_type: Dictionary of results grouped by benchmark type
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot or None
    """
    plt.figure(figsize=(10, 6))
    
    benchmark_names = []
    throughputs = []
    colors = []
    
    # Color mapping for benchmark types
    color_map = {
        "graph": "blue",
        "vector": "green",
        "hybrid": "red",
        "other": "gray"
    }
    
    # Extract throughput data
    for benchmark_type, results in results_by_type.items():
        for result in results:
            name = result.get("benchmark_name", "Unknown")
            stats = result.get("stats", {})
            
            if "throughput_ops" in stats:
                benchmark_names.append(name)
                throughputs.append(stats["throughput_ops"])
                colors.append(color_map.get(benchmark_type, "gray"))
    
    if not benchmark_names:
        return None
    
    # Create bar chart
    plt.bar(benchmark_names, throughputs, color=colors)
    plt.ylabel("Throughput (ops/sec)")
    plt.title("Benchmark Throughput Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def _generate_memory_usage_plot(results_by_type: Dict[str, List[Dict[str, Any]]], output_dir: str) -> Optional[str]:
    """
    Generate memory usage comparison plot.
    
    Args:
        results_by_type: Dictionary of results grouped by benchmark type
        output_dir: Directory to save the plot
        
    Returns:
        Path to saved plot or None
    """
    plt.figure(figsize=(10, 6))
    
    benchmark_names = []
    memory_usage = []
    colors = []
    
    # Color mapping for benchmark types
    color_map = {
        "graph": "blue",
        "vector": "green",
        "hybrid": "red",
        "other": "gray"
    }
    
    # Extract memory usage data
    for benchmark_type, results in results_by_type.items():
        for result in results:
            name = result.get("benchmark_name", "Unknown")
            stats = result.get("stats", {})
            
            if "memory_mb" in stats:
                benchmark_names.append(name)
                memory_usage.append(stats["memory_mb"])
                colors.append(color_map.get(benchmark_type, "gray"))
    
    if not benchmark_names:
        return None
    
    # Create bar chart
    plt.bar(benchmark_names, memory_usage, color=colors)
    plt.ylabel("Memory Usage (MB)")
    plt.title("Benchmark Memory Usage Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "memory_usage_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def _generate_hybrid_scaling_plots(hybrid_results: List[Dict[str, Any]], output_dir: str) -> List[str]:
    """
    Generate scaling plots for hybrid benchmarks.
    
    Args:
        hybrid_results: List of hybrid benchmark results
        output_dir: Directory to save plots
        
    Returns:
        List of saved plot file paths
    """
    plot_files = []
    
    for result in hybrid_results:
        benchmark_name = result.get("benchmark_name", "unknown")
        
        if "benchmark_hybrid_system_scalability" in benchmark_name:
            # Extract scaling data
            stats = result.get("stats", {})
            custom = stats.get("custom", {})
            
            doc_times = custom.get("doc_processing_times", [])
            search_times = custom.get("search_times", [])
            entity_counts = custom.get("entity_counts", [])
            relation_counts = custom.get("relation_counts", [])
            memory_usage = custom.get("memory_usage", [])
            
            # 1. Document processing time vs. data size
            if doc_times:
                plt.figure(figsize=(8, 5))
                sizes, times = zip(*doc_times)
                plt.plot(sizes, times, 'o-', color='blue')
                plt.xlabel("Number of Documents")
                plt.ylabel("Avg. Processing Time (ms)")
                plt.title("Document Processing Time vs. Data Size")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_path = os.path.join(output_dir, f"{benchmark_name}_doc_processing.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)
            
            # 2. Search time vs. data size
            if search_times:
                plt.figure(figsize=(8, 5))
                sizes, times = zip(*search_times)
                plt.plot(sizes, times, 'o-', color='green')
                plt.xlabel("Number of Documents")
                plt.ylabel("Avg. Search Time (ms)")
                plt.title("Search Time vs. Data Size")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_path = os.path.join(output_dir, f"{benchmark_name}_search_time.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)
            
            # 3. Entity and relation counts vs. data size
            if entity_counts and relation_counts:
                plt.figure(figsize=(8, 5))
                sizes1, counts1 = zip(*entity_counts)
                sizes2, counts2 = zip(*relation_counts)
                
                plt.plot(sizes1, counts1, 'o-', color='blue', label='Entities')
                plt.plot(sizes2, counts2, 'o-', color='red', label='Relations')
                plt.xlabel("Number of Documents")
                plt.ylabel("Count")
                plt.title("Entity and Relation Counts vs. Data Size")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_path = os.path.join(output_dir, f"{benchmark_name}_entity_relation_counts.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)
            
            # 4. Memory usage vs. data size
            if memory_usage:
                plt.figure(figsize=(8, 5))
                sizes, usage = zip(*memory_usage)
                plt.plot(sizes, usage, 'o-', color='purple')
                plt.xlabel("Number of Documents")
                plt.ylabel("Memory Usage (MB)")
                plt.title("Memory Usage vs. Data Size")
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plot_path = os.path.join(output_dir, f"{benchmark_name}_memory_usage.png")
                plt.savefig(plot_path)
                plt.close()
                plot_files.append(plot_path)
    
    return plot_files


def generate_report(results_dir: str, output_path: Optional[str] = None) -> str:
    """
    Generate a comprehensive HTML report from benchmark results.
    
    Args:
        results_dir: Directory with benchmark result files
        output_path: Optional path for the HTML report
        
    Returns:
        Path to the generated report
    """
    if output_path is None:
        output_path = os.path.join(results_dir, "benchmark_report.html")
    
    # Load result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".json")]
    
    if not result_files:
        return ""
    
    # Load config
    config = {}
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Load results
    results = []
    for filename in result_files:
        if filename == "config.json":
            continue
            
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    
    # Group results by type
    results_by_type = {}
    for result in results:
        benchmark_type = result.get("benchmark_type", "other")
        
        if benchmark_type not in results_by_type:
            results_by_type[benchmark_type] = []
            
        results_by_type[benchmark_type].append(result)
    
    # Generate HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px; }}
            .chart {{ width: 100%; max-width: 800px; margin: 20px 0; }}
            .section {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Report</h1>
        <div class="summary">
            <h2>Summary</h2>
            <p>Benchmark run at: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p>Total benchmarks: {len(results)}</p>
    """
    
    # Add config info
    if config:
        html += f"""
            <h3>Configuration</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
        """
        
        for key, value in config.items():
            if isinstance(value, dict):
                continue
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            
        html += "</table>"
    
    html += "</div>"
    
    # Add sections for each benchmark type
    for benchmark_type, type_results in results_by_type.items():
        html += f"""
        <div class="section">
            <h2>{benchmark_type.capitalize()} Benchmarks</h2>
            <table>
                <tr>
                    <th>Benchmark</th>
                    <th>Latency (ms)</th>
                    <th>Throughput (ops/s)</th>
                    <th>Memory (MB)</th>
                </tr>
        """
        
        for result in type_results:
            name = result.get("benchmark_name", "Unknown")
            stats = result.get("stats", {})
            
            latency = stats.get("latency", {}).get("mean_ms", "N/A")
            throughput = stats.get("throughput_ops", "N/A")
            memory = stats.get("memory_mb", "N/A")
            
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{latency:.2f if isinstance(latency, (int, float)) else latency}</td>
                    <td>{throughput:.2f if isinstance(throughput, (int, float)) else throughput}</td>
                    <td>{memory:.2f if isinstance(memory, (int, float)) else memory}</td>
                </tr>
            """
        
        html += "</table>"
        
        # Add charts
        chart_dir = os.path.join(results_dir, "plots")
        if os.path.exists(chart_dir):
            charts = [f for f in os.listdir(chart_dir) if benchmark_type in f.lower() and f.endswith(".png")]
            
            if charts:
                html += "<h3>Charts</h3>"
                
                for chart in charts:
                    chart_path = os.path.join("plots", chart)
                    html += f'<img class="chart" src="{chart_path}" alt="{chart}">'
        
        html += "</div>"
    
    # Add comparison charts
    comparison_charts = ["latency_comparison.png", "throughput_comparison.png", "memory_usage_comparison.png"]
    chart_dir = os.path.join(results_dir, "plots")
    
    if os.path.exists(chart_dir):
        existing_charts = [c for c in comparison_charts if os.path.exists(os.path.join(chart_dir, c))]
        
        if existing_charts:
            html += """
            <div class="section">
                <h2>Comparison Charts</h2>
            """
            
            for chart in existing_charts:
                chart_path = os.path.join("plots", chart)
                html += f'<img class="chart" src="{chart_path}" alt="{chart}">'
                
            html += "</div>"
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path 