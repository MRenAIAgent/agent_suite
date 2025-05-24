#!/usr/bin/env python3
"""
RAG Benchmark Report Generator

This script generates a detailed report on RAG system performance,
including visualizations of benchmark results.
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure plot style
plt.style.use('ggplot')
sns.set(font_scale=1.2)

class RagBenchmarkReport:
    """Generate comprehensive report from RAG benchmark results"""
    
    def __init__(self, results_file: str, output_dir: str = "benchmark_reports"):
        """
        Initialize the report generator.
        
        Args:
            results_file: Path to benchmark results JSON file
            output_dir: Directory to save the generated report
        """
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark results
        with open(self.results_file, 'r') as f:
            self.benchmark_data = json.load(f)
        
        # Extract components
        self.config = self.benchmark_data.get("config", {})
        self.metrics = self.benchmark_data.get("metrics", {})
        self.results = self.benchmark_data.get("results", [])
        
        # Convert results to DataFrame for easier analysis
        self.results_df = self._create_results_dataframe()
    
    def _create_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Returns:
            DataFrame of benchmark results
        """
        data = []
        for result in self.results:
            data.append({
                "question": result.get("question", ""),
                "context_relevance": result.get("metrics", {}).get("context_relevance", 0),
                "answer_relevance": result.get("metrics", {}).get("answer_relevance", 0),
                "factual_correctness": result.get("metrics", {}).get("factual_correctness", 0),
                "retrieval_latency": result.get("metrics", {}).get("retrieval_latency", 0),
                "generation_latency": result.get("metrics", {}).get("generation_latency", 0),
                "total_latency": result.get("metrics", {}).get("total_latency", 0),
                "token_usage": result.get("metrics", {}).get("token_usage", {}).get("total", 0),
                "context_length": sum(len(ctx.split()) for ctx in result.get("retrieved_context", [])),
                "answer_length": len(result.get("answer", "").split())
            })
        
        return pd.DataFrame(data)
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Returns:
            Path to the generated report
        """
        # Create report components
        self._generate_summary_metrics_chart()
        self._generate_latency_distribution_chart()
        self._generate_correlation_heatmap()
        self._generate_per_query_performance()
        self._generate_system_performance_radar()
        
        # Create HTML report
        report_path = self._generate_html_report()
        
        return str(report_path)
    
    def _generate_summary_metrics_chart(self) -> None:
        """Generate summary metrics bar chart"""
        plt.figure(figsize=(10, 6))
        
        # Extract key metrics
        metrics = [
            ("Retrieval Accuracy", self.metrics.get("retrieval_accuracy", 0)),
            ("Answer Relevance", self.metrics.get("answer_relevance", 0)),
            ("Factual Correctness", self.metrics.get("factual_correctness", 0)),
            ("Context Relevance", self.metrics.get("context_relevance", 0)),
            ("Summary Score", self.metrics.get("summary_score", 0))
        ]
        
        labels, values = zip(*metrics)
        
        # Create bar chart
        bars = plt.bar(labels, values, color=sns.color_palette("muted"))
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.ylim(0, 1.1)
        plt.title("RAG System Performance Metrics")
        plt.ylabel("Score (0-1)")
        plt.tight_layout()
        
        # Save chart
        plt.savefig(self.output_dir / "summary_metrics.png", dpi=300)
        plt.close()
    
    def _generate_latency_distribution_chart(self) -> None:
        """Generate latency distribution chart"""
        plt.figure(figsize=(12, 8))
        
        # Extract latency data
        latency_data = {
            "Retrieval": self.results_df["retrieval_latency"],
            "Generation": self.results_df["generation_latency"],
            "Total": self.results_df["total_latency"]
        }
        
        # Create box plot
        sns.boxplot(data=pd.DataFrame(latency_data))
        
        # Add mean latency values
        for i, (label, values) in enumerate(latency_data.items()):
            mean_val = values.mean()
            plt.text(i, mean_val + 0.01, f"Mean: {mean_val:.3f}s", 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.title("Latency Distribution by Processing Stage")
        plt.ylabel("Time (seconds)")
        plt.tight_layout()
        
        # Save chart
        plt.savefig(self.output_dir / "latency_distribution.png", dpi=300)
        plt.close()
        
        # Create detailed latency percentiles table
        latency_percentiles = {
            "metric": ["retrieval", "generation", "total"],
            "mean": [
                self.metrics.get("latency", {}).get("retrieval", {}).get("mean", 0),
                self.metrics.get("latency", {}).get("generation", {}).get("mean", 0),
                self.metrics.get("latency", {}).get("total", {}).get("mean", 0)
            ],
            "p50": [
                self.metrics.get("latency", {}).get("retrieval", {}).get("p50", 0),
                self.metrics.get("latency", {}).get("generation", {}).get("p50", 0),
                self.metrics.get("latency", {}).get("total", {}).get("p50", 0)
            ],
            "p95": [
                self.metrics.get("latency", {}).get("retrieval", {}).get("p95", 0),
                self.metrics.get("latency", {}).get("generation", {}).get("p95", 0),
                self.metrics.get("latency", {}).get("total", {}).get("p95", 0)
            ],
            "p99": [
                self.metrics.get("latency", {}).get("retrieval", {}).get("p99", 0),
                self.metrics.get("latency", {}).get("generation", {}).get("p99", 0),
                self.metrics.get("latency", {}).get("total", {}).get("p99", 0)
            ]
        }
        
        # Save as CSV
        pd.DataFrame(latency_percentiles).to_csv(self.output_dir / "latency_percentiles.csv", index=False)
    
    def _generate_correlation_heatmap(self) -> None:
        """Generate correlation heatmap between different metrics"""
        plt.figure(figsize=(10, 8))
        
        # Select columns for correlation analysis
        corr_columns = [
            "context_relevance", "answer_relevance", "factual_correctness", 
            "retrieval_latency", "generation_latency", "total_latency", 
            "token_usage", "context_length", "answer_length"
        ]
        
        # Compute correlation matrix
        corr_matrix = self.results_df[corr_columns].corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                   vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        
        plt.title("Correlation Between Performance Metrics")
        plt.tight_layout()
        
        # Save chart
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300)
        plt.close()
    
    def _generate_per_query_performance(self) -> None:
        """Generate per-query performance chart"""
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        queries = self.results_df["question"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
        metrics = ["context_relevance", "answer_relevance", "factual_correctness"]
        
        # Create bar chart
        x = np.arange(len(queries))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1)
            plt.bar(x + offset, self.results_df[metric], width, 
                   label=metric.replace("_", " ").title())
        
        plt.xlabel("Query")
        plt.ylabel("Score (0-1)")
        plt.title("Performance Metrics by Query")
        plt.xticks(x, queries, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        plt.savefig(self.output_dir / "per_query_performance.png", dpi=300)
        plt.close()
    
    def _generate_system_performance_radar(self) -> None:
        """Generate radar chart for system performance"""
        plt.figure(figsize=(8, 8))
        
        # Prepare data
        metrics = [
            "Retrieval Accuracy", 
            "Answer Relevance", 
            "Factual Correctness",
            "Context Relevance", 
            "Latency Efficiency",
            "Token Efficiency"
        ]
        
        # Convert latency to efficiency (1 - normalized latency)
        latency_mean = self.metrics.get("latency", {}).get("total", {}).get("mean", 0)
        max_acceptable_latency = 2.0  # Adjust as needed
        latency_efficiency = max(0, 1 - (latency_mean / max_acceptable_latency))
        
        # Convert token usage to efficiency (1 - normalized token usage)
        token_mean = self.metrics.get("token_usage", {}).get("mean", 0)
        max_acceptable_tokens = 2000  # Adjust as needed
        token_efficiency = max(0, 1 - (token_mean / max_acceptable_tokens))
        
        values = [
            self.metrics.get("retrieval_accuracy", 0),
            self.metrics.get("answer_relevance", 0),
            self.metrics.get("factual_correctness", 0),
            self.metrics.get("context_relevance", 0),
            latency_efficiency,
            token_efficiency
        ]
        
        # Add first value at the end to close the polygon
        values += [values[0]]
        metrics += [metrics[0]]
        
        # Create radar chart
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))  # Close the polygon
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(angles[:-1] * 180/np.pi, metrics[:-1])
        
        plt.ylim(0, 1)
        plt.title("RAG System Performance Profile")
        plt.tight_layout()
        
        # Save chart
        plt.savefig(self.output_dir / "system_performance_radar.png", dpi=300)
        plt.close()
    
    def _generate_html_report(self) -> Path:
        """
        Generate an HTML report with all charts and analysis.
        
        Returns:
            Path to the generated HTML report
        """
        report_path = self.output_dir / "rag_benchmark_report.html"
        
        # Calculate summary statistics
        avg_retrieval_latency = self.metrics.get("latency", {}).get("retrieval", {}).get("mean", 0)
        avg_generation_latency = self.metrics.get("latency", {}).get("generation", {}).get("mean", 0)
        avg_total_latency = self.metrics.get("latency", {}).get("total", {}).get("mean", 0)
        
        retrieval_accuracy = self.metrics.get("retrieval_accuracy", 0)
        answer_relevance = self.metrics.get("answer_relevance", 0)
        factual_correctness = self.metrics.get("factual_correctness", 0)
        context_relevance = self.metrics.get("context_relevance", 0)
        summary_score = self.metrics.get("summary_score", 0)
        
        # Format metrics for display
        def format_metric(value):
            return f"{value:.4f}"
        
        # Generate recommendations based on metrics
        recommendations = []
        
        if retrieval_accuracy < 0.7:
            recommendations.append("Improve retrieval accuracy by enhancing the vector search algorithm or using a better embedding model.")
            
        if answer_relevance < 0.7:
            recommendations.append("Enhance answer relevance by improving the prompt engineering or using a better language model.")
            
        if factual_correctness < 0.7:
            recommendations.append("Improve factual correctness by adding more high-quality documents to the knowledge base.")
            
        if avg_total_latency > 1.0:
            recommendations.append("Optimize system latency by implementing caching or using more efficient storage backends.")
            
        if not recommendations:
            recommendations.append("The RAG system is performing well. Consider expanding its capabilities with more advanced features.")
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Benchmark Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: inline-block;
                    text-align: center;
                    margin: 10px 20px;
                    min-width: 120px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric-label {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    border-bottom: 1px solid #ddd;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .chart {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .chart img {{
                    max-width: 100%;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-radius: 5px;
                }}
                .recommendations {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .recommendations ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <h1>RAG Benchmark Report</h1>
            
            <div class="summary-box">
                <h2>Summary</h2>
                <div class="metric">
                    <div class="metric-value">{format_metric(summary_score)}</div>
                    <div class="metric-label">Overall Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_metric(retrieval_accuracy)}</div>
                    <div class="metric-label">Retrieval Accuracy</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_metric(answer_relevance)}</div>
                    <div class="metric-label">Answer Relevance</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_metric(factual_correctness)}</div>
                    <div class="metric-label">Factual Correctness</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{format_metric(avg_total_latency)}s</div>
                    <div class="metric-label">Avg. Latency</div>
                </div>
            </div>
            
            <h2>System Configuration</h2>
            <table>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
                {"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in self.config.items())}
            </table>
            
            <h2>Performance Metrics</h2>
            
            <div class="chart">
                <h3>Summary Metrics</h3>
                <img src="summary_metrics.png" alt="Summary Metrics">
            </div>
            
            <div class="chart">
                <h3>System Performance Profile</h3>
                <img src="system_performance_radar.png" alt="System Performance Profile">
            </div>
            
            <div class="chart">
                <h3>Latency Distribution</h3>
                <img src="latency_distribution.png" alt="Latency Distribution">
                <p>
                    <strong>Average Latency Breakdown:</strong>
                    Retrieval: {avg_retrieval_latency:.4f}s | 
                    Generation: {avg_generation_latency:.4f}s | 
                    Total: {avg_total_latency:.4f}s
                </p>
            </div>
            
            <div class="chart">
                <h3>Correlation Between Metrics</h3>
                <img src="correlation_heatmap.png" alt="Correlation Heatmap">
            </div>
            
            <div class="chart">
                <h3>Performance by Query</h3>
                <img src="per_query_performance.png" alt="Per Query Performance">
            </div>
            
            <h2>Test Results</h2>
            <table>
                <tr>
                    <th>Query</th>
                    <th>Context Relevance</th>
                    <th>Answer Relevance</th>
                    <th>Factual Correctness</th>
                    <th>Total Latency (s)</th>
                </tr>
                {"".join(f"<tr><td>{row['question'][:50]}...</td><td>{row['context_relevance']:.4f}</td><td>{row['answer_relevance']:.4f}</td><td>{row['factual_correctness']:.4f}</td><td>{row['total_latency']:.4f}</td></tr>" for _, row in self.results_df.iterrows())}
            </table>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                    {"".join(f"<li>{recommendation}</li>" for recommendation in recommendations)}
                </ul>
            </div>
            
            <footer>
                <p>Generated by RAG Benchmark Report Generator</p>
            </footer>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path

def main():
    """Generate a RAG benchmark report from results"""
    parser = argparse.ArgumentParser(description="RAG Benchmark Report Generator")
    parser.add_argument("results_file", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default="benchmark_reports", help="Directory to save the report")
    args = parser.parse_args()
    
    # Generate report
    report_generator = RagBenchmarkReport(
        results_file=args.results_file,
        output_dir=args.output_dir
    )
    
    report_path = report_generator.generate_report()
    print(f"Benchmark report generated: {report_path}")
    print(f"All report files saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 