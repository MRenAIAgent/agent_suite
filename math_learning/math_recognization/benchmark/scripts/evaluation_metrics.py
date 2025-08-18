#!/usr/bin/env python3
"""
📊 Evaluation Metrics for Math Test Analysis System Benchmarking

Provides comprehensive evaluation metrics for measuring the accuracy and 
performance of our math test analysis system against ground truth data.

Key Metrics:
- OCR Accuracy (Character, Word, Expression level)
- Question Detection Accuracy
- Answer Extraction Accuracy  
- Error Analysis Accuracy
- Processing Speed and Efficiency
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import difflib
import numpy as np
from collections import defaultdict

@dataclass
class EvaluationResult:
    """Single evaluation result."""
    metric_name: str
    score: float
    max_score: float
    accuracy: float
    details: Dict[str, Any]
    
@dataclass
class BenchmarkResult:
    """Complete benchmark result."""
    dataset_name: str
    system_name: str
    evaluation_date: str
    total_samples: int
    processing_time: float
    metrics: List[EvaluationResult]
    overall_accuracy: float
    summary: Dict[str, Any]

class TextSimilarityMetrics:
    """Text similarity and accuracy metrics."""
    
    @staticmethod
    def character_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate character-level accuracy."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
            
        # Normalize strings
        pred_clean = re.sub(r'\s+', ' ', predicted.strip().lower())
        truth_clean = re.sub(r'\s+', ' ', ground_truth.strip().lower()) 
        
        # Calculate character-level accuracy using edit distance
        matcher = difflib.SequenceMatcher(None, truth_clean, pred_clean)
        return matcher.ratio()
    
    @staticmethod
    def word_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate word-level accuracy."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
            
        pred_words = set(predicted.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 1.0 if not pred_words else 0.0
            
        intersection = pred_words.intersection(truth_words)
        union = pred_words.union(truth_words)
        
        return len(intersection) / len(union) if union else 1.0
    
    @staticmethod
    def semantic_accuracy(predicted: str, ground_truth: str) -> float:
        """Calculate semantic accuracy for mathematical expressions."""
        # Normalize mathematical expressions
        def normalize_math(text: str) -> str:
            text = text.lower().strip()
            # Common mathematical normalizations
            text = re.sub(r'\s+', '', text)  # Remove spaces
            text = re.sub(r'\*', '', text)   # Remove multiplication signs
            text = re.sub(r'(\d)([a-z])', r'\1*\2', text)  # Add implicit multiplication
            return text
        
        pred_norm = normalize_math(predicted)
        truth_norm = normalize_math(ground_truth)
        
        return 1.0 if pred_norm == truth_norm else 0.0

class OCREvaluationMetrics:
    """OCR-specific evaluation metrics."""
    
    def __init__(self):
        self.similarity = TextSimilarityMetrics()
    
    def evaluate_expression_recognition(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationResult:
        """Evaluate mathematical expression recognition accuracy."""
        if not ground_truth:
            return EvaluationResult("expression_recognition", 0, 0, 0, {"error": "No ground truth data"})
        
        total_score = 0
        max_score = len(ground_truth)
        details = {
            "total_expressions": max_score,
            "correct_expressions": 0,
            "character_accuracy": [],
            "semantic_accuracy": [],
            "by_difficulty": defaultdict(list),
            "by_domain": defaultdict(list)
        }
        
        # Create prediction lookup
        pred_lookup = {p.get("id", f"pred_{i}"): p for i, p in enumerate(predictions)}
        
        for gt_item in ground_truth:
            gt_id = gt_item.get("id", "")
            gt_text = gt_item.get("text_ground_truth", "")
            gt_latex = gt_item.get("latex_ground_truth", "")
            difficulty = gt_item.get("difficulty", "unknown")
            domain = gt_item.get("domain", "unknown")
            
            pred_item = pred_lookup.get(gt_id, {})
            pred_text = pred_item.get("extracted_text", "")
            
            # Calculate accuracies
            char_acc = self.similarity.character_accuracy(pred_text, gt_text)
            semantic_acc = self.similarity.semantic_accuracy(pred_text, gt_text)
            
            details["character_accuracy"].append(char_acc)
            details["semantic_accuracy"].append(semantic_acc)
            details["by_difficulty"][difficulty].append(semantic_acc)
            details["by_domain"][domain].append(semantic_acc)
            
            if semantic_acc > 0.9:  # Consider >90% as correct
                details["correct_expressions"] += 1
                total_score += 1
        
        accuracy = total_score / max_score if max_score > 0 else 0
        
        # Calculate summary statistics
        details["avg_character_accuracy"] = np.mean(details["character_accuracy"]) if details["character_accuracy"] else 0
        details["avg_semantic_accuracy"] = np.mean(details["semantic_accuracy"]) if details["semantic_accuracy"] else 0
        
        return EvaluationResult(
            metric_name="expression_recognition",
            score=total_score,
            max_score=max_score,
            accuracy=accuracy,
            details=details
        )

class QuestionDetectionMetrics:
    """Question detection and parsing metrics."""
    
    def evaluate_question_detection(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationResult:
        """Evaluate question detection accuracy."""
        if not ground_truth:
            return EvaluationResult("question_detection", 0, 0, 0, {"error": "No ground truth data"})
        
        total_score = 0
        max_score = 0
        details = {
            "total_papers": len(ground_truth),
            "questions_detected": 0,
            "questions_missed": 0,
            "false_positives": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "by_subject": defaultdict(list)
        }
        
        for i, gt_paper in enumerate(ground_truth):
            gt_questions = gt_paper.get("ground_truth", {}).get("questions", [])
            max_score += len(gt_questions)
            
            pred_paper = predictions[i] if i < len(predictions) else {}
            pred_questions = pred_paper.get("questions", [])
            
            subject = gt_paper.get("subject", "unknown")
            
            # Simple matching based on question text similarity
            matched = 0
            for gt_q in gt_questions:
                gt_text = gt_q.get("question_text", "")
                best_match = 0
                
                for pred_q in pred_questions:
                    pred_text = pred_q.get("question", "")
                    similarity = TextSimilarityMetrics.word_accuracy(pred_text, gt_text)
                    best_match = max(best_match, similarity)
                
                if best_match > 0.7:  # Consider >70% similarity as match
                    matched += 1
                    total_score += 1
            
            details["questions_detected"] += matched
            details["questions_missed"] += len(gt_questions) - matched
            details["false_positives"] += max(0, len(pred_questions) - matched)
            details["by_subject"][subject].append(matched / len(gt_questions) if gt_questions else 0)
        
        # Calculate precision, recall, F1
        total_detected = details["questions_detected"]
        total_predicted = total_detected + details["false_positives"]
        total_actual = total_detected + details["questions_missed"]
        
        details["precision"] = total_detected / total_predicted if total_predicted > 0 else 0
        details["recall"] = total_detected / total_actual if total_actual > 0 else 0
        details["f1_score"] = 2 * details["precision"] * details["recall"] / (details["precision"] + details["recall"]) if (details["precision"] + details["recall"]) > 0 else 0
        
        accuracy = total_score / max_score if max_score > 0 else 0
        
        return EvaluationResult(
            metric_name="question_detection",
            score=total_score,
            max_score=max_score,
            accuracy=accuracy,
            details=details
        )

class AnswerExtractionMetrics:
    """Answer extraction and validation metrics."""
    
    def evaluate_answer_extraction(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationResult:
        """Evaluate answer extraction accuracy."""
        if not ground_truth:
            return EvaluationResult("answer_extraction", 0, 0, 0, {"error": "No ground truth data"})
        
        total_score = 0
        max_score = 0
        details = {
            "total_answers": 0,
            "correct_extractions": 0,
            "by_answer_type": defaultdict(list),
            "by_subject": defaultdict(list)
        }
        
        for i, gt_paper in enumerate(ground_truth):
            gt_questions = gt_paper.get("ground_truth", {}).get("questions", [])
            pred_paper = predictions[i] if i < len(predictions) else {}
            pred_questions = pred_paper.get("questions", [])
            
            subject = gt_paper.get("subject", "unknown")
            
            for j, gt_q in enumerate(gt_questions):
                max_score += 1
                details["total_answers"] += 1
                
                gt_answer = gt_q.get("correct_answer", "")
                answer_type = gt_q.get("answer_type", "unknown")
                
                pred_answer = ""
                if j < len(pred_questions):
                    pred_answer = pred_questions[j].get("answer", "")
                
                # Calculate answer accuracy
                accuracy = TextSimilarityMetrics.semantic_accuracy(pred_answer, gt_answer)
                
                details["by_answer_type"][answer_type].append(accuracy)
                details["by_subject"][subject].append(accuracy)
                
                if accuracy > 0.9:  # Consider >90% as correct
                    total_score += 1
                    details["correct_extractions"] += 1
        
        accuracy = total_score / max_score if max_score > 0 else 0
        
        return EvaluationResult(
            metric_name="answer_extraction",
            score=total_score,
            max_score=max_score,
            accuracy=accuracy,
            details=details
        )

class ErrorAnalysisMetrics:
    """Error analysis evaluation metrics."""
    
    def evaluate_error_analysis(self, predictions: List[Dict], ground_truth: List[Dict]) -> EvaluationResult:
        """Evaluate error analysis accuracy."""
        if not ground_truth:
            return EvaluationResult("error_analysis", 0, 0, 0, {"error": "No ground truth data"})
        
        total_score = 0
        max_score = len(ground_truth)
        details = {
            "total_solutions": max_score,
            "correct_classifications": 0,
            "error_type_accuracy": defaultdict(list),
            "root_cause_accuracy": defaultdict(list),
            "confidence_correlation": []
        }
        
        for i, gt_solution in enumerate(ground_truth):
            gt_error = gt_solution.get("error_analysis", {})
            gt_error_type = gt_error.get("error_type")
            gt_root_cause = gt_error.get("root_cause", "")
            gt_is_correct = gt_solution.get("is_correct", False)
            
            pred_solution = predictions[i] if i < len(predictions) else {}
            pred_error = pred_solution.get("error_analysis", {})
            pred_error_type = pred_error.get("error_type")
            pred_root_cause = pred_error.get("root_cause", "")
            pred_is_correct = pred_solution.get("is_correct", True)
            
            # Check correctness classification
            correctness_match = (gt_is_correct == pred_is_correct)
            
            # Check error type classification (only for incorrect answers)
            error_type_match = True
            if not gt_is_correct:
                error_type_match = (gt_error_type == pred_error_type)
                details["error_type_accuracy"][gt_error_type or "none"].append(1.0 if error_type_match else 0.0)
            
            # Check root cause analysis (only for incorrect answers)
            root_cause_match = True
            if not gt_is_correct and gt_root_cause:
                cause_similarity = TextSimilarityMetrics.word_accuracy(pred_root_cause, gt_root_cause)
                root_cause_match = cause_similarity > 0.6  # Consider >60% similarity as match
                details["root_cause_accuracy"][gt_error_type or "none"].append(cause_similarity)
            
            # Overall accuracy
            if correctness_match and error_type_match and root_cause_match:
                total_score += 1
                details["correct_classifications"] += 1
        
        accuracy = total_score / max_score if max_score > 0 else 0
        
        return EvaluationResult(
            metric_name="error_analysis",
            score=total_score,
            max_score=max_score,
            accuracy=accuracy,
            details=details
        )

class BenchmarkEvaluator:
    """Main benchmark evaluator."""
    
    def __init__(self):
        self.ocr_metrics = OCREvaluationMetrics()
        self.question_metrics = QuestionDetectionMetrics()
        self.answer_metrics = AnswerExtractionMetrics()
        self.error_metrics = ErrorAnalysisMetrics()
    
    def evaluate_complete_system(self, predictions: Dict[str, Any], ground_truth: Dict[str, Any], 
                                dataset_name: str, system_name: str = "Math Test Analysis System") -> BenchmarkResult:
        """Evaluate the complete system against ground truth."""
        start_time = time.time()
        
        metrics = []
        
        # Evaluate OCR (if expression data available)
        if "expressions" in ground_truth and "expressions" in predictions:
            ocr_result = self.ocr_metrics.evaluate_expression_recognition(
                predictions["expressions"], ground_truth["expressions"]
            )
            metrics.append(ocr_result)
        
        # Evaluate Question Detection (if test paper data available)
        if "test_papers" in ground_truth and "test_papers" in predictions:
            question_result = self.question_metrics.evaluate_question_detection(
                predictions["test_papers"], ground_truth["test_papers"]
            )
            metrics.append(question_result)
            
            # Evaluate Answer Extraction
            answer_result = self.answer_metrics.evaluate_answer_extraction(
                predictions["test_papers"], ground_truth["test_papers"]
            )
            metrics.append(answer_result)
        
        # Evaluate Error Analysis (if solution data available)
        if "solutions" in ground_truth and "solutions" in predictions:
            error_result = self.error_metrics.evaluate_error_analysis(
                predictions["solutions"], ground_truth["solutions"]
            )
            metrics.append(error_result)
        
        processing_time = time.time() - start_time
        
        # Calculate overall accuracy
        total_score = sum(m.score for m in metrics)
        max_total_score = sum(m.max_score for m in metrics)
        overall_accuracy = total_score / max_total_score if max_total_score > 0 else 0
        
        # Create summary
        summary = {
            "metrics_evaluated": len(metrics),
            "total_samples": sum(m.max_score for m in metrics),
            "processing_speed": f"{processing_time:.2f}s",
            "metric_breakdown": {m.metric_name: m.accuracy for m in metrics}
        }
        
        return BenchmarkResult(
            dataset_name=dataset_name,
            system_name=system_name,
            evaluation_date=datetime.now().isoformat(),
            total_samples=summary["total_samples"],
            processing_time=processing_time,
            metrics=metrics,
            overall_accuracy=overall_accuracy,
            summary=summary
        )
    
    def save_results(self, result: BenchmarkResult, output_path: Path) -> None:
        """Save benchmark results to JSON file."""
        result_dict = {
            "dataset_name": result.dataset_name,
            "system_name": result.system_name,
            "evaluation_date": result.evaluation_date,
            "total_samples": result.total_samples,
            "processing_time": result.processing_time,
            "overall_accuracy": result.overall_accuracy,
            "summary": result.summary,
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "score": m.score,
                    "max_score": m.max_score,
                    "accuracy": m.accuracy,
                    "details": m.details
                }
                for m in result.metrics
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"📊 Benchmark results saved to: {output_path}")
    
    def print_summary(self, result: BenchmarkResult) -> None:
        """Print a formatted summary of benchmark results."""
        print("\n" + "="*60)
        print(f"🎯 BENCHMARK RESULTS: {result.dataset_name}")
        print("="*60)
        print(f"System: {result.system_name}")
        print(f"Date: {result.evaluation_date}")
        print(f"Total Samples: {result.total_samples}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Overall Accuracy: {result.overall_accuracy:.1%}")
        print("\n" + "-"*40)
        print("METRIC BREAKDOWN:")
        print("-"*40)
        
        for metric in result.metrics:
            print(f"{metric.metric_name.replace('_', ' ').title()}: {metric.accuracy:.1%} ({metric.score}/{metric.max_score})")
        
        print("="*60) 