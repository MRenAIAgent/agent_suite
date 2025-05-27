"""
Analytics Dashboard module.

This module provides visualization and analytics for the personalized learning system.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .personalized_learning import PersonalizedLearningSystem, LearningPattern, WeaknessProfile
from .user_model import LearningGraph
from ..knowledge_graph.graph import KnowledgeGraph


@dataclass
class ProgressMetrics:
    """Comprehensive progress metrics for a user."""
    total_concepts_attempted: int
    concepts_mastered: int
    concepts_struggling: int
    overall_mastery_score: float
    learning_velocity: float
    retention_rate: float
    consistency_score: float
    time_spent_learning: float  # in minutes
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_concepts_attempted': self.total_concepts_attempted,
            'concepts_mastered': self.concepts_mastered,
            'concepts_struggling': self.concepts_struggling,
            'overall_mastery_score': self.overall_mastery_score,
            'learning_velocity': self.learning_velocity,
            'retention_rate': self.retention_rate,
            'consistency_score': self.consistency_score,
            'time_spent_learning': self.time_spent_learning
        }


@dataclass
class CategoryAnalysis:
    """Analysis of performance in a specific category."""
    category_name: str
    concepts_count: int
    avg_mastery: float
    avg_success_rate: float
    strength_level: str  # 'strong', 'moderate', 'weak'
    recommended_focus: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'category_name': self.category_name,
            'concepts_count': self.concepts_count,
            'avg_mastery': self.avg_mastery,
            'avg_success_rate': self.avg_success_rate,
            'strength_level': self.strength_level,
            'recommended_focus': self.recommended_focus
        }


class LearningAnalyticsDashboard:
    """Comprehensive learning analytics dashboard."""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Initialize the analytics dashboard.
        
        Args:
            knowledge_graph: The knowledge graph containing concept relationships
        """
        self.knowledge_graph = knowledge_graph
        self.personalized_system = PersonalizedLearningSystem(knowledge_graph)
    
    def generate_progress_report(self, learning_graph: LearningGraph) -> Dict[str, Any]:
        """
        Generate a comprehensive progress report for a user.
        
        Args:
            learning_graph: The user's learning graph
            
        Returns:
            Dictionary containing comprehensive progress information
        """
        # Calculate basic metrics
        metrics = self._calculate_progress_metrics(learning_graph)
        
        # Analyze learning patterns
        patterns = self.personalized_system.analyze_learning_patterns(learning_graph)
        
        # Detect weaknesses
        weaknesses = self.personalized_system.detect_weaknesses(learning_graph)
        
        # Generate insights
        insights = self.personalized_system.generate_learning_insights(learning_graph)
        
        # Analyze performance by category
        category_analysis = self._analyze_by_category(learning_graph, patterns)
        
        # Get recommendations
        recommendations = self.personalized_system.get_adaptive_recommendations(learning_graph)
        
        # Calculate learning trajectory
        trajectory = self._calculate_learning_trajectory(learning_graph)
        
        return {
            'user_id': learning_graph.user_id,
            'report_date': datetime.now().isoformat(),
            'progress_metrics': metrics.to_dict(),
            'category_analysis': [ca.to_dict() for ca in category_analysis],
            'learning_patterns': {k: v.to_dict() for k, v in patterns.items()},
            'weaknesses': [w.to_dict() for w in weaknesses],
            'insights': [i.to_dict() for i in insights],
            'recommendations': recommendations,
            'learning_trajectory': trajectory,
            'mastery_distribution': self._get_mastery_distribution(learning_graph),
            'concept_network_analysis': self._analyze_concept_network(learning_graph)
        }
    
    def track_learning_journey(self, learning_graph: LearningGraph, 
                             days_back: int = 30) -> Dict[str, Any]:
        """
        Track the user's learning journey over time.
        
        Args:
            learning_graph: The user's learning graph
            days_back: Number of days to look back
            
        Returns:
            Dictionary containing journey tracking information
        """
        # Extract historical data
        historical_data = self._extract_historical_data(learning_graph, days_back)
        
        # Calculate daily progress
        daily_progress = self._calculate_daily_progress(historical_data)
        
        # Identify learning streaks
        streaks = self._identify_learning_streaks(historical_data)
        
        # Calculate momentum
        momentum = self._calculate_learning_momentum(daily_progress)
        
        return {
            'journey_period_days': days_back,
            'daily_progress': daily_progress,
            'learning_streaks': streaks,
            'momentum_score': momentum,
            'concepts_learned_timeline': self._get_concepts_timeline(learning_graph, days_back),
            'difficulty_progression': self._track_difficulty_progression(learning_graph, days_back)
        }
    
    def generate_weakness_analysis(self, learning_graph: LearningGraph) -> Dict[str, Any]:
        """
        Generate detailed analysis of user weaknesses.
        
        Args:
            learning_graph: The user's learning graph
            
        Returns:
            Dictionary containing detailed weakness analysis
        """
        weaknesses = self.personalized_system.detect_weaknesses(learning_graph)
        
        # Group weaknesses by type
        weakness_by_type = defaultdict(list)
        for weakness in weaknesses:
            weakness_by_type[weakness.weakness_type].append(weakness)
        
        # Analyze weakness clusters
        clusters = self._identify_weakness_clusters(weaknesses)
        
        # Generate intervention plan
        intervention_plan = self._create_intervention_plan(weaknesses)
        
        return {
            'total_weaknesses': len(weaknesses),
            'weakness_by_type': {k: [w.to_dict() for w in v] for k, v in weakness_by_type.items()},
            'weakness_clusters': clusters,
            'intervention_plan': intervention_plan,
            'priority_order': [w.concept_id for w in weaknesses],
            'estimated_improvement_time': self._estimate_improvement_time(weaknesses)
        }
    
    def compare_with_peers(self, learning_graph: LearningGraph, 
                          peer_graphs: List[LearningGraph]) -> Dict[str, Any]:
        """
        Compare user's performance with peers.
        
        Args:
            learning_graph: The user's learning graph
            peer_graphs: List of peer learning graphs
            
        Returns:
            Dictionary containing peer comparison analysis
        """
        if not peer_graphs:
            return {'error': 'No peer data available for comparison'}
        
        # Calculate user metrics
        user_metrics = self._calculate_progress_metrics(learning_graph)
        
        # Calculate peer metrics
        peer_metrics = [self._calculate_progress_metrics(pg) for pg in peer_graphs]
        
        # Compare performance
        comparison = self._compare_performance(user_metrics, peer_metrics)
        
        # Identify relative strengths and weaknesses
        relative_analysis = self._analyze_relative_performance(learning_graph, peer_graphs)
        
        return {
            'user_metrics': user_metrics.to_dict(),
            'peer_average': self._calculate_average_metrics(peer_metrics),
            'percentile_ranking': comparison,
            'relative_strengths': relative_analysis['strengths'],
            'relative_weaknesses': relative_analysis['weaknesses'],
            'improvement_opportunities': relative_analysis['opportunities']
        }
    
    def predict_future_performance(self, learning_graph: LearningGraph,
                                 prediction_days: int = 30) -> Dict[str, Any]:
        """
        Predict future learning performance based on current patterns.
        
        Args:
            learning_graph: The user's learning graph
            prediction_days: Number of days to predict ahead
            
        Returns:
            Dictionary containing performance predictions
        """
        patterns = self.personalized_system.analyze_learning_patterns(learning_graph)
        
        # Calculate current learning velocity
        avg_velocity = np.mean([p.learning_velocity for p in patterns.values()]) if patterns else 0.5
        
        # Predict concepts that will be mastered
        predicted_mastery = self._predict_concept_mastery(learning_graph, patterns, prediction_days)
        
        # Predict potential struggles
        predicted_struggles = self._predict_struggles(learning_graph, patterns)
        
        # Estimate time to goals
        goal_estimates = self._estimate_time_to_goals(learning_graph, patterns)
        
        return {
            'prediction_period_days': prediction_days,
            'predicted_concepts_mastered': predicted_mastery,
            'potential_struggles': predicted_struggles,
            'estimated_learning_velocity': avg_velocity,
            'time_to_goals': goal_estimates,
            'confidence_intervals': self._calculate_prediction_confidence(patterns)
        }
    
    def _calculate_progress_metrics(self, learning_graph: LearningGraph) -> ProgressMetrics:
        """Calculate comprehensive progress metrics."""
        total_attempted = len(learning_graph.concept_mastery)
        mastered = len(learning_graph.get_mastered_concepts(threshold=0.7))
        struggling = len(learning_graph.get_struggling_concepts(threshold=0.3))
        
        # Calculate overall mastery score
        if total_attempted > 0:
            overall_mastery = sum(
                data['mastery'] for data in learning_graph.concept_mastery.values()
            ) / total_attempted
        else:
            overall_mastery = 0.0
        
        # Calculate learning velocity and retention
        patterns = self.personalized_system.analyze_learning_patterns(learning_graph)
        if patterns:
            avg_velocity = np.mean([p.learning_velocity for p in patterns.values()])
            avg_retention = np.mean([p.retention_rate for p in patterns.values()])
        else:
            avg_velocity = 0.5
            avg_retention = 0.7
        
        # Calculate consistency score (how regular is the learning)
        consistency = self._calculate_consistency_score(learning_graph)
        
        # Estimate time spent (simplified calculation)
        total_exercises = sum(len(attempts) for attempts in learning_graph.exercise_history.values())
        estimated_time = total_exercises * 3.0  # Assume 3 minutes per exercise
        
        return ProgressMetrics(
            total_concepts_attempted=total_attempted,
            concepts_mastered=mastered,
            concepts_struggling=struggling,
            overall_mastery_score=overall_mastery,
            learning_velocity=avg_velocity,
            retention_rate=avg_retention,
            consistency_score=consistency,
            time_spent_learning=estimated_time
        )
    
    def _analyze_by_category(self, learning_graph: LearningGraph,
                           patterns: Dict[str, LearningPattern]) -> List[CategoryAnalysis]:
        """Analyze performance by concept category."""
        category_data = defaultdict(list)
        
        # Group concepts by category
        for concept_id, pattern in patterns.items():
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                mastery = learning_graph.get_mastery(concept_id)
                category_data[concept.category].append({
                    'mastery': mastery,
                    'success_rate': pattern.success_rate,
                    'concept_id': concept_id
                })
        
        analyses = []
        for category, data in category_data.items():
            if len(data) >= 2:  # Need at least 2 concepts for meaningful analysis
                avg_mastery = np.mean([d['mastery'] for d in data])
                avg_success = np.mean([d['success_rate'] for d in data])
                
                # Determine strength level
                if avg_mastery > 0.7 and avg_success > 0.7:
                    strength_level = 'strong'
                    focus = ['Explore advanced topics', 'Help others in this area']
                elif avg_mastery > 0.5 and avg_success > 0.5:
                    strength_level = 'moderate'
                    focus = ['Practice more challenging problems', 'Review fundamentals']
                else:
                    strength_level = 'weak'
                    focus = ['Focus on basic concepts', 'Get additional help', 'Practice fundamentals']
                
                analyses.append(CategoryAnalysis(
                    category_name=category,
                    concepts_count=len(data),
                    avg_mastery=avg_mastery,
                    avg_success_rate=avg_success,
                    strength_level=strength_level,
                    recommended_focus=focus
                ))
        
        return sorted(analyses, key=lambda a: a.avg_mastery, reverse=True)
    
    def _calculate_learning_trajectory(self, learning_graph: LearningGraph) -> Dict[str, Any]:
        """Calculate the user's learning trajectory over time."""
        trajectory_data = []
        
        # Collect all historical mastery data
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            history = mastery_data.get('history', [])
            for entry in history:
                trajectory_data.append({
                    'timestamp': entry['timestamp'],
                    'concept_id': concept_id,
                    'mastery': entry['mastery']
                })
            
            # Add current state
            trajectory_data.append({
                'timestamp': mastery_data['last_practiced'],
                'concept_id': concept_id,
                'mastery': mastery_data['mastery']
            })
        
        # Sort by timestamp
        trajectory_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate trend
        if len(trajectory_data) >= 2:
            masteries = [d['mastery'] for d in trajectory_data]
            trend = np.polyfit(range(len(masteries)), masteries, 1)[0]  # Linear trend
        else:
            trend = 0.0
        
        return {
            'data_points': len(trajectory_data),
            'trend_slope': trend,
            'trajectory_data': trajectory_data[-20:],  # Last 20 data points
            'improvement_rate': max(0, trend * 100)  # Percentage improvement per unit time
        }
    
    def _get_mastery_distribution(self, learning_graph: LearningGraph) -> Dict[str, int]:
        """Get distribution of mastery levels."""
        distribution = {
            'beginner': 0,      # 0.0 - 0.3
            'developing': 0,    # 0.3 - 0.6
            'proficient': 0,    # 0.6 - 0.8
            'advanced': 0       # 0.8 - 1.0
        }
        
        for mastery_data in learning_graph.concept_mastery.values():
            mastery = mastery_data['mastery']
            if mastery < 0.3:
                distribution['beginner'] += 1
            elif mastery < 0.6:
                distribution['developing'] += 1
            elif mastery < 0.8:
                distribution['proficient'] += 1
            else:
                distribution['advanced'] += 1
        
        return distribution
    
    def _analyze_concept_network(self, learning_graph: LearningGraph) -> Dict[str, Any]:
        """Analyze the user's progress through the concept network."""
        mastered_concepts = set(learning_graph.get_mastered_concepts())
        struggling_concepts = set(learning_graph.get_struggling_concepts())
        
        # Find bottleneck concepts (prerequisites that are blocking progress)
        bottlenecks = []
        for concept in self.knowledge_graph.get_all_concepts():
            if concept.id in struggling_concepts:
                dependents = self.knowledge_graph.get_dependent_concepts(concept.id)
                if len(dependents) > 2:  # Concept blocks multiple others
                    bottlenecks.append({
                        'concept_id': concept.id,
                        'blocked_concepts': len(dependents),
                        'impact_score': len(dependents) * (1 - learning_graph.get_mastery(concept.id))
                    })
        
        # Sort bottlenecks by impact
        bottlenecks.sort(key=lambda b: b['impact_score'], reverse=True)
        
        return {
            'mastered_count': len(mastered_concepts),
            'struggling_count': len(struggling_concepts),
            'bottlenecks': bottlenecks[:5],  # Top 5 bottlenecks
            'network_coverage': len(mastered_concepts) / len(self.knowledge_graph.get_all_concepts())
        }
    
    def _calculate_consistency_score(self, learning_graph: LearningGraph) -> float:
        """Calculate how consistent the user's learning pattern is."""
        # Simplified consistency calculation based on exercise frequency
        if not learning_graph.exercise_history:
            return 0.0
        
        # Get all exercise timestamps
        timestamps = []
        for attempts in learning_graph.exercise_history.values():
            for attempt in attempts:
                timestamps.append(attempt['timestamp'])
        
        if len(timestamps) < 2:
            return 0.5
        
        # Calculate time gaps between exercises
        timestamps.sort()
        gaps = []
        for i in range(1, len(timestamps)):
            try:
                t1 = datetime.fromisoformat(timestamps[i-1])
                t2 = datetime.fromisoformat(timestamps[i])
                gap_hours = (t2 - t1).total_seconds() / 3600
                gaps.append(gap_hours)
            except:
                continue
        
        if not gaps:
            return 0.5
        
        # Consistency is higher when gaps are more uniform
        gap_std = np.std(gaps)
        gap_mean = np.mean(gaps)
        
        if gap_mean == 0:
            return 0.5
        
        coefficient_of_variation = gap_std / gap_mean
        consistency = max(0.0, min(1.0, 1.0 - coefficient_of_variation / 2.0))
        
        return consistency
    
    def _extract_historical_data(self, learning_graph: LearningGraph, days_back: int) -> List[Dict]:
        """Extract historical learning data for the specified period."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        historical_data = []
        
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            history = mastery_data.get('history', [])
            for entry in history:
                try:
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    if timestamp >= cutoff_date:
                        historical_data.append({
                            'date': timestamp.date(),
                            'concept_id': concept_id,
                            'mastery': entry['mastery'],
                            'timestamp': timestamp
                        })
                except:
                    continue
        
        return sorted(historical_data, key=lambda x: x['timestamp'])
    
    def _calculate_daily_progress(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate daily progress metrics."""
        daily_progress = defaultdict(list)
        
        for entry in historical_data:
            daily_progress[entry['date'].isoformat()].append(entry['mastery'])
        
        # Calculate average mastery per day
        return {
            date: np.mean(masteries) for date, masteries in daily_progress.items()
        }
    
    def _identify_learning_streaks(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """Identify learning streaks and patterns."""
        if not historical_data:
            return {'current_streak': 0, 'longest_streak': 0, 'total_active_days': 0}
        
        # Get unique learning days
        learning_days = set(entry['date'] for entry in historical_data)
        learning_days = sorted(learning_days)
        
        if not learning_days:
            return {'current_streak': 0, 'longest_streak': 0, 'total_active_days': 0}
        
        # Calculate streaks
        current_streak = 0
        longest_streak = 0
        temp_streak = 1
        
        for i in range(1, len(learning_days)):
            if (learning_days[i] - learning_days[i-1]).days == 1:
                temp_streak += 1
            else:
                longest_streak = max(longest_streak, temp_streak)
                temp_streak = 1
        
        longest_streak = max(longest_streak, temp_streak)
        
        # Check if current streak is active (learned today or yesterday)
        today = datetime.date.today()
        if learning_days and (today - learning_days[-1]).days <= 1:
            current_streak = temp_streak
        
        return {
            'current_streak': current_streak,
            'longest_streak': longest_streak,
            'total_active_days': len(learning_days)
        }
    
    def _calculate_learning_momentum(self, daily_progress: Dict[str, float]) -> float:
        """Calculate learning momentum based on recent progress."""
        if len(daily_progress) < 2:
            return 0.5
        
        # Get recent progress values
        recent_values = list(daily_progress.values())[-7:]  # Last 7 days
        
        if len(recent_values) < 2:
            return 0.5
        
        # Calculate trend
        x = range(len(recent_values))
        trend = np.polyfit(x, recent_values, 1)[0]
        
        # Normalize to 0-1 scale
        momentum = max(0.0, min(1.0, trend * 10 + 0.5))
        return momentum
    
    def _get_concepts_timeline(self, learning_graph: LearningGraph, days_back: int) -> List[Dict]:
        """Get timeline of concepts learned."""
        timeline = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            try:
                last_practiced = datetime.fromisoformat(mastery_data['last_practiced'])
                if last_practiced >= cutoff_date and mastery_data['mastery'] >= 0.7:
                    concept = self.knowledge_graph.get_concept(concept_id)
                    timeline.append({
                        'date': last_practiced.date().isoformat(),
                        'concept_id': concept_id,
                        'concept_name': concept.name if concept else concept_id,
                        'mastery_level': mastery_data['mastery']
                    })
            except:
                continue
        
        return sorted(timeline, key=lambda x: x['date'])
    
    def _track_difficulty_progression(self, learning_graph: LearningGraph, days_back: int) -> Dict[str, Any]:
        """Track how the user progresses through difficulty levels."""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        difficulty_data = []
        
        for concept_id, mastery_data in learning_graph.concept_mastery.items():
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                try:
                    last_practiced = datetime.fromisoformat(mastery_data['last_practiced'])
                    if last_practiced >= cutoff_date:
                        difficulty_data.append({
                            'date': last_practiced.date(),
                            'difficulty': concept.difficulty,
                            'mastery': mastery_data['mastery']
                        })
                except:
                    continue
        
        # Group by difficulty level
        difficulty_progression = defaultdict(list)
        for entry in difficulty_data:
            difficulty_progression[entry['difficulty']].append(entry['mastery'])
        
        # Calculate average mastery per difficulty level
        avg_by_difficulty = {
            level: np.mean(masteries) for level, masteries in difficulty_progression.items()
        }
        
        return {
            'difficulty_distribution': dict(difficulty_progression),
            'average_mastery_by_difficulty': avg_by_difficulty,
            'progression_trend': self._calculate_difficulty_trend(avg_by_difficulty)
        }
    
    def _calculate_difficulty_trend(self, avg_by_difficulty: Dict[int, float]) -> str:
        """Calculate the trend in difficulty progression."""
        if len(avg_by_difficulty) < 2:
            return 'insufficient_data'
        
        sorted_difficulties = sorted(avg_by_difficulty.items())
        
        # Check if mastery decreases with difficulty (expected pattern)
        decreasing = all(
            sorted_difficulties[i][1] >= sorted_difficulties[i+1][1] 
            for i in range(len(sorted_difficulties)-1)
        )
        
        if decreasing:
            return 'normal_progression'
        else:
            return 'irregular_progression'
    
    def _identify_weakness_clusters(self, weaknesses: List[WeaknessProfile]) -> List[Dict[str, Any]]:
        """Identify clusters of related weaknesses."""
        clusters = []
        processed = set()
        
        for weakness in weaknesses:
            if weakness.concept_id in processed:
                continue
            
            # Find related weaknesses
            cluster = [weakness]
            processed.add(weakness.concept_id)
            
            for other_weakness in weaknesses:
                if (other_weakness.concept_id not in processed and 
                    weakness.concept_id in other_weakness.related_concepts):
                    cluster.append(other_weakness)
                    processed.add(other_weakness.concept_id)
            
            if len(cluster) > 1:  # Only include actual clusters
                clusters.append({
                    'cluster_size': len(cluster),
                    'concepts': [w.concept_id for w in cluster],
                    'avg_severity': np.mean([w.severity for w in cluster]),
                    'dominant_type': max(set(w.weakness_type for w in cluster), 
                                       key=lambda x: sum(1 for w in cluster if w.weakness_type == x))
                })
        
        return sorted(clusters, key=lambda c: c['avg_severity'], reverse=True)
    
    def _create_intervention_plan(self, weaknesses: List[WeaknessProfile]) -> Dict[str, Any]:
        """Create a structured intervention plan."""
        if not weaknesses:
            return {'phases': [], 'estimated_duration': 0}
        
        # Group weaknesses by severity
        critical = [w for w in weaknesses if w.severity > 0.7]
        moderate = [w for w in weaknesses if 0.4 <= w.severity <= 0.7]
        mild = [w for w in weaknesses if w.severity < 0.4]
        
        phases = []
        
        # Phase 1: Address critical weaknesses
        if critical:
            phases.append({
                'phase': 1,
                'title': 'Address Critical Weaknesses',
                'concepts': [w.concept_id for w in critical],
                'estimated_weeks': len(critical) * 2,
                'interventions': list(set(
                    intervention for w in critical for intervention in w.suggested_interventions
                ))
            })
        
        # Phase 2: Improve moderate weaknesses
        if moderate:
            phases.append({
                'phase': 2,
                'title': 'Strengthen Moderate Areas',
                'concepts': [w.concept_id for w in moderate],
                'estimated_weeks': len(moderate) * 1.5,
                'interventions': list(set(
                    intervention for w in moderate for intervention in w.suggested_interventions
                ))
            })
        
        # Phase 3: Polish mild weaknesses
        if mild:
            phases.append({
                'phase': 3,
                'title': 'Polish and Reinforce',
                'concepts': [w.concept_id for w in mild],
                'estimated_weeks': len(mild) * 1,
                'interventions': list(set(
                    intervention for w in mild for intervention in w.suggested_interventions
                ))
            })
        
        total_duration = sum(phase['estimated_weeks'] for phase in phases)
        
        return {
            'phases': phases,
            'estimated_duration_weeks': total_duration,
            'priority_order': [w.concept_id for w in weaknesses]
        }
    
    def _estimate_improvement_time(self, weaknesses: List[WeaknessProfile]) -> Dict[str, float]:
        """Estimate time needed to improve each weakness."""
        estimates = {}
        
        for weakness in weaknesses:
            # Base time estimate on severity and type
            base_time = weakness.severity * 10  # Base hours
            
            # Adjust based on weakness type
            if weakness.weakness_type == 'conceptual':
                multiplier = 1.5  # Conceptual issues take longer
            elif weakness.weakness_type == 'procedural':
                multiplier = 1.2
            else:  # computational
                multiplier = 1.0
            
            estimated_hours = base_time * multiplier
            estimates[weakness.concept_id] = estimated_hours
        
        return estimates
    
    def _compare_performance(self, user_metrics: ProgressMetrics, 
                           peer_metrics: List[ProgressMetrics]) -> Dict[str, float]:
        """Compare user performance with peers."""
        if not peer_metrics:
            return {}
        
        comparisons = {}
        
        # Compare each metric
        metrics_to_compare = [
            'overall_mastery_score', 'learning_velocity', 'retention_rate', 'consistency_score'
        ]
        
        for metric in metrics_to_compare:
            user_value = getattr(user_metrics, metric)
            peer_values = [getattr(pm, metric) for pm in peer_metrics]
            
            # Calculate percentile
            better_than = sum(1 for pv in peer_values if user_value > pv)
            percentile = (better_than / len(peer_values)) * 100
            
            comparisons[metric] = percentile
        
        return comparisons
    
    def _calculate_average_metrics(self, peer_metrics: List[ProgressMetrics]) -> Dict[str, float]:
        """Calculate average metrics across peers."""
        if not peer_metrics:
            return {}
        
        return {
            'overall_mastery_score': np.mean([pm.overall_mastery_score for pm in peer_metrics]),
            'learning_velocity': np.mean([pm.learning_velocity for pm in peer_metrics]),
            'retention_rate': np.mean([pm.retention_rate for pm in peer_metrics]),
            'consistency_score': np.mean([pm.consistency_score for pm in peer_metrics])
        }
    
    def _analyze_relative_performance(self, learning_graph: LearningGraph,
                                    peer_graphs: List[LearningGraph]) -> Dict[str, List[str]]:
        """Analyze relative strengths and weaknesses compared to peers."""
        # This is a simplified analysis - in practice, you'd do more sophisticated comparison
        user_patterns = self.personalized_system.analyze_learning_patterns(learning_graph)
        
        # Calculate user's category performance
        user_categories = defaultdict(list)
        for concept_id, pattern in user_patterns.items():
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                user_categories[concept.category].append(pattern.success_rate)
        
        user_category_avg = {
            cat: np.mean(rates) for cat, rates in user_categories.items()
        }
        
        # Compare with peers (simplified)
        relative_strengths = []
        relative_weaknesses = []
        opportunities = []
        
        for category, avg_rate in user_category_avg.items():
            if avg_rate > 0.7:
                relative_strengths.append(category)
            elif avg_rate < 0.4:
                relative_weaknesses.append(category)
            else:
                opportunities.append(category)
        
        return {
            'strengths': relative_strengths,
            'weaknesses': relative_weaknesses,
            'opportunities': opportunities
        }
    
    def _predict_concept_mastery(self, learning_graph: LearningGraph,
                               patterns: Dict[str, LearningPattern],
                               prediction_days: int) -> List[Dict[str, Any]]:
        """Predict which concepts will be mastered in the given timeframe."""
        predictions = []
        
        # Get concepts at learning boundary
        boundary_concepts = self.personalized_system.gap_analyzer.find_learning_boundary(learning_graph)
        
        for concept_id in boundary_concepts:
            concept = self.knowledge_graph.get_concept(concept_id)
            if concept:
                current_mastery = learning_graph.get_mastery(concept_id)
                
                # Estimate learning rate based on similar concepts
                similar_patterns = [p for p in patterns.values() 
                                  if self.knowledge_graph.get_concept(p.concept_id) and
                                  self.knowledge_graph.get_concept(p.concept_id).difficulty == concept.difficulty]
                
                if similar_patterns:
                    avg_velocity = np.mean([p.learning_velocity for p in similar_patterns])
                else:
                    avg_velocity = 0.5
                
                # Predict mastery improvement
                days_to_mastery = (0.7 - current_mastery) / (avg_velocity * 0.1)  # Simplified calculation
                
                if days_to_mastery <= prediction_days:
                    predictions.append({
                        'concept_id': concept_id,
                        'concept_name': concept.name,
                        'current_mastery': current_mastery,
                        'predicted_mastery': min(1.0, current_mastery + avg_velocity * 0.1 * prediction_days),
                        'estimated_days_to_mastery': days_to_mastery,
                        'confidence': min(0.9, avg_velocity)
                    })
        
        return sorted(predictions, key=lambda p: p['estimated_days_to_mastery'])
    
    def _predict_struggles(self, learning_graph: LearningGraph,
                         patterns: Dict[str, LearningPattern]) -> List[Dict[str, Any]]:
        """Predict concepts where the user might struggle."""
        potential_struggles = []
        
        # Look for patterns that indicate potential struggles
        for concept_id, pattern in patterns.items():
            if (pattern.learning_velocity < 0.3 or 
                pattern.retention_rate < 0.4 or 
                pattern.success_rate < 0.5):
                
                concept = self.knowledge_graph.get_concept(concept_id)
                if concept:
                    # Find dependent concepts that might be affected
                    dependents = self.knowledge_graph.get_dependent_concepts(concept_id)
                    
                    for dependent in dependents:
                        if learning_graph.get_mastery(dependent.id) < 0.3:  # Not yet attempted much
                            potential_struggles.append({
                                'concept_id': dependent.id,
                                'concept_name': dependent.name,
                                'risk_factor': 1.0 - pattern.success_rate,
                                'reason': f"Struggles with prerequisite: {concept.name}"
                            })
        
        return sorted(potential_struggles, key=lambda s: s['risk_factor'], reverse=True)[:5]
    
    def _estimate_time_to_goals(self, learning_graph: LearningGraph,
                              patterns: Dict[str, LearningPattern]) -> Dict[str, float]:
        """Estimate time to reach various learning goals."""
        if not patterns:
            return {}
        
        avg_velocity = np.mean([p.learning_velocity for p in patterns.values()])
        
        # Define goals
        goals = {
            'next_mastery': 0.7,  # Next concept to 70% mastery
            'grade_level_complete': 0.8,  # All grade-level concepts to 80%
            'advanced_ready': 0.9   # Ready for advanced topics
        }
        
        estimates = {}
        
        # Get concepts at learning boundary
        boundary_concepts = self.personalized_system.gap_analyzer.find_learning_boundary(learning_graph)
        
        if boundary_concepts:
            # Estimate time to next mastery
            next_concept_id = boundary_concepts[0]
            current_mastery = learning_graph.get_mastery(next_concept_id)
            days_to_next = (goals['next_mastery'] - current_mastery) / (avg_velocity * 0.05)
            estimates['next_mastery_days'] = max(1, days_to_next)
        
        # Estimate time for other goals (simplified)
        estimates['grade_level_complete_weeks'] = len(boundary_concepts) * 2
        estimates['advanced_ready_months'] = len(boundary_concepts) * 0.5
        
        return estimates
    
    def _calculate_prediction_confidence(self, patterns: Dict[str, LearningPattern]) -> Dict[str, float]:
        """Calculate confidence intervals for predictions."""
        if not patterns:
            return {'low': 0.3, 'high': 0.7}
        
        # Calculate variance in learning patterns
        velocities = [p.learning_velocity for p in patterns.values()]
        velocity_std = np.std(velocities)
        
        # Higher variance = lower confidence
        base_confidence = 0.7
        confidence_adjustment = velocity_std * 0.5
        
        confidence = max(0.1, base_confidence - confidence_adjustment)
        
        return {
            'low': max(0.1, confidence - 0.2),
            'high': min(0.9, confidence + 0.2)
        } 