#!/usr/bin/env python3
"""
Multi-User Learning Simulation System

This script creates multiple users with different learning profiles, simulates their
learning journeys with realistic exercise attempts, and provides comprehensive
analysis and recommendations for each user.
"""

import sys
import os
import json
import random
import tempfile
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem
from math_learning.algebra_learning import AlgebraLearningSystem


class UserProfile:
    """Represents a user's learning characteristics and preferences."""
    
    def __init__(self, user_id: str, name: str, learning_style: str, 
                 base_ability: float, concept_preferences: Dict[str, float]):
        self.user_id = user_id
        self.name = name
        self.learning_style = learning_style  # "visual", "analytical", "practical", "mixed"
        self.base_ability = base_ability  # 0.0 to 1.0
        self.concept_preferences = concept_preferences  # concept_category -> preference_modifier
        self.learning_speed = random.uniform(0.7, 1.3)  # How fast they learn
        self.consistency = random.uniform(0.6, 0.9)  # How consistent their performance is


class MultiUserSimulation:
    """Simulates learning for multiple users with different profiles."""
    
    def __init__(self):
        self.algebra_system = AlgebraLearningSystem()
        self.personalized_system = PersonalizedLearningSystem(self.algebra_system.knowledge_graph)
        self.users: Dict[str, Tuple[UserProfile, LearningGraph]] = {}
        self.simulation_results: Dict[str, Dict[str, Any]] = {}
        
    def create_user_profiles(self) -> List[UserProfile]:
        """Create diverse user profiles for simulation."""
        profiles = [
            UserProfile(
                user_id="alice_2024",
                name="Alice Chen",
                learning_style="analytical",
                base_ability=0.75,
                concept_preferences={
                    "Number Sense": 0.8,
                    "Algebra": 0.9,
                    "Geometry": 0.6,
                    "Statistics": 0.7
                }
            ),
            UserProfile(
                user_id="bob_2024", 
                name="Bob Martinez",
                learning_style="visual",
                base_ability=0.65,
                concept_preferences={
                    "Number Sense": 0.7,
                    "Algebra": 0.5,
                    "Geometry": 0.9,
                    "Statistics": 0.6
                }
            ),
            UserProfile(
                user_id="carol_2024",
                name="Carol Johnson",
                learning_style="practical",
                base_ability=0.55,
                concept_preferences={
                    "Number Sense": 0.9,
                    "Algebra": 0.4,
                    "Geometry": 0.7,
                    "Statistics": 0.8
                }
            ),
            UserProfile(
                user_id="david_2024",
                name="David Kim",
                learning_style="mixed",
                base_ability=0.85,
                concept_preferences={
                    "Number Sense": 0.8,
                    "Algebra": 0.8,
                    "Geometry": 0.8,
                    "Statistics": 0.7
                }
            ),
            UserProfile(
                user_id="emma_2024",
                name="Emma Wilson",
                learning_style="analytical",
                base_ability=0.45,
                concept_preferences={
                    "Number Sense": 0.6,
                    "Algebra": 0.7,
                    "Geometry": 0.5,
                    "Statistics": 0.9
                }
            ),
            UserProfile(
                user_id="frank_2024",
                name="Frank Rodriguez",
                learning_style="visual",
                base_ability=0.70,
                concept_preferences={
                    "Number Sense": 0.8,
                    "Algebra": 0.6,
                    "Geometry": 0.9,
                    "Statistics": 0.5
                }
            ),
            UserProfile(
                user_id="grace_2024",
                name="Grace Thompson",
                learning_style="practical",
                base_ability=0.60,
                concept_preferences={
                    "Number Sense": 0.9,
                    "Algebra": 0.5,
                    "Geometry": 0.6,
                    "Statistics": 0.8
                }
            ),
            UserProfile(
                user_id="henry_2024",
                name="Henry Lee",
                learning_style="mixed",
                base_ability=0.80,
                concept_preferences={
                    "Number Sense": 0.7,
                    "Algebra": 0.9,
                    "Geometry": 0.7,
                    "Statistics": 0.6
                }
            ),
            UserProfile(
                user_id="iris_2024",
                name="Iris Patel",
                learning_style="analytical",
                base_ability=0.90,
                concept_preferences={
                    "Number Sense": 0.8,
                    "Algebra": 0.9,
                    "Geometry": 0.8,
                    "Statistics": 0.9
                }
            ),
            UserProfile(
                user_id="jack_2024",
                name="Jack Brown",
                learning_style="visual",
                base_ability=0.50,
                concept_preferences={
                    "Number Sense": 0.7,
                    "Algebra": 0.4,
                    "Geometry": 0.8,
                    "Statistics": 0.5
                }
            )
        ]
        return profiles
    
    def calculate_success_probability(self, profile: UserProfile, concept, difficulty: float, 
                                    current_mastery: float, session_num: int) -> float:
        """Calculate the probability of success for a user on a specific exercise."""
        # Base success rate from user's ability
        base_rate = profile.base_ability
        
        # Adjust for concept preference
        category_preference = profile.concept_preferences.get(concept.category, 0.7)
        base_rate *= category_preference
        
        # Adjust for current mastery (higher mastery = higher success rate)
        mastery_boost = current_mastery * 0.4
        
        # Adjust for difficulty (higher difficulty = lower success rate)
        difficulty_penalty = difficulty * 0.3
        
        # Learning progression (users get better over time)
        progress_boost = min(0.2, session_num * 0.02)
        
        # Apply learning speed
        speed_factor = 1.0 + (profile.learning_speed - 1.0) * 0.3
        
        # Calculate final probability
        success_prob = (base_rate + mastery_boost + progress_boost - difficulty_penalty) * speed_factor
        
        # Add consistency factor (some randomness)
        consistency_range = (1.0 - profile.consistency) * 0.3
        random_factor = random.uniform(-consistency_range, consistency_range)
        success_prob += random_factor
        
        # Clamp to reasonable bounds
        return max(0.05, min(0.95, success_prob))
    
    def simulate_user_learning_journey(self, profile: UserProfile, num_sessions: int = 12) -> LearningGraph:
        """Simulate a complete learning journey for a user."""
        print(f"\n🎓 Simulating learning journey for {profile.name}")
        print(f"   Learning Style: {profile.learning_style}")
        print(f"   Base Ability: {profile.base_ability:.1%}")
        print(f"   Learning Speed: {profile.learning_speed:.2f}x")
        print("-" * 60)
        
        # Create learning graph
        lg = LearningGraph(profile.user_id, profile.name)
        
        # Get concepts to work with (focus on basic to intermediate)
        all_concepts = self.algebra_system.knowledge_graph.get_all_concepts()
        available_concepts = [c for c in all_concepts if c.difficulty <= 3]
        
        # Simulate learning sessions over time
        for session_num in range(1, num_sessions + 1):
            print(f"  📅 Session {session_num:2d}: ", end="")
            
            # Number of exercises per session (varies by user and session)
            base_exercises = 8 + int(profile.learning_speed * 5)
            exercises_this_session = random.randint(base_exercises - 3, base_exercises + 5)
            
            session_successes = 0
            session_attempts = 0
            
            for exercise_num in range(exercises_this_session):
                # Select concept based on current mastery and preferences
                concept = self.select_concept_for_user(profile, lg, available_concepts, session_num)
                
                if not concept:
                    continue
                
                # Calculate exercise difficulty (progressive)
                base_difficulty = 0.3 + (session_num - 1) * 0.04
                difficulty = base_difficulty + random.uniform(-0.1, 0.2)
                difficulty = max(0.1, min(0.9, difficulty))
                
                # Get current mastery
                current_mastery = lg.get_mastery(concept.id)
                
                # Calculate success probability
                success_prob = self.calculate_success_probability(
                    profile, concept, difficulty, current_mastery, session_num
                )
                
                # Determine success
                success = random.random() < success_prob
                
                # Record exercise attempt
                exercise_id = f"session_{session_num:02d}_ex_{exercise_num:02d}_{concept.id}"
                lg.record_exercise_attempt(exercise_id, concept.id, success, difficulty, 1.0)
                
                session_attempts += 1
                if success:
                    session_successes += 1
            
            # Show session summary
            session_rate = session_successes / session_attempts if session_attempts > 0 else 0
            print(f"{session_successes:2d}/{session_attempts:2d} ({session_rate:.1%}) - "
                  f"Concepts: {len(set(lg.concept_mastery.keys()))}")
        
        print(f"  ✅ Journey complete: {len(lg.exercise_history)} total exercises")
        return lg
    
    def select_concept_for_user(self, profile: UserProfile, lg: LearningGraph, 
                               available_concepts: List, session_num: int):
        """Select an appropriate concept for the user to work on."""
        # Early sessions: focus on basic concepts
        if session_num <= 3:
            candidates = [c for c in available_concepts if c.difficulty <= 1]
        elif session_num <= 6:
            candidates = [c for c in available_concepts if c.difficulty <= 2]
        else:
            candidates = available_concepts
        
        if not candidates:
            candidates = available_concepts
        
        # Weight selection by mastery level and preferences
        weights = []
        for concept in candidates:
            mastery = lg.get_mastery(concept.id)
            
            # Prefer concepts that are partially learned but not mastered
            if mastery < 0.2:
                mastery_weight = 0.8  # High weight for new concepts
            elif mastery < 0.7:
                mastery_weight = 1.0  # Highest weight for learning concepts
            else:
                mastery_weight = 0.3  # Lower weight for mastered concepts
            
            # Apply user preferences
            category_pref = profile.concept_preferences.get(concept.category, 0.7)
            
            # Final weight
            weight = mastery_weight * category_pref
            weights.append(weight)
        
        # Select concept based on weights
        if weights:
            return random.choices(candidates, weights=weights)[0]
        else:
            return random.choice(candidates) if candidates else None
    
    def analyze_user_performance(self, profile: UserProfile, lg: LearningGraph) -> Dict[str, Any]:
        """Analyze a user's learning performance and generate insights."""
        print(f"\n🔍 Analyzing performance for {profile.name}")
        print("-" * 50)
        
        # Basic statistics
        total_exercises = len(lg.exercise_history)
        total_successes = sum(
            1 for attempts in lg.exercise_history.values() 
            for attempt in attempts if attempt['result']
        )
        overall_success_rate = total_successes / max(1, sum(len(attempts) for attempts in lg.exercise_history.values()))
        
        # Concept mastery analysis
        mastered_concepts = lg.get_mastered_concepts(threshold=0.7)
        struggling_concepts = lg.get_struggling_concepts(threshold=0.3)
        
        # Learning patterns analysis
        patterns = self.personalized_system.analyze_learning_patterns(lg)
        weaknesses = self.personalized_system.detect_weaknesses(lg)
        recommendations = self.personalized_system.get_adaptive_recommendations(lg, num_recommendations=5)
        insights = self.personalized_system.generate_learning_insights(lg)
        
        # Concept category performance
        category_performance = {}
        for concept_id, mastery_data in lg.concept_mastery.items():
            concept = self.algebra_system.knowledge_graph.get_concept(concept_id)
            if concept:
                category = concept.category
                if category not in category_performance:
                    category_performance[category] = []
                category_performance[category].append(mastery_data['mastery'])
        
        # Calculate average mastery per category
        category_averages = {
            category: sum(masteries) / len(masteries)
            for category, masteries in category_performance.items()
        }
        
        analysis = {
            'user_profile': {
                'user_id': profile.user_id,
                'name': profile.name,
                'learning_style': profile.learning_style,
                'base_ability': profile.base_ability,
                'learning_speed': profile.learning_speed,
                'consistency': profile.consistency
            },
            'performance_metrics': {
                'total_exercises': total_exercises,
                'overall_success_rate': overall_success_rate,
                'concepts_tracked': len(lg.concept_mastery),
                'mastered_concepts': len(mastered_concepts),
                'struggling_concepts': len(struggling_concepts)
            },
            'mastery_analysis': {
                'mastered_concepts': mastered_concepts,
                'struggling_concepts': struggling_concepts,
                'category_performance': category_averages
            },
            'learning_patterns': {
                concept_id: {
                    'attempts': pattern.attempts,
                    'success_rate': pattern.success_rate,
                    'learning_velocity': pattern.learning_velocity,
                    'retention_rate': pattern.retention_rate
                }
                for concept_id, pattern in patterns.items()
            },
            'weaknesses': [
                {
                    'concept_id': w.concept_id,
                    'weakness_type': w.weakness_type,
                    'severity': w.severity,
                    'confidence_level': w.confidence_level,
                    'suggested_interventions': w.suggested_interventions
                }
                for w in weaknesses
            ],
            'recommendations': [
                {
                    'type': rec['type'],
                    'concept_id': rec['concept_id'],
                    'priority': rec['priority'],
                    'reason': rec['reason']
                }
                for rec in recommendations
            ],
            'insights': [
                {
                    'type': insight.insight_type,
                    'title': insight.title,
                    'description': insight.description,
                    'confidence': insight.confidence
                }
                for insight in insights
            ]
        }
        
        # Print summary
        print(f"  📊 Total Exercises: {total_exercises}")
        print(f"  ✅ Success Rate: {overall_success_rate:.1%}")
        print(f"  🎯 Concepts Mastered: {len(mastered_concepts)}")
        print(f"  ⚠️  Struggling Concepts: {len(struggling_concepts)}")
        print(f"  🔍 Weaknesses Detected: {len(weaknesses)}")
        print(f"  💡 Recommendations: {len(recommendations)}")
        print(f"  🧠 Insights Generated: {len(insights)}")
        
        return analysis
    
    def generate_user_report(self, profile: UserProfile, lg: LearningGraph, analysis: Dict[str, Any]) -> str:
        """Generate a detailed report for a user."""
        report = f"""
{'='*80}
LEARNING REPORT FOR {profile.name.upper()}
{'='*80}

👤 USER PROFILE:
   • User ID: {profile.user_id}
   • Learning Style: {profile.learning_style}
   • Base Ability: {profile.base_ability:.1%}
   • Learning Speed: {profile.learning_speed:.2f}x average
   • Consistency: {profile.consistency:.1%}

📊 PERFORMANCE SUMMARY:
   • Total Exercises Completed: {analysis['performance_metrics']['total_exercises']}
   • Overall Success Rate: {analysis['performance_metrics']['overall_success_rate']:.1%}
   • Concepts Tracked: {analysis['performance_metrics']['concepts_tracked']}
   • Concepts Mastered (≥70%): {analysis['performance_metrics']['mastered_concepts']}
   • Struggling Concepts (≤30%): {analysis['performance_metrics']['struggling_concepts']}

🎯 MASTERY BY CATEGORY:
"""
        
        for category, avg_mastery in analysis['mastery_analysis']['category_performance'].items():
            mastery_level = "🟢 Strong" if avg_mastery >= 0.7 else "🟡 Developing" if avg_mastery >= 0.4 else "🔴 Needs Work"
            report += f"   • {category}: {avg_mastery:.1%} {mastery_level}\n"
        
        report += f"""
✅ MASTERED CONCEPTS:
"""
        mastered = analysis['mastery_analysis']['mastered_concepts']
        if mastered:
            for concept_id in mastered[:10]:  # Show top 10
                concept = self.algebra_system.knowledge_graph.get_concept(concept_id)
                mastery = lg.get_mastery(concept_id)
                concept_name = concept.name if concept else concept_id
                report += f"   • {concept_name}: {mastery:.1%}\n"
            if len(mastered) > 10:
                report += f"   ... and {len(mastered) - 10} more\n"
        else:
            report += "   • No concepts fully mastered yet\n"
        
        report += f"""
⚠️  STRUGGLING CONCEPTS:
"""
        struggling = analysis['mastery_analysis']['struggling_concepts']
        if struggling:
            for concept_id in struggling:
                concept = self.algebra_system.knowledge_graph.get_concept(concept_id)
                mastery = lg.get_mastery(concept_id)
                concept_name = concept.name if concept else concept_id
                report += f"   • {concept_name}: {mastery:.1%}\n"
        else:
            report += "   • No concepts identified as struggling\n"
        
        report += f"""
🔍 DETECTED WEAKNESSES:
"""
        weaknesses = analysis['weaknesses']
        if weaknesses:
            for weakness in weaknesses[:5]:  # Show top 5
                concept = self.algebra_system.knowledge_graph.get_concept(weakness['concept_id'])
                concept_name = concept.name if concept else weakness['concept_id']
                report += f"   • {concept_name}: {weakness['weakness_type']} (severity: {weakness['severity']:.1%})\n"
                report += f"     Interventions: {', '.join(weakness['suggested_interventions'][:2])}\n"
        else:
            report += "   • No significant weaknesses detected\n"
        
        report += f"""
💡 ADAPTIVE RECOMMENDATIONS:
"""
        recommendations = analysis['recommendations']
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                concept = self.algebra_system.knowledge_graph.get_concept(rec['concept_id'])
                concept_name = concept.name if concept else rec['concept_id']
                report += f"   {i}. {concept_name} ({rec['type']}) - Priority: {rec['priority']:.1%}\n"
                report += f"      Reason: {rec['reason']}\n"
        else:
            report += "   • No specific recommendations at this time\n"
        
        report += f"""
🧠 LEARNING INSIGHTS:
"""
        insights = analysis['insights']
        if insights:
            for insight in insights:
                report += f"   • {insight['title']}\n"
                report += f"     {insight['description']} (confidence: {insight['confidence']:.1%})\n"
        else:
            report += "   • Continue practicing to generate more insights\n"
        
        report += f"""
📈 NEXT STEPS:
   1. Focus on struggling concepts with targeted practice
   2. Follow adaptive recommendations in priority order
   3. Continue building on mastered concepts to maintain retention
   4. Consider adjusting study approach based on learning insights

{'='*80}
"""
        return report
    
    def run_simulation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run the complete multi-user simulation."""
        print("🚀 MULTI-USER LEARNING SIMULATION")
        print("="*80)
        
        # Create user profiles
        profiles = self.create_user_profiles()
        print(f"👥 Created {len(profiles)} user profiles")
        
        # Simulate learning for each user
        all_results = {}
        
        for i, profile in enumerate(profiles, 1):
            print(f"\n[{i}/{len(profiles)}] Processing {profile.name}...")
            
            # Simulate learning journey
            lg = self.simulate_user_learning_journey(profile)
            
            # Analyze performance
            analysis = self.analyze_user_performance(profile, lg)
            
            # Store results
            self.users[profile.user_id] = (profile, lg)
            self.simulation_results[profile.user_id] = analysis
            all_results[profile.user_id] = {
                'profile': profile,
                'learning_graph': lg,
                'analysis': analysis
            }
        
        # Generate summary statistics
        summary = self.generate_simulation_summary()
        
        # Save results if requested
        if save_results:
            self.save_simulation_results(all_results, summary)
        
        # Print final summary
        self.print_simulation_summary(summary)
        
        return {
            'users': all_results,
            'summary': summary
        }
    
    def generate_simulation_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all users."""
        if not self.simulation_results:
            return {}
        
        # Collect metrics across all users
        all_success_rates = []
        all_mastered_counts = []
        all_struggling_counts = []
        all_exercise_counts = []
        learning_style_performance = {}
        category_performance_by_style = {}
        
        for user_id, analysis in self.simulation_results.items():
            profile = self.users[user_id][0]
            
            # Basic metrics
            all_success_rates.append(analysis['performance_metrics']['overall_success_rate'])
            all_mastered_counts.append(analysis['performance_metrics']['mastered_concepts'])
            all_struggling_counts.append(analysis['performance_metrics']['struggling_concepts'])
            all_exercise_counts.append(analysis['performance_metrics']['total_exercises'])
            
            # Learning style analysis
            style = profile.learning_style
            if style not in learning_style_performance:
                learning_style_performance[style] = []
            learning_style_performance[style].append(analysis['performance_metrics']['overall_success_rate'])
            
            # Category performance by learning style
            if style not in category_performance_by_style:
                category_performance_by_style[style] = {}
            
            for category, performance in analysis['mastery_analysis']['category_performance'].items():
                if category not in category_performance_by_style[style]:
                    category_performance_by_style[style][category] = []
                category_performance_by_style[style][category].append(performance)
        
        # Calculate averages
        avg_success_rate = sum(all_success_rates) / len(all_success_rates)
        avg_mastered = sum(all_mastered_counts) / len(all_mastered_counts)
        avg_struggling = sum(all_struggling_counts) / len(all_struggling_counts)
        avg_exercises = sum(all_exercise_counts) / len(all_exercise_counts)
        
        # Learning style averages
        style_averages = {
            style: sum(rates) / len(rates)
            for style, rates in learning_style_performance.items()
        }
        
        return {
            'total_users': len(self.simulation_results),
            'overall_metrics': {
                'avg_success_rate': avg_success_rate,
                'avg_mastered_concepts': avg_mastered,
                'avg_struggling_concepts': avg_struggling,
                'avg_exercises_completed': avg_exercises
            },
            'learning_style_performance': style_averages,
            'category_performance_by_style': category_performance_by_style,
            'individual_results': self.simulation_results
        }
    
    def print_simulation_summary(self, summary: Dict[str, Any]):
        """Print a comprehensive summary of the simulation results."""
        print(f"\n\n🎯 SIMULATION SUMMARY")
        print("="*80)
        
        print(f"👥 Total Users Simulated: {summary['total_users']}")
        print(f"📊 Average Success Rate: {summary['overall_metrics']['avg_success_rate']:.1%}")
        print(f"🎯 Average Concepts Mastered: {summary['overall_metrics']['avg_mastered_concepts']:.1f}")
        print(f"⚠️  Average Struggling Concepts: {summary['overall_metrics']['avg_struggling_concepts']:.1f}")
        print(f"📝 Average Exercises Completed: {summary['overall_metrics']['avg_exercises_completed']:.0f}")
        
        print(f"\n📚 PERFORMANCE BY LEARNING STYLE:")
        for style, avg_rate in summary['learning_style_performance'].items():
            print(f"   • {style.title()}: {avg_rate:.1%} success rate")
        
        print(f"\n🏆 TOP PERFORMERS:")
        # Sort users by success rate
        sorted_users = sorted(
            summary['individual_results'].items(),
            key=lambda x: x[1]['performance_metrics']['overall_success_rate'],
            reverse=True
        )
        
        for i, (user_id, analysis) in enumerate(sorted_users[:3], 1):
            profile = self.users[user_id][0]
            success_rate = analysis['performance_metrics']['overall_success_rate']
            mastered = analysis['performance_metrics']['mastered_concepts']
            print(f"   {i}. {profile.name}: {success_rate:.1%} success, {mastered} concepts mastered")
        
        print(f"\n📈 USERS NEEDING SUPPORT:")
        for i, (user_id, analysis) in enumerate(sorted_users[-3:], 1):
            profile = self.users[user_id][0]
            success_rate = analysis['performance_metrics']['overall_success_rate']
            struggling = analysis['performance_metrics']['struggling_concepts']
            print(f"   {i}. {profile.name}: {success_rate:.1%} success, {struggling} struggling concepts")
    
    def save_simulation_results(self, all_results: Dict[str, Any], summary: Dict[str, Any]):
        """Save simulation results to files."""
        print(f"\n💾 Saving simulation results...")
        
        # Create output directory
        output_dir = "simulation_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual user reports
        for user_id, result in all_results.items():
            profile = result['profile']
            lg = result['learning_graph']
            analysis = result['analysis']
            
            # Generate and save detailed report
            report = self.generate_user_report(profile, lg, analysis)
            report_file = os.path.join(output_dir, f"{user_id}_report.txt")
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Save learning graph data
            lg_file = os.path.join(output_dir, f"{user_id}_learning_graph.json")
            lg.save_to_file(lg_file)
            
            # Save analysis data
            analysis_file = os.path.join(output_dir, f"{user_id}_analysis.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
        
        # Save summary
        summary_file = os.path.join(output_dir, "simulation_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {output_dir}/")
        print(f"   📄 Individual reports: {len(all_results)} files")
        print(f"   📊 Learning graphs: {len(all_results)} files")
        print(f"   🔍 Analysis data: {len(all_results)} files")
        print(f"   📈 Summary report: 1 file")


def main():
    """Run the multi-user simulation."""
    print("🎓 Multi-User Learning Simulation System")
    print("This will create 10 users with different learning profiles,")
    print("simulate their learning journeys, and provide detailed analysis.")
    print()
    
    # Create and run simulation
    simulation = MultiUserSimulation()
    results = simulation.run_simulation(save_results=True)
    
    print(f"\n🎉 Simulation Complete!")
    print(f"Check the 'simulation_output' directory for detailed reports.")


if __name__ == "__main__":
    main() 