#!/usr/bin/env python3
"""
User Analytics Dashboard

Provides comprehensive analytics and insights for the multi-user learning simulation,
including performance comparisons, learning pattern analysis, and recommendation summaries.
"""

import sys
import os
import json
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem

class UserAnalyticsDashboard:
    """Comprehensive analytics dashboard for multi-user learning data."""
    
    def __init__(self):
        self.users_data = []
        self.learning_systems = {}
        
    def load_user_data(self, simulation_data: Dict[str, Any]):
        """Load user data from simulation results."""
        self.users_data = []
        
        # Extract individual user results from the simulation data
        individual_results = simulation_data.get('individual_results', {})
        
        for user_id, user_data in individual_results.items():
            # Load the learning graph from separate file
            try:
                with open(f'simulation_output/{user_id}_learning_graph.json', 'r') as f:
                    learning_graph_data = json.load(f)
            except FileNotFoundError:
                print(f"⚠️  Warning: Learning graph file not found for {user_id}")
                learning_graph_data = {}
            
            # Convert the simulation data format to our expected format
            user_info = {
                'user_id': user_id,
                'name': user_data['user_profile']['name'],
                'learning_style': user_data['user_profile']['learning_style'],
                'base_ability': user_data['user_profile']['base_ability'],
                'learning_speed': user_data['user_profile']['learning_speed'],
                'performance_metrics': user_data['performance_metrics'],
                'mastery_analysis': user_data['mastery_analysis'],
                'learning_patterns': user_data.get('learning_patterns', {}),
                'weaknesses': user_data.get('weaknesses', {}),
                'recommendations': user_data.get('recommendations', []),
                'insights': user_data.get('insights', []),
                'learning_graph': learning_graph_data
            }
            self.users_data.append(user_info)
        
        print(f"✅ Loaded data for {len(self.users_data)} users")
        
        # Create learning systems for each user
        for user_data in self.users_data:
            user_id = user_data['user_id']
            if user_data['learning_graph']:
                try:
                    learning_graph = LearningGraph.from_dict(user_data['learning_graph'])
                    self.learning_systems[user_id] = PersonalizedLearningSystem(learning_graph)
                except Exception as e:
                    print(f"⚠️  Warning: Could not create learning system for {user_id}: {e}")
            else:
                print(f"⚠️  Warning: No learning graph data for {user_id}")
    
    def generate_comprehensive_report(self):
        """Generate and display comprehensive analytics report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE USER ANALYTICS DASHBOARD")
        print("=" * 80)
        
        # Overall Performance Analysis
        overall_stats = self.analyze_overall_performance()
        print(f"\n📈 OVERALL PERFORMANCE SUMMARY")
        print("-" * 50)
        print(f"👥 Total Users Analyzed: {overall_stats['total_users']}")
        print(f"📊 Average Success Rate: {overall_stats['avg_success_rate']:.1f}%")
        print(f"📊 Median Success Rate: {overall_stats['median_success_rate']:.1f}%")
        print(f"📊 Success Rate Std Dev: {overall_stats['success_rate_std']:.1f}%")
        print(f"🎯 Average Exercises Completed: {overall_stats['avg_exercises']:.0f}")
        print(f"✅ Average Concepts Mastered: {overall_stats['avg_mastered']:.1f}")
        print(f"⚠️  Average Struggling Concepts: {overall_stats['avg_struggling']:.1f}")
        
        # Learning Style Analysis
        style_analysis = self.analyze_learning_styles()
        print(f"\n🎨 LEARNING STYLE PERFORMANCE")
        print("-" * 50)
        for style, data in sorted(style_analysis.items(), key=lambda x: x[1]['avg_success_rate'], reverse=True):
            print(f"📚 {style.title()}: {data['avg_success_rate']:.1f}% avg ({data['user_count']} users)")
        
        # Individual User Analysis
        print(f"\n👤 INDIVIDUAL USER ANALYSIS")
        print("-" * 50)
        
        # Sort users by success rate
        sorted_users = sorted(self.users_data, 
                            key=lambda x: x['performance_metrics']['overall_success_rate'], 
                            reverse=True)
        
        print("🏆 TOP PERFORMERS:")
        for i, user in enumerate(sorted_users[:3], 1):
            metrics = user['performance_metrics']
            print(f"   {i}. {user['name']} ({user['learning_style']})")
            print(f"      ✅ {metrics['overall_success_rate']*100:.1f}% success rate")
            print(f"      🎯 {metrics['mastered_concepts']} concepts mastered")
            print(f"      📝 {metrics['total_exercises']} exercises completed")
        
        print("\n📈 USERS NEEDING SUPPORT:")
        for i, user in enumerate(sorted_users[-3:], 1):
            metrics = user['performance_metrics']
            print(f"   {i}. {user['name']} ({user['learning_style']})")
            print(f"      ⚠️  {metrics['overall_success_rate']*100:.1f}% success rate")
            print(f"      🔍 {metrics['struggling_concepts']} struggling concepts")
            print(f"      📝 {metrics['total_exercises']} exercises completed")
        
        # Detailed User Profiles
        print(f"\n📋 DETAILED USER PROFILES")
        print("-" * 50)
        
        for user in sorted_users:
            self.print_user_summary(user)
        
        # Recommendations Summary
        print(f"\n💡 SYSTEM RECOMMENDATIONS SUMMARY")
        print("-" * 50)
        
        all_recommendations = []
        for user in self.users_data:
            recommendations = user.get('recommendations', [])
            if isinstance(recommendations, list):
                all_recommendations.extend(recommendations)
        
        if all_recommendations:
            print(f"📊 Total Recommendations Generated: {len(all_recommendations)}")
            print("🔍 Common Recommendation Types:")
            
            # Count recommendation types (simplified)
            rec_types = {}
            for rec in all_recommendations:
                if isinstance(rec, dict) and 'type' in rec:
                    rec_type = rec['type']
                    rec_types[rec_type] = rec_types.get(rec_type, 0) + 1
            
            for rec_type, count in sorted(rec_types.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {rec_type}: {count} recommendations")
        else:
            print("⚠️  No recommendations data available")
        
        print("\n" + "=" * 80)

    def analyze_overall_performance(self) -> Dict[str, Any]:
        """Analyze overall performance across all users."""
        # Calculate average success rates
        success_rates = []
        total_exercises = []
        mastered_concepts = []
        struggling_concepts = []
        
        for user in self.users_data:
            metrics = user['performance_metrics']
            success_rates.append(metrics['overall_success_rate'] * 100)
            total_exercises.append(metrics['total_exercises'])
            mastered_concepts.append(metrics['mastered_concepts'])
            struggling_concepts.append(metrics['struggling_concepts'])
        
        return {
            'avg_success_rate': statistics.mean(success_rates),
            'median_success_rate': statistics.median(success_rates),
            'success_rate_std': statistics.stdev(success_rates) if len(success_rates) > 1 else 0,
            'avg_exercises': statistics.mean(total_exercises),
            'avg_mastered': statistics.mean(mastered_concepts),
            'avg_struggling': statistics.mean(struggling_concepts),
            'total_users': len(self.users_data)
        }

    def analyze_learning_styles(self) -> Dict[str, Any]:
        """Analyze performance by learning style."""
        style_performance = {}
        
        for user in self.users_data:
            style = user['learning_style']
            success_rate = user['performance_metrics']['overall_success_rate'] * 100
            
            if style not in style_performance:
                style_performance[style] = []
            style_performance[style].append(success_rate)
        
        # Calculate averages for each style
        style_averages = {}
        for style, rates in style_performance.items():
            style_averages[style] = {
                'avg_success_rate': statistics.mean(rates),
                'user_count': len(rates),
                'success_rates': rates
            }
        
        return style_averages

    def print_user_summary(self, user: Dict[str, Any]):
        """Print a detailed summary for a single user."""
        metrics = user['performance_metrics']
        mastery = user['mastery_analysis']
        
        print(f"\n🔹 {user['name']} ({user['user_id']})")
        print(f"   📚 Learning Style: {user['learning_style'].title()}")
        print(f"   🎯 Base Ability: {user['base_ability']*100:.0f}%")
        print(f"   ⚡ Learning Speed: {user['learning_speed']:.2f}x")
        print(f"   📊 Success Rate: {metrics['overall_success_rate']*100:.1f}%")
        print(f"   📝 Total Exercises: {metrics['total_exercises']}")
        print(f"   ✅ Concepts Mastered: {metrics['mastered_concepts']}")
        print(f"   ⚠️  Struggling Concepts: {metrics['struggling_concepts']}")
        
        # Show some mastered concepts
        if mastery.get('mastered_concepts'):
            mastered_sample = mastery['mastered_concepts'][:3]
            print(f"   🏆 Sample Mastered: {', '.join(mastered_sample)}")
        
        # Show some struggling concepts
        if mastery.get('struggling_concepts'):
            struggling_sample = mastery['struggling_concepts'][:3]
            print(f"   🔍 Sample Struggling: {', '.join(struggling_sample)}")
        
        # Show recommendations if available
        recommendations = user.get('recommendations', [])
        if recommendations and len(recommendations) > 0:
            print(f"   💡 Recommendations: {len(recommendations)} generated")
        
        # Show insights if available
        insights = user.get('insights', [])
        if insights and len(insights) > 0:
            print(f"   🧠 Insights: {len(insights)} generated")

def main():
    """Main function to run the analytics dashboard."""
    # Load the simulation results
    try:
        with open('simulation_output/simulation_summary.json', 'r') as f:
            simulation_data = json.load(f)
        
        dashboard = UserAnalyticsDashboard()
        dashboard.load_user_data(simulation_data)
        dashboard.generate_comprehensive_report()
        
        print("\n" + "=" * 80)
        print("📋 Analytics dashboard completed successfully!")
        print("=" * 80)
        
    except FileNotFoundError:
        print("❌ Error: simulation_output/simulation_summary.json not found.")
        print("Please run multi_user_simulation.py first to generate the data.")
    except Exception as e:
        print(f"❌ Error loading simulation data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 