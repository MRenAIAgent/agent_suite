"""
Geometry Learning System Demo.

This script demonstrates how to use the geometry learning system
with inheritance-based components.
"""

from .geometry_learning_system import GeometryLearningSystem


def main():
    """Demonstrate the geometry learning system."""
    print("🔷 GEOMETRY LEARNING SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Initialize the geometry learning system
    print("🔧 Initializing geometry learning system...")
    geometry_system = GeometryLearningSystem()
    
    # Show system statistics
    print("\n📊 System Statistics:")
    stats = geometry_system.get_system_statistics()
    print(f"   📚 Knowledge Graph: {stats['knowledge_graph_stats']['total_concepts']} concepts")
    print(f"   📝 Exercise Bank: {stats['exercise_bank_stats']['total_exercises']} exercises")
    print(f"   📈 Average Exercise Difficulty: {stats['exercise_bank_stats']['average_difficulty']:.1f}")
    
    # Create a student learning graph
    print("\n👤 Creating student learning graph...")
    student_graph = geometry_system.create_student_learning_graph("student_001", "Alice")
    print(f"   Created learning graph for: {student_graph.name}")
    
    # Simulate some learning activity
    print("\n📝 Simulating geometry learning activity...")
    
    # Record some exercise attempts with geometry-specific metadata
    exercises = [
        ("ex_001", "geom_basic_shapes", True, "visual-identification", True, False),
        ("ex_002", "geom_basic_shapes", True, "calculation", False, False),
        ("ex_003", "geom_angles", False, "calculation", False, False),
        ("ex_004", "geom_angles", True, "visual-identification", True, False),
        ("ex_005", "geom_triangles", True, "construction", True, True),
        ("ex_006", "geom_area_perimeter", False, "calculation", False, False),
        ("ex_007", "geom_area_perimeter", True, "word-problem", True, False),
    ]
    
    for exercise_id, concept_id, result, exercise_type, visual_aids, construction in exercises:
        geometry_system.record_geometry_exercise_attempt(
            student_graph, exercise_id, concept_id, result, exercise_type, visual_aids, construction
        )
        status = "✅" if result else "❌"
        print(f"   {status} {exercise_id}: {concept_id} ({exercise_type})")
    
    # Analyze geometry gaps
    print("\n🔍 Analyzing geometry knowledge gaps...")
    gap_analysis = geometry_system.analyze_student_geometry_gaps(student_graph)
    
    print(f"   📊 Gap Summary:")
    summary = gap_analysis['gap_summary']
    print(f"      • Total gaps: {summary['total_gaps']}")
    print(f"      • Visual learning gaps: {summary['visual_gaps']}")
    print(f"      • Spatial reasoning gaps: {summary['spatial_gaps']}")
    print(f"      • Construction skill gaps: {summary['construction_gaps']}")
    
    print(f"\n   🎯 Learning Boundary: {len(gap_analysis['learning_boundary'])} concepts ready to learn")
    
    # Show recommended strategies
    print(f"\n   💡 Recommended Learning Strategies:")
    for strategy in gap_analysis['recommended_strategies'][:3]:
        print(f"      • {strategy}")
    
    # Get exercise recommendations
    print("\n📋 Getting geometry exercise recommendations...")
    recommendations = geometry_system.get_geometry_exercise_recommendations(
        student_graph, max_exercises=3
    )
    
    print(f"   Found {len(recommendations)} recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec.exercise.title}")
        print(f"      📝 {rec.reason}")
        print(f"      🎯 Category: {rec.geometry_category}")
        print(f"      👁️  Visual aids: {'Yes' if rec.requires_visual_aids else 'No'}")
        print(f"      🔨 Construction: {'Yes' if rec.requires_construction else 'No'}")
        print()
    
    # Test learning style recommendations
    print("🎨 Testing learning style recommendations...")
    for style in ['visual', 'kinesthetic', 'analytical']:
        style_recs = geometry_system.get_geometry_exercise_recommendations(
            student_graph, max_exercises=2, learning_style=style
        )
        print(f"   {style.title()} learner: {len(style_recs)} recommendations")
    
    # Get comprehensive insights
    print("\n🧠 Generating student geometry insights...")
    insights = geometry_system.get_student_geometry_insights(student_graph)
    
    print("   📈 Performance Analysis:")
    perf = insights['performance_analysis']
    
    # Visual learning effectiveness
    visual_eff = perf['visual_learning_effectiveness']
    print(f"      • Visual aid effectiveness: {visual_eff['visual_aid_effectiveness']:.2f}")
    
    # Construction performance
    construction = perf['construction_performance']
    print(f"      • Construction success rate: {construction['construction_success_rate']:.1%}")
    
    # Learning recommendations
    print(f"\n   🎯 Personalized Recommendations:")
    for rec in insights['learning_recommendations']:
        print(f"      • {rec}")
    
    # Show concept details
    print("\n📖 Geometry Concept Details Example:")
    concept_details = geometry_system.get_geometry_concept_details("Basic Shapes")
    if 'error' not in concept_details:
        print(f"   📝 Concept: {concept_details['name']}")
        print(f"   📊 Difficulty: {concept_details['difficulty']}/5")
        print(f"   ⏱️  Time to master: {concept_details['time_to_master']} minutes")
        print(f"   🎓 Grade level: {concept_details['grade_level']}")
        print(f"   📚 Related exercises: {concept_details['total_exercises']}")
    
    print("\n" + "=" * 60)
    print("🎉 GEOMETRY LEARNING SYSTEM DEMO COMPLETE!")
    print("=" * 60)
    
    print("\n✨ Key Features Demonstrated:")
    print("   ✅ Geometry knowledge graph initialization")
    print("   ✅ Personal learning graph with geometry-specific tracking")
    print("   ✅ Visual and spatial learning preference analysis")
    print("   ✅ Construction skill assessment")
    print("   ✅ Geometry-specific gap analysis")
    print("   ✅ Learning style-based recommendations")
    print("   ✅ Comprehensive geometry learning insights")
    print("   ✅ Exercise tracking with geometry metadata")


if __name__ == "__main__":
    main() 