"""
Integrated Diagnostic Demo - Knowledge Graph + Error Analysis
Demonstrates how student errors map to specific concept gaps and prerequisite chains.
"""

import random
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# Simplified knowledge graph representation
@dataclass
class Concept:
    id: str
    name: str
    difficulty: int
    time_to_master: int
    category: str
    prerequisites: List[str]

class ErrorType(Enum):
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPUTATIONAL = "computational"
    TRANSLATION = "translation"

@dataclass
class ErrorMapping:
    wrong_answer: str
    concept_id: str
    concept_name: str
    error_type: ErrorType
    error_description: str
    remediation_hint: str
    frequency: float
    prerequisite_gaps: List[str]  # Which prerequisites are likely missing

@dataclass
class TestQuestion:
    id: str
    question_text: str
    correct_answer: str
    difficulty: int
    primary_concepts: List[str]
    common_errors: List[ErrorMapping]
    source: str = ""

@dataclass
class StudentResponse:
    question_id: str
    student_answer: str
    is_correct: bool
    error_mapping: Optional[ErrorMapping]
    confidence_level: int

@dataclass
class LearningPath:
    concept_id: str
    concept_name: str
    current_mastery: float
    target_mastery: float
    prerequisites_needed: List[str]
    estimated_study_time: int
    priority: int

class IntegratedDiagnosticSystem:
    """Integrated system that combines knowledge graph with error analysis."""
    
    def __init__(self):
        self.concepts = self._create_concept_graph()
        self.questions = self._create_integrated_questions()
        self.concept_map = {c.id: c for c in self.concepts}
    
    def _create_concept_graph(self) -> List[Concept]:
        """Create a simplified version of our K-12 algebra knowledge graph."""
        return [
            # Foundation concepts
            Concept("COUNTING", "Counting Up/Down", 1, 30, "Number Sense", []),
            Concept("ADDITION_FACTS", "Addition Facts", 1, 45, "Number Sense", ["COUNTING"]),
            Concept("MULTIPLICATION_FACTS", "Multiplication Facts", 2, 60, "Number Sense", ["ADDITION_FACTS"]),
            Concept("FRACTIONS", "Fractions as Part-Whole", 2, 90, "Number Sense", ["MULTIPLICATION_FACTS"]),
            
            # Pre-algebra
            Concept("ORDER_OF_OPERATIONS", "Order of Operations", 2, 45, "Pre-Algebra", ["MULTIPLICATION_FACTS"]),
            Concept("DISTRIBUTIVE_PROPERTY", "Distributive Property", 3, 60, "Pre-Algebra", ["MULTIPLICATION_FACTS"]),
            Concept("COMBINE_LIKE_TERMS", "Combine Like Terms", 3, 75, "Pre-Algebra", ["DISTRIBUTIVE_PROPERTY"]),
            Concept("ONE_STEP_EQUATIONS", "One-Step Linear Equations", 3, 60, "Pre-Algebra", ["ORDER_OF_OPERATIONS"]),
            Concept("TWO_STEP_EQUATIONS", "Two-Step Linear Equations", 4, 90, "Pre-Algebra", ["ONE_STEP_EQUATIONS", "COMBINE_LIKE_TERMS"]),
            
            # Advanced concepts
            Concept("QUADRATIC_FORMULA", "Quadratic Formula", 4, 120, "Quadratics", ["TWO_STEP_EQUATIONS", "ORDER_OF_OPERATIONS"]),
            Concept("SYSTEMS_EQUATIONS", "Systems of Linear Equations", 4, 150, "Linear Functions", ["TWO_STEP_EQUATIONS"]),
            Concept("FUNCTION_COMPOSITION", "Function Composition", 4, 120, "Function Tools", ["ORDER_OF_OPERATIONS"]),
            Concept("EXPONENTIAL_GROWTH", "Exponential Growth", 4, 135, "Exponentials", ["MULTIPLICATION_FACTS"]),
            Concept("RATIONAL_EXPRESSIONS", "Rational Expressions", 4, 180, "Polynomials", ["DISTRIBUTIVE_PROPERTY", "FRACTIONS"])
        ]
    
    def _create_integrated_questions(self) -> List[TestQuestion]:
        """Create test questions with detailed prerequisite error mapping."""
        return [
            TestQuestion(
                id="INT_001",
                question_text="Solve: 3x + 7 = 22",
                correct_answer="x = 5",
                difficulty=4,
                primary_concepts=["TWO_STEP_EQUATIONS"],
                source="Integrated Assessment",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="x = 29",
                        concept_id="TWO_STEP_EQUATIONS",
                        concept_name="Two-Step Linear Equations",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Added 7 to both sides instead of subtracting",
                        remediation_hint="To isolate x, subtract 7 from both sides first",
                        frequency=0.25,
                        prerequisite_gaps=["ONE_STEP_EQUATIONS"]
                    ),
                    ErrorMapping(
                        wrong_answer="x = 15",
                        concept_id="TWO_STEP_EQUATIONS", 
                        concept_name="Two-Step Linear Equations",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Subtracted 7 correctly but forgot to divide by 3",
                        remediation_hint="After getting 3x = 15, divide both sides by 3",
                        frequency=0.20,
                        prerequisite_gaps=["ONE_STEP_EQUATIONS"]
                    ),
                    ErrorMapping(
                        wrong_answer="x = 7.33",
                        concept_id="ORDER_OF_OPERATIONS",
                        concept_name="Order of Operations",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Divided 22 by 3 first, ignoring order of operations",
                        remediation_hint="Follow order of operations: handle addition/subtraction before division",
                        frequency=0.15,
                        prerequisite_gaps=["ORDER_OF_OPERATIONS"]
                    )
                ]
            ),
            
            TestQuestion(
                id="INT_002",
                question_text="Simplify: 2(x + 3) + 4x",
                correct_answer="6x + 6",
                difficulty=3,
                primary_concepts=["DISTRIBUTIVE_PROPERTY", "COMBINE_LIKE_TERMS"],
                source="Integrated Assessment",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="2x + 3 + 4x",
                        concept_id="DISTRIBUTIVE_PROPERTY",
                        concept_name="Distributive Property",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Didn't distribute the 2 to both terms inside parentheses",
                        remediation_hint="Multiply 2 by both x and 3: 2(x + 3) = 2x + 6",
                        frequency=0.30,
                        prerequisite_gaps=["MULTIPLICATION_FACTS"]
                    ),
                    ErrorMapping(
                        wrong_answer="2x + 6 + 4x",
                        concept_id="COMBINE_LIKE_TERMS",
                        concept_name="Combine Like Terms",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Distributed correctly but didn't combine like terms",
                        remediation_hint="Combine the x terms: 2x + 4x = 6x",
                        frequency=0.25,
                        prerequisite_gaps=["ADDITION_FACTS"]
                    ),
                    ErrorMapping(
                        wrong_answer="6x + 3",
                        concept_id="DISTRIBUTIVE_PROPERTY",
                        concept_name="Distributive Property",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Combined like terms correctly but made arithmetic error: 2×3 = 6, not 3",
                        remediation_hint="Check multiplication: 2 × 3 = 6",
                        frequency=0.20,
                        prerequisite_gaps=["MULTIPLICATION_FACTS"]
                    )
                ]
            ),
            
            TestQuestion(
                id="INT_003",
                question_text="If f(x) = x + 2 and g(x) = 3x, find f(g(1))",
                correct_answer="5",
                difficulty=4,
                primary_concepts=["FUNCTION_COMPOSITION"],
                source="Integrated Assessment",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="6",
                        concept_id="FUNCTION_COMPOSITION",
                        concept_name="Function Composition",
                        error_type=ErrorType.CONCEPTUAL,
                        error_description="Calculated g(f(1)) instead of f(g(1))",
                        remediation_hint="f(g(1)) means first find g(1), then apply f to that result",
                        frequency=0.35,
                        prerequisite_gaps=["ORDER_OF_OPERATIONS"]
                    ),
                    ErrorMapping(
                        wrong_answer="3",
                        concept_id="FUNCTION_COMPOSITION",
                        concept_name="Function Composition", 
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Found g(1) = 3 correctly but didn't apply f",
                        remediation_hint="After finding g(1) = 3, substitute into f: f(3) = 3 + 2 = 5",
                        frequency=0.25,
                        prerequisite_gaps=["ORDER_OF_OPERATIONS"]
                    ),
                    ErrorMapping(
                        wrong_answer="4",
                        concept_id="ADDITION_FACTS",
                        concept_name="Addition Facts",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Calculated correctly until final step: 3 + 2 = 4 instead of 5",
                        remediation_hint="Check addition: 3 + 2 = 5",
                        frequency=0.15,
                        prerequisite_gaps=["ADDITION_FACTS"]
                    )
                ]
            ),
            
            TestQuestion(
                id="INT_004",
                question_text="A recipe calls for 2/3 cup of flour. If you want to make 1.5 times the recipe, how much flour do you need?",
                correct_answer="1 cup",
                difficulty=3,
                primary_concepts=["FRACTIONS"],
                source="Integrated Assessment",
                common_errors=[
                    ErrorMapping(
                        wrong_answer="2/4.5 cup",
                        concept_id="FRACTIONS",
                        concept_name="Fractions as Part-Whole",
                        error_type=ErrorType.PROCEDURAL,
                        error_description="Divided instead of multiplying: 2/3 ÷ 1.5",
                        remediation_hint="To scale up a recipe, multiply the amount by the scale factor",
                        frequency=0.30,
                        prerequisite_gaps=["MULTIPLICATION_FACTS"]
                    ),
                    ErrorMapping(
                        wrong_answer="3/3 cup",
                        concept_id="FRACTIONS",
                        concept_name="Fractions as Part-Whole",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Added 1.5 to 2/3 instead of multiplying",
                        remediation_hint="Scaling means multiplying: (2/3) × 1.5",
                        frequency=0.25,
                        prerequisite_gaps=["MULTIPLICATION_FACTS"]
                    ),
                    ErrorMapping(
                        wrong_answer="1/3 cup",
                        concept_id="FRACTIONS",
                        concept_name="Fractions as Part-Whole",
                        error_type=ErrorType.COMPUTATIONAL,
                        error_description="Calculation error in fraction multiplication",
                        remediation_hint="(2/3) × (3/2) = 6/6 = 1, not 1/3",
                        frequency=0.20,
                        prerequisite_gaps=["FRACTIONS", "MULTIPLICATION_FACTS"]
                    )
                ]
            )
        ]
    
    def find_prerequisite_chain(self, concept_id: str) -> List[str]:
        """Find the complete prerequisite chain for a concept."""
        if concept_id not in self.concept_map:
            return []
        
        visited = set()
        chain = []
        
        def dfs(cid):
            if cid in visited:
                return
            visited.add(cid)
            
            concept = self.concept_map[cid]
            for prereq in concept.prerequisites:
                dfs(prereq)
                if prereq not in chain:
                    chain.append(prereq)
            
            if cid not in chain:
                chain.append(cid)
        
        dfs(concept_id)
        return chain
    
    def simulate_student_response(self, question: TestQuestion, mastery_levels: Dict[str, float]) -> StudentResponse:
        """Simulate student response based on concept mastery levels."""
        
        # Calculate success probability based on prerequisite mastery
        success_factors = []
        
        for concept_id in question.primary_concepts:
            if concept_id in self.concept_map:
                prereq_chain = self.find_prerequisite_chain(concept_id)
                
                # Check mastery of all prerequisites
                prereq_mastery = []
                for prereq in prereq_chain:
                    mastery = mastery_levels.get(prereq, 0.0)
                    prereq_mastery.append(mastery)
                
                # Success depends on weakest prerequisite
                if prereq_mastery:
                    min_mastery = min(prereq_mastery)
                    success_factors.append(min_mastery)
        
        # Overall success probability
        if success_factors:
            success_prob = min(success_factors)
        else:
            success_prob = 0.5
        
        # Add some randomness
        success_prob += random.uniform(-0.1, 0.1)
        success_prob = max(0.05, min(0.95, success_prob))
        
        is_correct = random.random() < success_prob
        
        if is_correct:
            return StudentResponse(
                question_id=question.id,
                student_answer=question.correct_answer,
                is_correct=True,
                error_mapping=None,
                confidence_level=random.randint(4, 5)
            )
        else:
            # Select error based on which prerequisites are weakest
            if question.common_errors:
                # Weight errors by how weak the prerequisite gaps are
                error_weights = []
                for error in question.common_errors:
                    weight = error.frequency
                    
                    # Increase weight if prerequisite gaps have low mastery
                    for gap in error.prerequisite_gaps:
                        if gap in mastery_levels:
                            gap_mastery = mastery_levels[gap]
                            weight *= (1.0 - gap_mastery + 0.1)  # Lower mastery = higher error probability
                    
                    error_weights.append(weight)
                
                # Normalize weights
                total_weight = sum(error_weights)
                if total_weight > 0:
                    error_weights = [w/total_weight for w in error_weights]
                    error = random.choices(question.common_errors, weights=error_weights)[0]
                else:
                    error = random.choice(question.common_errors)
                
                return StudentResponse(
                    question_id=question.id,
                    student_answer=error.wrong_answer,
                    is_correct=False,
                    error_mapping=error,
                    confidence_level=random.randint(1, 3)
                )
            else:
                return StudentResponse(
                    question_id=question.id,
                    student_answer="unknown",
                    is_correct=False,
                    error_mapping=None,
                    confidence_level=1
                )
    
    def analyze_learning_gaps(self, responses: List[StudentResponse]) -> Tuple[List[str], List[LearningPath]]:
        """Analyze responses to identify learning gaps and create personalized learning paths."""
        
        # Identify concepts with errors
        error_concepts = set()
        prerequisite_gaps = set()
        
        for response in responses:
            if response.error_mapping:
                error_concepts.add(response.error_mapping.concept_id)
                prerequisite_gaps.update(response.error_mapping.prerequisite_gaps)
        
        # Find all concepts that need attention (errors + prerequisites)
        all_gaps = error_concepts.union(prerequisite_gaps)
        
        # Create learning paths
        learning_paths = []
        for concept_id in all_gaps:
            if concept_id in self.concept_map:
                concept = self.concept_map[concept_id]
                
                # Determine current mastery (lower if errors found)
                current_mastery = 0.3 if concept_id in error_concepts else 0.6
                
                # Find prerequisites that also need work
                prereq_chain = self.find_prerequisite_chain(concept_id)
                prerequisites_needed = [p for p in prereq_chain if p in all_gaps and p != concept_id]
                
                # Calculate priority (higher for foundational concepts)
                priority = 5 - concept.difficulty  # Lower difficulty = higher priority
                if concept_id in error_concepts:
                    priority += 2  # Boost priority for concepts with direct errors
                
                learning_path = LearningPath(
                    concept_id=concept_id,
                    concept_name=concept.name,
                    current_mastery=current_mastery,
                    target_mastery=0.8,
                    prerequisites_needed=prerequisites_needed,
                    estimated_study_time=concept.time_to_master,
                    priority=priority
                )
                learning_paths.append(learning_path)
        
        # Sort by priority (higher first)
        learning_paths.sort(key=lambda x: x.priority, reverse=True)
        
        return list(all_gaps), learning_paths
    
    def run_integrated_diagnostic(self, student_name: str, mastery_profile: Dict[str, float]):
        """Run integrated diagnostic with knowledge graph analysis."""
        
        print(f"\n{'='*80}")
        print(f"INTEGRATED DIAGNOSTIC ASSESSMENT: {student_name}")
        print(f"Knowledge Graph + Error Analysis Integration")
        print(f"{'='*80}")
        
        print(f"\n📊 STUDENT MASTERY PROFILE:")
        for concept_id, mastery in sorted(mastery_profile.items()):
            concept_name = self.concept_map.get(concept_id, {}).name if concept_id in self.concept_map else concept_id
            mastery_bar = "█" * int(mastery * 10) + "░" * (10 - int(mastery * 10))
            print(f"   {concept_name:<25} [{mastery_bar}] {mastery:.1f}")
        
        # Run assessment
        responses = []
        print(f"\n📝 ASSESSMENT IN PROGRESS...")
        
        for question in self.questions:
            response = self.simulate_student_response(question, mastery_profile)
            responses.append(response)
            
            status = "✅" if response.is_correct else "❌"
            print(f"   {status} {question.id}: {question.question_text}")
            print(f"      Student: '{response.student_answer}' | Correct: '{question.correct_answer}'")
            
            if response.error_mapping:
                print(f"      🚨 Error: {response.error_mapping.error_description}")
                print(f"      🔗 Prerequisite Gaps: {', '.join(response.error_mapping.prerequisite_gaps)}")
                print(f"      💡 Remediation: {response.error_mapping.remediation_hint}")
        
        # Analyze gaps and create learning paths
        all_gaps, learning_paths = self.analyze_learning_gaps(responses)
        
        # Performance summary
        correct_count = sum(1 for r in responses if r.is_correct)
        accuracy = correct_count / len(responses) * 100
        
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print(f"   Questions Attempted: {len(responses)}")
        print(f"   Correct Answers: {correct_count}")
        print(f"   Overall Accuracy: {accuracy:.1f}%")
        
        if all_gaps:
            print(f"\n🎯 KNOWLEDGE GRAPH ANALYSIS:")
            print(f"   Concepts Needing Attention: {len(all_gaps)}")
            
            # Show prerequisite chains for main concepts
            main_concepts = [r.error_mapping.concept_id for r in responses if r.error_mapping]
            for concept_id in set(main_concepts):
                if concept_id in self.concept_map:
                    concept = self.concept_map[concept_id]
                    chain = self.find_prerequisite_chain(concept_id)
                    
                    print(f"\n   📚 {concept.name} Prerequisite Chain:")
                    for i, prereq_id in enumerate(chain):
                        if prereq_id in self.concept_map:
                            prereq = self.concept_map[prereq_id]
                            indent = "   " + "  " * i
                            status = "❌" if prereq_id in all_gaps else "✅"
                            print(f"{indent}{status} {prereq.name} (Difficulty: {prereq.difficulty})")
            
            print(f"\n🛤️ PERSONALIZED LEARNING PATH:")
            total_study_time = 0
            
            for i, path in enumerate(learning_paths[:8], 1):  # Show top 8 priorities
                priority_icon = "🔥" if path.priority >= 6 else "⚠️" if path.priority >= 4 else "📝"
                mastery_gap = path.target_mastery - path.current_mastery
                
                print(f"\n   {priority_icon} {i}. {path.concept_name}")
                print(f"      Current Mastery: {path.current_mastery:.1f} → Target: {path.target_mastery:.1f}")
                print(f"      Estimated Study Time: {path.estimated_study_time} minutes")
                print(f"      Priority Score: {path.priority}")
                
                if path.prerequisites_needed:
                    prereq_names = [self.concept_map[p].name for p in path.prerequisites_needed if p in self.concept_map]
                    print(f"      Prerequisites First: {', '.join(prereq_names)}")
                
                total_study_time += path.estimated_study_time
            
            print(f"\n⏱️ TOTAL ESTIMATED STUDY TIME: {total_study_time} minutes ({total_study_time/60:.1f} hours)")
            
            # Learning sequence recommendation
            print(f"\n📋 RECOMMENDED LEARNING SEQUENCE:")
            foundation_concepts = [p for p in learning_paths if p.priority >= 6]
            intermediate_concepts = [p for p in learning_paths if 4 <= p.priority < 6]
            advanced_concepts = [p for p in learning_paths if p.priority < 4]
            
            if foundation_concepts:
                print(f"   🏗️ FOUNDATION (Start Here):")
                for concept in foundation_concepts[:3]:
                    print(f"      • {concept.concept_name}")
            
            if intermediate_concepts:
                print(f"   🔧 SKILL BUILDING:")
                for concept in intermediate_concepts[:3]:
                    print(f"      • {concept.concept_name}")
            
            if advanced_concepts:
                print(f"   🚀 ADVANCED TOPICS:")
                for concept in advanced_concepts[:2]:
                    print(f"      • {concept.concept_name}")
        
        else:
            print(f"\n✅ EXCELLENT! No significant knowledge gaps detected.")
            print(f"   🎓 Ready for advanced topics in algebra!")

def main():
    """Run the integrated diagnostic demonstration."""
    
    print("🧮 INTEGRATED DIAGNOSTIC SYSTEM")
    print("Knowledge Graph + Error Analysis Integration")
    print("Demonstrates prerequisite tracking and personalized learning paths")
    
    system = IntegratedDiagnosticSystem()
    
    # Test different student profiles with varying mastery levels
    students = [
        ("Emma (Strong Foundation)", {
            "COUNTING": 0.9, "ADDITION_FACTS": 0.9, "MULTIPLICATION_FACTS": 0.8,
            "FRACTIONS": 0.7, "ORDER_OF_OPERATIONS": 0.8, "DISTRIBUTIVE_PROPERTY": 0.7,
            "COMBINE_LIKE_TERMS": 0.6, "ONE_STEP_EQUATIONS": 0.7, "TWO_STEP_EQUATIONS": 0.5,
            "QUADRATIC_FORMULA": 0.3, "SYSTEMS_EQUATIONS": 0.4, "FUNCTION_COMPOSITION": 0.3
        }),
        
        ("Jake (Gaps in Foundation)", {
            "COUNTING": 0.9, "ADDITION_FACTS": 0.7, "MULTIPLICATION_FACTS": 0.5,
            "FRACTIONS": 0.4, "ORDER_OF_OPERATIONS": 0.6, "DISTRIBUTIVE_PROPERTY": 0.4,
            "COMBINE_LIKE_TERMS": 0.3, "ONE_STEP_EQUATIONS": 0.5, "TWO_STEP_EQUATIONS": 0.3,
            "QUADRATIC_FORMULA": 0.2, "SYSTEMS_EQUATIONS": 0.2, "FUNCTION_COMPOSITION": 0.2
        }),
        
        ("Maya (Advanced but Inconsistent)", {
            "COUNTING": 0.9, "ADDITION_FACTS": 0.9, "MULTIPLICATION_FACTS": 0.9,
            "FRACTIONS": 0.8, "ORDER_OF_OPERATIONS": 0.6, "DISTRIBUTIVE_PROPERTY": 0.8,
            "COMBINE_LIKE_TERMS": 0.7, "ONE_STEP_EQUATIONS": 0.8, "TWO_STEP_EQUATIONS": 0.7,
            "QUADRATIC_FORMULA": 0.6, "SYSTEMS_EQUATIONS": 0.5, "FUNCTION_COMPOSITION": 0.4
        })
    ]
    
    for name, mastery_profile in students:
        # Set seed for reproducible results
        random.seed(hash(name) % 1000)
        system.run_integrated_diagnostic(name, mastery_profile)
    
    print(f"\n{'='*80}")
    print("🎉 INTEGRATED DIAGNOSTIC DEMONSTRATION COMPLETE!")
    print("="*80)
    
    print(f"\nKey Features Demonstrated:")
    print("✅ Knowledge graph prerequisite tracking")
    print("✅ Error mapping to specific concept gaps")
    print("✅ Prerequisite chain analysis")
    print("✅ Personalized learning path generation")
    print("✅ Priority-based study recommendations")
    print("✅ Mastery level visualization")
    print("✅ Estimated study time calculations")
    
    print(f"\nEducational Impact:")
    print("🎯 Identifies root causes of student errors")
    print("📚 Creates optimal learning sequences")
    print("⏱️ Provides realistic time estimates")
    print("🔄 Enables adaptive curriculum design")
    print("📊 Supports data-driven instruction")

if __name__ == "__main__":
    main() 