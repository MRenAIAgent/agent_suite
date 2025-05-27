"""
Algebra Exercises - Middle School Level

This module contains exercises for middle school algebra concepts (Grades 6-8).
"""

from exercises.exercise import Exercise
from exercises.exercise_bank import ExerciseBank


def create_middle_school_exercises() -> ExerciseBank:
    """
    Create an exercise bank with middle school algebra exercises.
    
    Returns:
        ExerciseBank: The exercise bank with middle school algebra exercises
    """
    bank = ExerciseBank(name="Middle School Algebra Exercises")
    
    # Variables and Expressions
    variables_ex1 = Exercise(
        title="Evaluate an Expression",
        problem="Evaluate the expression 3x + 2 when x = 4.",
        solution="14",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["Substitute x = 4 into the expression", "Calculate 3 × 4 + 2"]
    )
    variables_ex1.add_concept_relationship("variables", weight=1.0, relationship_type="primary")
    variables_ex1.add_concept_relationship("multiplication", weight=0.5, relationship_type="foundational")
    variables_ex1.add_concept_relationship("addition", weight=0.4, relationship_type="foundational")
    bank.add_exercise(variables_ex1)
    
    variables_ex2 = Exercise(
        title="Write an Expression",
        problem="Write an algebraic expression for 'seven more than twice a number y'.",
        solution="2y + 7",
        difficulty=3,
        estimated_time=4,
        format_type="free-response",
        hints=["'Twice a number y' means 2y", "Seven more than that means add 7"]
    )
    variables_ex2.add_concept_relationship("variables", weight=0.9, relationship_type="primary")
    bank.add_exercise(variables_ex2)
    
    # One-Step Equations
    equations_one_step_ex1 = Exercise(
        title="Solve One-Step Equation: Addition",
        problem="Solve for x: x + 5 = 12",
        solution="x = 7",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["To isolate x, subtract 5 from both sides", "Think: what plus 5 equals 12?"]
    )
    equations_one_step_ex1.add_concept_relationship("equations_one_step", weight=1.0, relationship_type="primary")
    equations_one_step_ex1.add_concept_relationship("variables", weight=0.7, relationship_type="foundational")
    equations_one_step_ex1.add_concept_relationship("subtraction", weight=0.5, relationship_type="foundational")
    bank.add_exercise(equations_one_step_ex1)
    
    equations_one_step_ex2 = Exercise(
        title="Solve One-Step Equation: Multiplication",
        problem="Solve for y: 3y = 18",
        solution="y = 6",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["To isolate y, divide both sides by 3", "Think: 3 times what number equals 18?"]
    )
    equations_one_step_ex2.add_concept_relationship("equations_one_step", weight=1.0, relationship_type="primary")
    equations_one_step_ex2.add_concept_relationship("variables", weight=0.7, relationship_type="foundational")
    equations_one_step_ex2.add_concept_relationship("division", weight=0.6, relationship_type="foundational")
    bank.add_exercise(equations_one_step_ex2)
    
    # Two-Step Equations
    equations_two_step_ex1 = Exercise(
        title="Solve Two-Step Equation",
        problem="Solve for x: 2x + 3 = 11",
        solution="x = 4",
        difficulty=3,
        estimated_time=5,
        format_type="free-response",
        hints=["First subtract 3 from both sides", "Then divide both sides by 2"]
    )
    equations_two_step_ex1.add_concept_relationship("equations_two_step", weight=1.0, relationship_type="primary")
    equations_two_step_ex1.add_concept_relationship("equations_one_step", weight=0.7, relationship_type="foundational")
    bank.add_exercise(equations_two_step_ex1)
    
    equations_two_step_ex2 = Exercise(
        title="Word Problem: Two-Step Equation",
        problem="A movie ticket costs $8 plus a $2 booking fee. If Sophia spent $42 on tickets, how many tickets did she buy?",
        solution="5 tickets",
        difficulty=4,
        estimated_time=6,
        format_type="free-response",
        hints=["Let x = number of tickets", "Cost = 8x + 2", "Set up the equation: 8x + 2 = 42"]
    )
    equations_two_step_ex2.add_concept_relationship("equations_two_step", weight=0.9, relationship_type="primary")
    equations_two_step_ex2.add_concept_relationship("variables", weight=0.6, relationship_type="foundational")
    bank.add_exercise(equations_two_step_ex2)
    
    # Inequalities
    inequalities_ex1 = Exercise(
        title="Solve an Inequality",
        problem="Solve for x: 2x - 3 > 7",
        solution="x > 5",
        difficulty=3,
        estimated_time=5,
        format_type="free-response",
        hints=["Add 3 to both sides: 2x > 10", "Divide both sides by 2: x > 5"]
    )
    inequalities_ex1.add_concept_relationship("inequalities", weight=1.0, relationship_type="primary")
    inequalities_ex1.add_concept_relationship("equations_one_step", weight=0.7, relationship_type="foundational")
    bank.add_exercise(inequalities_ex1)
    
    inequalities_ex2 = Exercise(
        title="Graph an Inequality",
        problem="Graph the solution to x ≤ 4 on a number line.",
        solution="A number line with a closed circle at 4 and shading to the left",
        difficulty=3,
        estimated_time=4,
        format_type="graphing",
        hints=["Mark the point x = 4 with a closed (filled) circle", 
               "Shade the number line to the left (for all values less than or equal to 4)"]
    )
    inequalities_ex2.add_concept_relationship("inequalities", weight=0.9, relationship_type="primary")
    inequalities_ex2.add_concept_relationship("coordinate_plane", weight=0.4, relationship_type="secondary")
    bank.add_exercise(inequalities_ex2)
    
    # Coordinate Plane
    coordinate_plane_ex1 = Exercise(
        title="Plot Points",
        problem="Plot the points A(2, 3), B(-1, 4), and C(0, -2) on the coordinate plane.",
        solution="Points plotted at (2, 3), (-1, 4), and (0, -2)",
        difficulty=3,
        estimated_time=4,
        format_type="graphing",
        hints=["For each point (x, y), move x units horizontally from the origin", 
               "Then move y units vertically"]
    )
    coordinate_plane_ex1.add_concept_relationship("coordinate_plane", weight=1.0, relationship_type="primary")
    bank.add_exercise(coordinate_plane_ex1)
    
    coordinate_plane_ex2 = Exercise(
        title="Find Coordinates",
        problem="What are the coordinates of point P shown on the coordinate plane?",
        solution="(3, -2) [Note: This would be accompanied by an image showing the point]",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["Find how far the point is horizontally from the origin (x-coordinate)", 
               "Find how far the point is vertically from the origin (y-coordinate)"]
    )
    coordinate_plane_ex2.add_concept_relationship("coordinate_plane", weight=0.9, relationship_type="primary")
    bank.add_exercise(coordinate_plane_ex2)
    
    # Proportional Relationships
    proportional_ex1 = Exercise(
        title="Unit Rate",
        problem="If 3 pounds of apples cost $4.50, what is the cost per pound?",
        solution="$1.50 per pound",
        difficulty=3,
        estimated_time=3,
        format_type="free-response",
        hints=["Divide the total cost by the number of pounds", "Calculate $4.50 ÷ 3"]
    )
    proportional_ex1.add_concept_relationship("proportional_relationships", weight=1.0, relationship_type="primary")
    proportional_ex1.add_concept_relationship("division", weight=0.6, relationship_type="foundational")
    bank.add_exercise(proportional_ex1)
    
    proportional_ex2 = Exercise(
        title="Proportional Relationship",
        problem="Is y = 3x a proportional relationship? Explain why or why not.",
        solution="Yes, it is proportional because it passes through the origin (0,0) and has a constant ratio y/x = 3.",
        difficulty=3,
        estimated_time=5,
        format_type="free-response",
        hints=["A proportional relationship can be written as y = kx", 
               "Check if the equation passes through the origin (0,0)", 
               "Check if the ratio y/x is constant"]
    )
    proportional_ex2.add_concept_relationship("proportional_relationships", weight=0.9, relationship_type="primary")
    proportional_ex2.add_concept_relationship("variables", weight=0.5, relationship_type="foundational")
    bank.add_exercise(proportional_ex2)
    
    # Linear Functions
    linear_functions_ex1 = Exercise(
        title="Identify Slope and Y-intercept",
        problem="Identify the slope and y-intercept of the line y = -2x + 5.",
        solution="Slope = -2, y-intercept = 5",
        difficulty=4,
        estimated_time=3,
        format_type="free-response",
        hints=["The slope-intercept form is y = mx + b", 
               "m represents the slope", 
               "b represents the y-intercept"]
    )
    linear_functions_ex1.add_concept_relationship("linear_functions", weight=1.0, relationship_type="primary")
    linear_functions_ex1.add_concept_relationship("coordinate_plane", weight=0.6, relationship_type="foundational")
    bank.add_exercise(linear_functions_ex1)
    
    linear_functions_ex2 = Exercise(
        title="Graph a Linear Function",
        problem="Graph the linear function y = 2x - 3.",
        solution="A straight line with y-intercept at (0, -3) and slope 2",
        difficulty=4,
        estimated_time=6,
        format_type="graphing",
        hints=["Find the y-intercept (0, -3)", 
               "Use the slope to find another point: from the y-intercept, move 1 unit right and 2 units up", 
               "Draw a line through these points"]
    )
    linear_functions_ex2.add_concept_relationship("linear_functions", weight=0.9, relationship_type="primary")
    linear_functions_ex2.add_concept_relationship("coordinate_plane", weight=0.7, relationship_type="foundational")
    bank.add_exercise(linear_functions_ex2)
    
    # Systems of Linear Equations
    systems_linear_ex1 = Exercise(
        title="Solve a System by Substitution",
        problem="Solve the system of equations:\ny = 2x + 1\ny = -x + 7",
        solution="x = 2, y = 5",
        difficulty=4,
        estimated_time=7,
        format_type="free-response",
        hints=["Since both equations have y, set them equal: 2x + 1 = -x + 7", 
               "Solve for x: 3x = 6, so x = 2", 
               "Substitute x = 2 into either equation to find y"]
    )
    systems_linear_ex1.add_concept_relationship("systems_linear", weight=1.0, relationship_type="primary")
    systems_linear_ex1.add_concept_relationship("linear_functions", weight=0.8, relationship_type="foundational")
    systems_linear_ex1.add_concept_relationship("equations_two_step", weight=0.5, relationship_type="foundational")
    bank.add_exercise(systems_linear_ex1)
    
    systems_linear_ex2 = Exercise(
        title="Word Problem: Systems of Equations",
        problem="A movie theater sold 200 tickets for a show. Adult tickets cost $12 and child tickets cost $8. The total revenue was $2,160. How many of each type of ticket were sold?",
        solution="80 adult tickets and 120 child tickets",
        difficulty=5,
        estimated_time=10,
        format_type="free-response",
        hints=["Let a = number of adult tickets and c = number of child tickets", 
               "Write two equations: a + c = 200 and 12a + 8c = 2,160", 
               "Solve the system of equations"]
    )
    systems_linear_ex2.add_concept_relationship("systems_linear", weight=0.9, relationship_type="primary")
    systems_linear_ex2.add_concept_relationship("equations_two_step", weight=0.6, relationship_type="foundational")
    bank.add_exercise(systems_linear_ex2)
    
    return bank 