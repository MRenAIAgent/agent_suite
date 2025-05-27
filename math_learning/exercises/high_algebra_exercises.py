"""
Algebra Exercises - High School Level

This module contains exercises for high school algebra concepts (Grades 9-12).
"""

from .exercise import Exercise
from .exercise_bank import ExerciseBank


def create_high_school_exercises() -> ExerciseBank:
    """
    Create an exercise bank with high school algebra exercises.
    
    Returns:
        ExerciseBank: The exercise bank with high school algebra exercises
    """
    bank = ExerciseBank(name="High School Algebra Exercises")
    
    # Exponents and Roots
    exponents_ex1 = Exercise(
        title="Simplify Exponential Expression",
        problem="Simplify the expression: (2^3)^2",
        solution="64",
        difficulty=4,
        estimated_time=3,
        format_type="free-response",
        hints=["Use the power rule: (x^a)^b = x^(a*b)", "Calculate 2^6"]
    )
    exponents_ex1.add_concept_relationship("exponents_roots", weight=1.0, relationship_type="primary")
    exponents_ex1.add_concept_relationship("multiplication", weight=0.5, relationship_type="foundational")
    bank.add_exercise(exponents_ex1)
    
    exponents_ex2 = Exercise(
        title="Solve Square Root Equation",
        problem="Solve for x: √(x - 5) = 4",
        solution="x = 21",
        difficulty=4,
        estimated_time=4,
        format_type="free-response",
        hints=["Square both sides of the equation", "Solve the resulting equation"]
    )
    exponents_ex2.add_concept_relationship("exponents_roots", weight=0.9, relationship_type="primary")
    exponents_ex2.add_concept_relationship("equations_one_step", weight=0.6, relationship_type="foundational")
    bank.add_exercise(exponents_ex2)
    
    # Polynomials
    polynomials_ex1 = Exercise(
        title="Add Polynomials",
        problem="Add the polynomials: (3x^2 + 2x - 5) + (x^2 - 4x + 7)",
        solution="4x^2 - 2x + 2",
        difficulty=4,
        estimated_time=4,
        format_type="free-response",
        hints=["Group like terms", "Combine coefficients of terms with the same exponent"]
    )
    polynomials_ex1.add_concept_relationship("polynomials", weight=1.0, relationship_type="primary")
    polynomials_ex1.add_concept_relationship("variables", weight=0.7, relationship_type="foundational")
    polynomials_ex1.add_concept_relationship("addition", weight=0.5, relationship_type="foundational")
    bank.add_exercise(polynomials_ex1)
    
    polynomials_ex2 = Exercise(
        title="Multiply Polynomials",
        problem="Multiply: (x + 3)(x - 2)",
        solution="x^2 + x - 6",
        difficulty=4,
        estimated_time=5,
        format_type="free-response",
        hints=["Use the distributive property", "Multiply each term in the first parenthesis by each term in the second"]
    )
    polynomials_ex2.add_concept_relationship("polynomials", weight=0.9, relationship_type="primary")
    polynomials_ex2.add_concept_relationship("variables", weight=0.7, relationship_type="foundational")
    polynomials_ex2.add_concept_relationship("multiplication", weight=0.6, relationship_type="foundational")
    bank.add_exercise(polynomials_ex2)
    
    # Factoring
    factoring_ex1 = Exercise(
        title="Factor Trinomial",
        problem="Factor completely: x^2 + 5x + 6",
        solution="(x + 2)(x + 3)",
        difficulty=4,
        estimated_time=5,
        format_type="free-response",
        hints=["Find two numbers that multiply to give 6 and add to give 5", 
               "These numbers are 2 and 3", 
               "Write as (x + 2)(x + 3)"]
    )
    factoring_ex1.add_concept_relationship("factoring", weight=1.0, relationship_type="primary")
    factoring_ex1.add_concept_relationship("polynomials", weight=0.8, relationship_type="foundational")
    bank.add_exercise(factoring_ex1)
    
    factoring_ex2 = Exercise(
        title="Factor by Grouping",
        problem="Factor by grouping: xy - 3x + 2y - 6",
        solution="(x + 2)(y - 3)",
        difficulty=4,
        estimated_time=6,
        format_type="free-response",
        hints=["Group the terms: (xy - 3x) + (2y - 6)", 
               "Factor out common factors from each group: x(y - 3) + 2(y - 3)", 
               "Factor out the common binomial: (y - 3)(x + 2)"]
    )
    factoring_ex2.add_concept_relationship("factoring", weight=0.9, relationship_type="primary")
    factoring_ex2.add_concept_relationship("polynomials", weight=0.7, relationship_type="foundational")
    bank.add_exercise(factoring_ex2)
    
    # Quadratic Functions
    quadratic_ex1 = Exercise(
        title="Find Vertex of Quadratic Function",
        problem="Find the vertex of the quadratic function f(x) = 2x^2 - 12x + 19",
        solution="Vertex: (3, 1)",
        difficulty=4,
        estimated_time=6,
        format_type="free-response",
        hints=["Use the formula x = -b/(2a) to find the x-coordinate of the vertex", 
               "For f(x) = 2x^2 - 12x + 19, a = 2 and b = -12", 
               "Substitute the x-value back into the function to find the y-coordinate"]
    )
    quadratic_ex1.add_concept_relationship("quadratic_functions", weight=1.0, relationship_type="primary")
    quadratic_ex1.add_concept_relationship("polynomials", weight=0.7, relationship_type="foundational")
    quadratic_ex1.add_concept_relationship("coordinate_plane", weight=0.5, relationship_type="foundational")
    bank.add_exercise(quadratic_ex1)
    
    quadratic_ex2 = Exercise(
        title="Analyze Quadratic Function",
        problem="For the function f(x) = -x^2 + 4x - 3, determine: (a) whether it opens up or down, (b) the axis of symmetry, (c) the y-intercept, and (d) the x-intercepts if they exist.",
        solution="(a) Opens down (a = -1), (b) Axis of symmetry: x = 2, (c) y-intercept: (0, -3), (d) x-intercepts: (1, 0) and (3, 0)",
        difficulty=5,
        estimated_time=10,
        format_type="free-response",
        hints=["The coefficient of x^2 determines whether it opens up (positive) or down (negative)", 
               "The axis of symmetry is x = -b/(2a)", 
               "The y-intercept is f(0)", 
               "For x-intercepts, set f(x) = 0 and solve"]
    )
    quadratic_ex2.add_concept_relationship("quadratic_functions", weight=0.9, relationship_type="primary")
    quadratic_ex2.add_concept_relationship("polynomials", weight=0.7, relationship_type="foundational")
    quadratic_ex2.add_concept_relationship("factoring", weight=0.6, relationship_type="secondary")
    bank.add_exercise(quadratic_ex2)
    
    # Quadratic Formula
    quadratic_formula_ex1 = Exercise(
        title="Solve Using Quadratic Formula",
        problem="Solve the equation 2x^2 - 5x + 2 = 0 using the quadratic formula.",
        solution="x = (5 + √9)/4 or x = (5 - √9)/4, which simplifies to x = 2 or x = 1/2",
        difficulty=4,
        estimated_time=6,
        format_type="free-response",
        hints=["The quadratic formula is x = (-b ± √(b^2 - 4ac))/(2a)", 
               "Identify a = 2, b = -5, c = 2", 
               "Substitute into the formula and simplify"]
    )
    quadratic_formula_ex1.add_concept_relationship("quadratic_formula", weight=1.0, relationship_type="primary")
    quadratic_formula_ex1.add_concept_relationship("quadratic_functions", weight=0.8, relationship_type="foundational")
    quadratic_formula_ex1.add_concept_relationship("exponents_roots", weight=0.7, relationship_type="foundational")
    bank.add_exercise(quadratic_formula_ex1)
    
    quadratic_formula_ex2 = Exercise(
        title="Quadratic Word Problem",
        problem="A ball is thrown upward from the ground with an initial velocity of 64 feet per second. The height h (in feet) of the ball after t seconds is given by h = 64t - 16t^2. When will the ball hit the ground?",
        solution="The ball will hit the ground after 4 seconds",
        difficulty=5,
        estimated_time=8,
        format_type="free-response",
        hints=["The ball hits the ground when h = 0", 
               "Solve the equation 64t - 16t^2 = 0", 
               "Factor out 16t: 16t(4 - t) = 0"]
    )
    quadratic_formula_ex2.add_concept_relationship("quadratic_formula", weight=0.9, relationship_type="primary")
    quadratic_formula_ex2.add_concept_relationship("quadratic_functions", weight=0.8, relationship_type="foundational")
    quadratic_formula_ex2.add_concept_relationship("factoring", weight=0.7, relationship_type="secondary")
    bank.add_exercise(quadratic_formula_ex2)
    
    # Rational Expressions
    rational_ex1 = Exercise(
        title="Simplify Rational Expression",
        problem="Simplify the rational expression: (x^2 - 4)/(x - 2)",
        solution="x + 2 for x ≠ 2",
        difficulty=5,
        estimated_time=5,
        format_type="free-response",
        hints=["Factor the numerator: (x^2 - 4) = (x - 2)(x + 2)", 
               "Cancel the common factor (x - 2)", 
               "Remember to state the restriction x ≠ 2"]
    )
    rational_ex1.add_concept_relationship("rational_expressions", weight=1.0, relationship_type="primary")
    rational_ex1.add_concept_relationship("factoring", weight=0.8, relationship_type="foundational")
    rational_ex1.add_concept_relationship("division", weight=0.6, relationship_type="foundational")
    bank.add_exercise(rational_ex1)
    
    rational_ex2 = Exercise(
        title="Add Rational Expressions",
        problem="Add the rational expressions: x/(x+1) + 2/(x-1)",
        solution="(x^2 + x - 2)/((x+1)(x-1))",
        difficulty=5,
        estimated_time=8,
        format_type="free-response",
        hints=["Find a common denominator: (x+1)(x-1)", 
               "Rewrite each fraction with this denominator", 
               "Add the numerators and simplify if possible"]
    )
    rational_ex2.add_concept_relationship("rational_expressions", weight=0.9, relationship_type="primary")
    rational_ex2.add_concept_relationship("polynomials", weight=0.7, relationship_type="foundational")
    rational_ex2.add_concept_relationship("fractions_intro", weight=0.5, relationship_type="foundational")
    bank.add_exercise(rational_ex2)
    
    # Exponential Functions
    exp_functions_ex1 = Exercise(
        title="Evaluate Exponential Function",
        problem="Evaluate the function f(x) = 3(2^x) when x = -1, 0, 1, and 2.",
        solution="f(-1) = 1.5, f(0) = 3, f(1) = 6, f(2) = 12",
        difficulty=4,
        estimated_time=5,
        format_type="free-response",
        hints=["Substitute each x-value into the function", 
               "Calculate 2^x for each value, then multiply by 3"]
    )
    exp_functions_ex1.add_concept_relationship("exponential_functions", weight=1.0, relationship_type="primary")
    exp_functions_ex1.add_concept_relationship("exponents_roots", weight=0.8, relationship_type="foundational")
    bank.add_exercise(exp_functions_ex1)
    
    exp_functions_ex2 = Exercise(
        title="Exponential Growth Problem",
        problem="A city's population was 50,000 in 2010 and grows at a rate of 3% per year. Write an exponential function to model the population P(t) after t years since 2010. What will the population be in 2025?",
        solution="P(t) = 50,000(1.03)^t; In 2025 (t = 15), the population will be approximately 77,898",
        difficulty=5,
        estimated_time=8,
        format_type="free-response",
        hints=["The exponential growth formula is P(t) = P₀(1 + r)^t", 
               "Initial population P₀ = 50,000 and growth rate r = 0.03", 
               "For 2025, calculate P(15)"]
    )
    exp_functions_ex2.add_concept_relationship("exponential_functions", weight=0.9, relationship_type="primary")
    exp_functions_ex2.add_concept_relationship("exponents_roots", weight=0.7, relationship_type="foundational")
    bank.add_exercise(exp_functions_ex2)
    
    # Logarithmic Functions
    log_functions_ex1 = Exercise(
        title="Evaluate Logarithmic Expression",
        problem="Evaluate the expression: log₃(27)",
        solution="3",
        difficulty=5,
        estimated_time=4,
        format_type="free-response",
        hints=["Recall that log_b(x) = y means b^y = x", 
               "So log₃(27) = y means 3^y = 27", 
               "Find y by determining what power of 3 equals 27"]
    )
    log_functions_ex1.add_concept_relationship("logarithmic_functions", weight=1.0, relationship_type="primary")
    log_functions_ex1.add_concept_relationship("exponential_functions", weight=0.8, relationship_type="foundational")
    log_functions_ex1.add_concept_relationship("exponents_roots", weight=0.7, relationship_type="foundational")
    bank.add_exercise(log_functions_ex1)
    
    log_functions_ex2 = Exercise(
        title="Solve Logarithmic Equation",
        problem="Solve for x: log(x) + log(x - 3) = 1",
        solution="x = 4",
        difficulty=5,
        estimated_time=8,
        format_type="free-response",
        hints=["Use the property log(a) + log(b) = log(ab)", 
               "Rewrite as log(x(x - 3)) = 1", 
               "Since log(10) = 1, this means x(x - 3) = 10", 
               "Expand to x² - 3x = 10 and solve the quadratic equation"]
    )
    log_functions_ex2.add_concept_relationship("logarithmic_functions", weight=0.9, relationship_type="primary")
    log_functions_ex2.add_concept_relationship("exponential_functions", weight=0.8, relationship_type="foundational")
    log_functions_ex2.add_concept_relationship("quadratic_functions", weight=0.6, relationship_type="secondary")
    bank.add_exercise(log_functions_ex2)
    
    # Function Transformations
    transformations_ex1 = Exercise(
        title="Describe Transformations",
        problem="Describe the transformations required to transform f(x) = x² to g(x) = (x - 2)² + 3.",
        solution="Shift right 2 units, then shift up 3 units",
        difficulty=4,
        estimated_time=5,
        format_type="free-response",
        hints=["(x - 2)² represents a horizontal shift 2 units to the right", 
               "(x - 2)² + 3 represents a vertical shift 3 units up"]
    )
    transformations_ex1.add_concept_relationship("function_transformations", weight=1.0, relationship_type="primary")
    transformations_ex1.add_concept_relationship("quadratic_functions", weight=0.7, relationship_type="secondary")
    bank.add_exercise(transformations_ex1)
    
    transformations_ex2 = Exercise(
        title="Graph Transformed Function",
        problem="Sketch the graph of h(x) = |x - 1| + 2.",
        solution="V-shaped graph with vertex at (1, 2) and increasing in both directions",
        difficulty=4,
        estimated_time=6,
        format_type="graphing",
        hints=["Start with the graph of f(x) = |x|", 
               "Shift it right 1 unit to get g(x) = |x - 1|", 
               "Shift it up 2 units to get h(x) = |x - 1| + 2"]
    )
    transformations_ex2.add_concept_relationship("function_transformations", weight=0.9, relationship_type="primary")
    transformations_ex2.add_concept_relationship("coordinate_plane", weight=0.6, relationship_type="foundational")
    bank.add_exercise(transformations_ex2)
    
    return bank 