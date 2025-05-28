"""
Algebra Knowledge Graph for K-12 Education.

This module builds a comprehensive knowledge graph for algebra topics
in K-12 education, organizing concepts by functional categories and establishing
prerequisite relationships.
"""

from .graph import KnowledgeGraph
from .concept import Concept


# Category prefixes for concept IDs
NS = "NS"  # Number Sense
AT = "AT"  # Algebraic Thinking
LF = "LF"  # Linear Functions
EXP = "EXP"  # Exponential
POL = "POL"  # Polynomials
Q = "Q"    # Quadratics
FN = "FN"  # Function Tools


def build_algebra_knowledge_graph() -> KnowledgeGraph:
    """
    Build a comprehensive knowledge graph for K-12 algebra concepts.
    
    Returns:
        KnowledgeGraph: The algebra knowledge graph
    """
    # Initialize the knowledge graph
    graph = KnowledgeGraph(name="K-12 Algebra")
    
    # Number Sense Concepts
    number_sense_concepts = {
        "counting": Concept(
            name="Counting Up/Down",
            description="Count forward or backward by 1s within 1,000.",
            difficulty=1,
            time_to_master=30,
            category="Number Sense",
            concept_id=f"{NS}-01",
            examples=["Count from 345 up to 356.", "Count backward from 120 down to 107."]
        ),
        "place_value": Concept(
            name="Place Value to 1,000,000",
            description="Identify the value of each digit up to the millions place.",
            difficulty=1,
            time_to_master=60,
            category="Number Sense",
            concept_id=f"{NS}-02",
            examples=["What is the value of 7 in 275,412?", "Write 603,000 in expanded form."]
        ),
        "commutative_property": Concept(
            name="Commutative Property",
            description="Understand a+b=b+a and a×b=b×a.",
            difficulty=1,
            time_to_master=60,
            category="Number Sense",
            concept_id=f"{NS}-03",
            examples=["Show that 5+9 = 9+5.", "Is 3×7 equal to 7×3? Explain."]
        ),
        "associative_property": Concept(
            name="Associative Property",
            description="Understand (a+b)+c=a+(b+c) and (ab)c=a(bc).",
            difficulty=1,
            time_to_master=60,
            category="Number Sense",
            concept_id=f"{NS}-04",
            examples=["Rewrite 2+(4+6) without parentheses.", "Does (3×5)×2 equal 3×(5×2)?"]
        ),
        "identity_inverse_properties": Concept(
            name="Identity & Inverse Properties",
            description="Additive identity 0, multiplicative identity 1, and inverses.",
            difficulty=1,
            time_to_master=60,
            category="Number Sense",
            concept_id=f"{NS}-05",
            examples=["Find the additive inverse of 8.", "What is the multiplicative inverse of 4⁄5?"]
        ),
        "multiplication_facts": Concept(
            name="Multiplication Facts 0–12",
            description="Recall single-digit products quickly.",
            difficulty=2,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-06",
            examples=["Give 8×7 in under 3 seconds.", "Complete the table: 9×__ = 63."]
        ),
        "division_facts": Concept(
            name="Division Facts 0–12",
            description="Recall single-digit division facts quickly.",
            difficulty=2,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-07",
            examples=["What is 56 ÷ 8?", "Complete: __ ÷ 9 = 6."]
        ),
        "fractions_intro": Concept(
            name="Fractions as Part–Whole",
            description="Model and name simple fractions.",
            difficulty=2,
            time_to_master=120,
            category="Number Sense",
            concept_id=f"{NS}-08",
            examples=["Shade 3⁄8 of a rectangle.", "Write the fraction for 2 red marbles out of 5."]
        ),
        "equivalent_fractions": Concept(
            name="Equivalent Fractions",
            description="Generate and recognize equal fractions.",
            difficulty=2,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-09",
            examples=["Fill in: 3⁄5 = ___⁄15.", "Are 4⁄6 and 2⁄3 equivalent?"]
        ),
        "fraction_decimal_conversion": Concept(
            name="Fraction-Decimal Conversion",
            description="Convert terminating fractions to decimals and back.",
            difficulty=2,
            time_to_master=120,
            category="Number Sense",
            concept_id=f"{NS}-10",
            examples=["Write 3⁄8 as a decimal (round to 0.001).", "Express 0.75 as a fraction in simplest form."]
        ),
        "greatest_common_factor": Concept(
            name="Greatest Common Factor",
            description="Find GCF of two whole numbers ≤100.",
            difficulty=2,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-11",
            examples=["Find GCF(18, 30).", "What is the GCF of 48 and 64?"]
        ),
        "least_common_multiple": Concept(
            name="Least Common Multiple",
            description="Find LCM of two whole numbers ≤50.",
            difficulty=2,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-12",
            examples=["Compute LCM(6, 15).", "Find the LCM of 9 and 12."]
        ),
        "integer_addition_subtraction": Concept(
            name="Integer Addition/Subtraction",
            description="Add and subtract positive and negative integers.",
            difficulty=2,
            time_to_master=120,
            category="Number Sense",
            concept_id=f"{NS}-13",
            examples=["Evaluate −7 + 12.", "Find 9 − (−4)."]
        ),
        "integer_multiplication_division": Concept(
            name="Integer Multiplication/Division",
            description="Multiply and divide signed integers.",
            difficulty=3,
            time_to_master=120,
            category="Number Sense",
            concept_id=f"{NS}-14",
            examples=["Compute (−6)(−3).", "Evaluate −48 ÷ 6."]
        ),
        "absolute_value": Concept(
            name="Absolute Value as Distance",
            description="Interpret |a| as distance from 0 on the number line.",
            difficulty=2,
            time_to_master=60,
            category="Number Sense",
            concept_id=f"{NS}-15",
            examples=["Find |−11|.", "Which has greater absolute value: −5 or 4?"]
        ),
        "order_of_operations": Concept(
            name="Order of Operations",
            description="Apply PEMDAS to simplify numeric expressions.",
            difficulty=3,
            time_to_master=90,
            category="Number Sense",
            concept_id=f"{NS}-16",
            examples=["Simplify 3+4×2.", "Evaluate (5+3)²−6."]
        ),
    }
    
    # Algebraic Thinking Concepts
    algebraic_concepts = {
        "translate_expressions": Concept(
            name="Translate Expressions",
            description="Convert word problems to algebraic expressions",
            difficulty=2,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-17",
            examples=["'5 more than x' → x + 5", "'twice a number' → 2n"],
        ),
        "like_terms": Concept(
            name="Identify Like Terms",
            description="Spot terms with identical variable part.",
            difficulty=2,
            time_to_master=60,
            category="Pre-Algebra",
            concept_id=f"{AT}-18",
            examples=["Circle the like terms: 3x, 7, −2x, 4x.", "How many like-term groups in 5y² − 4y + 3y² + 6?"]
        ),
        "distributive_property": Concept(
            name="Distributive Property",
            description="Expand a(b+c) and factor out common factor.",
            difficulty=3,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-19",
            examples=["Expand 3(2x−5).", "Factor 15x+10."]
        ),
        "combine_like_terms": Concept(
            name="Combine Like Terms",
            description="Add/subtract coefficients of like terms.",
            difficulty=3,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-20",
            examples=["Simplify 4x + 7x − 3.", "Combine: 5y² − 2y + 7y²."]
        ),
        "evaluate_expression": Concept(
            name="Evaluate Expression",
            description="Substitute values and calculate algebraic expressions",
            difficulty=3,
            time_to_master=60,
            category="Pre-Algebra",
            concept_id=f"{AT}-21",
            examples=["If x = 3, then 2x + 5 = 11", "Evaluate 3a - b when a = 4, b = 2"],
        ),
        "one_step_add_subtract": Concept(
            name="One-Step Equations (+/−)",
            description="Solve x±a=b.",
            difficulty=2,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-22",
            examples=["Solve x−7=12.", "Find m+5=−2."]
        ),
        "one_step_multiply_divide": Concept(
            name="One-Step Equations (×/÷)",
            description="Solve ax=b and x/a=b.",
            difficulty=2,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-23",
            examples=["Solve 4p=28.", "Find x/5 = −3."]
        ),
        "two_step_equations": Concept(
            name="Two-Step Linear Equations",
            description="Solve ax+b=c.",
            difficulty=3,
            time_to_master=120,
            category="Pre-Algebra",
            concept_id=f"{AT}-24",
            examples=["Solve 3x+4=19.", "Find y/2 −7 =5."]
        ),
        "multi_step_equations": Concept(
            name="Multi-Step Linear Equations",
            description="Equations requiring distribution, combining terms, etc.",
            difficulty=4,
            time_to_master=150,
            category="Pre-Algebra",
            concept_id=f"{AT}-25",
            examples=["Solve 2(3x−1)=4x+10.", "Find k−4−2k=11."]
        ),
        "proportional_relationships": Concept(
            name="Proportional Relationships & k",
            description="y = kx relationships and constant of proportionality",
            difficulty=4,
            time_to_master=75,
            category="Pre-Algebra",
            concept_id=f"{AT}-26",
            examples=["y = 3x has k = 3", "Distance = rate × time"],
        ),
        "unit_rate_slope": Concept(
            name="Unit Rate & Slope Concept",
            description="Interpret slope as rate of change.",
            difficulty=3,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-27",
            examples=["You travel 180 km in 3 h. Unit rate?", "From the graph find slope as rise/run."]
        ),
        "coordinate_plane": Concept(
            name="Coordinate Plane Plotting",
            description="Plot ordered pairs in four quadrants.",
            difficulty=2,
            time_to_master=90,
            category="Pre-Algebra",
            concept_id=f"{AT}-28",
            examples=["Plot (−3,4).", "Which quadrant is (5,−2)?"]
        ),
        "distance_formula": Concept(
            name="Distance Formula",
            description="Find distance between two plane points.",
            difficulty=3,
            time_to_master=120,
            category="Pre-Algebra",
            concept_id=f"{AT}-29",
            examples=["Distance between (2,3) and (7,11).", "Is segment AB of length √40? A(1,2) B(5,8)"]
        ),
    }
    
    # Linear Functions Concepts
    linear_concepts = {
        "slope_two_points": Concept(
            name="Slope from Two Points",
            description="Compute m = (y₂−y₁)/(x₂−x₁).",
            difficulty=3,
            time_to_master=90,
            category="Linear",
            concept_id=f"{LF}-30",
            examples=["Slope through (3,−1) and (9,5).", "Find slope of a horizontal line."]
        ),
        "slope_intercept_form": Concept(
            name="Slope-Intercept Form",
            description="Write/graph y=mx+b.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-31",
            examples=["Graph y=−2x+3.", "Write equation with m=4, b=−7."]
        ),
        "point_slope_form": Concept(
            name="Point-Slope Form",
            description="Use y−y₁=m(x−x₁).",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-32",
            examples=["Equation through (4,5) slope 2.", "Convert y−3=3(x+1) to slope-intercept."]
        ),
        "convert_linear_forms": Concept(
            name="Convert Linear Forms",
            description="Switch between standard, slope-intercept, point-slope.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-33",
            examples=["Convert 2x−3y=6 to y=mx+b.", "Write Ax+By=C from y=½x−4."]
        ),
        "parallel_perpendicular_slopes": Concept(
            name="Parallel & Perpendicular Slopes",
            description="Identify/produce lines with given relationships.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-34",
            examples=["Line l: y=3x−2. Give slope of line parallel to l.", "Equation of line through (2,1) perpendicular to y=−¼x+5."]
        ),
        "graph_one_var_inequality": Concept(
            name="Graph Linear Inequality (1-var)",
            description="Graph solutions to x > 3, x ≤ -2 on number line",
            difficulty=3,
            time_to_master=45,
            category="Linear",
            concept_id=f"{LF}-35",
            examples=["x > 3: open circle, arrow right", "x ≤ -2: closed circle, arrow left"],
        ),
        "graph_two_var_inequality": Concept(
            name="Graph Linear Inequality (2-var)",
            description="Shade half-plane solution.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-36",
            examples=["Graph y<−½x+4.", "Is (3,0) a solution of 2x−y≥5?"]
        ),
        "system_graphing": Concept(
            name="System – Graphing",
            description="Solve a linear system by intersecting lines.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-37",
            examples=["Graph and solve y=2x+1 and y=−x+7.", "Estimate intersection of y=0.5x−4 and y=3x+2."]
        ),
        "system_substitution": Concept(
            name="System – Substitution",
            description="Solve 2-eq system using substitution.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-38",
            examples=["Solve y=4x−5 and 2x+y=7.", "System: x=3y−2, 2x−y=8."]
        ),
        "system_elimination": Concept(
            name="System – Elimination",
            description="Solve using addition/elimination.",
            difficulty=3,
            time_to_master=120,
            category="Linear",
            concept_id=f"{LF}-39",
            examples=["Solve 3x+2y=22 and 5x−2y=18.", "System: 4a−b=9, 6a+3b=33."]
        ),
        "system_solution_classification": Concept(
            name="System – Solution Classification",
            description="Decide one, none, or infinitely many solutions.",
            difficulty=3,
            time_to_master=90,
            category="Linear",
            concept_id=f"{LF}-40",
            examples=["Classify 2x−3y=6 and 4x−6y=12.", "Give example of inconsistent system."]
        ),
    }
    
    # Exponential Concepts
    exponential_concepts = {
        "exponent_product_rule": Concept(
            name="Exponent Product Rule",
            description="a^m · a^n = a^{m+n}.",
            difficulty=3,
            time_to_master=45,
            category="Exponential",
            concept_id=f"{EXP}-41",
            examples=["Simplify x³·x⁵.", "Compute 2⁴·2³."]
        ),
        "power_of_power": Concept(
            name="Power of a Power",
            description="(a^m)^n = a^{mn}.",
            difficulty=3,
            time_to_master=45,
            category="Exponential",
            concept_id=f"{EXP}-42",
            examples=["Simplify (x²)³.", "Evaluate (3²)⁴."]
        ),
        "power_of_product": Concept(
            name="Power of a Product",
            description="(ab)^n = a^n b^n.",
            difficulty=3,
            time_to_master=45,
            category="Exponential",
            concept_id=f"{EXP}-43",
            examples=["Rewrite (2x)³.", "Expand (5ab)²."]
        ),
        "zero_exponent": Concept(
            name="Zero Exponent",
            description="a^0 = 1 for a≠0.",
            difficulty=3,
            time_to_master=30,
            category="Exponential",
            concept_id=f"{EXP}-44",
            examples=["Evaluate 7⁰.", "Simplify (3x)⁰."]
        ),
        "negative_exponents": Concept(
            name="Negative Integer Exponents",
            description="a^{−n}=1/a^n.",
            difficulty=3,
            time_to_master=60,
            category="Exponential",
            concept_id=f"{EXP}-45",
            examples=["Rewrite 2^{−3}.", "Simplify x^{−2} y^{−1}."]
        ),
        "fractional_exponents": Concept(
            name="Fractional Exponents (Roots)",
            description="a^{1/n}=ⁿ√a, a^{m/n}=ⁿ√a^m.",
            difficulty=3,
            time_to_master=60,
            category="Exponential",
            concept_id=f"{EXP}-46",
            examples=["Evaluate 27^{1/3}.", "Rewrite √[4]{x³} as an exponent."]
        ),
        "simplify_radicals": Concept(
            name="Simplify Radical Expressions",
            description="Combine like radicals & rationalize.",
            difficulty=3,
            time_to_master=60,
            category="Exponential",
            concept_id=f"{EXP}-47",
            examples=["Simplify 3√5 + 2√5.", "Rationalize 5/√2."]
        ),
        "scientific_notation": Concept(
            name="Scientific Notation",
            description="Write and compute with a×10^n.",
            difficulty=2,
            time_to_master=45,
            category="Exponential",
            concept_id=f"{EXP}-48",
            examples=["Express 0.00032 in sci-notation.", "(3×10⁵)(2×10⁻³)=?"]
        ),
    }
    
    # Polynomial Concepts
    polynomial_concepts = {
        "add_subtract_polynomials": Concept(
            name="Add/Subtract Polynomials",
            description="Combine like terms across polynomials.",
            difficulty=2,
            time_to_master=45,
            category="Polynomials",
            concept_id=f"{POL}-49",
            examples=["(2x²+3x) + (x²−4x).", "Subtract (3a²−2a+1) from (5a²+4)."]
        ),
        "multiply_monomial_polynomial": Concept(
            name="Multiply Monomial·Polynomial",
            description="Distribute a monomial across a polynomial.",
            difficulty=3,
            time_to_master=60,
            category="Polynomials",
            concept_id=f"{POL}-50",
            examples=["3x(4x²−5x+2).", "−2a²(6a−7)."]
        ),
        "binomial_product": Concept(
            name="Binomial Product (FOIL)",
            description="Multiply two binomials.",
            difficulty=3,
            time_to_master=60,
            category="Polynomials",
            concept_id=f"{POL}-51",
            examples=["(x+5)(x−3).", "(2a−7)(a+4)."]
        ),
        "factor_gcf": Concept(
            name="Factor GCF",
            description="Factor out greatest common factor.",
            difficulty=3,
            time_to_master=60,
            category="Polynomials",
            concept_id=f"{POL}-52",
            examples=["Factor 12x²y+8xy.", "Factor 6a³−9a²."]
        ),
        "factor_trinomials_a1": Concept(
            name="Factor Trinomials a=1",
            description="Factor x²+bx+c.",
            difficulty=3,
            time_to_master=75,
            category="Polynomials",
            concept_id=f"{POL}-53",
            examples=["Factor x²+5x+6.", "Factor x²−x−12."]
        ),
        "factor_trinomials_an1": Concept(
            name="Factor Trinomials a≠1",
            description="Factor ax²+bx+c, a≠1.",
            difficulty=4,
            time_to_master=90,
            category="Polynomials",
            concept_id=f"{POL}-54",
            examples=["Factor 2x²+7x+3.", "Factor 3x²−11x−4."]
        ),
        "special_products": Concept(
            name="Special Products & Diff Squares",
            description="Recognize (a±b)² and a²−b².",
            difficulty=3,
            time_to_master=60,
            category="Polynomials",
            concept_id=f"{POL}-55",
            examples=["Expand (3y−4)².", "Factor 25x²−9."]
        ),
        "polynomial_division": Concept(
            name="Polynomial Division (Synthetic)",
            description="Divide by x−k and find remainder.",
            difficulty=4,
            time_to_master=75,
            category="Polynomials",
            concept_id=f"{POL}-56",
            examples=["Divide x³−2x²+5 by x−2.", "Use synthetic division to find f(3) for f(x)=2x³+x²−5."]
        ),
    }
    
    # Quadratic Concepts
    quadratic_concepts = {
        "vertex_standard_form": Concept(
            name="Vertex from Standard Form",
            description="Find vertex of y=ax²+bx+c.",
            difficulty=4,
            time_to_master=60,
            category="Quadratic",
            concept_id=f"{Q}-57",
            examples=["Vertex of y=2x²−8x+3.", "Given vertex (h,k) compute axis of symmetry."]
        ),
        "completing_square": Concept(
            name="Completing the Square",
            description="Rewrite ax²+bx+c into a(x−h)²+k.",
            difficulty=4,
            time_to_master=90,
            category="Quadratic",
            concept_id=f"{Q}-58",
            examples=["Complete square: x²+6x+5.", "Rewrite 2x²−4x+7 in vertex form."]
        ),
        "quadratic_formula": Concept(
            name="Quadratic Formula",
            description="Solve ax²+bx+c=0 using −b±√(b²−4ac)/2a.",
            difficulty=4,
            time_to_master=75,
            category="Quadratic",
            concept_id=f"{Q}-59",
            examples=["Solve 3x²+2x−1=0.", "Find exact roots of x²−5x−6=0."]
        ),
        "discriminant": Concept(
            name="Discriminant & Roots",
            description="Analyze b²−4ac for number/type of roots.",
            difficulty=3,
            time_to_master=45,
            category="Quadratic",
            concept_id=f"{Q}-60",
            examples=["Discriminant of 4x²+4x+1.", "How many real roots does x²−7x+12 have?"]
        ),
        "graph_parabola": Concept(
            name="Graph Parabola (Vertex Form)",
            description="Graph y=a(x−h)²+k quickly.",
            difficulty=3,
            time_to_master=60,
            category="Quadratic",
            concept_id=f"{Q}-61",
            examples=["Graph y=−(x+2)²+3.", "Identify vertex and direction of y=2(x−1)²−4."]
        ),
    }
    
    # Function Tools Concepts
    function_concepts = {
        "function_notation": Concept(
            name="Function Notation",
            description="Use f(x) and evaluate.",
            difficulty=3,
            time_to_master=45,
            category="Function Tools",
            concept_id=f"{FN}-62",
            examples=["If f(x)=2x²−3, find f(4).", "Let g(x)=|x−5|. Evaluate g(1)."]
        ),
        "domain_range": Concept(
            name="Domain & Range",
            description="Find possible x (domain) and f(x) (range).",
            difficulty=3,
            time_to_master=60,
            category="Function Tools",
            concept_id=f"{FN}-63",
            examples=["Domain of f(x)=1/(x−2).", "Range of y=|x|."]
        ),
        "function_composition": Concept(
            name="Composition of Functions",
            description="Compute (f∘g)(x)=f(g(x)).",
            difficulty=3,
            time_to_master=75,
            category="Function Tools",
            concept_id=f"{FN}-64",
            examples=["If f(x)=x+1, g(x)=x², find (f∘g)(2).", "Find (g∘f)(x) for same f,g."]
        ),
        "function_transformations": Concept(
            name="Function Transformations",
            description="Shifts, reflections, stretches of parent graphs.",
            difficulty=4,
            time_to_master=90,
            category="Function Tools",
            concept_id=f"{FN}-65",
            examples=["Graph y=−2√(x−3)+5 from y=√x.", "Describe transformation from y=x² to y=(x−4)²−7."]
        ),
    }
    
    # Add all concepts to the graph
    for concept_dict in [number_sense_concepts, algebraic_concepts, linear_concepts, 
                        exponential_concepts, polynomial_concepts, quadratic_concepts,
                        function_concepts]:
        for concept_id, concept in concept_dict.items():
            graph.add_concept(concept)
    
    # === PREREQUISITE RELATIONSHIPS ===
    # Basic number sense foundations
    graph.add_prerequisite("NS-01", "NS-02")  # Counting → Place Value
    graph.add_prerequisite("NS-01", "NS-03")  # Counting → Commutative
    graph.add_prerequisite("NS-01", "NS-04")  # Counting → Associative
    graph.add_prerequisite("NS-03", "NS-05")  # Commutative → Identity/Inverse
    graph.add_prerequisite("NS-04", "NS-05")  # Associative → Identity/Inverse
    graph.add_prerequisite("NS-02", "NS-06")  # Place Value → Multiplication Facts
    graph.add_prerequisite("NS-06", "NS-07")  # Multiplication → Division Facts
    
    # Fraction sequence
    graph.add_prerequisite("NS-07", "NS-08")  # Division Facts → Fractions Part-Whole
    graph.add_prerequisite("NS-08", "NS-09")  # Fractions Part-Whole → Equivalent Fractions
    graph.add_prerequisite("NS-09", "NS-10")  # Equivalent Fractions → Fraction-Decimal
    
    # Advanced number concepts
    graph.add_prerequisite("NS-05", "NS-11")  # Identity/Inverse → GCF
    graph.add_prerequisite("NS-05", "NS-12")  # Identity/Inverse → LCM
    graph.add_prerequisite("NS-06", "NS-13")  # Multiplication Facts → Integer Add/Sub
    graph.add_prerequisite("NS-13", "NS-14")  # Integer Add/Sub → Integer Mult/Div
    graph.add_prerequisite("NS-13", "NS-15")  # Integer Add/Sub → Absolute Value
    graph.add_prerequisite("NS-14", "NS-16")  # Integer Mult/Div → Order of Operations
    
    # FIXED: Algebraic Thinking - proper logical order
    graph.add_prerequisite("NS-06", "AT-19")  # Multiplication Facts → Distributive Property
    graph.add_prerequisite("NS-10", "AT-17")  # Fraction-Decimal → Identify Like Terms  
    graph.add_prerequisite("AT-17", "AT-18")  # Identify Like Terms → Translate Expressions
    graph.add_prerequisite("AT-19", "AT-21")  # Distributive Property → Evaluate Expression
    graph.add_prerequisite("AT-18", "AT-21")  # Translate Expressions → Evaluate Expression
    graph.add_prerequisite("AT-21", "AT-20")  # Evaluate Expression → Combine Like Terms
    graph.add_prerequisite("AT-20", "AT-22")  # Combine Like Terms → One-Step Equations (+/−)
    graph.add_prerequisite("AT-20", "AT-23")  # Combine Like Terms → One-Step Equations (×/÷)
    
    # Equation solving sequence
    graph.add_prerequisite("NS-15", "AT-22")  # Absolute Value → One-Step (+/-)
    graph.add_prerequisite("AT-20", "AT-22")  # Evaluate Expression → One-Step (+/-)
    graph.add_prerequisite("AT-20", "AT-23")  # Evaluate Expression → One-Step (×/÷)
    graph.add_prerequisite("AT-22", "AT-24")  # One-Step (+/-) → Two-Step
    graph.add_prerequisite("AT-23", "AT-24")  # One-Step (×/÷) → Two-Step
    graph.add_prerequisite("AT-24", "AT-25")  # Two-Step → Multi-Step
    graph.add_prerequisite("NS-16", "AT-24")  # Order of Operations → Two-Step
    
    # Proportional relationships and coordinate plane
    graph.add_prerequisite("AT-25", "AT-26")  # Multi-Step → Proportional Relationships
    graph.add_prerequisite("AT-26", "AT-27")  # Proportional → Unit Rate/Slope
    graph.add_prerequisite("NS-15", "AT-28")  # Absolute Value → Coordinate Plane
    graph.add_prerequisite("AT-28", "AT-29")  # Coordinate Plane → Distance Formula
    
    # Linear Functions
    graph.add_prerequisite("AT-27", "LF-30")  # Unit Rate/Slope → Slope from Two Points
    graph.add_prerequisite("AT-28", "LF-30")  # Coordinate Plane → Slope from Two Points
    graph.add_prerequisite("AT-29", "LF-30")  # Distance Formula → Slope from Two Points
    graph.add_prerequisite("LF-30", "LF-31")  # Slope from Two Points → Slope-Intercept
    graph.add_prerequisite("LF-30", "LF-32")  # Slope from Two Points → Point-Slope
    graph.add_prerequisite("LF-31", "LF-33")  # Slope-Intercept → Convert Forms
    graph.add_prerequisite("LF-32", "LF-33")  # Point-Slope → Convert Forms
    graph.add_prerequisite("LF-33", "LF-34")  # Convert Forms → Parallel/Perpendicular
    
    # Linear inequalities
    graph.add_prerequisite("AT-22", "LF-35")  # One-Step Equations → 1-var Inequality
    graph.add_prerequisite("LF-31", "LF-35")  # Slope-Intercept → 1-var Inequality
    graph.add_prerequisite("LF-31", "LF-36")  # Slope-Intercept → 2-var Inequality
    
    # Systems of equations
    graph.add_prerequisite("LF-31", "LF-37")  # Slope-Intercept → System Graphing
    graph.add_prerequisite("LF-37", "LF-38")  # System Graphing → Substitution
    graph.add_prerequisite("LF-37", "LF-39")  # System Graphing → Elimination
    graph.add_prerequisite("LF-38", "LF-40")  # Substitution → Solution Classification
    graph.add_prerequisite("LF-39", "LF-40")  # Elimination → Solution Classification
    graph.add_prerequisite("LF-36", "LF-40")  # 2-var Inequality → Solution Classification
    
    # Exponential rules sequence
    graph.add_prerequisite("NS-14", "EXP-41")  # Integer Mult/Div → Product Rule
    graph.add_prerequisite("EXP-41", "EXP-42")  # Product Rule → Power of Power
    graph.add_prerequisite("EXP-41", "EXP-43")  # Product Rule → Power of Product
    graph.add_prerequisite("EXP-42", "EXP-44")  # Power of Power → Zero Exponent
    graph.add_prerequisite("EXP-43", "EXP-44")  # Power of Product → Zero Exponent
    graph.add_prerequisite("EXP-44", "EXP-45")  # Zero Exponent → Negative Exponents
    graph.add_prerequisite("EXP-45", "EXP-46")  # Negative → Fractional Exponents
    graph.add_prerequisite("EXP-46", "EXP-47")  # Fractional → Simplify Radicals
    graph.add_prerequisite("EXP-41", "EXP-48")  # Product Rule → Scientific Notation
    
    # Polynomial operations
    graph.add_prerequisite("AT-21", "POL-49")  # Combine Like Terms → Add/Subtract Polynomials
    graph.add_prerequisite("AT-19", "POL-50")  # Distributive → Multiply Monomial·Polynomial
    graph.add_prerequisite("EXP-41", "POL-50")  # Product Rule → Multiply Monomial·Polynomial
    graph.add_prerequisite("POL-50", "POL-51")  # Monomial·Polynomial → FOIL
    graph.add_prerequisite("NS-11", "POL-52")  # GCF → Factor GCF
    graph.add_prerequisite("NS-12", "POL-52")  # LCM → Factor GCF
    graph.add_prerequisite("POL-52", "POL-53")  # Factor GCF → Factor Trinomials a=1
    graph.add_prerequisite("POL-53", "POL-54")  # Factor a=1 → Factor a≠1
    graph.add_prerequisite("POL-51", "POL-55")  # FOIL → Special Products
    graph.add_prerequisite("POL-55", "POL-56")  # Special Products → Polynomial Division
    
    # Quadratics
    graph.add_prerequisite("POL-54", "Q-57")  # Factor a≠1 → Vertex from Standard
    graph.add_prerequisite("POL-49", "Q-58")  # Add/Subtract Polynomials → Completing Square
    graph.add_prerequisite("POL-56", "Q-58")  # Polynomial Division → Completing Square
    graph.add_prerequisite("Q-58", "Q-59")  # Completing Square → Quadratic Formula
    graph.add_prerequisite("Q-59", "Q-60")  # Quadratic Formula → Discriminant
    graph.add_prerequisite("Q-57", "Q-61")  # Vertex → Graph Quadratics
    graph.add_prerequisite("Q-60", "Q-61")  # Discriminant → Graph Quadratics
    
    # Function Tools
    graph.add_prerequisite("LF-31", "FN-62")  # Slope-Intercept → Function Notation
    graph.add_prerequisite("EXP-41", "FN-62")  # Product Rule → Function Notation
    graph.add_prerequisite("LF-40", "FN-63")  # Solution Classification → Domain & Range
    graph.add_prerequisite("FN-62", "FN-63")  # Function Notation → Domain & Range
    graph.add_prerequisite("FN-63", "FN-64")  # Domain & Range → Composition
    graph.add_prerequisite("FN-64", "FN-65")  # Composition → Transformations
    graph.add_prerequisite("Q-57", "FN-65")  # Vertex → Transformations
    graph.add_prerequisite("EXP-47", "FN-65")  # Simplify Radicals → Transformations
    
    return graph


class AlgebraGraph:
    """
    Simple wrapper class for the algebra knowledge graph.
    
    This provides a class-based interface for tests and other code
    that expects an AlgebraGraph class rather than just a function.
    """
    
    def __init__(self):
        """Initialize the algebra graph by building it."""
        self.graph = build_algebra_knowledge_graph()
    
    @property
    def knowledge_graph(self):
        """Provide access to the underlying knowledge graph for compatibility."""
        return self.graph
    
    def get_concept(self, concept_id: str):
        """Get a concept by ID."""
        return self.graph.get_concept(concept_id)
    
    def get_all_concepts(self):
        """Get all concepts."""
        return self.graph.get_all_concepts()
    
    def get_prerequisites(self, concept_id: str):
        """Get prerequisites for a concept."""
        return self.graph.get_prerequisites(concept_id)
    
    def get_dependent_concepts(self, concept_id: str):
        """Get dependent concepts."""
        return self.graph.get_dependent_concepts(concept_id)
    
    def find_learning_path(self, start_concept: str, end_concept: str):
        """Find learning path between concepts."""
        return self.graph.find_learning_path(start_concept, end_concept)
    
    def calculate_centrality(self, concept_id: str):
        """Calculate centrality of a concept."""
        return self.graph.calculate_centrality(concept_id)
    
    def get_central_concepts(self, n: int = 10):
        """Get the most central concepts."""
        return self.graph.get_central_concepts(n)
    
    def save_to_file(self, filename: str):
        """Save the graph to a file."""
        return self.graph.save_to_file(filename)


if __name__ == "__main__":
    # Build the graph and save it to a file
    graph = build_algebra_knowledge_graph()
    
    # Print some statistics
    print(f"Algebra Knowledge Graph: {len(graph.concepts)} concepts")
    
    # Identify the most central concepts
    central_concepts = graph.get_central_concepts(10)
    print("\nMost central concepts:")
    for concept in central_concepts:
        print(f"- {concept.name} (Centrality: {graph.calculate_centrality(concept.id):.2f})")
    
    # Save the graph to a file
    graph.save_to_file("algebra_knowledge_graph.json")
    print("\nSaved knowledge graph to algebra_knowledge_graph.json") 