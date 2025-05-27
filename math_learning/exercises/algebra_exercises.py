"""
Elementary Algebra Exercises

This module creates exercises for elementary algebra concepts.
"""

from typing import List
from .exercise import Exercise
from .exercise_bank import ExerciseBank


def create_elementary_exercises() -> ExerciseBank:
    """
    Create an exercise bank with elementary-level algebra exercises.
    
    Returns:
        ExerciseBank: The exercise bank with elementary algebra exercises
    """
    bank = ExerciseBank(name="Elementary Algebra Exercises")
    
    # Counting Exercises
    counting_ex1 = Exercise(
        title="Count the Objects",
        problem="Count the number of stars: ★ ★ ★ ★ ★ ★ ★",
        solution="7",
        difficulty=1,
        estimated_time=1,
        format_type="free-response",
        hints=["Count one by one", "Point to each star as you count"]
    )
    counting_ex1.add_concept_relationship("counting", weight=1.0, relationship_type="primary")
    bank.add_exercise(counting_ex1)
    
    counting_ex2 = Exercise(
        title="Count by Twos",
        problem="Fill in the missing numbers: 2, 4, 6, ___, ___, 12",
        solution="8, 10",
        difficulty=1,
        estimated_time=2,
        format_type="fill-in-blank",
        hints=["Count by adding 2 each time", "After 6, add 2 to get..."]
    )
    counting_ex2.add_concept_relationship("counting", weight=0.9, relationship_type="primary")
    counting_ex2.add_concept_relationship("patterns", weight=0.3, relationship_type="secondary")
    bank.add_exercise(counting_ex2)
    
    # Addition Exercises
    addition_ex1 = Exercise(
        title="Simple Addition",
        problem="What is 3 + 5?",
        solution="8",
        difficulty=1,
        estimated_time=1,
        format_type="free-response",
        hints=["Count 3 and then count 5 more", "Use your fingers to help"]
    )
    addition_ex1.add_concept_relationship("addition", weight=1.0, relationship_type="primary")
    bank.add_exercise(addition_ex1)
    
    addition_ex2 = Exercise(
        title="Word Problem: Addition",
        problem="Sarah has 4 apples. Her friend gives her 3 more apples. How many apples does Sarah have now?",
        solution="7 apples",
        difficulty=1,
        estimated_time=2,
        format_type="free-response",
        hints=["Start with the number of apples Sarah has", "Add the number of apples her friend gives her"]
    )
    addition_ex2.add_concept_relationship("addition", weight=0.9, relationship_type="primary")
    addition_ex2.add_concept_relationship("counting", weight=0.3, relationship_type="foundational")
    bank.add_exercise(addition_ex2)
    
    addition_ex3 = Exercise(
        title="Double-Digit Addition",
        problem="What is 24 + 35?",
        solution="59",
        difficulty=2,
        estimated_time=3,
        format_type="free-response",
        hints=["Add the ones digits: 4 + 5", "Add the tens digits: 2 + 3", "Combine your answers"]
    )
    addition_ex3.add_concept_relationship("addition", weight=0.9, relationship_type="primary")
    addition_ex3.add_concept_relationship("place_value", weight=0.7, relationship_type="secondary")
    bank.add_exercise(addition_ex3)
    
    # Subtraction Exercises
    subtraction_ex1 = Exercise(
        title="Simple Subtraction",
        problem="What is 8 - 3?",
        solution="5",
        difficulty=1,
        estimated_time=1,
        format_type="free-response",
        hints=["Start with 8 and count backwards 3 times", "Use your fingers to help"]
    )
    subtraction_ex1.add_concept_relationship("subtraction", weight=1.0, relationship_type="primary")
    bank.add_exercise(subtraction_ex1)
    
    subtraction_ex2 = Exercise(
        title="Word Problem: Subtraction",
        problem="John has 7 candies. He eats 2 candies. How many candies does John have left?",
        solution="5 candies",
        difficulty=1,
        estimated_time=2,
        format_type="free-response",
        hints=["Start with the number of candies John has", "Subtract the number of candies he eats"]
    )
    subtraction_ex2.add_concept_relationship("subtraction", weight=0.9, relationship_type="primary")
    subtraction_ex2.add_concept_relationship("counting", weight=0.3, relationship_type="foundational")
    bank.add_exercise(subtraction_ex2)
    
    # Multiplication Exercises
    multiplication_ex1 = Exercise(
        title="Simple Multiplication",
        problem="What is 4 × 3?",
        solution="12",
        difficulty=2,
        estimated_time=2,
        format_type="free-response",
        hints=["Think of 4 groups with 3 items in each", "Add: 3 + 3 + 3 + 3"]
    )
    multiplication_ex1.add_concept_relationship("multiplication", weight=1.0, relationship_type="primary")
    bank.add_exercise(multiplication_ex1)
    
    multiplication_ex2 = Exercise(
        title="Word Problem: Multiplication",
        problem="There are 5 bags. Each bag has 4 marbles. How many marbles are there in total?",
        solution="20 marbles",
        difficulty=2,
        estimated_time=3,
        format_type="free-response",
        hints=["Multiply the number of bags by the number of marbles in each bag", "5 × 4 = ?"]
    )
    multiplication_ex2.add_concept_relationship("multiplication", weight=0.9, relationship_type="primary")
    multiplication_ex2.add_concept_relationship("addition", weight=0.4, relationship_type="foundational")
    bank.add_exercise(multiplication_ex2)
    
    # Division Exercises
    division_ex1 = Exercise(
        title="Simple Division",
        problem="What is 10 ÷ 2?",
        solution="5",
        difficulty=2,
        estimated_time=2,
        format_type="free-response",
        hints=["Think of dividing 10 items into 2 equal groups", "How many items will be in each group?"]
    )
    division_ex1.add_concept_relationship("division", weight=1.0, relationship_type="primary")
    bank.add_exercise(division_ex1)
    
    division_ex2 = Exercise(
        title="Word Problem: Division",
        problem="15 cookies are shared equally among 3 friends. How many cookies does each friend get?",
        solution="5 cookies",
        difficulty=2,
        estimated_time=3,
        format_type="free-response",
        hints=["Divide the total number of cookies by the number of friends", "15 ÷ 3 = ?"]
    )
    division_ex2.add_concept_relationship("division", weight=0.9, relationship_type="primary")
    division_ex2.add_concept_relationship("multiplication", weight=0.4, relationship_type="foundational")
    bank.add_exercise(division_ex2)
    
    # Fractions Exercises
    fractions_ex1 = Exercise(
        title="Identify the Fraction",
        problem="What fraction of the shape is shaded? □□□■",
        solution="1/4",
        difficulty=2,
        estimated_time=2,
        format_type="free-response",
        hints=["Count the total number of parts", "Count the number of shaded parts", 
               "The fraction is (shaded parts)/(total parts)"]
    )
    fractions_ex1.add_concept_relationship("fractions_intro", weight=1.0, relationship_type="primary")
    bank.add_exercise(fractions_ex1)
    
    fractions_ex2 = Exercise(
        title="Comparing Fractions",
        problem="Which fraction is larger: 3/4 or 1/2?",
        solution="3/4",
        difficulty=3,
        estimated_time=3,
        format_type="multiple-choice",
        hints=["Convert to the same denominator", "Compare 3/4 and 2/4"]
    )
    fractions_ex2.add_concept_relationship("fractions_intro", weight=0.9, relationship_type="primary")
    fractions_ex2.add_concept_relationship("division", weight=0.4, relationship_type="foundational")
    bank.add_exercise(fractions_ex2)
    
    # Patterns Exercises
    patterns_ex1 = Exercise(
        title="Complete the Pattern",
        problem="What are the next two numbers in the pattern? 5, 10, 15, 20, ___, ___",
        solution="25, 30",
        difficulty=2,
        estimated_time=3,
        format_type="fill-in-blank",
        hints=["Look at how much each number increases by", "Add 5 to each number to get the next one"]
    )
    patterns_ex1.add_concept_relationship("patterns", weight=1.0, relationship_type="primary")
    patterns_ex1.add_concept_relationship("addition", weight=0.4, relationship_type="foundational")
    bank.add_exercise(patterns_ex1)
    
    patterns_ex2 = Exercise(
        title="Find the Rule",
        problem="What is the rule for this pattern? 3, 6, 9, 12, 15",
        solution="Add 3 each time" or "Multiply by 3",
        difficulty=2,
        estimated_time=4,
        format_type="free-response",
        hints=["Find the difference between consecutive numbers", "Is there a common factor?"]
    )
    patterns_ex2.add_concept_relationship("patterns", weight=0.9, relationship_type="primary")
    patterns_ex2.add_concept_relationship("multiplication", weight=0.5, relationship_type="secondary")
    bank.add_exercise(patterns_ex2)
    
    # Place Value Exercises
    place_value_ex1 = Exercise(
        title="Identify Place Value",
        problem="In the number 357, what is the value of the digit 5?",
        solution="50",
        difficulty=2,
        estimated_time=2,
        format_type="free-response",
        hints=["Look at the position of the digit", "The middle position represents tens"]
    )
    place_value_ex1.add_concept_relationship("place_value", weight=1.0, relationship_type="primary")
    bank.add_exercise(place_value_ex1)
    
    place_value_ex2 = Exercise(
        title="Write in Expanded Form",
        problem="Write 426 in expanded form.",
        solution="400 + 20 + 6",
        difficulty=2,
        estimated_time=3,
        format_type="free-response",
        hints=["Break down the number by place value", "Hundreds + Tens + Ones"]
    )
    place_value_ex2.add_concept_relationship("place_value", weight=0.9, relationship_type="primary")
    place_value_ex2.add_concept_relationship("addition", weight=0.3, relationship_type="secondary")
    bank.add_exercise(place_value_ex2)
    
    return bank 