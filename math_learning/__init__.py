"""
Math Learning System

A comprehensive personalized math learning platform with knowledge graphs,
adaptive recommendations, and learning analytics.
"""

from .learning_graph import LearningGraph, PersonalizedLearningSystem
from .knowledge_graph import KnowledgeGraph
from .recommendation import GapAnalyzer
from .exercises import Exercise, ExerciseBank
from .algebra_learning import AlgebraLearningSystem

__version__ = "1.0.0"
__all__ = [
    'LearningGraph', 
    'PersonalizedLearningSystem',
    'KnowledgeGraph', 
    'GapAnalyzer',
    'Exercise',
    'ExerciseBank',
    'AlgebraLearningSystem'
] 