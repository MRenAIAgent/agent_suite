import sys
sys.path.append('math_learning')
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem
from unittest.mock import Mock

# Create mock knowledge graph
kg = Mock()
concepts = {}
for category, concept_ids in [('Number Sense', ['NS-01']), ('Algebra', ['ALG-01']), ('Geometry', ['GEO-01'])]:
    for concept_id in concept_ids:
        concept = Mock()
        concept.id = concept_id
        concept.name = f'{category} Concept {concept_id.split("-")[1]}'
        concept.category = category
        concept.difficulty = 1
        concept.prerequisites = set()
        concept.dependents = set()
        concepts[concept.id] = concept

kg.concepts = concepts
kg.get_concept = lambda cid: concepts.get(cid)

# Create learning graph with exercise history
lg = LearningGraph('test_user', kg)
lg.concept_mastery = {
    'NS-01': {'mastery': 0.8, 'confidence': 0.7, 'last_updated': '2023-01-01', 'history': []},
    'ALG-01': {'mastery': 0.6, 'confidence': 0.5, 'last_updated': '2023-01-02', 'history': []},
    'GEO-01': {'mastery': 0.4, 'confidence': 0.3, 'last_updated': '2023-01-03', 'history': []}
}

lg.exercise_history = {
    'exercise_1': [
        {'concept_id': 'NS-01', 'result': True, 'timestamp': '2023-01-01T10:00:00'},
        {'concept_id': 'NS-01', 'result': True, 'timestamp': '2023-01-01T10:05:00'},
        {'concept_id': 'NS-01', 'result': False, 'timestamp': '2023-01-01T10:10:00'},
        {'concept_id': 'NS-01', 'result': True, 'timestamp': '2023-01-01T10:15:00'}
    ],
    'exercise_2': [
        {'concept_id': 'ALG-01', 'result': False, 'timestamp': '2023-01-02T10:00:00'},
        {'concept_id': 'ALG-01', 'result': False, 'timestamp': '2023-01-02T10:05:00'},
        {'concept_id': 'ALG-01', 'result': True, 'timestamp': '2023-01-02T10:10:00'},
        {'concept_id': 'ALG-01', 'result': False, 'timestamp': '2023-01-02T10:15:00'}
    ],
    'exercise_3': [
        {'concept_id': 'GEO-01', 'result': False, 'timestamp': '2023-01-03T10:00:00'},
        {'concept_id': 'GEO-01', 'result': False, 'timestamp': '2023-01-03T10:05:00'}
    ]
}

# Create personalized system and analyze
ps = PersonalizedLearningSystem(kg)
patterns = ps.analyze_learning_patterns(lg)
print('Patterns found:', len(patterns))
for concept_id, pattern in patterns.items():
    print(f'{concept_id}: success_rate={pattern.success_rate}, learning_velocity={pattern.learning_velocity}')
    weakness_type, severity = ps._classify_weakness(pattern, lg, concept_id)
    print(f'  -> weakness_type={weakness_type}, severity={severity}')

weaknesses = ps.detect_weaknesses(lg)
print('Weaknesses found:', len(weaknesses)) 