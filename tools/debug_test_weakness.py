import sys
sys.path.append('math_learning')
from math_learning.learning_graph.user_model import LearningGraph
from math_learning.learning_graph.personalized_learning import PersonalizedLearningSystem
from unittest.mock import Mock

# Create the exact same mock knowledge graph as in the test
kg = Mock()
concepts = {}
concept_objects = []

# Create concepts in different categories with enough per category
categories = {
    'Number Sense': ['NS-01', 'NS-02', 'NS-03'],
    'Algebra': ['ALG-01', 'ALG-02', 'ALG-03'],
    'Geometry': ['GEO-01', 'GEO-02']
}

for category, concept_ids in categories.items():
    for concept_id in concept_ids:
        concept = Mock()
        concept.id = concept_id
        concept.name = f'{category} Concept {concept_id.split("-")[1]}'
        concept.category = category
        concept.difficulty = 1
        concept.prerequisites = set()
        concept.dependents = set()
        concepts[concept.id] = concept
        concept_objects.append(concept)

# Add the test concepts
for i in range(3):
    concept_id = f'concept_{i}'
    concept = Mock()
    concept.id = concept_id
    concept.name = f'Test Concept {i}'
    concept.category = 'Test'
    concept.difficulty = 1
    concept.prerequisites = set()
    concept.dependents = set()
    concepts[concept.id] = concept
    concept_objects.append(concept)

kg.concepts = concepts
kg.get_concept = lambda cid: concepts.get(cid)
kg.get_all_concepts = lambda: concept_objects
kg.get_prerequisites = lambda cid: []
kg.get_dependent_concepts = lambda cid: []

# Create learning graph exactly like the test
lg = LearningGraph('test_user', kg)

# Simulate learning history exactly like the test
test_concepts = ['concept_0', 'concept_1', 'concept_2']

for i, concept_id in enumerate(test_concepts):
    # Different performance patterns for each concept
    success_rate = 0.8 - (i * 0.2)  # Decreasing success rate
    print(f'Concept {concept_id}: success_rate = {success_rate}')
    
    for j in range(10):
        exercise_id = f'ex_{concept_id}_{j}'
        success = j < (success_rate * 10)  # First exercises succeed
        difficulty = 0.5 + (j * 0.05)  # Increasing difficulty
        
        lg.record_exercise_attempt(exercise_id, concept_id, success, difficulty, 1.0)

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