from typing import List

class ExerciseBank:
    def get_exercises_for_concept(self, concept_id: str) -> List[Exercise]:
        """
        Get exercises that test a specific concept.

        Args:
            concept_id: The ID of the concept

        Returns:
            List of exercises that test the concept
        """
        # Get exercise IDs from the mapping, or empty list if none
        exercise_ids = self.concept_to_exercises.get(concept_id, [])
        
        # Debug output
        print(f"Looking for exercises with concept ID: {concept_id}")
        print(f"Found exercise IDs in mapping: {exercise_ids}")
        print(f"Available exercise IDs: {list(self.exercises.keys())}")
        
        # Filter out any missing exercise IDs
        valid_ids = [ex_id for ex_id in exercise_ids if ex_id in self.exercises]
        
        # If no valid IDs from mapping, fallback to checking all exercises
        if not valid_ids:
            print(f"No exercises found in mapping for concept {concept_id}, scanning all exercises...")
            for ex_id, exercise in self.exercises.items():
                # Check if this exercise is related to the concept
                for rel in exercise.concept_relationships:
                    if rel["concept_id"] == concept_id:
                        valid_ids.append(ex_id)
                        print(f"Found exercise {ex_id} related to concept {concept_id}")
                        break
        
        # Return exercises with the valid IDs
        return [self.exercises[ex_id] for ex_id in valid_ids] 