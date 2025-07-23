#!/usr/bin/env python3
"""
Simulate semantic embedding similarity between two sentences.

This script demonstrates how embedding models like OpenAI's ada-002 
would compare sentences, focusing on semantic meaning rather than 
just word overlap.
"""

import numpy as np
from collections import Counter

def basic_cosine_similarity(text1, text2):
    """Calculate basic cosine similarity using bag of words."""
    # Tokenize texts
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    # Get all unique words
    all_tokens = set(tokens1 + tokens2)
    
    # Create count vectors
    vector1 = [tokens1.count(token) for token in all_tokens]
    vector2 = [tokens2.count(token) for token in all_tokens]
    
    # Convert to numpy arrays
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)
    
    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def simulated_semantic_similarity(text1, text2):
    """
    Simulate semantic similarity that embeddings would capture.
    
    This is a simplified simulation that attempts to account for semantic
    differences like negation, which real embeddings would capture better.
    """
    # Basic similarity as a starting point
    basic_sim = basic_cosine_similarity(text1, text2)
    
    # Check for critical semantic differences
    t1_tokens = text1.lower().split()
    t2_tokens = text2.lower().split()
    
    # Check for negation (not, isn't, doesn't, etc.)
    # If one has a negation and the other doesn't, they are likely semantically opposite
    negation_words = ["not", "isn't", "doesn't", "don't", "can't", "cannot", "never", "no"]
    has_negation1 = any(word in t1_tokens for word in negation_words)
    has_negation2 = any(word in t2_tokens for word in negation_words)
    
    # Different products/models mentioned?
    products1 = [word for word in t1_tokens if word.startswith("bm") or word.startswith("xm")]
    products2 = [word for word in t2_tokens if word.startswith("bm") or word.startswith("xm")]
    different_products = set(products1) != set(products2)
    
    # Adjust similarity based on semantic differences
    if has_negation1 != has_negation2:
        # Opposite meanings due to negation - embeddings would catch this
        # Reduce similarity significantly
        return basic_sim * 0.4  # Much lower similarity due to opposite meaning
    elif different_products:
        # Different products discussed - embeddings would understand this difference
        return basic_sim * 0.8  # Somewhat lower similarity due to different products
    else:
        return basic_sim

def main():
    # The two sentences to compare
    sentence1 = "bm2000 is compatible with xm2030"
    sentence2 = "bm2030 is not compatible with xm2030"
    
    # Calculate both types of similarity
    basic_sim = basic_cosine_similarity(sentence1, sentence2)
    semantic_sim = simulated_semantic_similarity(sentence1, sentence2)
    
    # Print results
    print(f"Sentence 1: \"{sentence1}\"")
    print(f"Sentence 2: \"{sentence2}\"")
    print(f"\nBasic cosine similarity (bag of words): {basic_sim:.4f}")
    print(f"Simulated semantic similarity (like ada-002): {semantic_sim:.4f}")
    
    print("\nExplanation:")
    print("1. Basic cosine similarity only considers word overlap")
    print("2. Semantic embedding models like OpenAI's ada-002 would understand:")
    print("   - The semantic impact of negations ('not')")
    print("   - Different product references ('bm2000' vs 'bm2030')")
    print("   - That these sentences have nearly opposite meanings")
    print("\nNote: This is a simulation - real embedding models would provide")
    print("more accurate semantic similarity scores by understanding context")
    print("and meaning, not just word presence.")

if __name__ == "__main__":
    main() 