#!/usr/bin/env python3
"""
Compare similarity between different product compatibility statements.
"""

import numpy as np

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
    """
    # Basic similarity as a starting point
    basic_sim = basic_cosine_similarity(text1, text2)
    
    # Check for critical semantic differences
    t1_tokens = text1.lower().split()
    t2_tokens = text2.lower().split()
    
    # Check for negation
    negation_words = ["not", "isn't", "doesn't", "don't", "can't", "cannot", "never", "no"]
    has_negation1 = any(word in t1_tokens for word in negation_words)
    has_negation2 = any(word in t2_tokens for word in negation_words)
    
    # Different products/models mentioned?
    products1 = [word for word in t1_tokens 
                if any(word.startswith(prefix) for prefix in ["sx", "bm", "xm"])]
    products2 = [word for word in t2_tokens 
                if any(word.startswith(prefix) for prefix in ["sx", "bm", "xm"])]
    different_products = set(products1) != set(products2)
    
    # Adjust similarity based on semantic differences
    if has_negation1 != has_negation2:
        # Opposite meanings due to negation
        return basic_sim * 0.4
    elif different_products:
        # Embeddings would recognize different products have different properties
        # But in this case, the compatibility with the same item might be semantically similar
        return basic_sim * 0.85  # Slightly lower similarity due to different products
    else:
        return basic_sim

def display_token_vectors(text1, text2):
    """Display the token vectors for comparison."""
    # Tokenize
    tokens1 = text1.lower().split()
    tokens2 = text2.lower().split()
    
    # Get all unique tokens in sorted order
    all_tokens = sorted(set(tokens1 + tokens2))
    
    # Create vectors
    vector1 = [tokens1.count(token) for token in all_tokens]
    vector2 = [tokens2.count(token) for token in all_tokens]
    
    # Display
    print("\nToken vectors:")
    print(f"Tokens: {all_tokens}")
    print(f"Vector 1: {vector1}")
    print(f"Vector 2: {vector2}")

def main():
    # Original sentences
    original1 = "bm2000 is compatible with xm2030"
    original2 = "bm2030 is not compatible with xm2030"
    
    # New sentences
    sentence1 = "sx2030 is compitable with 12345678"
    sentence2 = "sx2000 is compitable with 12345678"
    
    # Calculate similarities for the new sentences
    basic_sim = basic_cosine_similarity(sentence1, sentence2)
    semantic_sim = simulated_semantic_similarity(sentence1, sentence2)
    
    # Print results
    print(f"Sentence 1: \"{sentence1}\"")
    print(f"Sentence 2: \"{sentence2}\"")
    print(f"\nBasic cosine similarity (bag of words): {basic_sim:.4f}")
    print(f"Estimated semantic similarity (like ada-002): {semantic_sim:.4f}")
    
    # Display token vectors
    display_token_vectors(sentence1, sentence2)
    
    # Comparison with original example
    print("\nComparison with original example:")
    print(f"Original sentence 1: \"{original1}\"")
    print(f"Original sentence 2: \"{original2}\"")
    orig_basic = basic_cosine_similarity(original1, original2)
    orig_semantic = simulated_semantic_similarity(original1, original2)
    print(f"Original basic similarity: {orig_basic:.4f}")
    print(f"Original semantic similarity: {orig_semantic:.4f}")
    
    print("\nAnalysis:")
    print("1. In the new example, both sentences state compatibility with the same item (12345678)")
    print("2. The only difference is the product model (sx2030 vs sx2000)")
    print("3. There are no negations or contradictions between the statements")
    print("4. Semantic embedding models would recognize higher similarity between these")
    print("   statements compared to the original example with a negation")
    print("5. Both statements make similar claims about compatibility, just about different products")

if __name__ == "__main__":
    main() 