import re

def rule_based_route(query: str):
    """Tier 0: Ultra-fast regex routing (No LLM, No Embeddings)"""
    q = query.lower()
    
    # Priority 1: Math/Logic (CODE)
    if any(word in q for word in ["calculate", "math", "code", "solve", "+", "-", "*", "/"]):
        return "CODE"
    
    # Priority 2: Summarization
    if any(word in q for word in ["summarize", "summary", "tl;dr"]):
        return "SUMMARIZE"
        
    # Priority 3: Facts/Entities
    if any(word in q for word in ["extract", "entities", "who is", "what is"]):
        return "ENTITY"
        
    return None

from sentence_transformers import SentenceTransformer, util
import torch

# Load the model once
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define Intent Prototypes
INTENT_PROTOTYPES = {
    "SUMMARIZE": "Please provide a brief summary of this text.",
    "ENTITY": "Extract all the names, dates, and locations from this paragraph.",
    "CODE": "Solve this mathematical equation or write a code snippet.",
    "ANOMALY": "Find any outliers or strange patterns in this dataset.",
    "STRUCTURED": "Convert this raw data into a clean JSON or table format."
}

# Pre-calculate prototype embeddings
PROTOTYPE_EMBEDDINGS = {k: model.encode(v, convert_to_tensor=True) for k, v in INTENT_PROTOTYPES.items()}

def semantic_route(query: str):
    """Tier 1: Semantic similarity routing using Cosine Similarity"""
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    best_intent = None
    max_score = 0.0
    
    for intent, proto_emb in PROTOTYPE_EMBEDDINGS.items():
        score = util.cos_sim(query_embedding, proto_emb).item()
        if score > max_score:
            max_score = score
            best_intent = intent
            
    return best_intent, max_score
