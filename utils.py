import re

def rule_based_route(query: str):
    """Tier 0: Ultra-fast regex routing (No LLM, No Embeddings)"""
    q = query.lower()
    if any(word in q for word in ["summarize", "summary", "tl;dr"]):
        return "SUMMARIZE"
    if any(word in q for word in ["extract", "entities", "who is", "what is"]):
        return "ENTITY"
    if any(word in q for word in ["calculate", "math", "code", "solve"]):
        return "CODE"
    return None

def semantic_route(query: str):
    """
    Tier 1 (Lite): String-based similarity.
    Replaced heavy SentenceTransformers for competition speed.
    """
    # Simply pass through to LLM for now to avoid loading heavy models
    return None, 0.0
