import os
from typing import Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import AgentState
from dotenv import load_dotenv
from utils import rule_based_route, semantic_route

load_dotenv()

class UniversalOutput(BaseModel):
    task: str
    status: str
    result: dict
    confidence: float
    error: Optional[str] = None

# --- CLOUD MODELS (Groq) ---
llm_70b = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
llm_8b = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# --- NODES ---

def classifier_node(state: AgentState):
    query = state["input"]
    rule_intent = rule_based_route(query)
    if rule_intent: return {"intent": rule_intent, "confidence": 1.0, "steps": ["Rule-routed"]}
    
    sem_intent, sem_score = semantic_route(query)
    if sem_intent and sem_score > 0.85: return {"intent": sem_intent, "confidence": sem_score, "steps": ["Semantic-routed"]}
    
    prompt = ChatPromptTemplate.from_template(
        "Classify the user intent into one of: SUMMARIZE, ENTITY, RAG, CODE, ANOMALY, STRUCTURED.\n"
        "Output ONLY the keyword, nothing else.\n"
        "Input: {input}"
    )
    response = llm_70b.invoke(prompt.format(input=query))
    content = response.content.strip().upper()
    
    for choice in ["SUMMARIZE", "ENTITY", "RAG", "CODE", "ANOMALY", "STRUCTURED"]:
        if choice in content:
            return {"intent": choice, "confidence": 0.9, "steps": [f"Groq-70B identified {choice}"]}
            
    return {"intent": "AMBIGUOUS", "confidence": 0.0, "steps": ["Groq failed to classify"]}

def code_solver_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an API serving exact answers for an evaluator. "
        "Rule 1: Solve the math/code problem correctly.\n"
        "Rule 2: Respond with EXACTLY one sentence. For addition, use 'The sum is X.'. For other operations, use the appropriate word (e.g., 'The difference is X.', 'The product is X.').\n"
        "Rule 3: NO conversational text, NO markdown, NO explanations.\n"
        "Problem: {input}"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"solution": response.content.strip()}, "steps": ["High-precision math solution generated"]}

def summarizer_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text in exactly one concise sentence. Do not add any conversational filler.\n\nText: {input}"
    )
    response = llm_8b.invoke(prompt.format(input=state["input"]))
    return {"result": {"summary": response.content.strip()}, "steps": ["Groq-8B summarized"]}

def entity_extractor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an exact string extractor. You MUST output ONLY the raw extracted entity value, with absolutely NO extra words, NO sentences, and NO punctuation at the end.\n"
        "Example 1: Extract date from 'Meeting on 12 March 2024'\nOutput: 12 March 2024\n"
        "Example 2: Extract email from 'Contact test@test.com'\nOutput: test@test.com\n\n"
        "Input: {input}\nOutput:"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"entities": response.content.strip()}, "steps": ["Groq-70B extracted entities"]}

def structured_processor_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        "You are an exact string extractor. You MUST output ONLY the raw extracted entity value, with absolutely NO extra words, NO sentences, and NO punctuation at the end.\n"
        "Example 1: Extract date from 'Meeting on 12 March 2024'\nOutput: 12 March 2024\n"
        "Example 2: Extract email from 'Contact test@test.com'\nOutput: test@test.com\n\n"
        "Input: {input}\nOutput:"
    )
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"analysis": response.content.strip()}, "steps": ["Groq-70B processed data"]}

def anomaly_detector_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template("Identify any anomalies in one concise sentence. No conversational filler.\n\nData: {input}")
    response = llm_70b.invoke(prompt.format(input=state["input"]))
    return {"result": {"anomalies": response.content.strip()}, "steps": ["Groq-70B detected anomalies"]}

def ambiguity_node(state: AgentState):
    return {"result": {"question": "Please clarify your request."}, "steps": ["Ambiguity detected"]}

def rag_node(state: AgentState):
    return {"result": {"answer": "Cloud RAG is disabled in Lite mode."}, "steps": ["RAG bypassed"]}

def validator_node(state: AgentState):
    if not state.get("result"): return {"error": "Processing failed", "steps": ["Validation failed"]}
    output = UniversalOutput(task=state.get("intent", "MULTI"), status="success", result=state["result"], confidence=state.get("confidence", 0.0))
    return {"result": output.model_dump(), "steps": ["Final validation passed"]}
