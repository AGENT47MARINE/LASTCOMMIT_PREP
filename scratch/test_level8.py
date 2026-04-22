
import sys
import os
from nodes import code_solver_node
from state import AgentState

def test_level_8():
    query = "The following is a transaction log. Extract the FIRST transaction greater than $100 made by a user whose name starts with 'S'. Log: - Alice paid $45 | Sam paid $80 | Steve paid $210 | Bob paid $310 - Sophie paid $95 | Sara paid $150 | Tom paid $500 | Sally paid $130"
    state = {"input": query, "intent": "CODE"}
    result = code_solver_node(state)
    print(f"Query: {query}")
    print(f"Result: {result['result']['solution']}")
    print(f"Reasoning: {result['reasoning']}")

if __name__ == "__main__":
    test_level_8()
