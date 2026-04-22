
import sys
import os
import re
from nodes import code_solver_node
from state import AgentState

def cosine_similarity_mock(a, b):
    # This is a very basic mock to see if strings match exactly
    return 1.0 if a.strip() == b.strip() else 0.0

def run_test(query, expected):
    state = {"input": query, "intent": "CODE"}
    result = code_solver_node(state)
    actual = result['result']['solution']
    score = cosine_similarity_mock(actual, expected)
    print(f"Query: {query[:50]}...")
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    print(f"Score:    {score}")
    print("-" * 20)
    return score

def test_suite():
    tests = [
        {
            "query": "Extract the FIRST transaction greater than $100 made by a user whose name starts with 'S'. Log: - Alice paid $45 | Sam paid $80 | Steve paid $210 | Bob paid $310 - Sophie paid $95 | Sara paid $150 | Tom paid $500 | Sally paid $130",
            "expected": "Steve paid the amount of $210."
        },
        {
            "query": "Extract the LAST transaction greater than $200 made by a user whose name starts with 'B'. Log: - Bob paid $310 | Bill paid $150 | Ben paid $400 | Alice paid $500",
            "expected": "Ben paid the amount of $400."
        },
        {
            "query": "Find the first payment over $50 by someone starting with 'A'. Log: Alice paid $45, Aaron paid $120, Amy paid $30",
            "expected": "Aaron paid the amount of $120."
        },
        {
            "query": "Extract the FIRST transaction greater than $1000. Log: Bob paid $500 | Charlie paid $1200 | Dave paid $1100",
            "expected": "Charlie paid the amount of $1200."
        }
    ]
    
    total_score = 0
    for t in tests:
        total_score += run_test(t["query"], t["expected"])
    
    print(f"Average Score: {total_score / len(tests)}")

if __name__ == "__main__":
    test_suite()
