import os
import sys
sys.path.append('.')
from nodes import classifier_node, code_solver_node
from dotenv import load_dotenv

load_dotenv()

queries = [
    "Apply rules in order to input number -5: Rule 1: If even -> double it. If odd -> add 10. Rule 2: If result > 20 -> subtract 5. Otherwise -> add 3. Rule 3: If final result divisible by 3 -> output \"FIZZ\". Otherwise -> output the number.",
    "Apply rules in order to input number 2: Rule 1: If positive -> add 5. If negative -> subtract 5. Rule 2: If result is greater than 10 -> subtract 1. Otherwise -> add 1. Rule 3: If final result divisible by 2 -> output \"BUZZ\". Otherwise -> output the number."
]

for q in queries:
    print(f"\nQuery: {q}")
    state = {"input": q, "steps": [], "result": {}, "confidence": 0}
    c_res = classifier_node(state)
    print(f"Intent: {c_res['intent']}")
    if c_res['intent'] == 'CODE':
        s_res = code_solver_node(state)
        print(f"Result: {s_res['result']}")
        print(f"Reasoning: {s_res['reasoning']}")
