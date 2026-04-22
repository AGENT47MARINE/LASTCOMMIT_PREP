import re

def _extract_actual_task(query: str) -> str:
    patterns = [
        r"(?is)actual\s+task\s*[:\-]\s*(.*)$",
        r"(?is)actual\s+question\s*[:\-]\s*(.*)$",
        r"(?is)task\s*[:\-]\s*(.*)$",
        r"(?is)question\s*[:\-]\s*(.*)$",
    ]

    best_task = query
    for pattern in patterns:
        matches = list(re.finditer(pattern, query))
        if matches:
            last_match = matches[-1]
            task = last_match.group(1).strip()
            task = task.strip(' "\'`')
            if task:
                best_task = task
    return best_task

q = "Apply these rules. Rule 1: If even, double. Rule 2: If result > 20, subtract 5. Task: return the final result for input 6."
print(f"Original: {q}")
print(f"Extracted: {_extract_actual_task(q)}")
