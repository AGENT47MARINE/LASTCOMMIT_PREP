# Skill: Prompt Engineering & Testing for a New Challenge Level

## 🎯 Purpose

When a new hackathon challenge level arrives, this skill teaches you how to:

1. **Write a tight, filler-free `SYSTEM_PROMPT`** inside `challenges/<id>/agent.py` that matches the new use case.
2. **Validate it immediately** using `test_conn.py` (LM Studio connectivity) and a local similarity test script modelled on `scratch/bench01.py` / `scratch/test_similarity.py`.

---

## 📥 Inputs You Need Before Starting

| Input | Where to get it | Example |
|---|---|---|
| **New use case** | Challenge brief / evaluator description | "Level 3: Single-sentence text summarisation" |
| **Conditions / constraints** | Challenge rules | "Return exactly one sentence. Preserve key facts." |
| **Sample test case** | Evaluator's example or inferred from the level | `Q: "The NEURON-12 … " → A: "NEURON-12 is a multi-agent …"` |
| **Expected output format** | From `skills.md` / evaluator schema | Plain string inside `{"output": "..."}` |

---

## 🔌 Step 0 — Verify LM Studio / Groq Connectivity

Before editing any prompt, confirm the model endpoint is reachable.

**File:** `LASTCOMMIT_TEST/test_conn.py`

```python
import requests

try:
    response = requests.get("http://127.0.0.1:1234/v1/models")
    print(f"Status: {response.status_code}")
    print(f"Models: {response.json()}")
except Exception as e:
    print(f"Connection Failed: {e}")
```

Run it:

```bash
cd LASTCOMMIT_TEST
python test_conn.py
```

✅ **Expected:** `Status: 200` and a list of loaded models.  
❌ **If it fails:** Make sure LM Studio is running with a model loaded **or** that `GROQ_API_KEY` is set in `.env` and you are testing against Groq instead.

---

## ✍️ Step 1 — Craft the New `SYSTEM_PROMPT`

Open `challenges/<new_id>/agent.py`. Locate the `SYSTEM_PROMPT` constant and **replace it entirely** — do not extend the old one, do not layer new rules on top of stale rules.

### The Golden Template

```python
SYSTEM_PROMPT = """\
You are a precise answer engine. <ONE sentence describing the task type>.

RULES:
1. <Core output rule — what to return and in what form.>
2. Never use conversational filler: no "Sure!", "Of course!", "Certainly!",
   "I think", "Here is", "Great question", or any similar phrases.
3. Never repeat or rephrase the question in your answer.
4. Do not add explanations, caveats, or extra commentary unless explicitly asked.
5. <Any format-specific rule: punctuation, separators, casing, length cap, etc.>
6. Output plain text only. No markdown, no bullet points, no headers,
   unless the question explicitly asks for a structured format.
"""
```

### How to Fill the Template Per Use Case

| Use case | Rule 1 | Rule 5 (extras) |
|---|---|---|
| **General QA / formatting** | "Answer the question directly and correctly, matching the implied format." | "Arithmetic answers use 'The \<noun\> is \<value\>.' (sum/difference/product/quotient/remainder)." |
| **Entity / value extraction** | "Return only the extracted value verbatim — nothing more, nothing less." | "Multiple values → comma-separated. Nothing extractable → return: None" |
| **Single-sentence summarisation** | "Summarise the input in exactly one concise sentence." | "Preserve all key facts. Do not invent or omit named entities." |
| **Structured-data processing** | "Return only the derived value or transformed structure as a plain string." | "Do not wrap in JSON or markdown. No labels like 'Result:' or 'Answer:'." |
| **Code / math solving** | "Solve the problem and return ONLY the final answer or code block." | "For arithmetic: 'The \<noun\> is \<value\>.' For code: return the raw code only." |
| **Anomaly detection** | "Identify anomalies in one concise sentence." | "If no anomaly exists, return: No anomaly detected." |
| **RAG (assets provided)** | "Answer using ONLY the information in the supplied context." | "If the answer is not in the context, return: Not found in context." |

### ❌ What NOT to do

- Do **not** keep the old challenge's rules if they contradict the new use case.
- Do **not** add "be helpful" or "be friendly" — those encourage filler.
- Do **not** add examples inside the system prompt (few-shot) unless cosine similarity is below 0.85 after two test runs; examples bloat token count.

---

## 🧪 Step 2 — Write a Local Test Script

Create `scratch/test_<challenge_id>.py`. Model it on `scratch/bench01.py`:

```python
"""
scratch/test_<challenge_id>.py
Smoke-test for challenges/<challenge_id>/agent.py
Run from project root: python -m scratch.test_<challenge_id>
"""
import math, importlib
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# ── Load the agent ───────────────────────────────────────────────────────────
agent = importlib.import_module("challenges.<challenge_id>.agent")

# ── Helpers ──────────────────────────────────────────────────────────────────
def cosine(a: str, b: str) -> float:
    v1 = Counter(a.lower().split())
    v2 = Counter(b.lower().split())
    common = set(v1) & set(v2)
    num = sum(v1[x] * v2[x] for x in common)
    den = (
        math.sqrt(sum(v**2 for v in v1.values()))
        * math.sqrt(sum(v**2 for v in v2.values()))
    )
    return num / den if den else 0.0

def jaccard_ngram(a: str, b: str, n: int = 2) -> float:
    def ngrams(text):
        t = text.lower()
        return set(t[i : i + n] for i in range(len(t) - n + 1))
    s1, s2 = ngrams(a), ngrams(b)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)

# ── Test cases ────────────────────────────────────────────────────────────────
# Format: (query, expected_output)
# Add the sample test case provided with the challenge FIRST.
# Then add edge cases that probe the conditions you wrote in the SYSTEM_PROMPT.
tests = [
    # ↓ Sample case from the challenge brief (paste exactly as given)
    (
        "<PASTE SAMPLE QUERY HERE>",
        "<PASTE EXPECTED OUTPUT HERE>",
    ),
    # ↓ Your additional edge cases
    # ("<query>", "<expected>"),
]

# ── Runner ────────────────────────────────────────────────────────────────────
total_cos = 0.0
for q, expected in tests:
    got = agent.run(q)
    cos = cosine(got, expected)
    jac = jaccard_ngram(got, expected)
    total_cos += cos
    icon = "✅" if cos > 0.85 else "⚠️"
    print(f"{icon} [{cos:.3f} cos | {jac:.3f} jac]")
    print(f"   Q        : {q}")
    print(f"   Expected : {expected}")
    print(f"   Got      : {got}")
    print()

print(f"Average cosine: {total_cos / len(tests):.3f}")
```

Replace every `<challenge_id>` placeholder with the real two-digit ID (e.g. `03`).

**Run from project root:**

```bash
python -m scratch.test_03
```

---

## 📊 Step 3 — Interpret Results & Iterate

| Cosine score | Verdict | Action |
|---|---|---|
| **≥ 0.99** | Perfect match 🎯 | Ship it. |
| **0.85 – 0.98** | High similarity ✅ | Acceptable. Optional minor prompt tune. |
| **0.60 – 0.84** | Moderate ⚠️ | Tighten Rule 1 — be more explicit about output format. |
| **< 0.60** | Low ❌ | Rewrite Rule 1 from scratch. Consider adding 1–2 few-shot examples *inside the user message*, not the system prompt. |

### Common failure patterns & fixes

| Symptom | Root cause | Fix |
|---|---|---|
| Answer has "The answer is…" prefix | Model echoes a label | Add to `_clean()`: `re.sub(r"^(The answer is|Answer):\s*", "", text, flags=re.IGNORECASE)` |
| Answer is too long / contains a preamble | System prompt too vague | Replace "answer the question" with "return ONLY the \<specific thing\>" |
| Answer format correct but wrong value | Model hallucinating | Check assets are fetched; add "Use ONLY the provided text." |
| Extraction returns full sentence instead of value | Missing verbatim instruction | Add "Return ONLY the extracted value. Do not form a sentence." |
| Arithmetic uses wrong operation noun | Missing the noun list | Paste the noun table from Challenge 01's `SYSTEM_PROMPT` Rule 1 |

---

## 🚀 Step 4 — Quick API Smoke Test (Optional but Recommended)

Start the gateway locally, then hit your new challenge endpoint:

```bash
# Terminal 1 — start gateway
uvicorn gateway:app --reload --port 8000

# Terminal 2 — fire the request
curl -s -X POST http://localhost:8000/challenge/<challenge_id> \
  -H "Content-Type: application/json" \
  -d '{"query": "<PASTE SAMPLE QUERY HERE>", "assets": []}' | python -m json.tool
```

Expected response:
```json
{
  "output": "<expected answer>"
}
```

If the response matches your expected output → you are ready to submit. If not, return to Step 2.

---

## 📝 Worked Example — Challenge 01 (General QA)

### Use case & conditions

> **Level 1 — General QA / Basic formatting.**  
> Return direct, tone-matched answers. Arithmetic uses "The \<noun\> is \<value\>." No filler.

### Sample test case (from the evaluator)

| Query | Expected output |
|---|---|
| `What is 10 + 15?` | `The sum is 25.` |

### Resulting `SYSTEM_PROMPT` (the one in `challenges/01/agent.py`)

```python
SYSTEM_PROMPT = """\
You are a precise answer engine. Your job is to answer questions directly \
and correctly, matching the tone and expected length of the question.

RULES:
1. Match the question's implied format and length:
   - A direct one-word question gets a direct one-word answer.
   - A question asking you to list or recite something gets a full list.
   - An arithmetic question (add, subtract, multiply, divide, etc.) answers \
     as a complete sentence: "The <operation_noun> is <value>." \
     where operation_noun is: addition→sum, subtraction→difference, \
     multiplication→product, division→quotient, modulo→remainder.
2. Never use conversational filler: no "Sure!", "Of course!", "Certainly!", \
   "I think", "Here is", "Great question", or any similar phrases.
3. Never repeat or rephrase the question in your answer.
4. Do not add explanations, caveats, or extra commentary unless the question \
   explicitly asks for them.
5. Match punctuation intuitively — use whatever punctuation a clean, \
   professional answer would naturally have for that question type.
6. Output plain text only. No markdown, no bullet points, no headers, unless \
   the question explicitly asks for a structured format.
"""
```

### Running the test

```bash
python -m scratch.bench01
```

```
✅ [1.000]  Q: What is 10 + 15?
         Expected : The sum is 25.
         Got      : The sum is 25.

✅ [1.000]  Q: Is 17 a prime number?
         Expected : Yes.
         Got      : Yes.

Average cosine: 1.000
```

---

## ✅ Checklist Before Moving to the Next Level

- [ ] `test_conn.py` returns `Status: 200`.
- [ ] `SYSTEM_PROMPT` in `challenges/<id>/agent.py` has been **replaced** (not appended).
- [ ] `scratch/test_<id>.py` contains the evaluator's sample case and at least one edge case.
- [ ] Average cosine from the test script is **≥ 0.85** (ideally ≥ 0.99).
- [ ] `curl` smoke test against `POST /challenge/<id>` returns the expected `{"output": "..."}`.
- [ ] `gateway.py` routes `/v1/answer` → correct challenge ID for the current active level.
