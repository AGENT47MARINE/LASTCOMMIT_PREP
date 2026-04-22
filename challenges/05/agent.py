from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# We use the recommended format from prompt_and_test.md
SYSTEM_PROMPT = """\
You are a precise answer engine. Determine the highest or lowest value from a given text comparison.

RULES:
1. Return ONLY the extracted name or value verbatim — nothing more, nothing less.
2. Never use conversational filler: no "Sure!", "Of course!", "Certainly!", "I think", "Here is", "Great question", or any similar phrases.
3. Never repeat or rephrase the question in your answer.
4. Do not add explanations, caveats, or extra commentary unless explicitly asked.
5. Do not include any punctuation at the end of your answer.
6. Output plain text only. No markdown, no bullet points, no headers, unless the question explicitly asks for a structured format.
"""

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def run(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Charlie ran 10km, David ran 12km. Who ran more?"),
        ("ai", "David"),
        ("human", "Product A costs $50, Product B costs $30. Which is cheaper?"),
        ("ai", "Product B"),
        ("human", "Team X has 5 points, Team Y has 8 points. Who is leading?"),
        ("ai", "Team Y"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return response.content.strip()
