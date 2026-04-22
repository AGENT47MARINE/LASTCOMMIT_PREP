import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

def solve_math_problem():
    # Initialize the Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # The problem description
    query = (
        "Let: p(x) = (x-1)(x-2)(x-3)(x-4)(x-5)(x-6) "
        "q(x) = (x-3)(x-4)(x-5)(x-6)(x-7)(x-8) "
        "Compute the degree of the GCD polynomial gcd(p(x), q(x)) over Q. "
        "Output ONLY the integer value. Do not include any text, reasoning, or punctuation."
    )

    # Use llama-3.3-70b-versatile (as verified available)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a mathematical solver. Your output must be strictly a single integer representing the answer. No explanations, no words."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0,
        max_tokens=10
    )

    result = completion.choices[0].message.content.strip()
    return result

if __name__ == "__main__":
    answer = solve_math_problem()
    print(answer)
