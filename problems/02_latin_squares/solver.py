from llm_client import GroqClient

class MathSolver:
    def __init__(self):
        self.client = GroqClient()
        self.system_prompt = "You are a mathematical expert. Output ONLY the final integer answer."

    def solve(self, query):
        # We append strict instructions to the query as well for redundancy
        formatted_query = f"{query} Output only the integer."
        return self.client.get_completion(self.system_prompt, formatted_query)
