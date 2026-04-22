from groq import Groq
from config import Config

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=Config.GROQ_API_KEY)

    def get_completion(self, system_prompt, user_query):
        completion = self.client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        return completion.choices[0].message.content.strip()
