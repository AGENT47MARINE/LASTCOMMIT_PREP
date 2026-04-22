import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Using the verified available model
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0
    MAX_TOKENS = 10
