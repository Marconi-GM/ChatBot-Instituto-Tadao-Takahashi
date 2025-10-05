import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"

def get_llm(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=temperature,
        api_key=GOOGLE_API_KEY
    )
