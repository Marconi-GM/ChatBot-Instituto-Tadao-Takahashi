from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Lê a chave da API
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Teste para ver se o seu ambiente foi configurado corretamente
llm = ChatGoogleGenerativeAI(
    model="gemma-3-27b-it",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

response_test = llm.invoke("O corinthians joga essa semana?")

print(response_test.content)