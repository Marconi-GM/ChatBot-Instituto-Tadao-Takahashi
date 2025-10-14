""" Responsabilidade: Centralizar o carregamento de configurações e chaves de API. 
Assim, não há necessidade de se repetir o código de carregar o .env em vários lugares.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

GEMINI_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

def load_api_key() -> str:
    """Carrega a chave da API do Google a partir do arquivo .env."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Verifique .env")
    else:
        print("Chave carregada com sucesso.\n")
    
    return GOOGLE_API_KEY
