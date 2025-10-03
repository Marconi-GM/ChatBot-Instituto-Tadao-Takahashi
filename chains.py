"""Responsabilidade: Definir e configurar as "ferramentas" lógicas do LangChain.
 A chain de triagem e a chain de RAG ficam aqui.
"""
from typing import Literal, List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from config import GEMINI_MODEL, load_api_key

# --- Configuração da Chain de Triagem ---
TRIAGEM_PROMPT = (
    "Você é um ajudante dos profissionais que atuam no ITT (Instituto Tadao Takahashi)"
    "que fornece informações sobre o estatuto do ITT e auxilia com dúvidas gerais. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas.\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações.\n'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(default_factory=list)

def get_triagem_chain():
    """Retorna uma chain configurada para a triagem inicial."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        google_api_key=load_api_key()
    )
    return llm.with_structured_output(TriagemOut)

# --- Configuração da Chain de RAG ---
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um ajudante dos profissionais que atuam no ITT (Instituto Tadao Takahashi) "
     "e responde perguntas sobre o estatuto do ITT. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se a resposta não estiver no contexto, responda apenas 'Não sei'."),
    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

def get_rag_chain():
    """Retorna uma chain configurada para responder perguntas com base em contexto (RAG)."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.3,
        google_api_key=load_api_key()
    )
    return create_stuff_documents_chain(llm, RAG_PROMPT_TEMPLATE)