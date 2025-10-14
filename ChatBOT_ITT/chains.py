"""Responsabilidade: Definir e configurar as "ferramentas" lógicas do LangChain.
 A chain de triagem e a chain de RAG ficam aqui.
"""
from typing import Literal, List, Dict
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from .config import GEMINI_MODEL, load_api_key

# --- Configuração da Chain de Triagem ---
TRIAGEM_PROMPT = (
    "Você é um classificador de perguntas para um assistente de IA do Instituto Tadao Takahashi (ITT). "
    "Sua função é decidir se uma pergunta pode ser respondida buscando informações no estatuto do ITT ou se ela é incompleta. "
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
        temperature=0.7,
        google_api_key=load_api_key()
    )
    return llm.with_structured_output(TriagemOut)

# # --- Configuração da Chain de RAG ---
# RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
#     ("system",
#      "Você é um assistente especialista no estatuto do Instituto Tadao Takahashi (ITT). "
#      "Sua tarefa é responder às perguntas do usuário de forma clara, completa e prestativa, baseando-se estritamente no contexto fornecido.\n\n"
#      "Regras Importantes:\n"
#      "1. Sintetize a Informação: Se o contexto contiver informações de diferentes seções do documento, combine-as para formar uma resposta coesa e completa.\n"
#      "2. Resposta Direta: Responda diretamente à pergunta do usuário.\n"
#      "3. Formatação: Use listas (bullet points) se for apropriado para organizar a informação e facilitar a leitura.\n"
#      "4. Fidelidade ao Contexto: NÃO adicione nenhuma informação que não esteja explicitamente no contexto fornecido.\n"
#      "5. Resposta Negativa: Se a resposta para a pergunta não puder ser encontrada no contexto, responda de forma clara e educada: 'Com base nos documentos que tenho acesso, não encontrei a resposta para a sua pergunta.'"),
#     ("human", "Pergunta do Usuário: {input}\n\nContexto Relevante dos Documentos:\n{context}")
# ])

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente especialista sobre o Instituto Tadao Takahashi (ITT). "
     "Sua missão é responder às perguntas do usuário de forma clara, completa e útil. "
     "Use o estatuto do ITT, fornecido no 'Contexto', como sua principal fonte de verdade, seguindo estas regras rigorosamente:\n\n"
     
     "--- PRINCÍPIO DE RACIOCÍNIO ---\n\n"
     
     "1. **Analise o Tipo de Pergunta:**\n"
     "   - **a) Perguntas Específicas:** Se a pergunta for sobre regras, políticas, finanças, procedimentos ou detalhes operacionais (geralmente com 'como', 'qual', 'pode', 'deve'), você deve se basear **ESTRITAMENTE** no contexto fornecido.\n"
     "   - **b) Perguntas Gerais:** Se a pergunta for introdutória ('o que é o ITT?', 'qual o objetivo do instituto?'), o contexto pode ser insuficiente. Neste caso, **VOCÊ PODE** usar seu conhecimento geral para dar uma resposta informativa, mas sempre priorize e integre qualquer informação relevante que encontrar no contexto.\n\n"
     
     "--- REGRAS DE EXECUÇÃO DA RESPOSTA ---\n\n"
     
     "2. **Sintetize a Informação:** Combine informações de diferentes partes do contexto para construir uma resposta coesa e completa. Não se limite a extrair trechos isolados.\n\n"
     
     "3. **Seja Direto e Formate Bem:** Responda diretamente à pergunta do usuário. Use listas (bullet points) para organizar informações complexas e facilitar a leitura.\n\n"
     
     "4. **Quando não souber:** Se a resposta para uma pergunta específica (tipo 1a) não puder ser encontrada no contexto, responda de forma educada: 'Com base no estatuto do ITT que tenho acesso, não encontrei uma resposta para sua pergunta.'"),
    
    ("human", "Pergunta do Usuário: {input}\n\nContexto do Estatuto do ITT:\n{context}")
])

def get_rag_chain():
    """Retorna uma chain configurada para responder perguntas com base em contexto (RAG)."""
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=0.7,
        google_api_key=load_api_key()
    )
    return create_stuff_documents_chain(llm, RAG_PROMPT_TEMPLATE)