from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from config import getAPIKey
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
    campos_faltantes: List[str] = Field(default_factory = list)


def getRetriever():
    
    docs = []
    data_path = Path(__file__).parent / "Documentos_ITT"

    for n in data_path.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"Carregado com sucesso arquivo {n.name}")
        except Exception as e:
            print(f"Erro ao carregar arquivo {n.name}: {e}")

    print(f"Total de documentos carregados: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key = getAPIKey()
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold":0.3, "k": 4}
        )


def getTriagemChain():
    
    llm_triagem = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature = 0,
        api_key = getAPIKey()
    )
    
    return llm_triagem.with_structured_output(TriagemOut)


def getDocumentChain():
    
    llm_triagem = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature = 0,
        api_key = getAPIKey()
    )
    
    prompt_rag = ChatPromptTemplate.from_messages([
        ("system",
        "Você é um ajudante dos profissionais que atuam no ITT (Instituto Tadao Takahashi) "
        "e responde perguntas sobre o estatuto do ITT. "
        "Responda SOMENTE com base no contexto fornecido. "
        "Se a resposta não estiver no contexto, responda apenas 'Não sei'."),
        ("human", "Pergunta: {input}\n\nContexto:\n{context}")
        ])

    return create_stuff_documents_chain(llm_triagem, prompt_rag)



