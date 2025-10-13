from dotenv import load_dotenv            
import os                                   
from pathlib import Path                    
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI           
from pydantic import BaseModel, Field    
from typing import Literal, List, Dict   
from langchain_core.messages import SystemMessage, HumanMessage         
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from .vectorDB import VectorDB


class ITTAgent: 
    def __init__(self):
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=GOOGLE_API_KEY,
        )

    def triagem(self, message : str):

        class TriagemOut(BaseModel):
            decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
            campos_faltantes: List[str] = Field(default_factory=list)

        triagem_chain = self.llm.with_structured_output(TriagemOut)

        with open("prompts.yaml", "r", encoding="utf-8") as f:
            TRIAGEM_PROMPT = yaml.safe_load(f)["prompts"]["triagem"]["content"]

        out : TriagemOut = triagem_chain.invoke([
            SystemMessage(content=TRIAGEM_PROMPT), 
            HumanMessage(content=message)
        ])

        return out.model_dump()
    

    def process_query(self, payload : dict):

        message = payload.get("message", "")

        with open("prompts.yaml", "r", encoding="utf-8") as f:
            RAG_PROMPT = yaml.safe_load(f)["prompts"]["prompt_rag"]

        messages = []

        for msg in RAG_PROMPT:
            messages.append((msg["role"], msg["content"]))

        rag_template = ChatPromptTemplate.from_messages(messages)

        db = VectorDB()
        docs = db.query(message)

        document_chain = create_stuff_documents_chain(self.llm, prompt=rag_template)

        out = document_chain.invoke(
            {"input": message, "context": docs}
        )

        docs_str = [doc.page_content for doc in docs]

        return {
            "completion": out,
            "source_documents": docs_str
        }
