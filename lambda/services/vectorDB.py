from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
import os

class VectorDB: 

    def __init__(self): 
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

    def query(self, message : str):

        if not os.path.exists("faiss_index"):
            raise ValueError("O índice FAISS não foi encontrado. Certifique-se de que o índice foi criado e salvo corretamente.")
        
        vectorstore = FAISS.load_local("faiss_index", self.embedder, allow_dangerous_deserialization=True)

        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                        search_kwargs={"score_threshold": 0.3, "k":4})
        
        docs = retriever.invoke(message)

        return docs
        
    def create_faiss_index(self, parent_folder : str):

        if not os.path.exists(parent_folder):
            raise ValueError(f"O diretório {parent_folder} não existe. Verifique o caminho e tente novamente.")
        
        parent_folder = Path(__file__).parent / parent_folder
        docs = []
        for file_path in parent_folder.glob("*.pdf"):
            try:
                loader = PyMuPDFLoader(str(file_path))
                docs.extend(loader.load())
                print(f"Carregado com sucesso o arquivo {file_path.name}")

            except Exception as e:
                print(f"Erro ao carregar arquivo {file_path.name}: {e}")

        print(f"Total de documentos carregados: {len(docs)}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        split_docs = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(split_docs, self.embedder)

        vectorstore.save_local("faiss_index")
        