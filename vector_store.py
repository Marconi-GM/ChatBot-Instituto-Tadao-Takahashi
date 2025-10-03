"""Responsabilidade: Gerenciar tudo relacionado aos seus documentos e ao banco de dados de vetores."""
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever

from config import EMBEDDING_MODEL, load_api_key

def get_retriever() -> VectorStoreRetriever:
    """
    Cria e retorna um retriever a partir dos documentos PDF.
    Esta função carrega os PDFs, os divide em chunks, cria os embeddings
    e inicializa um Vector Store FAISS.
    """
    GOOGLE_API_KEY = load_api_key()

    # 1. Carregas os Documentos
    pasta_pdfs = Path(__file__).parent / "Documentos_ITT"
    docs = []
    for pdf_path in pasta_pdfs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(pdf_path))
            docs.extend(loader.load())
            print(f"Carregado: {pdf_path.name}")
        except Exception as e:
            print(f"Erro ao carregar {pdf_path.name}:{e}")

    if not docs:
        raise RuntimeError("Nenhum documento foi carregado. Verifique a pasta de documentos.")

    # 2. Dividir em Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    # 3. Criar embeddings e vector store
    embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Criar o Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
        )
    
    return retriever