from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def carregar_pdfs(diretorio: str) -> list:
    docs = []
    data_path = Path(__file__).parent.parent / "Documentos_ITT"
    
    for n in data_path.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(n))
            docs.extend(loader.load())
            print(f"Carregado: {n.name}")
        except Exception as e:
            print(f"Erro ao carregar {n.name}: {e}")
    return docs
