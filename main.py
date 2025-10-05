from rag.load_docs import carregar_pdfs
from rag.embeddings import get_retriever
from rag.service import criar_document_chain
from workflow.builder import criar_workflow

def main():
    docs = carregar_pdfs("./pdfs")
    retriever = get_retriever(docs)
    document_chain = criar_document_chain()

    grafo = criar_workflow(retriever, document_chain)

    perguntas = [
        "Quais são os deveres do ITT?",
        "A diretoria pode se envolver com política?",
        "Quantas capivaras tem no rio São Francisco?"
    ]

    for p in perguntas:
        resultado = grafo.invoke({"pergunta": p})
        print(f"\nPERGUNTA: {p}")
        print(f"AÇÃO FINAL: {resultado.get('acao_final')}")
        print(f"RESPOSTA: {resultado.get('resposta')}")
        if resultado.get("citacoes"):
            print("CITAÇÕES:")
            for c in resultado["citacoes"]:
                print(f" - {c['documento']} (pág. {c['pagina']})")
        print("---------------------------------")

if __name__ == "__main__":
    main()
