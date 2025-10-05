from typing import Literal, List, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from dataProcessing import getRetriever
from rag.formatters import formatar_citacoes
from dataProcessing import TriagemOut, getTriagemChain, getDocumentChain

document_chain = getDocumentChain()
triagem_chain = getTriagemChain()


def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content = TRIAGEM_PROMPT),
        HumanMessage(content = mensagem)
    ])

    return saida.model_dump()

def perguntar_politica_RAG(pergunta: str) -> Dict:
    retriever = getRetriever()
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    answer = document_chain.invoke(
        {"input": pergunta,
        "context": docs_relacionados}
    )

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

    

if __name__ == '__main__':
    
    testes = [
        "Quais são os deveres do ITT?",
        "A diretoria pode se envolver com política?",
        "Quantas capivaras tem no rio São Francisco?"
    ]
    
    for msg_teste in testes:
        resposta = perguntar_politica_RAG(msg_teste)
        print(f"PERGUNTA: {msg_teste}")
        print(f"RESPOSTA: {resposta['answer']}")
        if resposta['contexto_encontrado']:
            print("CITAÇÕES:")
            for c in resposta['citacoes']:
                print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
                print(f"   Trecho: {c['trecho']}")
        print("------------------------------------")