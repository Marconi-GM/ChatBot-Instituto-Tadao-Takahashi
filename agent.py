"""Responsabilidade: Orquestrar o fluxo de trabalho. Este arquivo vai importar 
as chains e o retriever para definir os nós e as arestas do seu grafo de estados
(LangGraph). """
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, START, END
from chains import get_triagem_chain, get_rag_chain, TRIAGEM_PROMPT
from vector_store import get_retriever
from langchain_core.messages import SystemMessage, HumanMessage

# --- Definição do Estado do Agente ---
class AgentState(TypedDict, total=False):
    pergunta: str
    triagem: dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str

# --- Inicialização ---
triagem_chain = get_triagem_chain()
rag_chain = get_rag_chain()
retriever = get_retriever()

# --- Nós ---
def node_triagem(state: AgentState) -> AgentState:
    print(">> Nó: Triagem")
    pergunta = state["pergunta"]
    resposta = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=pergunta)
    ])
    return {"triagem": resposta.model_dump()}

def node_auto_resolver(state: AgentState) -> AgentState:
    print(">> Nó: Auto Resolver (RAG)")
    pergunta = state["pergunta"]
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"resposta": "Não sei.", "citacoes": [], "rag_sucesso": False}

    resposta_llm = rag_chain.invoke({"input": pergunta, "context": docs_relacionados})
    txt = (resposta_llm or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"resposta": "Não sei.", "citacoes": [], "rag_sucesso": False}

    return {"resposta": txt, "citacoes": docs_relacionados, "rag_sucesso": True}

def node_pedir_info(state: AgentState) -> AgentState:
    print(">> Nó: Pedir Informações")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ", ".join(faltantes) if faltantes else "mais detalhes sobre sua dúvida"
    return {"resposta": f"Para te ajudar melhor, por favor, forneça {detalhe}.",
            "citacoes": [], "acao_final": "PEDIR_INFO"}

# --- Condições ---
def decidir_pos_triagem(state: AgentState) -> str:
    print(">> Decisão: Pós-Triagem")
    return state["triagem"]["decisao"].lower()

# --- Compilação ---
def get_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("triagem", node_triagem)
    workflow.add_node("auto_resolver", node_auto_resolver)
    workflow.add_node("pedir_info", node_pedir_info)

    workflow.add_edge(START, "triagem")
    workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
        "auto_resolver": "auto_resolver",
        "pedir_info": "pedir_info"
    })
    workflow.add_edge("auto_resolver", END)
    workflow.add_edge("pedir_info", END)

    return workflow.compile()
