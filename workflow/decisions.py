from .nodes import AgentState

def decidir_pos_triagem(state: AgentState):
    """Decide o próximo passo após a triagem."""
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER":
        return "auto"
    if decisao == "PEDIR_INFO":
        return "info"

    return "info"


def decidir_pos_auto_resolver(state: AgentState):
    """Decide o que fazer após tentar o RAG."""
    if state.get("rag_sucesso"):
        print("RAG com sucesso - finalizando o fluxo.")
        return "ok"

    print("RAG não encontrou contexto - pedindo mais informações.")
    return "info"
