from langgraph.graph import StateGraph, START, END
from .nodes import AgentState, node_triagem, node_auto_resolver, node_pedir_info
from .decisions import decidir_pos_triagem, decidir_pos_auto_resolver

def criar_workflow(retriever, document_chain):
    workflow = StateGraph(AgentState)

    # Nós
    workflow.add_node("triagem", node_triagem)
    workflow.add_node("auto_resolver", lambda s: node_auto_resolver(s, retriever, document_chain))
    workflow.add_node("pedir_info", node_pedir_info)

    # Conexões
    workflow.add_edge(START, "triagem")

    workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
        "auto": "auto_resolver",
        "info": "pedir_info"
    })

    workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
        "ok": END,
        "info": "pedir_info"
    })

    workflow.add_edge("pedir_info", END)

    return workflow.compile()
