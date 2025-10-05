import streamlit as st
from rag.load_docs import carregar_pdfs
from rag.embeddings import get_retriever
from rag.service import criar_document_chain
from workflow.builder import criar_workflow 

# Configuração da página
st.set_page_config(
    page_title="Assistente LLM - Políticas Internas",
    page_icon="🤖",
    layout="centered"
)

# Inicialização do grafo (com cache para performance)
@st.cache_resource
def inicializar_grafo():
    docs = carregar_pdfs("./pdfs")
    retriever = get_retriever(docs)
    document_chain = criar_document_chain()
    return criar_workflow(retriever, document_chain)

grafo = inicializar_grafo()

# Título
st.title("💬 Assistente de Políticas Internas")
st.caption("Pergunte sobre políticas, regras ou procedimentos do ITT.")

# Inicializa histórico na sessão
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

# Exibe histórico do chat
for msg in st.session_state.mensagens:
    with st.chat_message(msg["autor"]):
        st.markdown(msg["texto"])

# Campo de entrada
pergunta = st.chat_input("Digite sua pergunta...")

# Quando o usuário envia
if pergunta:
    # Adiciona a pergunta ao histórico
    st.session_state.mensagens.append({"autor": "user", "texto": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Processa a pergunta no workflow
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            resultado = grafo.invoke({"pergunta": pergunta})
            resposta = resultado.get("resposta", "Desculpe, não consegui gerar uma resposta.")
            citacoes = resultado.get("citacoes", [])
            
            st.markdown(resposta)
            if citacoes:
                with st.expander("📄 Citações usadas"):
                    for c in citacoes:
                        st.markdown(f"- **{c['documento']}**, pág. {c['pagina']}")

    # Adiciona a resposta ao histórico
    st.session_state.mensagens.append({"autor": "assistant", "texto": resposta})
