import streamlit as st
from ChatBOT_ITT.agent import get_graph

# --- Configuração da Página ---
st.set_page_config(
    page_title="ChatBOT do ITT",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    /* Centraliza o título principal da aplicação */
    h1 {
        text-align: center;
    }

    /* Estiliza a caixa de texto do chat */
    .stChatInputContainer {
        max-width: 80%; /* Define uma largura máxima */
        margin: auto;   /* Centraliza o container */
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 ChatBOT do Instituto Tadao Takahashi")
st.caption("Faça perguntas sobre o estatuto e procedimentos do ITT.")

@st.cache_resource
def load_chatbot():
    """Carrega e compila o grafo do agente uma única vez."""
    print("\tInicializando o ChatBOT, por favor aguarde...")
    return get_graph()


chatbot = load_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "Assistente", "content": "Olá! Como posso ajudar com informações sobre o ITT hoje?"}]

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Digite sua pergunta aqui..."):
    # a) Adiciona a mensagem do usuário ao histórico.
    st.session_state.messages.append({"role": "user", "content": prompt})
    # b) Exibe a mensagem do usuário na tela instantaneamente.
    with st.chat_message("user"):
        st.markdown(prompt)

    # c) Prepara para mostrar a resposta do assistente.
    with st.chat_message("assistant"):
        # st.spinner cria uma animação de "carregando" para dar feedback
        # visual de que o bot está processando a informação.
        with st.spinner("Analisando sua pergunta..."):
            # d) AQUI ACONTECE A MÁGICA: Invoca seu agente LangGraph com a pergunta.
            resposta_final = chatbot.invoke({"pergunta": prompt})
            
            # e) Extrai as informações da resposta do seu agente.
            resposta_texto = resposta_final.get("resposta", "Desculpe, ocorreu um erro.")
            citacoes = resposta_final.get("citacoes")

            # f) Exibe a resposta principal na tela.
            st.markdown(resposta_texto)
            
            # g) Se houver citações, as exibe de forma organizada em um "expander".
            if citacoes:
                with st.expander("Fontes Consultadas"):
                    for doc in citacoes:
                        # Limpa o nome do arquivo para melhor visualização.
                        source = doc.metadata.get("source", "N/D").split("/")[-1]
                        page = doc.metadata.get("page", "N/D")
                        
                        st.markdown(f"**Documento:** `{source}` (página {page + 1})")
                        st.markdown(f"> _{doc.page_content}_")

    # h) Adiciona a resposta do bot ao histórico para que ela seja exibida
    # nas próximas interações.
    st.session_state.messages.append({"role": "assistant", "content": resposta_texto})