import streamlit as st
from dotenv import load_dotenv
import requests 
import json

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="ChatBot Instituto Tadao Takahashi",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main chat interface
st.title("🤖 ChatBot Instituto Tadao Takahashi")
st.markdown("*Tire suas dúvidas sobre o Instituto Tadao Takahashi*")
st.markdown("---")

# Chat container
chat_container = st.container(height=500)

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Digite sua pergunta sobre o Instituto Tadao Takahashi..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)
    
    # Generate and display assistant response
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            try:
                with st.spinner("Processando sua pergunta..."):
                    # Call the ITT API
                    response = requests.post(
                        "http://localhost:8000/query_response",
                        json={"message": prompt},
                        headers={"Content-Type": "application/json"}
                    )

                    response.raise_for_status()  
                    response_data = response.json()
                    
                    # Extract response and sources
                    bot_response = response_data["response"]["response"]
                    source_documents = response_data["response"]["source_documents"]

                # Display the main response
                message_placeholder.markdown(bot_response)
                
                # Display sources if available
                if source_documents:
                    with sources_placeholder.container():
                        st.markdown("📚 Fontes consultadas:")
                        for i, doc in enumerate(source_documents, 1):
                            st.markdown(f"<{doc[:200]}...>")
               
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": bot_response,
                    "sources": [
                        {
                            "page_content": doc,
                        } for doc in source_documents
                    ]
                })
                
            except requests.exceptions.RequestException as e:
                error_message = f"❌ Erro de conexão com a API: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })
            except json.JSONDecodeError as e:
                error_message = f"❌ Erro ao processar resposta da API: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })
            except Exception as e:
                error_message = f"❌ Erro inesperado: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })

# Sidebar with information
with st.sidebar:
    st.markdown("## ℹ️ Sobre")
    st.markdown("""
    Este chatbot foi desenvolvido para responder perguntas sobre o **Instituto Tadao Takahashi**.
    
    **Como usar:**
    1. Digite sua pergunta na caixa de texto
    2. Aguarde o processamento
    3. Veja a resposta e as fontes consultadas
    
    **Exemplos de perguntas:**
    - Quais são os principais objetivos do Instituto?
    - Que tipo de pesquisas são realizadas?
    - Como posso me envolver com o Instituto?
    """)
    
    st.markdown("---")
    st.markdown("*Powered by Google Gemini & LangChain*")
