from agent import get_graph

def main():
    """Função principal que executa o chatbot."""
    # Compila o grafo do agente
    chatbot = get_graph()
    
    print("Chatbot do ITT iniciado! Digite 'sair' para terminar.")
    
    # Loop de conversação
    while True:
        pergunta = input("Você: ")
        if pergunta.lower() == 'sair':
            break
        
        # Invoca o agente com a pergunta do usuário
        resposta_final = chatbot.invoke({"pergunta": pergunta})
        
        # Imprime a resposta para o usuário
        print(f"Chatbot: {resposta_final.get('resposta')}")
        
        # Opcional: Imprimir citações se houver
        if resposta_final.get("citacoes"):
            print("\nFontes encontradas:")
            for doc in resposta_final["citacoes"]:
                source = doc.metadata.get("source", "N/D")
                page = doc.metadata.get("page", "N/D")
                print(f"  - Documento: {source}, Página: {page}")
            print("-" * 20)

if __name__ == "__main__":
    main()