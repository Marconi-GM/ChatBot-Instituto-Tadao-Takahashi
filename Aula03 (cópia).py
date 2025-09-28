from dotenv import load_dotenv              # ler um arquivo .env e colocar as variáveis definidas nele dentro do processo Python
                                            # isso permite que você use .getenv para acessar segredos, sem colocar a chave diretamente no código
import os                                   # interface para o python interagir com o sistema operacional (aqui é usado para o os.getenv(nome_variavel))
from pathlib import Path                    # módulo interno do python para manipular caminhos de arquivos de forma portável
from langchain_google_genai import ChatGoogleGenerativeAI               # Classe principal da biblioteca lanchaingooglegenai. Ela permite realizar a conexão com a API do Gemini e permite que seu código python envie prompts e receba respostas do modelo
from pydantic import BaseModel, Field       # Pydantic é uma biblioteca para validação de dados. Aqui ela é usada para definir a estrutura de saída que você espera do Gemini. Ao criar a classe TriagemOut você está dizendo ao LangChain que a respota do LLM deve ser um JSON com os campos decisao, urgencia, etc...
from typing import Literal, List, Dict      # São ferramentas de tipagem do Python. Literal é especialmente útil aqui para restringir os valores possíveis de um campo (ex: decisao só pode ser "AUTO_RESOLVER", "PEDIR_INFO" ou "ABRIR_CHAMADO").
from langchain_core.messages import SystemMessage, HumanMessage         # SystemMessage e HumanMessage são classes que ajudam a estruturar a conversa com o LLM, diferenciando as instruções do sistema (o que você quer que o AI faça) da entrada do usuário.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def main() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Verifique .env")
    else:
        print("Chave carregada com sucesso.\n")



        """
            TRIAGEM_PROMTP é uma string que contém o conjunto de instruções detalhadas para a LLM.
            Sua função é programar o comportamento do LLM. Ele age como um manual de instruções ou um roteiro
            que diz o Gemini exatamente o que fazer.
            Ele:
            Define a Persona: "Você é um triador de Service Desk...". Isso coloca o LLM no contexto correto. Ele não é mais um assistente genérico, ele tem um trabalho específico.

            Define a Tarefa: "...retorne SOMENTE um JSON com:...". Esta é a instrução mais importante. Ele não deve conversar, não deve ser simpático, ele deve executar uma tarefa específica: gerar um objeto JSON.

            Define as Regras do Jogo: As seções que começam com "- **AUTO_RESOLVER**...", "- **PEDIR_INFO**...", etc., são a lógica de negócio. O prompt está ensinando ao LLM os critérios que ele deve usar para tomar uma decisão e preencher o campo "decisao" do JSON.
        """
        TRIAGEM_PROMPT = (
            "Você é um ajudante dos profissionais que atuam no ITT (Instituto Tadao Takahashi)"
            "que fornece informações sobre o estatuto do ITT e auxilia com dúvidas gerais. "
            "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
            "{\n"
            '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO",\n'
            '  "campos_faltantes": ["..."]\n'
            "}\n"
            "Regras:\n"
            '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
            '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
            "Analise a mensagem e decida a ação mais apropriada."
        )

        """
            É uma classe python que usa a biblioteca pydantic para definir um esquema de dados, ou seja, um molde para os dados.

            Sua função é ser a garantia de que as respostas do LLM virão no formato que esperamos.
            
            Ela tem duas funções principais:
            Definir a Estrutura: A classe diz que qualquer dado válido de "Triagem" precisa ter três campos: decisao, urgencia e campos_faltantes.

            Validar os Dados: Ela impõe regras estritas sobre esses campos. Usando Literal, ela garante que decisao só pode conter um dos três valores permitidos ("AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"), e o mesmo para urgencia. Isso evita que o LLM "invente" um status novo, como "TALVEZ_RESOLVER". Garante também que campos_faltantes seja sempre uma lista de strings.
        """
        class TriagemOut(BaseModel):
            decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO"]
            campos_faltantes: List[str] = Field(default_factory=list)

        llm_triagem = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=GOOGLE_API_KEY
            )

        """
        Como Eles se Linkam e Trabalham Juntos

A mágica acontece na linha:
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

Aqui está a conexão:

    Nós enviamos o TRIAGEM_PROMPT (o "pedido") para o LLM. O LLM lê as instruções e a mensagem do usuário e pensa: "Ok, com base nas regras, a decisão aqui é AUTO_RESOLVER e a urgência é BAIXA".

    O método .with_structured_output(TriagemOut) intercepta a resposta do LLM. Ele usa a classe TriagemOut (a "garantia") como um molde.

    Ele força a resposta do LLM a se encaixar perfeitamente nesse molde, gerando um JSON limpo e validado.

Analogia Final:

    TRIAGEM_PROMPT é a receita do bolo. Diz os ingredientes, as quantidades e os passos para o cozinheiro (LLM).

    class TriagemOut é a forma do bolo. Não importa o quão criativo o cozinheiro seja, o resultado final tem que ter o formato exato daquela forma.

Juntos, eles transformam o LLM de um gerador de texto imprevisível em um componente de software confiável que retorna dados estruturados e previsíveis, que podem ser usados com segurança pelo resto do seu programa (no caso, o LangGraph).        
        """
        triagem_chain = llm_triagem.with_structured_output(TriagemOut)

        def triagem(mensagem: str) -> Dict:

            """
                O processo acontece assim:

    Preparação (a linha ...with_structured_output...):

        Antes de começar o trabalho, você cria uma regra permanente para o seu Especialista. Você diz a ele: "Olha, a partir de agora, toda vez que eu te pedir uma análise, eu não quero um texto como resposta. Eu quero que você me entregue apenas este Formulário Padronizado (TriagemOut) perfeitamente preenchido".

        É isso que o .with_structured_output(TriagemOut) faz. Ele cria uma versão "especializada" do LLM (triagem_chain) que está programada para sempre responder no formato da classe TriagemOut. É aqui que o TriagemOut é ligado ao processo.

    Execução (a função triagem e o .invoke):

        Um novo pedido chega (mensagem).

        O Gerente (.invoke) entra em ação. Ele pega as Instruções (TRIAGEM_PROMPT) e o pedido do cliente (mensagem).

        Ele vai até o Especialista e diz: "Use estas Instruções para analisar este pedido e, como combinamos, me entregue o Formulário Padronizado preenchido".

        É neste momento que o TRIAGEM_PROMPT é ligado ao processo.

O LangChain, por baixo dos panos, combina tudo isso. Ele envia para a API do Gemini:

    A mensagem do usuário.

    O prompt do sistema (TRIAGEM_PROMPT).

    E instruções adicionais (geradas a partir da classe TriagemOut) que forçam o modelo a gerar sua resposta em um formato JSON que corresponda exatamente ao "formulário".

Em resumo:

A ligação não é você juntando os dois no seu código. A ligação acontece porque:

    Você cria uma chain que já sabe qual é o formato de saída obrigatório (TriagemOut).

    Depois, você envia para essa chain as instruções de como pensar (TRIAGEM_PROMPT).

O TRIAGEM_PROMPT guia o raciocínio do LLM, enquanto o TriagemOut (através do .with_structured_output) força a formatação da resposta final daquele raciocínio.

            """
            saida: TriagemOut = triagem_chain.invoke([SystemMessage(content=TRIAGEM_PROMPT),
                                                     HumanMessage(content=mensagem)])

            return saida.model_dump()

        testes = ["Como deve ser destinado o patrimônio do ITT?",
                  "Como deve ser tratada as dispesas do ITT?",
                  "Quais são os direitos dos associados titulares do ITT?",
                  "O Ary pode promover engajamento político ou apoiar candidaturas?",
                  "Quantas capivaras tem no rio São francisco?"
                  ]
        
        # for msg in testes:
        #     print(f"Pergunta: {msg}\n -> Resposta: {triagem(msg)}\n")

        pasta_pdfs = Path(__file__).parent / "PDFS_Aula_2"
        docs = list()
        for i in pasta_pdfs.glob("*.pdf"):
            try:
                loader = PyMuPDFLoader(str(i))
                docs.extend(loader.load())
                print(f"Carregado com sucesso o arquivo {i.name}")

            except Exception as e:
                print(f"Erro ao carregar arquivo {i.name}: {e}")

        print(f"Total de documentos carregados: {len(docs)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

        chunks = splitter.split_documents(docs)
        chunks = chunks[0:50]
        """
            Podemos usar pandas para splittar os documentos em parágrafos completos
            para não ter esse escape de overlap
            PESQUISAR DEPOIS
        """

        """
        
        Ótima questão. O conceito de "embedding" é a peça central que faz a busca por similaridade (e todo o RAG) funcionar.

Vamos dividir a resposta em duas partes: a definição conceitual e a aplicação prática no seu código.

1. O que é um Embedding? (A Definição)

Um embedding é uma representação numérica do significado de um texto. É uma forma de traduzir palavras, frases ou parágrafos, que os computadores não entendem, em uma lista de números (um vetor), que os computadores são ótimos em processar.

Pense nisso como um "GPS de Significados":

    Tradução para Coordenadas: Um modelo de embedding (como o models/gemini-embedding-001 do seu código) é treinado com uma quantidade massiva de texto da internet. Ele aprende as relações sutis entre as palavras. Quando você dá um texto a ele, ele o converte em um conjunto de coordenadas nesse "mapa de significados".

    Proximidade é Similaridade: A parte mais importante é que, neste mapa, textos com significados parecidos terão coordenadas muito próximas. Textos com significados diferentes terão coordenadas distantes.

        A frase "Qual o salário de um rei?" e "Quanto ganha um monarca?" estarão praticamente no mesmo ponto do mapa.

        Enquanto a frase "Qual o salário de um rei?" e "Qual a receita de bolo de chocolate?" estarão em continentes diferentes no mapa.

    São Apenas Números: No final, um embedding é apenas uma lista de números, por exemplo [0.01, -0.45, 0.89, ..., 0.12]. O que importa não são os números individuais, mas a posição que eles representam no "mapa" e sua distância em relação a outros pontos.

Em resumo: Embedding é a conversão de um texto em um vetor numérico onde a distância entre vetores representa a similaridade de significado entre os textos originais.
        
        """


        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",
                                                  google_api_key=GOOGLE_API_KEY
                                                  )
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", 
                                             search_kwargs={"score_threshold": 0.3, "k":4})
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain

        prompt_rag = ChatPromptTemplate.from_messages([
            ("system",
            "Você é um ajudante dos profissionais que atuam no ITT (Instituto Tadao Takahashi)"
            "que fornece informações sobre o estatuto do ITT e auxilia com dúvidas gerais. "
            "Responda SOMENTE com base no contexto fornecido. "
            "Se não houver base suficiente, responda apenas 'Não sei'."),

            ("human", "Pergunta: {input}\n\nContexto:\n{context}")
        ])

        document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)
        
        def perguntar_politica_RAG(pergunta: str) -> Dict:
            docs_relacionados = retriever.invoke(pergunta)
            if not docs_relacionados:
                return {"answer": "Não sei.",
                        "citacoes": "",
                        "contexto_encontrado": False}
            
            answer = document_chain.invoke({"input": pergunta,
                                            "context": docs_relacionados})
            
            txt = (answer or "").strip()

            if txt.rstrip(".!?") == "Não sei":
                return {"answer": "Não sei.",
                        "citacoes": "",
                        "contexto_encontrado": False}
            
            return {"answer": txt,
                    "citacoes": docs_relacionados,
                    "contexto_encontrado": True}
        
        testes = ["Como deve ser destinado o patrimônio do ITT?",
                  "Como deve ser tratada as dispesas do ITT?",
                  "Quais são os direitos dos associados titulares do ITT?",
                  "O Ary voluntario do ITT, pode promover engajamento político ou apoiar candidaturas?",
                  "Quantas capivaras tem no rio São francisco?"
                  ]

        for msg_teste in testes:
            resposta = perguntar_politica_RAG(msg_teste)
            print(f"PERGUNTA: {msg_teste}")
            print(f"RESPOSTA: {resposta['answer']}")
            if resposta['contexto_encontrado']:
                print("CITAÇÕES:")
                for c in resposta['citacoes']:
                    # metadados podem conter informações úteis como o caminho do arquivo e número da página
                    doc_nome = c.metadata.get("source", "Documento desconhecido")
                    pagina = c.metadata.get("page", "N/A")
                    print(f" - Documento: {doc_nome}, Página: {pagina}")
                    print(f"   Trecho: {c.page_content[:200]}...")  # imprime só os primeiros 200 caracteres para não ficar gigante
                print("------------------------------------")


        # from typing import TypedDict, Optional

        # class AgentState(TypedDict, total = False):
        #     pergunta: str
        #     triagem: dict
        #     resposta: Optional[str]
        #     citacoes: List[dict]
        #     rag_sucesso: bool
        #     acao_final: str

        # def node_triagem(state: AgentState) -> AgentState:
        #         print("Executando nó de triagem...")
        #         return {"triagem": triagem(state["pergunta"])}


        # def node_auto_resolver(state: AgentState) -> AgentState:
        #     print("Executando nó de auto_resolver...")
        #     resposta_rag = perguntar_politica_RAG(state["pergunta"])

        #     update: AgentState = {
        #         "resposta": resposta_rag["answer"],
        #         "citacoes": resposta_rag.get("citacoes", []),
        #         "rag_sucesso": resposta_rag["contexto_encontrado"],
        #     }

        #     if resposta_rag["contexto_encontrado"]:
        #         update["acao_final"] = "AUTO_RESOLVER"

        #     return update


        # def node_pedir_info(state: AgentState) -> AgentState:
        #     print("Executando nó de pedir_info...")
        #     faltantes = state["triagem"].get("campos_faltantes", [])
        #     if faltantes:
        #         detalhe = ",".join(faltantes)
        #     else:
        #         detalhe = "Tema e contexto específico"

        #     return {
        #         "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        #         "citacoes": [],
        #         "acao_final": "PEDIR_INFO"
        #     }

        # def node_abrir_chamado(state: AgentState) -> AgentState:
        #     print("Executando nó de abrir_chamado...")
        #     triagem = state["triagem"]

        #     return {
        #         "resposta": f"Abrindo chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
        #         "citacoes": [],
        #         "acao_final": "ABRIR_CHAMADO"
        #     }

        # KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

        # def decidir_pos_triagem(state: AgentState) -> str:
        #     print("Decidindo após a triagem...")
        #     decisao = state["triagem"]["decisao"]

        #     if decisao == "AUTO_RESOLVER": return "auto"
        #     if decisao == "PEDIR_INFO": return "info"
        #     if decisao == "ABRIR_CHAMADO": return "chamado"


        # def decidir_pos_auto_resolver(state: AgentState) -> str:
        #     print("Decidindo após o auto_resolver...")

        #     if state.get("rag_sucesso"):
        #         print("Rag com sucesso, finalizando o fluxo.")
        #         return "ok"

        #     state_da_pergunta = (state["pergunta"] or "").lower()

        #     if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        #         print("Rag falhou, mas foram encontradas keywords de abertura de ticket. Abrindo...")
        #         return "chamado"

        #     print("Rag falhou, sem keywords, vou pedir mais informações...")
        #     return "info"


        # from langgraph.graph import StateGraph, START, END

        # workflow = StateGraph(AgentState)

        # workflow.add_node("triagem", node_triagem)
        # workflow.add_node("auto_resolver", node_auto_resolver)
        # workflow.add_node("pedir_info", node_pedir_info)
        # workflow.add_node("abrir_chamado", node_abrir_chamado)

        # workflow.add_edge(START, "triagem")
        # workflow.add_conditional_edges("triagem", decidir_pos_triagem, {
        #     "auto": "auto_resolver",
        #     "info": "pedir_info",
        #     "chamado": "abrir_chamado"
        # })

        # workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
        #     "info": "pedir_info",
        #     "chamado": "abrir_chamado",
        #     "ok": END
        # })

        # workflow.add_edge("pedir_info", END)
        # workflow.add_edge("abrir_chamado", END)

        # grafo = workflow.compile()

        # from IPython.display import display, Image

        # graph_bytes = grafo.get_graph().draw_mermaid_png()
        # display(Image(graph_bytes))

        # testes = ["Posso reembolsar a internet?",
        #         "Quero mais 5 dias de trabalho remoto. Como faço?",
        #         "Posso reembolsar cursos ou treinamentos da Alura?",
        #         "É possível reembolsar certificações do Google Cloud?",
        #         "Posso obter o Google Gemini de graça?",
        #         "Qual é a palavra-chave da aula de hoje?",
        #         "Quantas capivaras tem no Rio Pinheiros?"]

        # for msg_test in testes:
        #         resposta_final = grafo.invoke({"pergunta": msg_test})

        #         triag = resposta_final.get("triagem", {})
        #         print(f"PERGUNTA: {msg_test}")
        #         print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")                    print(f"RESPOSTA: {resposta_final.get('resposta')}")
        #         if resposta_final.get("citacoes"):
        #                 print("CITAÇÕES:")
        #                 for citacao in resposta_final.get("citacoes"):
        #                     print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
        #                     print(f"   Trecho: {citacao['trecho']}")
        #                     print("------------------------------------")



if __name__ == "__main__":
     main()
     #LANGCHAIN
     #CHUNKS
     #LANGGRAPH