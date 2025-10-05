from langchain_core.messages import SystemMessage, HumanMessage
from .prompt import TRIAGEM_PROMPT
from .model import TriagemOut
from config import get_llm

llm_triagem = get_llm()
triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()
