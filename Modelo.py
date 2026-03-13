from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import numpy as np 
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage


#definindo o LLM que vamos usar (instalado localmente)
llm = OllamaLLM(model="gpt-oss:20b", num_predict=1024) # <- 1024, numero de tokens a serem gerados para resposta

#definir o schema de estado, mantendo historico de mensagens
class State(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]

#funcao de entrada.
def entrada_usuario(state:State) -> State:
    pergunta = input("Digite sua pergunta: ")
    if not pergunta.strip():
        raise ValueError("A pergunta não pode ser vazia.")
    pergunta_msg = HumanMessage(content=pergunta)
    return {"messages": [pergunta_msg]}

def processamento_resposta(state:State) -> State:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, HumanMessage):
        raise ValueError("A última mensagem deve ser do tipo HumanMessage.")
    pergunta = last_msg.content
    resposta = llm.invoke(pergunta)
    resposta_msg = AIMessage(f"A resposta para {pergunta} é: {resposta}")
    return {"messages": [resposta_msg]}

def saida_resposta(state:State) -> State:
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage):
        raise ValueError("A última mensagem deve ser do tipo AIMessage.")
    print(last_msg.content)
    return{}#retornamos vazio pois não precisamos manter o estado após a resposta

grafo = StateGraph(State)
grafo.add_node("entrada", entrada_usuario)
grafo.add_node("processamento", processamento_resposta)
grafo.add_node("saida", saida_resposta)

#denifinindo arestas dos fluxos
grafo.add_edge(START, "entrada")
grafo.add_edge("entrada", "processamento")
grafo.add_edge("processamento", "saida")
grafo.add_edge("saida", END)

compiled = grafo.compile()

initial_state: State = {"messages": []}

while True:
    try:
        result_state = compiled.invoke(initial_state)
        initial_state = result_state
    
    except Exception as e:
        print(f"Erro: {e}")
        break

    continuar = input("Deseja fazer outra pergunta? (s/n): ").strip().lower()
    if continuar != 's':
        print("Encerrando o programa.")
        break

print("Programa finalizado.")
for msg in initial_state["messages"]:
    role = "Usuário" if isinstance(msg, HumanMessage) else "Assistente"
    print(f"[{role}]: {msg.content}")