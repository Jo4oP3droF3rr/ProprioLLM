from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import numpy as np 
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

llm = OllamaLLM(model="gpt-oss:20b", num_predict=1024, temperature=0) 

class State(TypedDict):
    messages: Annotated[list[HumanMessage | AIMessage], add_messages]

def entrada_usuario(state:State) -> State:
    pergunta = input("Usuario: ")
    if isinstance(pergunta, str) and pergunta.strip() != "":
        return {"messages": [HumanMessage(content=pergunta)]}
    else:
        raise ValueError("Entrada inválida. Por favor, insira uma pergunta válida.")

def processamento_usuario(state:State) -> State:
    ultima_mensagem = state["messages"][-1]
    if isinstance(ultima_mensagem, HumanMessage):
        try:
            resposta = llm.invoke("Resposta gerada:" + ultima_mensagem.content)
            return {"messages": [AIMessage(content=resposta)]}
        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return {"messages": [AIMessage(content="Desculpe, ocorreu um erro ao processar sua pergunta. Tente novamente.")]}
    else:
        raise ValueError("Última mensagem não é do tipo HumanMessage. Verifique o estado atual.")
    
def saida_usuario(state:State) -> State:
    ultima_mensagem = state["messages"][-1]
    if hasattr(ultima_mensagem, 'content'):
        print(f"Agente de Atendimento: {ultima_mensagem.content}")
    else:
        print(ultima_mensagem)
    return state

grafo = StateGraph(State)
grafo.add_node ("entrada", entrada_usuario)
grafo.add_node ("processamento", processamento_usuario)
grafo.add_node ("saida", saida_usuario)

grafo.add_edge(START, "entrada")
grafo.add_edge("entrada", "processamento")
grafo.add_edge("processamento", "saida")
grafo.add_edge("saida", END)

compiled = grafo.compile()

print("Bem-vindo ao Agente de Atendimento! Digite sua pergunta abaixo:")

while True:
    try:
        compiled.invoke({"messages": []})
        continuar = input("Deseja fazer outra pergunta? (s/n): ")
        if continuar.lower() != 's':
            print("Obrigado por usar o Agente de Atendimento. Até a próxima!")
            break
    except Exception as e:  
        print(f"Ocorreu um erro: {e}. Tente novamente.")
    except KeyboardInterrupt:
        print("\nInterrupção detectada. Encerrando o Agente de Atendimento. Até a próxima!")
        break
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}. Tente novamente.")
    print("Agente de Atendimento encerrado.")


