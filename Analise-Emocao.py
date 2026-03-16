from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage

llm = OllamaLLM(model="gpt-oss:20b", num_predict=1024, temperature=0)

def analisar_sentimento(mensagem:str) -> str:
    prompt = (
        f"Classifique o sentimento baseado na mensagem, entre POSTIVO, NEGATIVO ou NEUTRO."
        f"Comentario: {mensagem}"
    )

    resposta = llm.invoke([HumanMessage(content=prompt)])

    return resposta.strip().upper()

while True:
    try:
        entrada = input("Digite uma mensagem para análise de sentimento (ou 'sair' para encerrar): ")
        if entrada.lower() == 'sair':
            print("Encerrando a análise de sentimento. Até a próxima!")
            break

        resultado = analisar_sentimento(entrada)
        print(f"Sentimento classificado: {resultado}\n")

    except Exception as e:
        print(f"Ocorreu um erro: {e}. Tente novamente.")