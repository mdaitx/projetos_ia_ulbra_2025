from langchain_ollama.llms import OllamaLLM  # import direto
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

llm = OllamaLLM(model="llama3:8b")        # ou "llama3:8b", "llama3:70b"

prompt = ChatPromptTemplate.from_template(
    """Você é um especialista em responder perguntas sobre jogos.

    Informações relevantes sobre os jogos:
    {reviews}

    Pergunta:
    {question}
    """
)

chain = prompt | llm

while True:
    print("\n-------------------------------")
    question = input("Faça sua pergunta (q para sair): ").strip()
    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)          # deve devolver texto, não objetos
    answer = chain.invoke({"reviews": reviews, "question": question})
    print("\n" + answer)