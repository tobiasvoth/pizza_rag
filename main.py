from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an exeprt in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    docs = retriever.invoke(question)
    
    context_text = ""
    for i, doc in enumerate(docs):
        context_text += f"\n[Rezension #{i+1}]:\nDatum: {doc.metadata.get('date')}\nBewertung: {doc.metadata.get('rating')} Sterne\nInhalt: {doc.page_content}\n"


    result = chain.invoke({"reviews": context_text, "question": question})
    
    print("\n--- ANTWORT DES EXPERTEN ---")
    print(result)
    
    print("\n--- VERWENDETE QUELLEN (METADATEN) ---")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] Datum: {doc.metadata.get('date')} | Rating: {doc.metadata.get('rating')}/5")