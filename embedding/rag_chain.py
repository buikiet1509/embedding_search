import os 

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import TOP_K, LLM_MODEL

SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on the provided documents.\n"
    "Rules:\n"
    "1. ONLY use information from the CONTEXT section. Do not use your own knowledge.\n"
    "2. If the context does not contain the answer, reply with exactly one sentence: "
    "\"No information found in the documents.\"\n"
    "3. After each point, cite the source file name in square brackets, e.g. [datafile.txt].\n"
    "4. Keep answers concise, in the same language as the question."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
])

LLM = ChatOllama(model=LLM_MODEL, temperature=0)

def format_docs(docs: str) -> str:
    part=[]
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        part.append(f"[{source}] {doc.page_content}")

    return "\n\n".join(part)

def answer_question(query: str, database) -> dict:
    docs = database.similarity_search(query, k=TOP_K)          # 1. retrieve
    context = format_docs(docs)                                # 2. build context
    messages = PROMPT.invoke({"context": context,              # 3. fill prompt
                              "question": query})
    response = LLM.invoke(messages)                            # 4. call LLM
    return {"answer": response.content}

