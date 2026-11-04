import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import LLM_MODEL, SCORE_THRESHOLD, TOP_K
from models import HybridRetriever

REFUSAL = "No information found in the documents."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions based on the provided documents.\n"
    "Rules:\n"
    "1. ONLY use information from the CONTEXT section. Do not use your own knowledge.\n"
    "2. If the context does not contain the answer, reply with exactly one sentence: "
    f"\"{REFUSAL}\"\n"
    "3. After each point, cite the source file name in square brackets, e.g. [datafile.txt].\n"
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "CONTEXT:\n{context}\n\nQUESTION: {question}"),
])

LLM = ChatOllama(model=LLM_MODEL, temperature=0)

def format_docs(docs: str) -> list:
    part=[]
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        part.append(f"[{source}] {doc.page_content}")

    return "\n\n".join(part)

def answer_question(query: str, retriever: HybridRetriever) -> dict:
    docs, top_score = retriever.search(query, k=TOP_K)        # 1. retrieve (FAISS + BM25)

    if top_score < SCORE_THRESHOLD:
        return {"answer": REFUSAL, "docs": docs,
                "top_score": top_score, "route": "refused"}

    context = format_docs(docs)                                # 2. build context
    messages = PROMPT.invoke({"context": context,              # 3. fill prompt
                              "question": query})
    response = LLM.invoke(messages)                            # 4. call LLM
    return {"answer": response.content, "docs": docs,
            "top_score": top_score, "route": "answered"}

