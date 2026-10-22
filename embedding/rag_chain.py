import os 

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import TOP_K, LLM_MODEL

SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu.\n"
    "Quy tắc:\n"
    "1. CHỈ dùng thông tin trong phần NGỮ CẢNH. Không dùng kiến thức riêng.\n"
    "2. Nếu ngữ cảnh không chứa câu trả lời, trả lời đúng một câu: "
    "\"Không tìm thấy thông tin trong tài liệu.\"\n"
    "3. Sau mỗi ý, ghi nguồn bằng tên file trong ngoặc vuông, ví dụ [datafile.txt].\n"
    "4. Trả lời ngắn gọn, bằng ngôn ngữ của câu hỏi."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}"),
])

LLM = ChatOllama(model=LLM_MODEL, temperature=0)

def format_docs(docs: str) -> str:
    part=[]
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        part.append(f"[{source}] {doc.page_content}")

    return "\n\n".join(part)

def answer_question(query: str, database) -> dict:
    docs = database.similarity_search(query, k=TOP_K)          # 1. tìm kiếm
    context = format_docs(docs)                                # 2. ghép ngữ cảnh
    messages = PROMPT.invoke({"context": context,              # 3. điền prompt
                              "question": query})
    response = LLM.invoke(messages)                            # 4. gọi LLM
    return {"answer": response.content}

