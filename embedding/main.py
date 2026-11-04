
from rag_chain import answer_question
from models import get_retriever


def main(DEBUG=True):
    retriever = get_retriever()
 
    while True:
        query = input("Enter your query (type stop to exit): ").strip()
        if query.lower() == "stop":
            break
        if query == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        answer = answer_question(query, retriever)
        print(answer["answer"])
if __name__ == "__main__":
    main()