import os

from index import get_database
from rag_chain import answer_question 

def main(DEBUG=True):
    database = get_database()
 
    while True:
        query = input("Enter your query (type stop to exit): ").strip()
        if query.lower() == "stop":
            break
        if query == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        answer = answer_question(query, database)
        print(answer["answer"])
if __name__ == "__main__":
    main()