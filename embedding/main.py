import os
 
from langchain_community.vectorstores import FAISS
 
from config import TOP_K
from index import get_database
 
 
def search(query: str, database: FAISS, k: int = TOP_K) -> list:
    results = database.similarity_search_with_score(query, k=k)
    for rank, (doc, score) in enumerate(results, start=1):
        source = os.path.basename(doc.metadata.get("source", "N/A"))
        print(f"Rank {rank} | score = {score:.3f} | source = {source}")
        print(doc.page_content, "\n")
    return results
 
 
def main():
    database = get_database()
 
    while True:
        query = input("Enter your query (type stop to exit): ").strip()
        if query.lower() == "stop":
            break
        if query == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        search(query, database)
if __name__ == "__main__":
    main()