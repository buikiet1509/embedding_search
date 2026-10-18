import os

from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(BASE_DIR, "..", "corpus_file")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")

def load_and_split_text(file_path: str) -> list:
    loader = DirectoryLoader(
        file_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50
    )
    docs = loader.load()
    chunks = chunker.split_documents(docs)
    return chunks

def _search(query: str, database: FAISS) -> list[tuple]:
    results = database.similarity_search_with_score(query, k=3)
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"Rank {rank}:")
        print(f"Score: {score}")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content: {doc.page_content}\n")

    return results

def _index_is_stale(corpus_dir: str, index_path: str) -> bool:  # [FIX #1] new helper: decide whether the saved index must be rebuilt
    index_file = os.path.join(index_path, "index.faiss")  # [FIX #1] the file whose timestamp marks when the index was built
    if not os.path.exists(index_file):  # [FIX #1] no index yet -> must build
        return True  # [FIX #1]
    index_mtime = os.path.getmtime(index_file)  # [FIX #1] time the index was last saved
    for root, _, files in os.walk(corpus_dir):  # [FIX #1] scan corpus folder and subfolders (same scope as glob="**/*.txt")
        if os.path.getmtime(root) > index_mtime:  # [FIX #1] folder mtime changes when a file is added/removed/renamed
            return True  # [FIX #1]
        for name in files:  # [FIX #1]
            if name.endswith(".txt") and os.path.getmtime(os.path.join(root, name)) > index_mtime:  # [FIX #1] a .txt edited after the index was built
                return True  # [FIX #1]
    return False  # [FIX #1] nothing changed since the index was built -> safe to load

def _index(data: list, embeddings: HuggingFaceEmbeddings) -> FAISS:
    database = FAISS.from_documents(
        data,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    database.save_local(FAISS_INDEX_PATH)
    return database

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", 
        encode_kwargs={"normalize_embeddings": True}
    )
    if _index_is_stale(CORPUS_DIR, FAISS_INDEX_PATH):  # [FIX #1] was `not os.path.exists(FAISS_INDEX_PATH)`, which never rebuilt after corpus changes
        print("Corpus changed or index missing -> rebuilding FAISS index...")  # [FIX #1] tell the user a rebuild is happening
        data = load_and_split_text(CORPUS_DIR)
        database = _index(data, embeddings)
    else:
        database = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, 
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            allow_dangerous_deserialization=True
        )
    while(True):
        query = input("Enter your query(type stop to exit): ").strip()
        if query.lower() == "stop":
            break
        if query == "":
            print("Query cannot be empty. Please enter a valid query.")
            continue
        _search(query, database)

if __name__ == "__main__":
    main()