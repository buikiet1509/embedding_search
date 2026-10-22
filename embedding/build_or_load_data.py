from config import CONFIG
from embedding_model import embedding 
from load_and_split_text import TextProcessor
from index import vector_index

from langchain_community.vectorstores import FAISS

embeddings = embedding.embeddings()

class build_load_data:
    def __init__(self,CORPUS_DIR, FAISS_INDEX_PATH, text_processor):
        self.CORPUS_DIR = CORPUS_DIR
        self.FAISS_INDEX_PATH = FAISS_INDEX_PATH
        self.text_processor = text_processor
        self.embeddings = embeddings
    if _index_is_stale(CORPUS_DIR, FAISS_INDEX_PATH):
        print("Corpus changed or index missing -> rebuilding FAISS index...")
        data = text_processor.load_and_split_text()
        database = vector_index(data, embeddings)
    else:
        database = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, 
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            allow_dangerous_deserialization=True
        )  