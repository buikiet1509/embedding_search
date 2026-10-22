import os
import json

from config import CONFIG, CORPUS_DIR, FAISS_INDEX_PATH
from embedding_model import get_embeddings
from load_and_split_text import TextProcessor

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

def index_is_stale(corpus_dir: str, index_path: str) -> bool:
    """True nếu cần build lại index."""
    index_file = os.path.join(index_path, "index.faiss")
    config_file = os.path.join(index_path, "config.json")
 
    if not os.path.exists(index_file):
        return True
    if not os.path.exists(config_file):
        return True
    with open(config_file, "r", encoding="utf-8") as file:
        saved_config = json.load(file)
    if saved_config != CONFIG:
        return True
 
    index_mtime = os.path.getmtime(index_file)
    for root, _, files in os.walk(corpus_dir):
        if os.path.getmtime(root) > index_mtime:
            return True
        for name in files:
            if name.endswith(".txt") and os.path.getmtime(os.path.join(root, name)) > index_mtime:
                return True
    return False
 
 
def build_index(chunks: list, embeddings, index_path: str) -> FAISS:
    """Chunk -> vector -> FAISS, lưu xuống đĩa và dán nhãn config.json."""
    database = FAISS.from_documents(
        chunks,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )
    database.save_local(index_path)
 
    config_file = os.path.join(index_path, "config.json")
    with open(config_file, "w", encoding="utf-8") as file:
        json.dump(CONFIG, file, ensure_ascii=False, indent=2)
 
    return database
 
 
def load_index(embeddings, index_path: str) -> FAISS:
    return FAISS.load_local(
        index_path,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        allow_dangerous_deserialization=True,
    )
 
 
def get_database(corpus_dir: str = CORPUS_DIR, index_path: str = FAISS_INDEX_PATH) -> FAISS:
    """Điểm vào duy nhất: build nếu cần, ngược lại load. main.py và evaluation.py đều gọi hàm này."""
    embeddings = get_embeddings()
 
    if index_is_stale(corpus_dir, index_path):
        print("Corpus/config thay đổi hoặc chưa có index -> build lại FAISS index...")
        processor = TextProcessor(corpus_dir)   # chỉ tạo khi thật sự cần build
        chunks = processor.load_and_split_text()
        return build_index(chunks, embeddings, index_path)
 
    return load_index(embeddings, index_path)