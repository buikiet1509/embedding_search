from config import CONFIG
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name = CONFIG["model_name"],
        encode_kwargs = {"normalize_embeddings": CONFIG["normalize_embeddings"]}
    )
