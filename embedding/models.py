"""Retrieval models: dense embeddings, sparse BM25, and the hybrid retriever that fuses them."""
import re
import unicodedata

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import CANDIDATE_K, CONFIG, RRF_K, TOP_K


# ---- Dense: embedding model -------------------------------------------------

def get_embeddings() -> HuggingFaceEmbeddings:
    """BAAI/bge-m3 with normalized vectors, so FAISS inner product = cosine."""
    return HuggingFaceEmbeddings(
        model_name=CONFIG["model_name"],
        encode_kwargs={"normalize_embeddings": CONFIG["normalize_embeddings"]},
    )


# ---- Sparse: BM25 model -----------------------------------------------------

def bm25_tokenize(text: str) -> list[str]:
    """NFC + lowercase + word tokens, so 'Việt Nam,' and 'việt nam' match."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())


def get_bm25retriever(docs: list[Document], k: int = TOP_K) -> BM25Retriever:
    """Keyword retriever over the given chunks (pass the same chunks FAISS indexed)."""
    return BM25Retriever.from_documents(docs, k=k, preprocess_func=bm25_tokenize)


# ---- Hybrid: FAISS + BM25 fused with Reciprocal Rank Fusion -----------------

def doc_key(doc: Document) -> tuple:
    return (doc.metadata.get("source"), doc.page_content)


class HybridRetriever:
    """Dense (FAISS) + sparse (BM25) retrieval, merged with Reciprocal Rank Fusion."""

    def __init__(self, database: FAISS):
        self.database = database
        chunks = list(database.docstore._dict.values())   # same chunks FAISS indexed, no re-split
        self.bm25 = get_bm25retriever(chunks, k=CANDIDATE_K)

    def dense_search(self, query: str, k: int = TOP_K) -> tuple[list[Document], float]:
        """FAISS only. Returns (docs, top-1 inner-product score)."""
        results = self.database.similarity_search_with_score(query, k=k)
        top_score = float(results[0][1]) if results else 0.0
        return [doc for doc, _ in results], top_score

    def sparse_search(self, query: str, k: int = TOP_K) -> list[Document]:
        """BM25 only."""
        self.bm25.k = k
        return self.bm25.invoke(query)

    def search(self, query: str, k: int = TOP_K) -> tuple[list[Document], float]:
        """FAISS + BM25 fused by RRF. The score returned is still the dense top-1,
        because BM25 scores are not on a fixed scale and can't share SCORE_THRESHOLD."""
        candidates = max(k, CANDIDATE_K)
        dense_docs, top_score = self.dense_search(query, candidates)
        sparse_docs = self.sparse_search(query, candidates)

        scores, docs = {}, {}
        for ranked in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranked, start=1):
                key = doc_key(doc)
                docs[key] = doc
                scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)

        fused = sorted(scores, key=scores.get, reverse=True)[:k]
        return [docs[key] for key in fused], top_score


def get_retriever() -> HybridRetriever:
    """Single entry point for main.py, rag_chain.py and evaluation.py."""
    from index import get_database   # imported here: index.py imports get_embeddings from this file
    return HybridRetriever(get_database())
