import os

CONFIG = {
	"chunk_size" : 128,
	"chunk_overlap": 20,
	"model_name": "BAAI/bge-m3",
	"normalize_embeddings": True,
	"glob": "**/*.txt",
	"length_unit": "tokens"
}

LLM_MODEL = "qwen2.5:7b" 


TOP_K = 3 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "corpus_file"))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
TESTSET_PATH = os.path.normpath(
    os.path.join(BASE_DIR, "..", "corpus_file", "test_set", "testset.json")
)