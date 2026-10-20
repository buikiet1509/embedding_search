from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import CONFIG

class TextProcessor:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
        self.loader = DirectoryLoader(
            corpus_dir,
            glob=CONFIG["glob"],
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )
        self.chunker = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"]
        )

    def load_and_split_text(self) -> list:
        docs = self.loader.load()
        chunks = self.chunker.split_documents(docs)
        return chunks
