# Embedding Search

A small semantic search engine over local text files. It splits documents into chunks, embeds them with [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (multilingual, so English and Vietnamese both work), stores the vectors in a FAISS index, and returns the top-k most relevant chunks for a query.

## How it works

1. **Load & split**: every `.txt` file in `corpus_file/` is loaded and split into 256-token chunks with a 20-token overlap, using the bge-m3 tokenizer.
2. **Embed**: chunks are encoded with normalized bge-m3 embeddings.
3. **Index**: vectors go into a FAISS index (inner-product similarity), which is saved to `embedding/faiss_index/`.
4. **Search**: a query is embedded the same way, and the top-k chunks are printed with their score and source file.

The index is rebuilt automatically when it is missing, when the settings in `config.py` change, or when any corpus file is newer than the index. Otherwise the saved index is loaded from disk.

## Project structure

```
corpus_file/
  datafile.txt            # sample corpus (English)
  vietnam_data.txt        # sample corpus (Vietnamese)
  test_set/testset.json   # labeled queries for evaluation
embedding/
  config.py               # chunking, model and path settings
  load_and_split_text.py  # loads and chunks the corpus
  embedding_model.py      # HuggingFace embedding model
  index.py                # build / load / staleness check for the FAISS index
  main.py                 # interactive search CLI
  evaluation.py           # retrieval evaluation (work in progress)
```

## Getting started

Requires Python 3.10+.

```bash
pip install langchain-community langchain-huggingface langchain-text-splitters \
            transformers sentence-transformers faiss-cpu
```

Run the search:

```bash
cd embedding
python main.py
```

```
Enter your query (type stop to exit): When did Mississippi become a state?
Rank 1 | score = <similarity> | source = datafile.txt
<matching chunk text>
```

The first run downloads the bge-m3 model and builds the index, so it takes a while.

## Configuration

Edit `embedding/config.py`:

| Setting | Default | Description |
|---|---|---|
| `chunk_size` | `256` | Chunk length in tokens |
| `chunk_overlap` | `20` | Overlap between chunks |
| `model_name` | `BAAI/bge-m3` | HuggingFace embedding model |
| `TOP_K` | `3` | Number of results per query |

To search your own documents, drop `.txt` files into `corpus_file/` and the index will rebuild on the next run.

## Evaluation

`corpus_file/test_set/testset.json` contains questions labeled as answerable or unanswerable, with their expected source files and evidence. `embedding/evaluation.py` loads this set to measure retrieval quality; it is still in progress.
