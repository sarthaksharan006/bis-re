# BIS Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI%2FML-RAG%20LLM-green)

This project is a retrieval-based BIS recommendation system built for a hackathon. It turns BIS source material into searchable chunks, embeds them with a sentence-transformer model, and uses FAISS to retrieve the most relevant standards for a query.

The repository also includes a Gradio-based RAG chatbot that retrieves supporting chunks and feeds them into a local GGUF LLM for answer generation.

## Features 

- **Bulk Inference Evaluation**: Run large-scale retrieval tests with latency measurement and standardized metrics (Hit Rate @3, MRR @5).
- **Interactive RAG Chatbot**: User-friendly Gradio interface for real-time queries, with retrieved standards displayed alongside LLM-generated answers.
- **Hallucination-Free Retrieval**: Queries are matched deterministically against embeddings; standard IDs are fetched directly from JSON. The LLM is used only for answer generation, never for retrieval, eliminating the risk of hallucinated standards.

## Installation

1. Clone the repository and navigate to the project:
```bash
git clone https://github.com/sarthaksharan006/bis-re.git
cd bis-re
```

2. Use Python 3.13 (`requires-python = ">=3.13,<3.14"`).

3. Install dependencies:
   - **uv method (preferred)**:
     - Install uv with `pip install uv`.
     - Run `uv sync` to install all dependencies and set up a virtual environment.
   - **pip method**:
     - Runtime dependencies are listed in `requirements.txt`.
     - Run `pip install -r requirements.txt`.

4. Download models:
```bash
python setup.py
```
    The script downloads the embedding model first, then prompts whether to download the optional GGUF LLM model.

## Quick Start

### Bulk Inference Evaluation

Evaluate retrieval quality across a test set:

```bash
python inference.py --input data/testsets/public_test_set.json --output results.json
python eval_script.py --results results.json
```

This runs retrieval for each query, measures latency, and computes standard hackathon metrics (Hit Rate @3, MRR @5, average latency).

### Interactive Chatbot with LLM

Launch the user-friendly Gradio chat interface:

```bash
python app.py
```

Open the browser link shown in the terminal. The chatbot retrieves relevant BIS standards and uses a local LLM to generate contextual answers.

**Example queries**:
- "What are the regulations for cement production?"
- "Tell me about steel standards in BIS"
- "What are the standards for glass panes?"

Each query triggers retrieval of relevant chunks, displays them in the UI, and generates a grounded LLM response.

## Full Pipeline

**Note**: Chunking and embedding scripts are included for reference and allow you to rebuild the pipeline from the BIS PDF. However, the dataset is pre-processed, so these are not required for evaluation or chatbot use.


### 1. Data Ingestion / Extraction

`src/extract_text.py` expects `data/dataset.pdf`. It extracts page text blocks with PyMuPDF, sorts them top-to-bottom and left-to-right, then cleans the text by:

- trimming content to the first useful `SECTION 1` marker, or the second occurrence if the PDF repeats headers,
- removing the known boilerplate string `SP  21 : 2005`,
- writing the cleaned output to `data/raw.txt`.

### 2. Chunking

`src/chunker.py` reads `data/raw.txt` and builds chunk records from `SUMMARY OF` blocks. For each block it:

- extracts a canonical standard ID such as `IS 456: 2000` or `IS 456 (Part 2): 2015`,
- derives a title and scope using heuristics around `Scope`, numbered bullets, and block boundaries,
- removes revision notes and excess whitespace,
- stores the result as `standard` and `content`.

The same script also builds a category map by scanning section headings and uppercase subsection titles, then attaches the resulting `category` to matching standards. Its output is `data/generated_chunks.json`.

### 3. Filtering

`src/filter_chunks.py` is a lightweight manual-quality-control helper. It groups:

- chunks whose `content` is shorter than 10 words,
- chunks missing a `category` field.

The script writes both groups to `data/filtered_chunks.json` and explicitly notes that manual editing is needed before renaming the corrected file to `chunks.json`.

### 4. Embedding Generation

`src/embedder.py` reads `data/chunks.json`, concatenates `category` and `content` for each entry when a category exists, and embeds the resulting text with `SentenceTransformer(EMBEDDING_MODEL_NAME)`.

The embeddings are converted to `float32`, inserted into a `faiss.IndexFlatL2`, and saved to `data/index.faiss`.

#### Indexing

Indexing is exact L2 nearest-neighbor search over the full embedded chunk set. There is no approximate index or reranking layer in the code.

### 5. Retrieval Logic

`src/retriever.py` loads the embedding model, FAISS index, and `data/chunks.json` at import time.

- `retrieve(query, k=5)` is used by `inference.py`.
- It lowercases the query, embeds it, searches FAISS, and returns only the matched standard IDs.
- It removes duplicate standard IDs while preserving order, then truncates to `k` number of results.
- `retrieve_for_llm(query, k=5)` is used by `app.py`.
- It searches for neighbors, ignores empty FAISS results, deduplicates by chunk `content`, and returns dictionaries containing `standard`, `content`, and `category`.

### 6. Inference / Output Formatting

`inference.py` is the evaluation entry point. It expects a JSON list of query objects, each with at least:

- `id`
- `query`
- `expected_standards` optional

For each item, it records the retrieved standards and the measured latency, then writes the final list as pretty-printed JSON.

### 7. Generation (RAG LLM)

`src/generator.py` loads a GGUF quantized model (`LiquidAI/LFM2-350M-GGUF`) using `llama_cpp`. The model is loaded once at import time for reuse across multiple chat requests.

- It accepts a user prompt and an optional system prompt from `src/constants.py`.
- Prompts are formatted in ChatML structure (`<|startoftext|>`, `<|im_start|>`, etc.).
- The LLM generates responses grounded in retrieved BIS standard chunks.
- Used by `app.py` for generating chat answers based on user queries and retrieved context.
- Supports configurable context window size (8000 tokens) and multi-threaded inference (4 threads).

## Project Structure

- `inference.py` - bulk evaluation script that runs retrieval for each input query and writes JSON results.
- `app.py` - Gradio chatbot UI that retrieves supporting chunks and generates answers with the local LLM.
- `eval_script.py` - offline scoring script for result files; computes Hit Rate @3, MRR @5, and average latency.
- `setup.py` - utility script that downloads the embedding model and optionally downloads the GGUF LLM.
- `src/extract_text.py` - extracts and cleans text from the BIS PDF into `data/raw.txt`.
- `src/chunker.py` - converts raw text into structured standards chunks and attaches section categories.
- `src/filter_chunks.py` - flags short chunks and chunks missing categories for manual cleanup.
- `src/embedder.py` - embeds chunks and builds the FAISS index saved in `data/index.faiss`.
- `src/retriever.py` - loads the model, index, and chunks; exposes retrieval functions for evaluation and chat.
- `src/generator.py` - loads the GGUF model with `llama_cpp` and formats prompts for generation.
- `src/constants.py` - central place for the embedding model name, GGUF model name, filename, and system prompt.
- `src/logger.py` - shared logger configuration with console formatting and a file log at `logs/bis-re.log`.
- `requirements.txt` - pinned runtime dependency list.
- `pyproject.toml` - project metadata, Python version constraint, and dependency declarations.

## Models / Libraries Used

- `sentence-transformers` - query and chunk embeddings with `BAAI/bge-base-en-v1.5`.
- `faiss-cpu` - vector similarity search over the embedded chunk set.
- `pymupdf` (`fitz`) - PDF text extraction.
- `llama-cpp-python` - local GGUF model loading and text generation.
- `huggingface-hub` - downloading the optional GGUF model file.
- `gradio` - browser-based chat interface.
- `numpy` - vector conversion before FAISS indexing and search.
- `tqdm` - progress bar during bulk inference.
- `torch` - dependency required by the sentence-transformer stack.

## Notes / Implementation Details

- **Hallucination-Free Architecture**: Retrieval is completely deterministic. The embedding model maps queries to vectors, FAISS finds the nearest neighbors, and standard IDs are looked up directly in `data/chunks.json`. No LLM is involved in this process, so retrieved standards are always factually accurate—never hallucinated.
- The embedding model loader and the retriever both try a local cache first, then download if needed.
- `src/generator.py` uses a manually formatted ChatML-style prompt and a fixed system prompt from `src/constants.py`.
- `app.py` keeps the retrieved chunks visible in the UI and also feeds the same context into the LLM prompt.
- Retrieval is deterministic and based on exact FAISS L2 search; there is no reranking or semantic post-processing beyond deduplication.
- The pipeline depends on prebuilt artifacts being present in `data/chunks.json` and `data/index.faiss`.
- `setup.py` does not build the index or process the PDF; it only downloads models.
