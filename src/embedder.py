import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .constants import EMBEDDING_MODEL_NAME
from .logger import get_logger

LOGGER = get_logger(__name__)

# This file builds the vector index that retrieval depends on.
# End-to-end flow:
# 1) Load the embedding model (SentenceTransformer).
# 2) Read chunk metadata from data/chunks.json (output of chunker.py).
# 3) Convert each chunk into a single text string for embedding.
# 4) Encode all strings into vectors and build a FAISS index.
# 5) Save the index to data/index.faiss for retriever.py to load.

LOGGER.info("Running embedder...")


def load_model(model_name=EMBEDDING_MODEL_NAME):
    # Load the embedding model by name from constants.
    # Try local cache first to avoid re-downloading large model files.
    try:
        LOGGER.info("Trying to find model locally...")
        model = SentenceTransformer(model_name, local_files_only=True)
        LOGGER.info("Loaded model from local cache.")
        return model

    except Exception:
        LOGGER.info("Model not found locally. Downloading...")
        try:
            model = SentenceTransformer(model_name)
            LOGGER.info("Model downloaded and cached.")
            return model
        except Exception as download_error:
            LOGGER.error("Error occurred while downloading model: %s", str(download_error))
            raise
    
# Load the model once at import time so repeated calls reuse it.
model = load_model()

# Load chunk metadata produced by chunker.py.
# Each entry in data is a dict with fields like: standard, content, category.
with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Combine category + content so the vector captures both topical label and body text.
# This improves retrieval because section headings are often the best signal.
texts = []
for item in data:
    content = item.get("content", "")
    category = item.get("category", "")
    # Concatenate both columns; if category exists, prepend it.
    # The resulting string is what gets embedded into a vector.
    combined = f"{category} {content}".strip() if category else content
    texts.append(combined)

LOGGER.info("Encoding embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# FAISS expects float32 vectors for efficient similarity search.
embeddings = np.array(embeddings).astype("float32")

# Build a flat (exact) L2 distance index and add all vectors.
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save the index to disk; retriever.py loads this file to serve queries.
faiss.write_index(index, "data/index.faiss")

LOGGER.info("Embeddings and index saved to data/index.faiss")