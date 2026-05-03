import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.constants import EMBEDDING_MODEL_NAME
from .logger import get_logger

LOGGER = get_logger(__name__)

# This module exposes two retrieval paths:
# - retrieve(): returns only standard IDs (used by inference.py for evaluation).
# - retrieve_for_llm(): returns standard + content + category (used by app.py for RAG).
# Both functions share the same in-memory embedding model, FAISS index, and chunks list.

LOGGER.info("Running retriever...")


def load_model(model_name=EMBEDDING_MODEL_NAME):
    # Embedding model used to vectorize user queries.
    # The name is stored in src/constants.py so it matches embedder.py.
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


# Load once at import time so app.py chat requests reuse the same model + index.
# The index file was created by embedder.py, and chunks.json was created by chunker.py.
model = load_model()
index = faiss.read_index("data/index.faiss")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def retrieve(query, k=5):
    # Basic retrieval used by inference.py; returns only the standard IDs.
    # k = final output size (controlled from inference.py). k=5 is default.
    query = query.lower()

    # 1) Convert query text -> embedding vector using SentenceTransformer.
    q_emb = model.encode([query])
    # 2) Cast to float32 to match FAISS expectations.
    q_emb = np.array(q_emb).astype("float32")

    # 3) FAISS nearest-neighbor search returns index positions.
    #    Those positions directly map to entries in chunks.json.
    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        # Lookup the matched chunk by index and return its standard ID.
        # This method prevents hallucinations.
        results.append(data[idx]["standard"])

    # Remove duplicates while preserving order in case multiple chunks
    # map to the same standard ID.
    results = list(dict.fromkeys(results))

    return results[:k]


def retrieve_for_llm(query, k=5):
    # RAG retrieval for app.py; returns dicts with standard/content/category.
    # This provides the LLM with both identifiers and the actual content text.
    query = query.lower()

    # 1) Convert query text -> embedding vector.
    q_emb = model.encode([query])
    # 2) Cast to float32 to satisfy FAISS.
    q_emb = np.array(q_emb).astype("float32")

    # Fetch extra results initially to ensure we have 'k' items after deduplication.
    # We may drop duplicates by content, so we query a larger pool.
    fetch_k = k * 2
    D, I = index.search(q_emb, fetch_k)

    results = []
    seen_contents = set()

    for idx in I[0]:
        # FAISS returns -1 for empty/missing neighbors
        if idx == -1:
            continue

        # Index position aligns with chunks.json; one lookup yields all fields.
        item = data[idx]
        content = item.get("content")

        # Deduplicate based on the chunk content to avoid repeated context.
        if content not in seen_contents:
            seen_contents.add(content)

            # Append the required dictionary format for app.py.
            results.append(
                {
                    "standard": item.get("standard"),
                    "content": content,
                    "category": item.get("category"),
                }
            )

        # Stop collecting once we hit the desired target size 'k'.
        if len(results) == k:
            break

    return results

    