import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.constants import EMBEDDING_MODEL_NAME
from .logger import get_logger

LOGGER = get_logger(__name__)

LOGGER.info("Running retriever...")


def load_model(model_name=EMBEDDING_MODEL_NAME):
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


# load everything once
model = load_model()
index = faiss.read_index("data/index.faiss")

with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def retrieve(query, k=5):
    # k = final output size (controlled from inference.py)
    query = query.lower()

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    D, I = index.search(q_emb, k)

    results = []
    for idx in I[0]:
        results.append(data[idx]["standard"])

    # Remove duplicates
    results = list(dict.fromkeys(results))

    return results[:k]


def retrieve_for_llm(query, k=5):
    query = query.lower()

    q_emb = model.encode([query])
    q_emb = np.array(q_emb).astype("float32")

    # Fetch extra results initially to ensure we have 'k' items after deduplication
    fetch_k = k * 2
    D, I = index.search(q_emb, fetch_k)

    results = []
    seen_contents = set()

    for idx in I[0]:
        # FAISS returns -1 for empty/missing neighbors
        if idx == -1:
            continue

        item = data[idx]
        content = item.get("content")

        # Deduplicate based on the chunk content
        if content not in seen_contents:
            seen_contents.add(content)

            # Append the required dictionary format
            results.append(
                {
                    "standard": item.get("standard"),
                    "content": content,
                    "category": item.get("category"),
                }
            )

        # Stop collecting once we hit the desired target size 'k'
        if len(results) == k:
            break

    return results

    