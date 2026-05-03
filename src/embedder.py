import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from .constants import EMBEDDING_MODEL_NAME
from .logger import get_logger

LOGGER = get_logger(__name__)

LOGGER.info("Running embedder...")


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
    
model = load_model()

# load your chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Combine the category and content so the vector captures both.
texts = []
for item in data:
    content = item.get("content", "")
    category = item.get("category", "")
    # concatenate both columns; if category exists, prepend it
    combined = f"{category} {content}".strip() if category else content
    texts.append(combined)

LOGGER.info("Encoding embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# save index + metadata
faiss.write_index(index, "data/index.faiss")

LOGGER.info("Embeddings and index saved to data/index.faiss")