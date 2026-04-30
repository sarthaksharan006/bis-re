import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_model(model_name="all-MiniLM-L6-v2"):
    try:
        print("Trying to load model locally...")
        model = SentenceTransformer(model_name, local_files_only=True)
        print("Loaded model from local cache")
        return model

    except Exception as e:
        print("Model not found locally. Downloading...")
        model = SentenceTransformer(model_name)
        print("Model downloaded and cached")
        return model
    
model = load_model()

# load your chunks
with open("data/chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["content"] for item in data]

print("Encoding embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

# build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# save index + metadata
faiss.write_index(index, "data/index.faiss")

print("Embeddings + index saved")